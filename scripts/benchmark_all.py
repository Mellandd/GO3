"""Multi-library benchmark orchestrator.

Runs three batteries (loading, term-pair, gene-pair) across every
available runner and emits comparable JSON + plots.

Term-pair mode is **closed term sets**: the requested "size" is a target
pair count translated to N terms such that C(N,2) >= size, then every
runner sees the SAME N-term workload. This is the only way to feed
TaxaGO (which produces an N×N matrix) a comparable workload.

Discovery: every runner's `is_available()` decides whether it participates.
Use `--only` / `--exclude` to override.

Examples:

    # Default: all available libs, default profile.
    python scripts/benchmark_all.py

    # Reproduce the paper-ready profile.
    python scripts/benchmark_all.py --paper-ready

    # Only Python libs.
    python scripts/benchmark_all.py --only go3,goatools,fastsemsim,pygosemsim

    # Skip the heaviest libs.
    python scripts/benchmark_all.py --exclude pygosemsim,simona
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Make the local `runners` package importable when running this script
# directly (`python scripts/benchmark_all.py`).
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from runners import available_runners  # noqa: E402
from runners._base import VALID_METHODS, RunResult, bootstrap_ci_median  # noqa: E402
from runners._gaf import (  # noqa: E402
    NAMESPACE_TO_ASPECT,
    parse_symbol_to_terms,
)

DEFAULT_TERM_PAIR_SIZES = [100, 1000, 5000, 10000]
DEFAULT_GENE_PAIR_SIZES = [25, 50, 100, 200]
DEFAULT_MATRIX_GENE_SIZES = [8, 12, 16]
PAPER_TERM_PAIR_SIZES = [100, 500, 1000, 2500, 5000]
PAPER_GENE_PAIR_SIZES = [10, 25, 50, 75, 100]
PAPER_MATRIX_GENE_SIZES = [6, 8, 10, 12, 14, 16]
PAPER_WARMUP = 2
PAPER_REPEATS = 5
PAPER_LOADING_REPEATS = 5
PAPER_THREADS = 8
PAPER_MIN_GENE_TERMS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_int_list(csv: str) -> list[int]:
    return [int(p.strip()) for p in csv.split(",") if p.strip()]


def n_choose_two(n: int) -> int:
    return (n * (n - 1)) // 2


def n_for_pair_target(target: int) -> int:
    """Smallest n such that C(n,2) >= target."""
    n = 2
    while n_choose_two(n) < target:
        n += 1
    return n


def all_unique_pairs(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            out.append((a, items[j]))
    return out


def default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent
    return root / "tests" / "go-basic.obo", root / "tests" / "goa_human.gaf"


def collect_system_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count_logical": os.cpu_count(),
        "python_version": sys.version,
    }


def parse_obo_version(obo_path: Path) -> dict[str, str]:
    """Read the OBO header (lines before the first `[Term]`) and pull out
    the fields that identify the release. Cheap — stops at first [Term]."""
    info: dict[str, str] = {}
    keys = {"format-version", "data-version", "ontology", "date",
            "default-namespace"}
    try:
        with open(obo_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("["):
                    break
                if ":" not in stripped:
                    continue
                key, _, value = stripped.partition(":")
                key = key.strip()
                if key in keys:
                    info[key] = value.strip()
    except OSError:
        pass
    return info


def parse_gaf_version(gaf_path: Path) -> dict[str, str]:
    """Read the GAF header (leading `!`-comment block) for version, date,
    generator, and source URL."""
    info: dict[str, str] = {}
    try:
        with open(gaf_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("!"):
                    break
                stripped = line.lstrip("!").strip()
                if not stripped or ":" not in stripped:
                    continue
                key, _, value = stripped.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key in {"gaf-version", "date-generated", "generated-by",
                          "url", "funding-source"}:
                    info[key] = value
    except OSError:
        pass
    return info


def speedup(faster_med: float, slower_med: float) -> float:
    if faster_med <= 0:
        return float("inf")
    return slower_med / faster_med


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_term_sets_for_sizes(
    candidate_terms: list[str],
    pair_targets: list[int],
    rng: random.Random,
) -> dict[int, list[str]]:
    """For each target pair count, pick a closed term set of size N such
    that C(N,2) >= target. Sets are independent samples from `candidate_terms`."""
    if not candidate_terms:
        raise RuntimeError("No candidate GO terms available for sampling.")
    out: dict[int, list[str]] = {}
    for target in pair_targets:
        n = n_for_pair_target(target)
        if n > len(candidate_terms):
            raise RuntimeError(
                f"Need {n} terms for {target} pairs, but only {len(candidate_terms)} candidates."
            )
        out[target] = rng.sample(candidate_terms, n)
    return out


def term_pairs_from_sets(term_sets: dict[int, list[str]]) -> dict[int, list[tuple[str, str]]]:
    """Translate closed term sets to all-vs-all pair lists.

    Returned key is the actual pair count (C(N,2)), not the requested target,
    so plots show the work that was actually done.
    """
    out: dict[int, list[tuple[str, str]]] = {}
    for _target, terms in term_sets.items():
        pairs = all_unique_pairs(terms)
        out[len(pairs)] = pairs
    return out


def sample_disjoint_pair_groups(
    items: list[str], sizes: list[int], rng: random.Random
) -> dict[int, list[tuple[str, str]]]:
    if not items:
        raise RuntimeError("No items available for pair sampling.")
    total = sum(sizes)
    if total > n_choose_two(len(items)):
        raise RuntimeError(
            f"Need {total} pairs but only {n_choose_two(len(items))} unique pairs are possible "
            f"from {len(items)} items."
        )
    seen: set[tuple[str, str]] = set()
    while len(seen) < total:
        a, b = rng.sample(items, 2)
        if a > b:
            a, b = b, a
        seen.add((a, b))
    pairs = list(seen)
    rng.shuffle(pairs)
    out: dict[int, list[tuple[str, str]]] = {}
    idx = 0
    for size in sizes:
        out[size] = pairs[idx: idx + size]
        idx += size
    return out


def select_gene_candidates(
    gene2terms: dict[str, list[str]],
    *,
    needed_pairs: int,
    min_gene_terms: int,
) -> tuple[list[str], dict[str, Any]]:
    ranked = sorted(gene2terms.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    needed = n_for_pair_target(needed_pairs)
    for threshold in range(max(1, min_gene_terms), 0, -1):
        eligible = [g for g, ts in ranked if len(ts) >= threshold]
        if len(eligible) < needed:
            continue
        target = needed_pairs * 2
        sel: list[str] = []
        for g in eligible:
            sel.append(g)
            if len(sel) >= needed and n_choose_two(len(sel)) >= target:
                break
        return sel, {
            "min_gene_terms_requested": min_gene_terms,
            "min_gene_terms_used": threshold,
            "candidate_count": len(sel),
            "eligible_before_cap": len(eligible),
        }
    raise RuntimeError(
        f"Not enough genes for {needed_pairs} unique pairs at any term-count threshold."
    )


def select_term_candidates_for_namespace(
    obo: Path, gaf: Path, namespace: str
) -> list[str]:
    """Return GO IDs annotated in `gaf` for the chosen namespace, ordered."""
    aspect = NAMESPACE_TO_ASPECT[namespace]
    gene2terms = parse_symbol_to_terms(gaf, namespace_aspect=aspect)
    seen: set[str] = set()
    for terms in gene2terms.values():
        seen.update(terms)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

LIB_STYLES: dict[str, dict[str, Any]] = {
    "go3":         {"label": "GO3",         "marker": "o", "linestyle": "-",  "color": "#1b9e77"},
    "goatools":    {"label": "GOATOOLS",     "marker": "s", "linestyle": "--", "color": "#d95f02"},
    "gosemsim":    {"label": "GOSemSim",    "marker": "^", "linestyle": ":",  "color": "#7570b3"},
    "fastsemsim":  {"label": "FastSemSim",  "marker": "D", "linestyle": "-.", "color": "#e7298a"},
    "pygosemsim":  {"label": "pygosemsim",  "marker": "v", "linestyle": "-",  "color": "#66a61e"},
    "simona":      {"label": "simona",      "marker": "P", "linestyle": "--", "color": "#a6761d"},
    "taxago":      {"label": "TaxaGO",      "marker": "*", "linestyle": "-",  "color": "#1f78b4"},
}


def paper_rc_params(paper_ready: bool) -> dict[str, Any]:
    if not paper_ready:
        return {}
    return {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }


def save_figure(fig, out_path: Path, paper_ready: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = 320 if paper_ready else 180
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if paper_ready:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")


def _ci_errs(
    values: list[float], lo: list[float], hi: list[float]
) -> tuple[list[float], list[float]]:
    """Convert parallel (median, ci_low, ci_high) lists into matplotlib's
    two-sided yerr format: [down, up] distances (always non-negative).
    NaN-tolerant — any NaN in a triple becomes a 0 error on that bar."""
    down: list[float] = []
    up: list[float] = []
    for v, l, h in zip(values, lo, hi):
        if math.isnan(v) or math.isnan(l) or math.isnan(h):
            down.append(0.0)
            up.append(0.0)
        else:
            down.append(max(0.0, v - l))
            up.append(max(0.0, h - v))
    return down, up


def plot_loading_summary(out_path: Path, loading: dict[str, dict[str, Any]], paper_ready: bool) -> None:
    # Exclude TaxaGO — it's a standalone binary, not comparable to library init
    loading = {k: v for k, v in loading.items() if k != "taxago"}
    libs = list(loading.keys())
    labels = [LIB_STYLES.get(lib, {}).get("label", loading[lib].get("display_name", lib)) for lib in libs]
    times = [float(loading[lib]["total_time_s"]) for lib in libs]
    time_ci_lo = [float(loading[lib].get("total_time_s_ci_low", float("nan"))) for lib in libs]
    time_ci_hi = [float(loading[lib].get("total_time_s_ci_high", float("nan"))) for lib in libs]
    peaks = [float(loading[lib].get("peak_rss_mb", float("nan"))) for lib in libs]
    peak_ci_lo = [float(loading[lib].get("peak_rss_mb_ci_low", float("nan"))) for lib in libs]
    peak_ci_hi = [float(loading[lib].get("peak_rss_mb_ci_high", float("nan"))) for lib in libs]
    colors = [LIB_STYLES.get(lib, {"color": "#888888"})["color"] for lib in libs]
    n_reps = max((int(loading[lib].get("n_repeats", 1)) for lib in libs), default=1)

    has_mem = any(not math.isnan(p) for p in peaks)
    n_panels = 2 if has_mem else 1
    with plt.rc_context(paper_rc_params(paper_ready)):
        fig, axes = plt.subplots(1, n_panels, figsize=(11.0 if n_panels == 2 else 6.5, 4.8))
        ax_time = axes[0] if n_panels == 2 else axes
        x = list(range(len(libs)))
        t_down, t_up = _ci_errs(times, time_ci_lo, time_ci_hi)
        bars = ax_time.bar(x, times, color=colors, edgecolor="#333", linewidth=0.6,
                           yerr=[t_down, t_up], capsize=4,
                           error_kw={"elinewidth": 1.1, "ecolor": "#222"})
        ax_time.set_title(f"Loading time (median, {n_reps} runs)")
        ax_time.set_xlabel("Library")
        ax_time.set_ylabel("Time (s)")
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(labels, rotation=20, ha="right")
        ax_time.grid(True, axis="y", linestyle="--", alpha=0.35)
        for bar, value, up in zip(bars, times, t_up):
            y = value + up
            ax_time.text(bar.get_x() + bar.get_width() / 2.0, y, f"{value:.2f}",
                         ha="center", va="bottom")

        if has_mem:
            ax_mem = axes[1]
            mem_vals = [0.0 if math.isnan(p) else p for p in peaks]
            m_down, m_up = _ci_errs(mem_vals, peak_ci_lo, peak_ci_hi)
            bars2 = ax_mem.bar(x, mem_vals, color=colors, edgecolor="#333", linewidth=0.6,
                               yerr=[m_down, m_up], capsize=4,
                               error_kw={"elinewidth": 1.1, "ecolor": "#222"})
            ax_mem.set_title(f"Peak memory (median, {n_reps} runs)")
            ax_mem.set_xlabel("Library")
            ax_mem.set_ylabel("Peak RSS (MB)")
            ax_mem.set_xticks(x)
            ax_mem.set_xticklabels(labels, rotation=20, ha="right")
            ax_mem.grid(True, axis="y", linestyle="--", alpha=0.35)
            for bar, value, raw, up in zip(bars2, mem_vals, peaks, m_up):
                label = "n/a" if math.isnan(raw) else f"{value:.1f}"
                ax_mem.text(bar.get_x() + bar.get_width() / 2.0, value + up, label,
                            ha="center", va="bottom")
        fig.tight_layout()
        save_figure(fig, out_path, paper_ready)
        plt.close(fig)


def plot_runtime_curves(
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    sizes: list[int],
    data: dict[str, list[dict[str, Any]]],
    paper_ready: bool = False,
) -> None:
    with plt.rc_context(paper_rc_params(paper_ready)):
        fig, ax = plt.subplots(figsize=(9.2, 5.4) if paper_ready else (10.0, 6.0))

        for lib, points in data.items():
            style = LIB_STYLES.get(lib, {"label": lib, "marker": "o", "linestyle": "-", "color": "#444"})
            point_map = {int(p["n"]): max(float(p["median_s"]), 1e-9) for p in points}
            ci_lo_map = {int(p["n"]): float(p.get("ci_low_s", float("nan"))) for p in points}
            ci_hi_map = {int(p["n"]): float(p.get("ci_high_s", float("nan"))) for p in points}
            xs = [s for s in sizes if s in point_map]
            ys = [point_map[s] for s in xs]
            if not xs:
                continue
            lo = [max(ci_lo_map.get(s, float("nan")), 1e-9) for s in xs]
            hi = [max(ci_hi_map.get(s, float("nan")), 1e-9) for s in xs]
            if all(not math.isnan(v) for v in lo + hi):
                ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.18, linewidth=0)
            ax.plot(xs, ys,
                    label=style["label"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=style["color"],
                    linewidth=2.4 if paper_ready else 2.0,
                    markersize=7 if paper_ready else 6)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Median runtime [s]")
        ax.set_yscale("log")
        if sizes and min(sizes) > 0 and len(set(sizes)) > 1:
            ax.set_xscale("log")
        if sizes:
            ticks = sorted(set(int(v) for v in sizes))
            tick_set = set(ticks)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _pos: f"{int(x)}" if int(round(x)) in tick_set else ""
                )
            )
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92)
        fig.tight_layout()
        save_figure(fig, out_path, paper_ready)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Battery runners
# ---------------------------------------------------------------------------

def run_loading_battery(
    runners: dict[str, type], *, obo: Path, gaf: Path, namespace: str, outdir: Path,
    paper_ready: bool, loading_repeats: int,
) -> dict[str, dict[str, Any]]:
    """Run each runner's loading path `loading_repeats` times (each call
    spawns a fresh subprocess where applicable, so every repeat pays the
    full import + OBO/GAF parse cost). Aggregate across repeats with a
    bootstrap 95% CI on the median."""
    results: dict[str, dict[str, Any]] = {}
    for name, cls in runners.items():
        if not cls.caps().loading:
            continue
        print(f"[loading] {name} (x{loading_repeats}) ...", flush=True)
        time_runs: list[float] = []
        peak_runs: list[float] = []
        last_payload: dict[str, Any] | None = None
        failed = False
        for i in range(max(1, loading_repeats)):
            try:
                payload = cls.loading(obo, gaf, namespace)
            except Exception as exc:
                print(f"[loading] {name}: FAILED (rep {i+1}) -> {exc}", flush=True)
                failed = True
                break
            t = payload.get("total_time_s")
            p = payload.get("peak_rss_mb")
            if t is not None and not (isinstance(t, float) and math.isnan(t)):
                time_runs.append(float(t))
            if p is not None and not (isinstance(p, float) and math.isnan(p)):
                peak_runs.append(float(p))
            last_payload = payload
            print(f"  rep {i+1}: {float(t):.2f}s, peak {float(p):.1f} MB", flush=True)
        if failed or last_payload is None:
            continue
        t_ci_lo, t_ci_hi = bootstrap_ci_median(time_runs)
        p_ci_lo, p_ci_hi = bootstrap_ci_median(peak_runs) if peak_runs else (float("nan"), float("nan"))
        agg = dict(last_payload)
        agg["display_name"] = agg.get("display_name", cls.display_name)
        agg["total_time_s"] = median(time_runs) if time_runs else float("nan")
        agg["total_time_s_runs"] = time_runs
        agg["total_time_s_ci_low"] = t_ci_lo
        agg["total_time_s_ci_high"] = t_ci_hi
        agg["peak_rss_mb"] = median(peak_runs) if peak_runs else float("nan")
        agg["peak_rss_mb_runs"] = peak_runs
        agg["peak_rss_mb_ci_low"] = p_ci_lo
        agg["peak_rss_mb_ci_high"] = p_ci_hi
        agg["n_repeats"] = len(time_runs)
        results[name] = agg
        print(f"[loading] {name}: median={agg['total_time_s']:.2f}s "
              f"[{t_ci_lo:.2f}, {t_ci_hi:.2f}] s, "
              f"peak median={agg['peak_rss_mb']:.1f} MB", flush=True)
    if results:
        plot_loading_summary(outdir / "benchmark_loading_time_memory.png", results, paper_ready)
    return results


def _serialize_points(points: dict[int, RunResult]) -> list[dict[str, Any]]:
    return [points[k].to_dict() for k in sorted(points)]


def run_term_pairs_battery(
    runners: dict[str, type], *, obo: Path, gaf: Path, namespace: str, method: str,
    pair_targets: list[int], rng: random.Random, warmup: int, repeats: int,
    threads: int | None, outdir: Path, paper_ready: bool,
) -> dict[str, Any]:
    candidates = select_term_candidates_for_namespace(obo, gaf, namespace)
    term_sets = sample_term_sets_for_sizes(candidates, pair_targets, rng)
    pairs_by_size = term_pairs_from_sets(term_sets)
    actual_sizes = sorted(pairs_by_size)

    runs: dict[str, list[dict[str, Any]]] = {}
    for name, cls in runners.items():
        caps = cls.caps()
        if not caps.supports_term(method):
            print(f"[term] {name}: skipped (no {method})", flush=True)
            continue
        print(f"[term] {name} on sizes={actual_sizes} ...", flush=True)
        try:
            points = cls.term_pairs(
                obo=obo, gaf=gaf, namespace=namespace, method=method,
                pairs_by_size=pairs_by_size,
                warmup=warmup, repeats=repeats, threads=threads,
                workdir=outdir / "tmp" / name,
            )
        except Exception as exc:
            print(f"[term] {name}: FAILED -> {exc}", flush=True)
            continue
        runs[name] = _serialize_points(points)
        for pt in points.values():
            print(f"  n={pt.n:>6}  median={pt.median_s:.4f}s  "
                  f"thr={pt.throughput_per_s:.1f}/s", flush=True)

    plot_runtime_curves(
        out_path=outdir / "benchmark_batch_similarity.png",
        title=f"Batch GO term similarity ({namespace}, {method})",
        xlabel="Number of GO term pairs",
        sizes=actual_sizes,
        data=runs,
        paper_ready=paper_ready,
    )

    return {
        "method": method,
        "pair_targets": pair_targets,
        "actual_sizes": actual_sizes,
        "term_set_sizes": {str(t): len(ts) for t, ts in term_sets.items()},
        "runs": runs,
    }


def run_gene_pairs_battery(
    runners: dict[str, type], *, obo: Path, gaf: Path, namespace: str, method: str,
    sizes: list[int], rng: random.Random, warmup: int, repeats: int,
    threads: int | None, min_gene_terms: int, outdir: Path, paper_ready: bool,
) -> dict[str, Any]:
    aspect = NAMESPACE_TO_ASPECT[namespace]
    gene2terms = parse_symbol_to_terms(gaf, namespace_aspect=aspect)
    genes, sel_meta = select_gene_candidates(
        gene2terms, needed_pairs=sum(sizes), min_gene_terms=min_gene_terms,
    )
    pairs_by_size = sample_disjoint_pair_groups(genes, sizes, rng)

    runs: dict[str, list[dict[str, Any]]] = {}
    for name, cls in runners.items():
        caps = cls.caps()
        if not caps.supports_gene(method):
            print(f"[gene] {name}: skipped (no gene-level {method})", flush=True)
            continue
        print(f"[gene] {name} on sizes={sizes} ...", flush=True)
        try:
            points = cls.gene_pairs(
                obo=obo, gaf=gaf, namespace=namespace, method=method,
                gene_pairs_by_size=pairs_by_size,
                gene2terms=gene2terms,
                warmup=warmup, repeats=repeats, threads=threads,
                workdir=outdir / "tmp" / name,
            )
        except Exception as exc:
            print(f"[gene] {name}: FAILED -> {exc}", flush=True)
            continue
        runs[name] = _serialize_points(points)
        for pt in points.values():
            print(f"  n={pt.n:>4}  median={pt.median_s:.4f}s  "
                  f"thr={pt.throughput_per_s:.1f}/s", flush=True)

    plot_runtime_curves(
        out_path=outdir / "benchmark_gene_batch_similarity.png",
        title=f"Batch gene similarity ({namespace}, {method}, BMA)",
        xlabel="Number of gene pairs",
        sizes=sizes,
        data=runs,
        paper_ready=paper_ready,
    )
    return {
        "method": method,
        "sizes": sizes,
        "candidate_selection": sel_meta,
        "runs": runs,
    }


def run_all_vs_all_genes_battery(
    runners: dict[str, type], *, obo: Path, gaf: Path, namespace: str, method: str,
    gene_sizes: list[int], rng: random.Random, warmup: int, repeats: int,
    threads: int | None, min_gene_terms: int, outdir: Path, paper_ready: bool,
) -> dict[str, Any]:
    aspect = NAMESPACE_TO_ASPECT[namespace]
    gene2terms = parse_symbol_to_terms(gaf, namespace_aspect=aspect)
    if not gene_sizes:
        return {"method": method, "gene_sizes": [], "runs": {}}

    max_pairs = max(n_choose_two(s) for s in gene_sizes)
    genes, sel_meta = select_gene_candidates(
        gene2terms, needed_pairs=max_pairs, min_gene_terms=min_gene_terms,
    )
    if len(genes) < max(gene_sizes):
        raise RuntimeError(
            f"Need {max(gene_sizes)} genes for all-vs-all but only {len(genes)} available."
        )
    selected = genes[: max(gene_sizes)]

    pairs_by_size: dict[int, list[tuple[str, str]]] = {}
    pair_count_by_n: dict[int, int] = {}
    for n in sorted(gene_sizes, reverse=True):
        subset = selected[:n]
        pairs = all_unique_pairs(subset)
        pairs_by_size[n] = pairs
        pair_count_by_n[n] = len(pairs)

    runs: dict[str, list[dict[str, Any]]] = {}
    for name, cls in runners.items():
        caps = cls.caps()
        if not caps.supports_gene(method):
            continue
        print(f"[matrix] {name} on gene_sizes={gene_sizes} ...", flush=True)
        try:
            points = cls.gene_pairs(
                obo=obo, gaf=gaf, namespace=namespace, method=method,
                gene_pairs_by_size=pairs_by_size,
                gene2terms=gene2terms,
                warmup=warmup, repeats=repeats, threads=threads,
                workdir=outdir / "tmp" / name,
            )
        except Exception as exc:
            print(f"[matrix] {name}: FAILED -> {exc}", flush=True)
            continue
        # Re-key by gene count instead of pair count for plotting consistency
        # with previous benchmark.
        rekeyed: dict[int, RunResult] = {}
        for n in gene_sizes:
            pcount = pair_count_by_n[n]
            if pcount in points:
                p = points[pcount]
                rekeyed[n] = RunResult(
                    n=n, median_s=p.median_s, min_s=p.min_s, max_s=p.max_s,
                    runs_s=p.runs_s, throughput_per_s=p.throughput_per_s,
                )
        runs[name] = _serialize_points(rekeyed)

    plot_runtime_curves(
        out_path=outdir / "benchmark_all_vs_all_gene_similarity.png",
        title=f"All-vs-all gene similarity ({namespace}, {method}, BMA)",
        xlabel="Number of genes in cohort",
        sizes=gene_sizes,
        data=runs,
        paper_ready=paper_ready,
    )
    return {
        "method": method,
        "gene_sizes": gene_sizes,
        "candidate_selection": sel_meta,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obo", type=Path, default=None)
    p.add_argument("--gaf", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("imgs"))
    p.add_argument("--namespace", choices=["BP", "MF", "CC"], default="BP")
    p.add_argument("--only", default=None,
                   help="Comma-separated runner names to include (default: all available)")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated runner names to exclude")
    p.add_argument("--term-pair-sizes", default=",".join(str(v) for v in DEFAULT_TERM_PAIR_SIZES))
    p.add_argument("--gene-pair-sizes", default=",".join(str(v) for v in DEFAULT_GENE_PAIR_SIZES))
    p.add_argument("--matrix-gene-sizes", default=",".join(str(v) for v in DEFAULT_MATRIX_GENE_SIZES))
    p.add_argument("--term-method", choices=sorted(VALID_METHODS), default="lin")
    p.add_argument("--gene-method", choices=sorted(VALID_METHODS), default="lin")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3,
                   help="Repeats per size group for term_pairs/gene_pairs/matrix batteries (after --warmup).")
    p.add_argument("--loading-repeats", type=int, default=3,
                   help="Repeats of the loading battery. Each repeat spawns a fresh "
                        "subprocess where applicable so import + parse cost counts every time.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--min-gene-terms", type=int, default=8)
    p.add_argument("--no-loading", action="store_true")
    p.add_argument("--no-term", action="store_true")
    p.add_argument("--no-gene", action="store_true")
    p.add_argument("--no-all-vs-all", action="store_true")
    p.add_argument("--paper-ready", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--replot", type=Path, default=None,
                   help="Path to an existing benchmark_results.json — regenerate plots without re-running benchmarks.")
    p.add_argument("--list-runners", action="store_true",
                   help="List discovered runners and exit.")
    return p


def apply_paper_profile(args: argparse.Namespace) -> None:
    if not args.paper_ready:
        return
    args.term_pair_sizes = ",".join(str(v) for v in PAPER_TERM_PAIR_SIZES)
    args.gene_pair_sizes = ",".join(str(v) for v in PAPER_GENE_PAIR_SIZES)
    args.matrix_gene_sizes = ",".join(str(v) for v in PAPER_MATRIX_GENE_SIZES)
    args.warmup = PAPER_WARMUP
    args.repeats = PAPER_REPEATS
    args.loading_repeats = PAPER_LOADING_REPEATS
    args.threads = PAPER_THREADS
    args.min_gene_terms = PAPER_MIN_GENE_TERMS


def replot_from_json(json_path: Path, outdir: Path, paper_ready: bool) -> int:
    """Regenerate all plots from an existing benchmark_results.json."""
    report = json.loads(json_path.read_text(encoding="utf-8"))
    outdir.mkdir(parents=True, exist_ok=True)

    if "loading" in report:
        plot_loading_summary(
            outdir / "benchmark_loading_time_memory.png",
            report["loading"], paper_ready,
        )
        print("Wrote benchmark_loading_time_memory.png")

    if "term_pairs" in report:
        tp = report["term_pairs"]
        plot_runtime_curves(
            out_path=outdir / "benchmark_batch_similarity.png",
            title=f"Batch GO term similarity ({report['namespace']}, {tp['method']})",
            xlabel="Number of GO term pairs",
            sizes=tp["actual_sizes"],
            data=tp["runs"],
            paper_ready=paper_ready,
        )
        print("Wrote benchmark_batch_similarity.png")

    if "gene_pairs" in report:
        gp = report["gene_pairs"]
        plot_runtime_curves(
            out_path=outdir / "benchmark_gene_batch_similarity.png",
            title=f"Batch gene similarity ({report['namespace']}, {gp['method']}, BMA)",
            xlabel="Number of gene pairs",
            sizes=gp["sizes"],
            data=gp["runs"],
            paper_ready=paper_ready,
        )
        print("Wrote benchmark_gene_batch_similarity.png")

    if "all_vs_all_gene" in report:
        av = report["all_vs_all_gene"]
        plot_runtime_curves(
            out_path=outdir / "benchmark_all_vs_all_gene_similarity.png",
            title=f"All-vs-all gene similarity ({report['namespace']}, {av['method']}, BMA)",
            xlabel="Number of genes in cohort",
            sizes=av["gene_sizes"],
            data=av["runs"],
            paper_ready=paper_ready,
        )
        print("Wrote benchmark_all_vs_all_gene_similarity.png")

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    apply_paper_profile(args)

    if args.replot:
        return replot_from_json(args.replot, args.outdir, args.paper_ready)

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    exclude = set(s.strip() for s in args.exclude.split(",")) if args.exclude else set()

    runners = available_runners(only=only)
    if exclude:
        runners = {n: c for n, c in runners.items() if n not in exclude}

    if args.list_runners:
        print("Discovered runners:")
        for name, cls in runners.items():
            caps = cls.caps()
            print(f"  - {name:<12} loading={caps.loading} "
                  f"term={sorted(caps.term_pair_methods)} gene={sorted(caps.gene_pair_methods)}"
                  + (f"  ({caps.notes})" if caps.notes else ""))
        return 0

    if not runners:
        raise SystemExit("No runners are available in this environment.")

    obo_default, gaf_default = default_paths()
    obo = args.obo or obo_default
    gaf = args.gaf or gaf_default
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    term_targets = sorted(set(parse_int_list(args.term_pair_sizes)))
    gene_sizes = sorted(set(parse_int_list(args.gene_pair_sizes)))
    matrix_gene_sizes = sorted(set(parse_int_list(args.matrix_gene_sizes)))

    report: dict[str, Any] = {
        "profile": "paper_ready" if args.paper_ready else "default",
        "namespace": args.namespace,
        "term_pair_targets": term_targets,
        "gene_pair_sizes": gene_sizes,
        "matrix_gene_sizes": matrix_gene_sizes,
        "term_method": args.term_method,
        "gene_method": args.gene_method,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "threads": args.threads,
        "min_gene_terms": args.min_gene_terms,
        "runners": {n: {
            "display_name": cls.display_name,
            "caps": {
                "loading": cls.caps().loading,
                "term_pair_methods": sorted(cls.caps().term_pair_methods),
                "gene_pair_methods": sorted(cls.caps().gene_pair_methods),
                "notes": cls.caps().notes,
            },
        } for n, cls in runners.items()},
        "system_metadata": collect_system_metadata(),
        "obo": str(obo),
        "gaf": str(gaf),
        "obo_version": parse_obo_version(obo),
        "gaf_version": parse_gaf_version(gaf),
        "loading_repeats": args.loading_repeats,
    }

    if not args.no_loading:
        loading = run_loading_battery(
            runners, obo=obo, gaf=gaf, namespace=args.namespace,
            outdir=outdir, paper_ready=args.paper_ready,
            loading_repeats=args.loading_repeats,
        )
        report["loading"] = loading

    if not args.no_term:
        report["term_pairs"] = run_term_pairs_battery(
            runners, obo=obo, gaf=gaf, namespace=args.namespace,
            method=args.term_method, pair_targets=term_targets,
            rng=random.Random(args.seed),
            warmup=args.warmup, repeats=args.repeats, threads=args.threads,
            outdir=outdir, paper_ready=args.paper_ready,
        )

    if not args.no_gene:
        report["gene_pairs"] = run_gene_pairs_battery(
            runners, obo=obo, gaf=gaf, namespace=args.namespace,
            method=args.gene_method, sizes=gene_sizes,
            rng=random.Random(args.seed + 1),
            warmup=args.warmup, repeats=args.repeats, threads=args.threads,
            min_gene_terms=args.min_gene_terms,
            outdir=outdir, paper_ready=args.paper_ready,
        )

    if not args.no_all_vs_all and matrix_gene_sizes:
        report["all_vs_all_gene"] = run_all_vs_all_genes_battery(
            runners, obo=obo, gaf=gaf, namespace=args.namespace,
            method=args.gene_method, gene_sizes=matrix_gene_sizes,
            rng=random.Random(args.seed + 2),
            warmup=args.warmup, repeats=args.repeats, threads=args.threads,
            min_gene_terms=args.min_gene_terms,
            outdir=outdir, paper_ready=args.paper_ready,
        )

    json_path = args.json or (outdir / "benchmark_results.json")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
