from __future__ import annotations

import argparse
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from statistics import median
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import go3
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import 'go3'. Run benchmarks from an environment with go3 installed, "
        "for example: ./venv/bin/python scripts/benchmark_go3vsgoatools.py ..."
    ) from exc

try:
    from goatools.anno.gaf_reader import GafReader
    from goatools.obo_parser import GODag
    from goatools.semantic import TermCounts, lin_sim, resnik_sim
    from goatools.semsim.termwise.wang import SsWang
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'goatools'. Install it in the active environment before running benchmarks."
    ) from exc

from demo_utils import default_paths

DEFAULT_TERM_PAIR_SIZES = [100, 1000, 5000, 10000]
DEFAULT_GENE_PAIR_SIZES = [25, 50, 100, 200]
DEFAULT_MATRIX_GENE_SIZES = [8, 12, 16]
PAPER_TERM_PAIR_SIZES = [1000, 5000, 20000]
PAPER_GENE_PAIR_SIZES = [25, 50, 100]
PAPER_MATRIX_GENE_SIZES = [8, 12]
PAPER_WARMUP = 2
PAPER_REPEATS = 5
PAPER_THREADS = 8
PAPER_MIN_GENE_TERMS = 8
VALID_METHODS = {"resnik", "lin", "wang"}
NAMESPACE_TO_FULL = {
    "BP": "biological_process",
    "MF": "molecular_function",
    "CC": "cellular_component",
}


def is_executable_on_path(name: str) -> bool:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return True
    return False


def _linux_mem_field_mb(field: str) -> float | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except FileNotFoundError:
        return None
    return None


def rss_mb() -> float:
    if sys.platform.startswith("linux"):
        value = _linux_mem_field_mb("VmRSS:")
        if value is not None:
            return value
    return float("nan")


def peak_rss_mb() -> float:
    if sys.platform.startswith("linux"):
        value = _linux_mem_field_mb("VmHWM:")
        if value is not None:
            return value
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak) / (1024.0 * 1024.0)
    return float(peak) / 1024.0


def parse_int_list(csv: str) -> list[int]:
    values: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def cli_arg_present(flag: str, argv: list[str]) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in argv)


def apply_paper_profile(args: argparse.Namespace, argv: list[str]) -> None:
    if not args.paper_ready:
        return

    if not cli_arg_present("--pair-sizes", argv) and not cli_arg_present("--term-pair-sizes", argv):
        args.term_pair_sizes = ",".join(str(v) for v in PAPER_TERM_PAIR_SIZES)
    if not cli_arg_present("--pair-sizes", argv) and not cli_arg_present("--gene-pair-sizes", argv):
        args.gene_pair_sizes = ",".join(str(v) for v in PAPER_GENE_PAIR_SIZES)
    if not cli_arg_present("--matrix-gene-sizes", argv):
        args.matrix_gene_sizes = ",".join(str(v) for v in PAPER_MATRIX_GENE_SIZES)
    if not cli_arg_present("--warmup", argv):
        args.warmup = PAPER_WARMUP
    if not cli_arg_present("--repeats", argv):
        args.repeats = PAPER_REPEATS
    if not cli_arg_present("--threads", argv):
        args.threads = PAPER_THREADS
    if not cli_arg_present("--min-gene-terms", argv):
        args.min_gene_terms = PAPER_MIN_GENE_TERMS


def package_version_or_none(pkg: str) -> str | None:
    try:
        return importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def cpu_model_name() -> str | None:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            return parts[1].strip()
        except FileNotFoundError:
            return None
    return platform.processor() or None


def total_memory_gb() -> float | None:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = float(parts[1])
                            return kb / (1024.0 * 1024.0)
        except FileNotFoundError:
            return None
    return None


def collect_system_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_model_name(),
        "cpu_count_logical": os.cpu_count(),
        "total_memory_gb": total_memory_gb(),
        "python_version": sys.version,
        "go3_version": package_version_or_none("go3"),
        "goatools_version": package_version_or_none("goatools"),
        "matplotlib_version": package_version_or_none("matplotlib"),
    }


def n_choose_two(n: int) -> int:
    return (n * (n - 1)) // 2


def min_items_for_unique_pairs(n_pairs: int) -> int:
    if n_pairs <= 0:
        return 0
    n = 2
    while n_choose_two(n) < n_pairs:
        n += 1
    return n


def sample_unique_unordered_pairs(items: list[str], n_pairs: int, rng: random.Random) -> list[tuple[str, str]]:
    if n_pairs <= 0:
        return []
    max_pairs = n_choose_two(len(items))
    if n_pairs > max_pairs:
        raise ValueError(f"Requested {n_pairs} unique pairs, but only {max_pairs} are possible.")
    pairs: set[tuple[str, str]] = set()
    while len(pairs) < n_pairs:
        a, b = rng.sample(items, 2)
        if a > b:
            a, b = b, a
        pairs.add((a, b))
    return sorted(pairs)


def sample_disjoint_pair_groups(
    items: list[str],
    sizes: list[int],
    rng: random.Random,
) -> dict[int, list[tuple[str, str]]]:
    sizes_local = list(sizes)
    total_needed = sum(sizes_local)
    max_pairs = n_choose_two(len(items))
    if total_needed > max_pairs:
        raise ValueError(
            f"Requested {total_needed} total disjoint pairs, but only {max_pairs} are possible."
        )

    all_pairs = sample_unique_unordered_pairs(items, total_needed, rng)
    rng.shuffle(all_pairs)

    out: dict[int, list[tuple[str, str]]] = {}
    idx = 0
    for size in sizes_local:
        out[size] = all_pairs[idx : idx + size]
        idx += size
    return out


def all_unique_pairs(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i, a in enumerate(items):
        for j in range(i + 1, len(items)):
            out.append((a, items[j]))
    return out


def time_runs(fn: Callable[[], Any], *, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        fn()

    runs: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)

    return {
        "runs_s": runs,
        "median_s": median(runs),
        "min_s": min(runs),
        "max_s": max(runs),
    }


def add_throughput(point: dict[str, Any], work_items: int) -> dict[str, Any]:
    med = max(float(point["median_s"]), 1e-12)
    point = dict(point)
    point["throughput_per_s"] = float(work_items) / med
    return point


def speedup_points(
    faster_points: list[dict[str, Any]],
    slower_points: list[dict[str, Any]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    faster = {int(p["n"]): float(p["median_s"]) for p in faster_points}
    slower = {int(p["n"]): float(p["median_s"]) for p in slower_points}
    shared = sorted(set(faster).intersection(slower))
    out: list[dict[str, Any]] = []
    for n in shared:
        fast = faster[n]
        slow = slower[n]
        if fast <= 0.0:
            speed = float("inf")
        else:
            speed = slow / fast
        out.append({"n": n, "speedup": speed, "label": label})
    return out


def summarize_speedup(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {}
    values = [float(p["speedup"]) for p in points]
    values_sorted = sorted(values)
    return {
        "n_points": len(values),
        "median_x": values_sorted[len(values_sorted) // 2],
        "min_x": values_sorted[0],
        "max_x": values_sorted[-1],
    }


def parse_gene_symbol2go(gaf_path: Path, godag: GODag, namespace: str) -> dict[str, list[str]]:
    ns_full = NAMESPACE_TO_FULL[namespace]
    gene2gos: dict[str, set[str]] = defaultdict(set)

    with open(gaf_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            qualifier = cols[3]
            go_id = cols[4]
            evidence = cols[6]
            if evidence == "ND" or "NOT" in qualifier:
                continue
            go_obj = godag.get(go_id)
            if go_obj is None:
                continue
            if getattr(go_obj, "namespace", None) != ns_full:
                continue
            if getattr(go_obj, "is_obsolete", False):
                continue
            gene_symbol = cols[2].strip()
            if not gene_symbol:
                continue
            gene2gos[gene_symbol].add(go_obj.id)

    return {gene: sorted(terms) for gene, terms in gene2gos.items() if terms}


def write_gosemsim_goanno_tsv(gaf_path: Path, out_path: Path) -> None:
    aspect_map = {"P": "BP", "F": "MF", "C": "CC"}
    with open(gaf_path, "r", encoding="utf-8", errors="ignore") as handle, open(
        out_path, "w", encoding="utf-8"
    ) as out:
        out.write("gene\tGO\tONTOLOGY\n")
        for line in handle:
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            qualifier = cols[3]
            go_id = cols[4]
            evidence = cols[6]
            aspect = cols[8].strip()
            if evidence == "ND" or "NOT" in qualifier:
                continue
            ontology = aspect_map.get(aspect)
            if not ontology:
                continue
            gene_symbol = cols[2]
            if not gene_symbol or not go_id:
                continue
            out.write(f"{gene_symbol}\t{go_id}\t{ontology}\n")


def bma_one_pass(go_terms1: list[str], go_terms2: list[str], sim: Callable[[str, str], float | None]) -> float:
    if not go_terms1 or not go_terms2:
        return 0.0
    total = float(len(go_terms1) + len(go_terms2))
    if total == 0.0:
        return 0.0
    col_max = [0.0] * len(go_terms2)
    sum_row_max = 0.0
    for t1 in go_terms1:
        row_max = 0.0
        for j, t2 in enumerate(go_terms2):
            value = sim(t1, t2) or 0.0
            if value > row_max:
                row_max = value
            if value > col_max[j]:
                col_max[j] = value
        sum_row_max += row_max
    return (sum_row_max + sum(col_max)) / total


def goatools_sim_factory(
    method: str,
    godag: GODag,
    termcounts: TermCounts,
    wang: SsWang | None,
) -> Callable[[str, str], float | None]:
    method = method.lower()
    if method == "resnik":
        return lambda a, b: resnik_sim(a, b, godag, termcounts)
    if method == "lin":
        return lambda a, b: lin_sim(a, b, godag, termcounts, dfltval=0.0)
    if method == "wang":
        if wang is None:
            raise ValueError("Wang helper not initialized")
        return lambda a, b: wang.get_sim(a, b)
    raise ValueError(f"Unsupported method: {method}")


def select_term_candidates(
    *,
    counter: Any,
    termcounts: TermCounts,
    godag: GODag,
    namespace: str,
    needed_pairs: int,
) -> tuple[list[str], dict[str, Any]]:
    ns_full = NAMESPACE_TO_FULL[namespace]
    needed_terms = min_items_for_unique_pairs(needed_pairs)
    goatools_goids = set(termcounts.goids)

    candidate_steps = [
        {"min_depth": 4, "min_ic": 0.5},
        {"min_depth": 3, "min_ic": 0.1},
        {"min_depth": 2, "min_ic": 0.0},
        {"min_depth": 0, "min_ic": 0.0},
    ]

    last_candidates: list[str] = []
    selected_meta: dict[str, Any] = {}

    for step in candidate_steps:
        min_depth = int(step["min_depth"])
        min_ic = float(step["min_ic"])
        cands: list[str] = []
        for go_id, ic in counter.ic.items():
            if ic <= min_ic:
                continue
            if go_id not in goatools_goids:
                continue
            obj = godag.get(go_id)
            if obj is None:
                continue
            if getattr(obj, "namespace", None) != ns_full:
                continue
            depth = getattr(obj, "depth", None)
            if depth is None:
                depth = 0
            if int(depth) < min_depth:
                continue
            cands.append(go_id)

        cands = sorted(set(cands))
        last_candidates = cands
        selected_meta = {
            "candidate_count": len(cands),
            "min_depth": min_depth,
            "min_ic": min_ic,
        }
        if len(cands) >= needed_terms:
            return cands, selected_meta

    raise RuntimeError(
        f"Not enough term candidates for namespace={namespace}. "
        f"Need >= {needed_terms} terms for {needed_pairs} unique pairs, got {len(last_candidates)}."
    )


def select_gene_candidates(
    gene2terms: dict[str, list[str]],
    *,
    needed_pairs: int,
    min_gene_terms: int,
) -> tuple[list[str], dict[str, Any]]:
    ranked = sorted(gene2terms.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    needed_genes = min_items_for_unique_pairs(needed_pairs)

    for threshold in range(max(1, min_gene_terms), 0, -1):
        eligible = [gene for gene, terms in ranked if len(terms) >= threshold]
        if len(eligible) < needed_genes:
            continue

        # Keep a margin so random pair sampling still has diversity.
        target_pairs = needed_pairs * 2
        selected: list[str] = []
        for gene in eligible:
            selected.append(gene)
            if len(selected) >= needed_genes and n_choose_two(len(selected)) >= target_pairs:
                break

        return selected, {
            "min_gene_terms_requested": min_gene_terms,
            "min_gene_terms_used": threshold,
            "candidate_count": len(selected),
            "eligible_before_cap": len(eligible),
        }

    raise RuntimeError(
        f"Not enough genes for {needed_pairs} unique pairs with at least one annotation term in namespace."
    )


def child_loading_metrics(lib: str, *, obo: Path, gaf: Path) -> dict[str, Any]:
    def step(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
        start = time.perf_counter()
        details = fn()
        elapsed = time.perf_counter() - start
        return {
            "name": name,
            "time_s": elapsed,
            "rss_mb": rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            "details": details,
        }

    steps: list[dict[str, Any]] = []

    if lib == "go3":
        steps.append(step("Load ontology", lambda: {"n_terms": len(go3.load_go_terms(str(obo)))}))
        annotations = None

        def _load_gaf() -> dict[str, Any]:
            nonlocal annotations
            annotations = go3.load_gaf(str(gaf))
            return {"n_annotations": len(annotations)}

        steps.append(step("Load annotations", _load_gaf))

        def _build_counter() -> dict[str, Any]:
            assert annotations is not None
            counter = go3.build_term_counter(annotations)
            return {"n_ic_terms": len(counter.ic)}

        steps.append(step("Build counter", _build_counter))

    elif lib == "goatools":
        godag = None
        id2gos = None

        def _load_obo() -> dict[str, Any]:
            nonlocal godag
            godag = GODag(str(obo), optional_attrs={"relationship"}, prt=None)
            return {"n_terms": len(godag)}

        steps.append(step("Load ontology", _load_obo))

        def _load_gaf() -> dict[str, Any]:
            nonlocal godag, id2gos
            assert godag is not None
            reader = GafReader(str(gaf), godag=godag, prt=None)
            id2gos = reader.get_id2gos_nss(prt=None)
            return {"n_objects": len(id2gos)}

        steps.append(step("Load annotations", _load_gaf))

        def _build_counter() -> dict[str, Any]:
            assert godag is not None and id2gos is not None
            termcounts = TermCounts(godag, id2gos, prt=None)
            return {"n_goids": len(termcounts.goids)}

        steps.append(step("Build counter", _build_counter))

    else:
        raise ValueError(f"Unknown lib '{lib}'")

    return {
        "lib": lib,
        "steps": steps,
        "total_time_s": sum(step_info["time_s"] for step_info in steps),
        "peak_rss_mb": max(step_info["peak_rss_mb"] for step_info in steps),
        "final_rss_mb": steps[-1]["rss_mb"] if steps else rss_mb(),
        "python": sys.version,
        "platform": sys.platform,
    }


def run_child_loading(lib: str, *, obo: Path, gaf: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "loading",
        "--lib",
        lib,
        "--obo",
        str(obo),
        "--gaf",
        str(gaf),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def build_r_env(r_libs_user: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if r_libs_user:
        env["R_LIBS_USER"] = r_libs_user
    return env


def run_gosemsim_loading(
    *,
    namespace: str,
    orgdb: str,
    anno_tsv: Path | None,
    r_libs_user: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not is_executable_on_path("Rscript"):
        return None, "Rscript not found on PATH"
    r_script = Path(__file__).with_name("benchmark_gosemsim.R")
    if not r_script.exists():
        return None, f"Missing helper script: {r_script}"

    cmd = [
        "Rscript",
        str(r_script),
        "--mode",
        "loading",
        "--ontology",
        namespace,
    ]
    if anno_tsv is not None:
        cmd.extend(["--anno-tsv", str(anno_tsv)])
    else:
        cmd.extend(["--orgdb", orgdb])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=build_r_env(r_libs_user),
        )
        payload = json.loads(proc.stdout.strip())
        return payload, None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = stderr if stderr else str(exc)
        return None, f"Rscript failed (loading): {detail}"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {exc}"


def run_gosemsim_pairs(
    *,
    mode: str,
    namespace: str,
    method: str,
    gosemsim_measure: str | None,
    orgdb: str,
    warmup: int,
    repeats: int,
    seed: int,
    pairs_tsv: Path,
    r_libs_user: str | None,
    anno_tsv: Path | None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not is_executable_on_path("Rscript"):
        return None, "Rscript not found on PATH"

    r_script = Path(__file__).with_name("benchmark_gosemsim.R")
    if not r_script.exists():
        return None, f"Missing helper script: {r_script}"

    chosen = (gosemsim_measure or method).lower()
    measure = {"resnik": "Resnik", "lin": "Lin", "wang": "Wang"}[chosen]
    cmd = [
        "Rscript",
        str(r_script),
        "--mode",
        mode,
        "--ontology",
        namespace,
        "--measure",
        measure,
        "--pairs-tsv",
        str(pairs_tsv),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
    ]
    if anno_tsv is not None:
        cmd.extend(["--anno-tsv", str(anno_tsv)])
    else:
        cmd.extend(["--orgdb", orgdb])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=build_r_env(r_libs_user),
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) < 2:
            raise RuntimeError(f"Unexpected output from GOSemSim helper:\n{proc.stdout}\n{proc.stderr}")
        header = lines[0].split("\t")
        if header[:2] != ["size", "median_s"]:
            raise RuntimeError(f"Unexpected header from GOSemSim helper: {lines[0]}")

        points: list[dict[str, Any]] = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            n = int(parts[0])
            med = float(parts[1])
            points.append(
                {
                    "n": n,
                    "median_s": med,
                    "throughput_per_s": float(n) / max(med, 1e-12),
                }
            )

        points.sort(key=lambda item: item["n"])
        return points, None
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = stderr if stderr else str(exc)
        if "enterRNGScope" in detail:
            detail = f"{detail}\nSuggestion: try --gosemsim-measure wang (IC methods can hit an Rcpp symbol issue)."
        return None, f"Rscript failed ({mode}): {detail}"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {exc}"


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


def save_figure(fig: Any, out_path: Path, *, paper_ready: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = 320 if paper_ready else 180
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if paper_ready:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")


def speedup_summary_text(data: dict[str, list[dict[str, Any]]]) -> str | None:
    if "go3" not in data or "goatools" not in data:
        return None
    points = speedup_points(data["go3"], data["goatools"], label="go3_vs_goatools")
    summary = summarize_speedup(points)
    if not summary:
        return None
    return (
        "Speedup go3 vs goatools (median/min/max): "
        f"{summary['median_x']:.2f}x / {summary['min_x']:.2f}x / {summary['max_x']:.2f}x"
    )


def plot_loading_summary(
    *,
    out_path: Path,
    loading: dict[str, dict[str, Any]],
    paper_ready: bool = False,
) -> None:
    libs = [lib for lib in ["go3", "goatools", "gosemsim"] if lib in loading]
    labels = [loading[lib].get("display_name", lib) for lib in libs]
    times = [float(loading[lib]["total_time_s"]) for lib in libs]
    peaks = [float(loading[lib]["peak_rss_mb"]) for lib in libs]
    colors = ["#1b9e77", "#d95f02", "#7570b3"][: len(libs)]

    with plt.rc_context(paper_rc_params(paper_ready)):
        fig, (ax_time, ax_mem) = plt.subplots(
            1,
            2,
            figsize=(11.0, 4.8) if paper_ready else (10.0, 6.0),
        )
        x = list(range(len(libs)))

        bars_time = ax_time.bar(x, times, color=colors, edgecolor="#333333", linewidth=0.6)
        ax_time.set_title("Loading and preprocessing time")
        ax_time.set_ylabel("Time (s)")
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(labels, rotation=15, ha="right")
        ax_time.grid(True, axis="y", linestyle="--", alpha=0.35)
        for bar, value in zip(bars_time, times):
            ax_time.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        bars_mem = ax_mem.bar(x, peaks, color=colors, edgecolor="#333333", linewidth=0.6)
        ax_mem.set_title("Peak memory during loading")
        ax_mem.set_ylabel("Peak RSS (MB)")
        ax_mem.set_xticks(x)
        ax_mem.set_xticklabels(labels, rotation=15, ha="right")
        ax_mem.grid(True, axis="y", linestyle="--", alpha=0.35)
        for bar, value in zip(bars_mem, peaks):
            ax_mem.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        if "go3" in loading and "goatools" in loading:
            go3_time = float(loading["go3"]["total_time_s"])
            goatools_time = float(loading["goatools"]["total_time_s"])
            go3_mem = float(loading["go3"]["peak_rss_mb"])
            goatools_mem = float(loading["goatools"]["peak_rss_mb"])
            if go3_time > 0.0 and go3_mem > 0.0:
                speedup = goatools_time / go3_time
                mem_ratio = goatools_mem / go3_mem
                fig.suptitle(
                    f"go3 vs goatools: {speedup:.2f}x faster and {mem_ratio:.2f}x lower peak memory",
                    y=1.02,
                )

        fig.tight_layout()
        save_figure(fig, out_path, paper_ready=paper_ready)
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
    styles = {
        "go3": {"label": "go3", "marker": "o", "linestyle": "-", "color": "#1b9e77"},
        "goatools": {"label": "goatools", "marker": "s", "linestyle": "--", "color": "#d95f02"},
        "gosemsim": {"label": "GOSemSim (R)", "marker": "^", "linestyle": ":", "color": "#7570b3"},
    }

    with plt.rc_context(paper_rc_params(paper_ready)):
        fig, ax = plt.subplots(figsize=(9.2, 5.4) if paper_ready else (10.0, 6.0))

        for lib in ["go3", "goatools", "gosemsim"]:
            if lib not in data:
                continue
            point_map = {int(item["n"]): max(float(item["median_s"]), 1e-9) for item in data[lib]}
            xs: list[int] = []
            ys: list[float] = []
            for size in sizes:
                if size in point_map:
                    xs.append(size)
                    ys.append(point_map[size])
            if not xs:
                continue
            style = styles[lib]
            ax.plot(
                xs,
                ys,
                label=style["label"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.4 if paper_ready else 2.0,
                markersize=7 if paper_ready else 6,
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Median runtime (s)")
        ax.set_yscale("log")
        if sizes and min(sizes) > 0 and len(set(sizes)) > 1:
            ax.set_xscale("log")
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92)

        summary_text = speedup_summary_text(data)
        if summary_text:
            fig.text(
                0.5,
                0.01,
                summary_text,
                va="bottom",
                ha="center",
                fontsize=10 if paper_ready else 9,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.25"},
            )
            fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        else:
            fig.tight_layout()
        save_figure(fig, out_path, paper_ready=paper_ready)
        plt.close(fig)


def prepare_context(*, obo: Path, gaf: Path, namespace: str) -> dict[str, Any]:
    # go3 objects
    go3.load_go_terms(str(obo))
    annotations = go3.load_gaf(str(gaf))
    counter = go3.build_term_counter(annotations)

    # goatools objects
    godag = GODag(str(obo), optional_attrs={"relationship"}, prt=None)
    reader = GafReader(str(gaf), godag=godag, prt=None)
    id2gos = reader.get_id2gos_nss(prt=None)
    termcounts = TermCounts(godag, id2gos, prt=None)

    gene2terms = parse_gene_symbol2go(gaf, godag, namespace)

    return {
        "counter": counter,
        "godag": godag,
        "termcounts": termcounts,
        "gene2terms": gene2terms,
    }


def benchmark_term_pairs(
    *,
    context: dict[str, Any],
    method: str,
    sizes: list[int],
    namespace: str,
    seed: int,
    warmup: int,
    repeats: int,
    include_gosemsim: bool,
    gosemsim_orgdb: str,
    r_libs_user: str | None,
    gosemsim_anno_tsv: Path | None,
    gosemsim_measure: str | None,
    outdir: Path,
) -> dict[str, Any]:
    counter = context["counter"]
    godag = context["godag"]
    termcounts = context["termcounts"]

    max_pairs = sum(sizes)
    terms, selection_meta = select_term_candidates(
        counter=counter,
        termcounts=termcounts,
        godag=godag,
        namespace=namespace,
        needed_pairs=max_pairs,
    )

    rng = random.Random(seed)
    pairs_by_size = sample_disjoint_pair_groups(terms, sizes, rng)

    all_terms = {term for pairs in pairs_by_size.values() for pair in pairs for term in pair}
    wang = SsWang(all_terms, godag, {"part_of"}) if method == "wang" else None
    goatools_sim = goatools_sim_factory(method, godag, termcounts, wang)

    # Prime single-call setup and caches once outside timed regions.
    first_pair = pairs_by_size[sizes[0]][0]
    go3.batch_similarity([first_pair[0]], [first_pair[1]], method, counter)
    _ = goatools_sim(first_pair[0], first_pair[1])

    points_go3_map: dict[int, dict[str, Any]] = {}
    points_goatools_map: dict[int, dict[str, Any]] = {}

    for size in sorted(sizes, reverse=True):
        pairs = pairs_by_size[size]
        list1 = [a for a, _ in pairs]
        list2 = [b for _, b in pairs]

        def _go3() -> None:
            _ = go3.batch_similarity(list1, list2, method, counter)

        def _goatools() -> None:
            _ = [goatools_sim(a, b) for a, b in pairs]

        p_go3 = add_throughput({"n": size, **time_runs(_go3, warmup=warmup, repeats=repeats)}, size)
        p_gt = add_throughput({"n": size, **time_runs(_goatools, warmup=warmup, repeats=repeats)}, size)
        points_go3_map[size] = p_go3
        points_goatools_map[size] = p_gt

    points_go3 = [points_go3_map[size] for size in sizes]
    points_goatools = [points_goatools_map[size] for size in sizes]

    result: dict[str, Any] = {
        "method": method,
        "sizes": sizes,
        "candidate_selection": selection_meta,
        "runs": {
            "go3": points_go3,
            "goatools": points_goatools,
        },
    }

    if include_gosemsim:
        pairs_tsv = outdir / f"gosemsim_term_pairs_{namespace}_{method}.tsv"
        with open(pairs_tsv, "w", encoding="utf-8") as handle:
            for size in sizes:
                for a, b in pairs_by_size[size]:
                    handle.write(f"{size}\t{a}\t{b}\n")

        points_r, err = run_gosemsim_pairs(
            mode="term",
            namespace=namespace,
            method=method,
            gosemsim_measure=gosemsim_measure,
            orgdb=gosemsim_orgdb,
            warmup=warmup,
            repeats=repeats,
            seed=seed,
            pairs_tsv=pairs_tsv,
            r_libs_user=r_libs_user,
            anno_tsv=gosemsim_anno_tsv,
        )
        if points_r is not None:
            result["runs"]["gosemsim"] = points_r
        if err is not None:
            result["gosemsim_error"] = err

    speedup = speedup_points(result["runs"]["go3"], result["runs"]["goatools"], label="go3_vs_goatools")
    result["go3_speedup_vs_goatools"] = speedup
    result["go3_speedup_summary"] = summarize_speedup(speedup)
    return result


def benchmark_gene_pairs(
    *,
    context: dict[str, Any],
    method: str,
    sizes: list[int],
    namespace: str,
    seed: int,
    warmup: int,
    repeats: int,
    min_gene_terms: int,
    include_gosemsim: bool,
    gosemsim_orgdb: str,
    r_libs_user: str | None,
    gosemsim_anno_tsv: Path | None,
    gosemsim_measure: str | None,
    outdir: Path,
) -> dict[str, Any]:
    counter = context["counter"]
    godag = context["godag"]
    termcounts = context["termcounts"]
    gene2terms = context["gene2terms"]

    max_pairs = sum(sizes)
    genes, selection_meta = select_gene_candidates(
        gene2terms,
        needed_pairs=max_pairs,
        min_gene_terms=min_gene_terms,
    )

    rng = random.Random(seed)
    pairs_by_size = sample_disjoint_pair_groups(genes, sizes, rng)

    if method == "wang":
        all_terms: set[str] = set()
        for pairs in pairs_by_size.values():
            for g1, g2 in pairs:
                all_terms.update(gene2terms.get(g1, []))
                all_terms.update(gene2terms.get(g2, []))
        wang = SsWang(all_terms, godag, {"part_of"})
    else:
        wang = None

    goatools_term_sim = goatools_sim_factory(method, godag, termcounts, wang)

    first_pair = pairs_by_size[sizes[0]][0]
    go3.compare_gene_pairs_batch([first_pair], namespace, method, "bma", counter)
    _ = bma_one_pass(
        gene2terms.get(first_pair[0], []),
        gene2terms.get(first_pair[1], []),
        goatools_term_sim,
    )

    points_go3_map: dict[int, dict[str, Any]] = {}
    points_goatools_map: dict[int, dict[str, Any]] = {}

    for size in sorted(sizes, reverse=True):
        pairs = pairs_by_size[size]

        def _go3() -> None:
            _ = go3.compare_gene_pairs_batch(pairs, namespace, method, "bma", counter)

        def _goatools() -> None:
            _ = [
                bma_one_pass(gene2terms.get(g1, []), gene2terms.get(g2, []), goatools_term_sim)
                for g1, g2 in pairs
            ]

        p_go3 = add_throughput({"n": size, **time_runs(_go3, warmup=warmup, repeats=repeats)}, size)
        p_gt = add_throughput({"n": size, **time_runs(_goatools, warmup=warmup, repeats=repeats)}, size)
        points_go3_map[size] = p_go3
        points_goatools_map[size] = p_gt

    points_go3 = [points_go3_map[size] for size in sizes]
    points_goatools = [points_goatools_map[size] for size in sizes]

    result: dict[str, Any] = {
        "method": method,
        "sizes": sizes,
        "candidate_selection": selection_meta,
        "runs": {
            "go3": points_go3,
            "goatools": points_goatools,
        },
    }

    if include_gosemsim:
        pairs_tsv = outdir / f"gosemsim_gene_pairs_{namespace}_{method}.tsv"
        with open(pairs_tsv, "w", encoding="utf-8") as handle:
            for size in sizes:
                for g1, g2 in pairs_by_size[size]:
                    terms1 = ",".join(gene2terms.get(g1, []))
                    terms2 = ",".join(gene2terms.get(g2, []))
                    handle.write(f"{size}\t{terms1}\t{terms2}\n")

        points_r, err = run_gosemsim_pairs(
            mode="gene",
            namespace=namespace,
            method=method,
            gosemsim_measure=gosemsim_measure,
            orgdb=gosemsim_orgdb,
            warmup=warmup,
            repeats=repeats,
            seed=seed,
            pairs_tsv=pairs_tsv,
            r_libs_user=r_libs_user,
            anno_tsv=gosemsim_anno_tsv,
        )
        if points_r is not None:
            result["runs"]["gosemsim"] = points_r
        if err is not None:
            result["gosemsim_error"] = err

    speedup = speedup_points(result["runs"]["go3"], result["runs"]["goatools"], label="go3_vs_goatools")
    result["go3_speedup_vs_goatools"] = speedup
    result["go3_speedup_summary"] = summarize_speedup(speedup)
    return result


def benchmark_all_vs_all_genes(
    *,
    context: dict[str, Any],
    method: str,
    gene_sizes: list[int],
    namespace: str,
    warmup: int,
    repeats: int,
    min_gene_terms: int,
) -> dict[str, Any]:
    counter = context["counter"]
    godag = context["godag"]
    termcounts = context["termcounts"]
    gene2terms = context["gene2terms"]

    if not gene_sizes:
        return {"method": method, "gene_sizes": [], "runs": {}}

    max_pairs = max(n_choose_two(size) for size in gene_sizes)
    genes, selection_meta = select_gene_candidates(
        gene2terms,
        needed_pairs=max_pairs,
        min_gene_terms=min_gene_terms,
    )

    max_gene_size = max(gene_sizes)
    if len(genes) < max_gene_size:
        raise RuntimeError(
            f"Requested all-vs-all benchmark for {max_gene_size} genes, but only {len(genes)} are available."
        )

    selected_genes = genes[:max_gene_size]

    if method == "wang":
        all_terms: set[str] = set()
        for gene in selected_genes:
            all_terms.update(gene2terms.get(gene, []))
        wang = SsWang(all_terms, godag, {"part_of"})
    else:
        wang = None
    goatools_term_sim = goatools_sim_factory(method, godag, termcounts, wang)

    points_go3_map: dict[int, dict[str, Any]] = {}
    points_goatools_map: dict[int, dict[str, Any]] = {}

    for n_genes in sorted(gene_sizes, reverse=True):
        genes_subset = selected_genes[:n_genes]
        pairs = all_unique_pairs(genes_subset)
        pair_count = len(pairs)

        def _go3() -> None:
            _ = go3.compare_gene_pairs_batch(pairs, namespace, method, "bma", counter)

        def _goatools() -> None:
            _ = [
                bma_one_pass(gene2terms.get(g1, []), gene2terms.get(g2, []), goatools_term_sim)
                for g1, g2 in pairs
            ]

        p_go3 = add_throughput(
            {
                "n": n_genes,
                "pair_count": pair_count,
                **time_runs(_go3, warmup=warmup, repeats=repeats),
            },
            pair_count,
        )
        p_gt = add_throughput(
            {
                "n": n_genes,
                "pair_count": pair_count,
                **time_runs(_goatools, warmup=warmup, repeats=repeats),
            },
            pair_count,
        )
        points_go3_map[n_genes] = p_go3
        points_goatools_map[n_genes] = p_gt

    ordered_gene_sizes = sorted(gene_sizes)
    points_go3 = [points_go3_map[n] for n in ordered_gene_sizes]
    points_goatools = [points_goatools_map[n] for n in ordered_gene_sizes]

    result = {
        "method": method,
        "gene_sizes": ordered_gene_sizes,
        "candidate_selection": selection_meta,
        "runs": {
            "go3": points_go3,
            "goatools": points_goatools,
        },
    }
    speedup = speedup_points(result["runs"]["go3"], result["runs"]["goatools"], label="go3_vs_goatools")
    result["go3_speedup_vs_goatools"] = speedup
    result["go3_speedup_summary"] = summarize_speedup(speedup)
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark go3 vs goatools for loading, batch term similarity, "
            "batch gene similarity, and all-vs-all gene workloads."
        ),
    )
    parser.add_argument("--obo", type=Path, default=None, help="Path to OBO file. Defaults to tests/go-basic.obo.")
    parser.add_argument("--gaf", type=Path, default=None, help="Path to GAF file. Defaults to tests/goa_human.gaf.")
    parser.add_argument("--outdir", type=Path, default=Path("imgs"), help="Output folder for plots and JSON.")
    parser.add_argument("--namespace", choices=["BP", "MF", "CC"], default="BP", help="GO subontology.")
    parser.add_argument(
        "--pair-sizes",
        default=None,
        help="Comma-separated sizes applied to both term and gene pair benchmarks (legacy shortcut).",
    )
    parser.add_argument(
        "--term-pair-sizes",
        default=",".join(str(v) for v in DEFAULT_TERM_PAIR_SIZES),
        help="Comma-separated sizes for term-pair benchmark.",
    )
    parser.add_argument(
        "--gene-pair-sizes",
        default=",".join(str(v) for v in DEFAULT_GENE_PAIR_SIZES),
        help="Comma-separated sizes for gene-pair benchmark.",
    )
    parser.add_argument(
        "--matrix-gene-sizes",
        default=",".join(str(v) for v in DEFAULT_MATRIX_GENE_SIZES),
        help="Comma-separated gene counts for all-vs-all benchmark.",
    )
    parser.add_argument(
        "--term-method",
        choices=sorted(VALID_METHODS),
        default="lin",
        help="Similarity method for term-pair benchmark.",
    )
    parser.add_argument(
        "--gene-method",
        choices=sorted(VALID_METHODS),
        default="lin",
        help="Similarity method for gene-pair and all-vs-all gene benchmarks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (not timed).")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per size (median reported).")
    parser.add_argument("--threads", type=int, default=8, help="Rayon threads for go3 (0 uses all cores).")
    parser.add_argument(
        "--min-gene-terms",
        type=int,
        default=8,
        help="Prefer genes with at least this many namespace terms for gene-level benchmarks.",
    )
    parser.add_argument("--include-gosemsim", action="store_true", help="Try to include GOSemSim (R) as third line.")
    parser.add_argument("--gosemsim-orgdb", default="org.Hs.eg.db", help="R OrgDb package for GOSemSim.")
    parser.add_argument(
        "--r-libs-user",
        default=None,
        help="Optional R library path (sets R_LIBS_USER for GOSemSim subprocesses).",
    )
    parser.add_argument(
        "--gosemsim-anno-tsv",
        default=None,
        help="Optional TSV with columns gene,GO,ONTOLOGY to build GOSemSim IC (overrides orgdb).",
    )
    parser.add_argument(
        "--gosemsim-measure",
        choices=sorted(VALID_METHODS),
        default=None,
        help="Optional GOSemSim measure override (default uses selected method).",
    )

    parser.add_argument("--no-term", action="store_true", help="Skip term-pair benchmark.")
    parser.add_argument("--no-gene", action="store_true", help="Skip gene-pair benchmark.")
    parser.add_argument("--no-all-vs-all", action="store_true", help="Skip all-vs-all gene benchmark.")
    parser.add_argument("--no-loading", action="store_true", help="Skip loading/memory benchmark.")
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help=(
            "Use paper-ready defaults (more repeats, publication-oriented size profile) "
            "and export high-resolution plots (PNG + SVG)."
        ),
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")

    parser.add_argument("--child", choices=["loading"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lib", choices=["go3", "goatools"], default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    raw_argv = sys.argv[1:]
    args = build_argparser().parse_args()
    apply_paper_profile(args, raw_argv)

    if args.child == "loading":
        if args.lib is None:
            raise SystemExit("--lib is required in --child loading mode")
        if args.obo is None or args.gaf is None:
            raise SystemExit("--obo and --gaf are required in --child loading mode")
        out = child_loading_metrics(args.lib, obo=args.obo, gaf=args.gaf)
        print(json.dumps(out))
        return 0

    gaf_default, obo_default = default_paths()
    obo = args.obo or obo_default
    gaf = args.gaf or gaf_default

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.pair_sizes:
        term_sizes = parse_int_list(args.pair_sizes)
        gene_sizes = parse_int_list(args.pair_sizes)
    else:
        term_sizes = parse_int_list(args.term_pair_sizes) if args.term_pair_sizes else list(DEFAULT_TERM_PAIR_SIZES)
        gene_sizes = parse_int_list(args.gene_pair_sizes) if args.gene_pair_sizes else list(DEFAULT_GENE_PAIR_SIZES)

    term_sizes = sorted(set(term_sizes))
    gene_sizes = sorted(set(gene_sizes))
    if not term_sizes:
        term_sizes = list(DEFAULT_TERM_PAIR_SIZES)
    if not gene_sizes:
        gene_sizes = list(DEFAULT_GENE_PAIR_SIZES)

    matrix_gene_sizes = parse_int_list(args.matrix_gene_sizes) if args.matrix_gene_sizes else []
    matrix_gene_sizes = sorted(set(matrix_gene_sizes))

    if args.threads is not None:
        go3.set_num_threads(int(args.threads))

    report: dict[str, Any] = {
        "profile": "paper_ready" if args.paper_ready else "default",
        "namespace": args.namespace,
        "term_pair_sizes": term_sizes,
        "gene_pair_sizes": gene_sizes,
        "matrix_gene_sizes": matrix_gene_sizes,
        "term_method": args.term_method,
        "gene_method": args.gene_method,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "threads": args.threads,
        "min_gene_terms": args.min_gene_terms,
        "r_libs_user": args.r_libs_user,
        "gosemsim_measure": args.gosemsim_measure,
        "python": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
        "system_metadata": collect_system_metadata(),
    }
    if args.pair_sizes:
        report["pair_sizes"] = term_sizes
    if args.paper_ready:
        report["paper_profile_reference"] = {
            "term_pair_sizes": PAPER_TERM_PAIR_SIZES,
            "gene_pair_sizes": PAPER_GENE_PAIR_SIZES,
            "matrix_gene_sizes": PAPER_MATRIX_GENE_SIZES,
            "warmup": PAPER_WARMUP,
            "repeats": PAPER_REPEATS,
            "threads": PAPER_THREADS,
            "min_gene_terms": PAPER_MIN_GENE_TERMS,
        }

    gosemsim_anno_tsv: Path | None = None
    if args.include_gosemsim:
        if args.gosemsim_anno_tsv:
            gosemsim_anno_tsv = Path(args.gosemsim_anno_tsv)
        else:
            gosemsim_anno_tsv = outdir / "gosemsim_goanno.tsv"
            write_gosemsim_goanno_tsv(gaf, gosemsim_anno_tsv)
        report["gosemsim_anno_tsv"] = str(gosemsim_anno_tsv)

    if not args.no_loading:
        loading_go3 = run_child_loading("go3", obo=obo, gaf=gaf)
        loading_go3["display_name"] = "go3"
        loading_goatools = run_child_loading("goatools", obo=obo, gaf=gaf)
        loading_goatools["display_name"] = "goatools"

        loading_map: dict[str, dict[str, Any]] = {
            "go3": loading_go3,
            "goatools": loading_goatools,
        }

        if args.include_gosemsim:
            gosemsim_loading, err = run_gosemsim_loading(
                namespace=args.namespace,
                orgdb=args.gosemsim_orgdb,
                anno_tsv=gosemsim_anno_tsv,
                r_libs_user=args.r_libs_user,
            )
            if gosemsim_loading is not None:
                gosemsim_loading["display_name"] = "GOSemSim"
                loading_map["gosemsim"] = gosemsim_loading
            if err is not None:
                report["gosemsim_loading_error"] = err

        report["loading"] = loading_map

        if "go3" in loading_map and "goatools" in loading_map:
            go3_time = float(loading_map["go3"]["total_time_s"])
            gt_time = float(loading_map["goatools"]["total_time_s"])
            go3_mem = float(loading_map["go3"]["peak_rss_mb"])
            gt_mem = float(loading_map["goatools"]["peak_rss_mb"])
            report["loading_summary"] = {
                "go3_vs_goatools_time_speedup_x": gt_time / max(go3_time, 1e-12),
                "go3_vs_goatools_peak_memory_reduction_x": gt_mem / max(go3_mem, 1e-12),
            }

        plot_loading_summary(
            out_path=outdir / "benchmark_loading_time_memory.png",
            loading=loading_map,
            paper_ready=bool(args.paper_ready),
        )

    run_term = not args.no_term
    run_gene = not args.no_gene
    run_all_vs_all = (not args.no_all_vs_all) and bool(matrix_gene_sizes)

    context: dict[str, Any] | None = None
    if run_term or run_gene or run_all_vs_all:
        context = prepare_context(obo=obo, gaf=gaf, namespace=args.namespace)

    if run_term:
        assert context is not None
        term_result = benchmark_term_pairs(
            context=context,
            method=args.term_method,
            sizes=term_sizes,
            namespace=args.namespace,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            include_gosemsim=bool(args.include_gosemsim),
            gosemsim_orgdb=args.gosemsim_orgdb,
            r_libs_user=args.r_libs_user,
            gosemsim_anno_tsv=gosemsim_anno_tsv,
            gosemsim_measure=args.gosemsim_measure,
            outdir=outdir,
        )
        report["term_pairs"] = term_result

        plot_runtime_curves(
            out_path=outdir / "benchmark_batch_similarity.png",
            title=f"Batch GO term similarity ({args.namespace}, {args.term_method})",
            xlabel="Number of GO term pairs",
            sizes=term_sizes,
            data=term_result["runs"],
            paper_ready=bool(args.paper_ready),
        )

    if run_gene:
        assert context is not None
        gene_result = benchmark_gene_pairs(
            context=context,
            method=args.gene_method,
            sizes=gene_sizes,
            namespace=args.namespace,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            min_gene_terms=args.min_gene_terms,
            include_gosemsim=bool(args.include_gosemsim),
            gosemsim_orgdb=args.gosemsim_orgdb,
            r_libs_user=args.r_libs_user,
            gosemsim_anno_tsv=gosemsim_anno_tsv,
            gosemsim_measure=args.gosemsim_measure,
            outdir=outdir,
        )
        report["gene_pairs"] = gene_result

        plot_runtime_curves(
            out_path=outdir / "benchmark_gene_batch_similarity.png",
            title=f"Batch gene similarity ({args.namespace}, {args.gene_method}, BMA)",
            xlabel="Number of gene pairs",
            sizes=gene_sizes,
            data=gene_result["runs"],
            paper_ready=bool(args.paper_ready),
        )

    if run_all_vs_all:
        assert context is not None
        all_vs_all_result = benchmark_all_vs_all_genes(
            context=context,
            method=args.gene_method,
            gene_sizes=matrix_gene_sizes,
            namespace=args.namespace,
            warmup=args.warmup,
            repeats=args.repeats,
            min_gene_terms=args.min_gene_terms,
        )
        report["all_vs_all_gene"] = all_vs_all_result

        plot_runtime_curves(
            out_path=outdir / "benchmark_all_vs_all_gene_similarity.png",
            title=f"All-vs-all gene similarity ({args.namespace}, {args.gene_method}, BMA)",
            xlabel="Number of genes in cohort",
            sizes=matrix_gene_sizes,
            data=all_vs_all_result["runs"],
            paper_ready=bool(args.paper_ready),
        )

    json_path = args.json or (outdir / "benchmark_results.json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
