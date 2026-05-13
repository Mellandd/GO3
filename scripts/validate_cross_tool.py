"""Cross-tool numerical validation of GO semantic similarity scores.

Runs every available library on exactly the same sample of term and gene
pairs and emits:

* `cross_tool_scores.json` — the raw per-tool score vectors (aligned by
  pair index) + metadata (sampled pairs, OBO/GAF versions, system info).
* `cross_tool_validation.csv` — pairwise agreement metrics (Pearson r,
  Spearman rho, max/mean absolute difference raw and min-max normalised,
  RMSE) for every (tool_a, tool_b, granularity, method) combination.
* (optional, --plot) `cross_tool_scatter_<granularity>_<method>.png` —
  a grid of go3-vs-other-tool scatter plots per method.

Designed for the paper's supplementary validation table. Sampling mirrors
`scripts/benchmark_all.py` so the term/gene sets are reproducible with
the same `--seed`.

Example:

    python scripts/validate_cross_tool.py \
        --term-pairs 1000 --gene-pairs 100 \
        --outdir imgs/validation --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from runners import available_runners  # noqa: E402
from runners._gaf import NAMESPACE_TO_ASPECT, parse_symbol_to_terms  # noqa: E402
from runners._proc import run_with_rusage  # noqa: E402
from runners._r import build_r_env, measure_for_method, write_anno_tsv  # noqa: E402

from benchmark_all import (  # noqa: E402
    all_unique_pairs,
    collect_system_metadata,
    default_paths,
    n_for_pair_target,
    parse_gaf_version,
    parse_obo_version,
    sample_disjoint_pair_groups,
    select_gene_candidates,
    select_term_candidates_for_namespace,
)


METHODS = ("resnik", "lin")
GRANULARITIES = ("term", "gene")

GOSEMSIM_R = SCRIPTS_DIR / "benchmark_gosemsim.R"
SIMONA_R = SCRIPTS_DIR / "runners" / "r_helpers" / "benchmark_simona.R"

# Display names — must match LIB_STYLES in benchmark_all.py exactly.
DISPLAY_NAMES: dict[str, str] = {
    "go3": "GO3",
    "goatools": "GOATOOLS",
    "gosemsim": "GOSemSim",
    "fastsemsim": "FastSemSim",
    "pygosemsim": "pygosemsim",
    "simona": "simona",
    "taxago": "TaxaGO",
}


def display_name(tool: str) -> str:
    return DISPLAY_NAMES.get(tool, tool)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).statistic)
    except Exception:
        return _pearson(_rankdata(a), _rankdata(b))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties broken as average), scipy-compatible."""
    try:
        from scipy.stats import rankdata as sp_rankdata

        return np.asarray(sp_rankdata(x, method="average"), dtype=float)
    except Exception:
        order = x.argsort(kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        return ranks


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def compute_metrics(a_raw: list[float], b_raw: list[float]) -> dict[str, float]:
    a = np.asarray(a_raw, dtype=float)
    b = np.asarray(b_raw, dtype=float)
    diff = np.abs(a - b)
    a_s = _minmax(a)
    b_s = _minmax(b)
    diff_s = np.abs(a_s - b_s)
    return {
        "n": int(a.size),
        "pearson": _pearson(a, b),
        "spearman": _spearman(a, b),
        "max_abs_diff": float(diff.max()) if a.size else float("nan"),
        "mean_abs_diff": float(diff.mean()) if a.size else float("nan"),
        "rmse": float(np.sqrt((diff ** 2).mean())) if a.size else float("nan"),
        "max_abs_diff_scaled": float(diff_s.max()) if a.size else float("nan"),
        "mean_abs_diff_scaled": float(diff_s.mean()) if a.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-tool score computation (each function returns (term_scores, gene_scores)
# aligned with the shared `term_pairs` / `gene_pairs` inputs).
#
# All functions reload the library fresh and recompute IC from the provided
# GAF so every tool sees the same corpus. Heavy imports are local so the
# script doesn't require every library to be installed.
# ---------------------------------------------------------------------------


def scores_go3(
    obo: Path, gaf: Path, namespace: str, method: str,
    term_pairs: list[tuple[str, str]], gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
) -> tuple[list[float], list[float]]:
    import go3  # type: ignore

    go3.load_go_terms(str(obo))
    annotations = go3.load_gaf(str(gaf))
    counter = go3.build_term_counter(annotations)

    list1 = [a for a, _ in term_pairs]
    list2 = [b for _, b in term_pairs]
    term_scores = [float(v) for v in go3.batch_similarity(list1, list2, method, counter)]
    gene_scores = [
        float(v)
        for v in go3.compare_gene_pairs_batch(gene_pairs, namespace, method, "bma", counter)
    ]
    return term_scores, gene_scores


def scores_goatools(
    obo: Path, gaf: Path, namespace: str, method: str,
    term_pairs: list[tuple[str, str]], gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
) -> tuple[list[float], list[float]]:
    from goatools.anno.gaf_reader import GafReader  # type: ignore
    from goatools.obo_parser import GODag  # type: ignore
    from goatools.semantic import TermCounts, lin_sim, resnik_sim  # type: ignore

    from runners.goatools_runner import _bma_one_pass

    godag = GODag(str(obo), optional_attrs={"relationship"}, prt=None)
    reader = GafReader(str(gaf), godag=godag, prt=None)
    id2gos = reader.get_id2gos_nss(prt=None)
    termcounts = TermCounts(godag, id2gos, prt=None)

    if method == "resnik":
        def sim(a: str, b: str) -> float:
            v = resnik_sim(a, b, godag, termcounts)
            return float(v) if v is not None else 0.0
    elif method == "lin":
        def sim(a: str, b: str) -> float:
            v = lin_sim(a, b, godag, termcounts, dfltval=0.0)
            return float(v) if v is not None else 0.0
    else:
        raise ValueError(f"Unsupported method: {method}")

    term_scores = [sim(a, b) for a, b in term_pairs]
    gene_scores = [
        float(_bma_one_pass(gene2terms.get(g1, []), gene2terms.get(g2, []), sim))
        for g1, g2 in gene_pairs
    ]
    return term_scores, gene_scores


def scores_fastsemsim(
    obo: Path, gaf: Path, namespace: str, method: str,
    term_pairs: list[tuple[str, str]], gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
) -> tuple[list[float], list[float]]:
    import fastsemsim  # type: ignore
    from fastsemsim.semsim import BMASemSim, LinSemSim, ResnikSemSim, SemSimUtils  # type: ignore

    ontology = fastsemsim.load_ontology(
        source_file=str(obo), ontology_type="GeneOntology", file_type="obo"
    )
    ac = fastsemsim.load_ac(
        ontology, source_file=str(gaf), file_type="gaf-2.0", species="human"
    )
    util = SemSimUtils(ontology, ac)
    try:
        util.det_IC_table()
    except Exception:
        pass

    if method == "resnik":
        term_sim = ResnikSemSim(ontology, ac, util)
    elif method == "lin":
        term_sim = LinSemSim(ontology, ac, util)
    else:
        raise ValueError(f"Unsupported method: {method}")
    bma = BMASemSim(ontology, ac, util)

    def _safe_term(a: str, b: str) -> float:
        try:
            v = term_sim.SemSim(a, b)
        except Exception:
            return 0.0
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return float(v)

    term_scores = [_safe_term(a, b) for a, b in term_pairs]
    gene_scores: list[float] = []
    for g1, g2 in gene_pairs:
        t1 = gene2terms.get(g1, [])
        t2 = gene2terms.get(g2, [])
        if not t1 or not t2:
            gene_scores.append(0.0)
            continue
        try:
            v = bma.SemSim(t1, t2, term_sim)
        except Exception:
            v = 0.0
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = 0.0
        gene_scores.append(float(v))
    return term_scores, gene_scores


def scores_pygosemsim(
    obo: Path, gaf: Path, namespace: str, method: str,
    term_pairs: list[tuple[str, str]], gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
) -> tuple[list[float], list[float]]:
    import functools

    from pygosemsim import annotation as py_annotation  # type: ignore
    from pygosemsim import graph as py_graph  # type: ignore
    from pygosemsim import similarity as py_similarity  # type: ignore
    from pygosemsim import term_set as py_term_set  # type: ignore

    from runners._gaf import parse_symbol_to_uniprot
    from runners.pygosemsim_runner import _stage_resources, _terms_for_symbol

    _stage_resources(obo, gaf)
    G = py_graph.from_resource("go-basic")
    py_graph.precalc_lower_bounds(G)
    annot = py_annotation.from_resource("goa_human")
    sym2uni = parse_symbol_to_uniprot(gaf)

    if method == "resnik":
        sim_fn = py_similarity.resnik
    elif method == "lin":
        sim_fn = py_similarity.lin
    else:
        raise ValueError(f"Unsupported method: {method}")

    def _safe_term(a: str, b: str) -> float:
        try:
            v = sim_fn(G, a, b)
        except Exception:
            return 0.0
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return float(v)

    term_scores = [_safe_term(a, b) for a, b in term_pairs]

    sf = functools.partial(py_term_set.sim_func, G, sim_fn)
    gene_scores: list[float] = []
    for g1, g2 in gene_pairs:
        t1 = _terms_for_symbol(g1, annot, sym2uni)
        t2 = _terms_for_symbol(g2, annot, sym2uni)
        if not t1 or not t2:
            gene_scores.append(0.0)
            continue
        try:
            v = py_term_set.sim_bma(t1, t2, sf)
        except Exception:
            v = 0.0
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = 0.0
        gene_scores.append(float(v))
    return term_scores, gene_scores


# ---- R-based tools --------------------------------------------------------


def _write_term_pairs_scores_tsv(path: Path, term_pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for a, b in term_pairs:
            handle.write(f"1\t{a}\t{b}\n")


def _write_gene_pairs_scores_tsv(
    path: Path,
    gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for g1, g2 in gene_pairs:
            t1 = ",".join(gene2terms.get(g1, []))
            t2 = ",".join(gene2terms.get(g2, []))
            handle.write(f"1\t{t1}\t{t2}\n")


def _run_r_scores(cmd: list[str]) -> list[float]:
    proc = run_with_rusage(cmd, env=build_r_env(None), check=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines or lines[0].strip() != "score":
        raise RuntimeError(
            f"Unexpected R --mode scores output.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    out: list[float] = []
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            out.append(0.0)
    return out


def scores_gosemsim(
    obo: Path, gaf: Path, namespace: str, method: str,
    term_pairs: list[tuple[str, str]], gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
    *, workdir: Path,
) -> tuple[list[float], list[float]]:
    wd = workdir / "gosemsim"
    wd.mkdir(parents=True, exist_ok=True)
    anno_tsv = wd / "anno.tsv"
    if not anno_tsv.exists():
        write_anno_tsv(gaf, anno_tsv)

    term_tsv = wd / f"term_pairs_{namespace}_{method}.tsv"
    _write_term_pairs_scores_tsv(term_tsv, term_pairs)
    gene_tsv = wd / f"gene_pairs_{namespace}_{method}.tsv"
    _write_gene_pairs_scores_tsv(gene_tsv, gene_pairs, gene2terms)

    base_cmd = [
        "Rscript", str(GOSEMSIM_R),
        "--ontology", namespace,
        "--measure", measure_for_method(method),
        "--anno-tsv", str(anno_tsv),
    ]
    term_scores = _run_r_scores(base_cmd + ["--mode", "scores-term", "--pairs-tsv", str(term_tsv)])
    gene_scores = _run_r_scores(base_cmd + ["--mode", "scores-gene", "--pairs-tsv", str(gene_tsv)])
    if len(term_scores) != len(term_pairs):
        raise RuntimeError(
            f"GOSemSim returned {len(term_scores)} term scores for {len(term_pairs)} pairs"
        )
    if len(gene_scores) != len(gene_pairs):
        raise RuntimeError(
            f"GOSemSim returned {len(gene_scores)} gene scores for {len(gene_pairs)} pairs"
        )
    return term_scores, gene_scores


def scores_simona(
    obo: Path, gaf: Path, namespace: str, method: str,
    term_pairs: list[tuple[str, str]], gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
    *, workdir: Path,
) -> tuple[list[float], list[float]]:
    wd = workdir / "simona"
    wd.mkdir(parents=True, exist_ok=True)
    anno_tsv = wd / "anno.tsv"
    if not anno_tsv.exists():
        write_anno_tsv(gaf, anno_tsv)

    term_tsv = wd / f"term_pairs_{namespace}_{method}.tsv"
    _write_term_pairs_scores_tsv(term_tsv, term_pairs)
    gene_tsv = wd / f"gene_pairs_{namespace}_{method}.tsv"
    _write_gene_pairs_scores_tsv(gene_tsv, gene_pairs, gene2terms)

    base_cmd = [
        "Rscript", str(SIMONA_R),
        "--ontology", namespace,
        "--measure", measure_for_method(method),
        "--obo", str(obo),
        "--anno-tsv", str(anno_tsv),
    ]
    term_scores = _run_r_scores(base_cmd + ["--mode", "scores-term", "--pairs-tsv", str(term_tsv)])
    gene_scores = _run_r_scores(base_cmd + ["--mode", "scores-gene", "--pairs-tsv", str(gene_tsv)])
    if len(term_scores) != len(term_pairs):
        raise RuntimeError(
            f"simona returned {len(term_scores)} term scores for {len(term_pairs)} pairs"
        )
    if len(gene_scores) != len(gene_pairs):
        raise RuntimeError(
            f"simona returned {len(gene_scores)} gene scores for {len(gene_pairs)} pairs"
        )
    return term_scores, gene_scores


def scores_taxago(
    obo: Path, gaf: Path, namespace: str, method: str,
    term_pairs: list[tuple[str, str]], gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
    *, workdir: Path,
) -> tuple[list[float], list[float]]:
    import os as _os

    from runners.taxago_runner import _binary, _bma, _matrix_path, _parse_matrix

    binary = _binary()
    if binary is None:
        raise RuntimeError(
            "taxago binary not found (set TAXAGO_SEMSIM_BIN or TAXAGO_BIN)"
        )
    wd = workdir / "taxago"
    wd.mkdir(parents=True, exist_ok=True)

    method_flag = method.lower()

    def _run(terms: list[str], tag: str) -> dict[tuple[str, str], float]:
        terms_file = wd / f"terms_{tag}.txt"
        terms_file.write_text("\n".join(terms) + "\n", encoding="utf-8")
        outdir = wd / f"out_{tag}"
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            binary,
            "-p",
            "-o", str(obo),
            "-t", str(terms_file),
            "-m", method_flag,
            "-i", "9606",
            "-d", str(outdir),
        ]
        proc = run_with_rusage(cmd, env=_os.environ.copy())
        if proc.returncode != 0:
            raise RuntimeError(f"taxago failed ({tag}):\n{proc.stderr}")
        return _parse_matrix(_matrix_path(outdir, method_flag))

    # Term pairs: single invocation over the union of terms in the closed
    # set; read back the N×N matrix and extract aligned scores.
    term_union: list[str] = []
    seen_t: set[str] = set()
    for a, b in term_pairs:
        if a not in seen_t:
            seen_t.add(a)
            term_union.append(a)
        if b not in seen_t:
            seen_t.add(b)
            term_union.append(b)
    term_matrix = _run(term_union, f"term_{method_flag}")
    term_scores: list[float] = []
    for a, b in term_pairs:
        v = term_matrix.get((a, b))
        if v is None:
            v = term_matrix.get((b, a), 0.0)
        term_scores.append(float(v))

    # Gene pairs: single invocation over the union of annotations across all
    # sampled gene pairs; BMA computed in Python (matches the runner).
    gene_union: list[str] = []
    seen_g: set[str] = set()
    for g1, g2 in gene_pairs:
        for t in gene2terms.get(g1, []) + gene2terms.get(g2, []):
            if t not in seen_g:
                seen_g.add(t)
                gene_union.append(t)
    gene_matrix = _run(gene_union, f"gene_{method_flag}")
    gene_scores: list[float] = []
    for g1, g2 in gene_pairs:
        t1 = gene2terms.get(g1, [])
        t2 = gene2terms.get(g2, [])
        if not t1 or not t2:
            gene_scores.append(0.0)
            continue
        gene_scores.append(float(_bma(t1, t2, gene_matrix)))
    return term_scores, gene_scores


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


SCORE_FUNCS: dict[str, Any] = {
    "go3": scores_go3,
    "goatools": scores_goatools,
    "fastsemsim": scores_fastsemsim,
    "pygosemsim": scores_pygosemsim,
    "gosemsim": scores_gosemsim,
    "simona": scores_simona,
    "taxago": scores_taxago,
}

# Tools that accept a `workdir` kwarg (R + taxago need one for scratch files).
NEEDS_WORKDIR = {"gosemsim", "simona", "taxago"}


def compute_all_scores(
    tool: str,
    *,
    obo: Path,
    gaf: Path,
    namespace: str,
    term_pairs: list[tuple[str, str]],
    gene_pairs: list[tuple[str, str]],
    gene2terms: dict[str, list[str]],
    workdir: Path,
) -> dict[str, dict[str, list[float]]]:
    """Return `{method: {"term": [...], "gene": [...]}}` for one tool."""
    fn = SCORE_FUNCS[tool]
    out: dict[str, dict[str, list[float]]] = {}
    for method in METHODS:
        print(f"  [{tool}/{method}] computing scores ...", flush=True)
        kwargs: dict[str, Any] = {}
        if tool in NEEDS_WORKDIR:
            kwargs["workdir"] = workdir
        term_scores, gene_scores = fn(
            obo, gaf, namespace, method, term_pairs, gene_pairs, gene2terms, **kwargs
        )
        out[method] = {"term": term_scores, "gene": gene_scores}
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scatter_grid(
    outdir: Path,
    tool_names: list[str],
    scores: dict[str, dict[str, dict[str, list[float]]]],
    pivot: str,
    paper_ready: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    others = [t for t in tool_names if t != pivot]
    if not others:
        return

    n_cols = min(3, len(others))
    n_rows = math.ceil(len(others) / n_cols)

    rc = {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    } if paper_ready else {}

    for granularity in GRANULARITIES:
        for method in METHODS:
            with plt.rc_context(rc):
                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(4.0 * n_cols, 3.8 * n_rows),
                    squeeze=False,
                )
                a = np.asarray(scores[pivot][method][granularity], dtype=float)
                for idx, other in enumerate(others):
                    ax = axes[idx // n_cols][idx % n_cols]
                    b = np.asarray(scores[other][method][granularity], dtype=float)
                    r = _pearson(a, b)
                    rho = _spearman(a, b)
                    ax.scatter(a, b, s=14, alpha=0.55, edgecolor="none",
                               color="#1b9e77")
                    lo = min(float(a.min()), float(b.min()))
                    hi = max(float(a.max()), float(b.max()))
                    if hi > lo:
                        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0,
                                color="#555", alpha=0.7)
                    ax.set_xlabel(display_name(pivot))
                    ax.set_ylabel(display_name(other))
                    ax.set_title(
                        f"{display_name(other)}  (r={r:.3f}, ρ={rho:.3f})",
                        loc="left",
                    )
                    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
                # Hide unused axes
                for k in range(len(others), n_rows * n_cols):
                    axes[k // n_cols][k % n_cols].axis("off")
                fig.suptitle(
                    f"{granularity.capitalize()}-level {method.capitalize()} — "
                    f"{display_name(pivot)} vs others  (n={len(a)})",
                    y=1.02,
                )
                fig.tight_layout()
                out_path = outdir / f"cross_tool_scatter_{granularity}_{method}.png"
                fig.savefig(out_path, dpi=320 if paper_ready else 180,
                            bbox_inches="tight")
                if paper_ready:
                    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
                plt.close(fig)
                print(f"  wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--obo", type=Path, default=None)
    p.add_argument("--gaf", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=Path("imgs/validation"))
    p.add_argument("--namespace", choices=["BP", "MF", "CC"], default="BP")
    p.add_argument("--term-pairs", type=int, default=1000,
                   help="Target number of term pairs (closed set with C(N,2) >= target).")
    p.add_argument("--gene-pairs", type=int, default=100,
                   help="Number of gene pairs to sample.")
    p.add_argument("--min-gene-terms", type=int, default=8,
                   help="Minimum annotation depth for candidate genes.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", default=None,
                   help="Comma-separated tool names to include.")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated tool names to exclude.")
    p.add_argument("--plot", action="store_true",
                   help="Write scatter plots (go3 vs each other tool per method).")
    p.add_argument("--pivot", default="go3",
                   help="Tool used as the x-axis in scatter plots (default: go3).")
    p.add_argument("--paper-ready", action="store_true",
                   help="Use publication-grade styling and emit SVG next to PNG.")
    p.add_argument("--json", type=Path, default=None,
                   help="Path for the raw scores JSON (default: outdir/cross_tool_scores.json).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Path for the validation CSV (default: outdir/cross_tool_validation.csv).")
    return p


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "granularity", "method", "tool_a", "tool_b", "n",
        "pearson", "spearman",
        "max_abs_diff", "mean_abs_diff", "rmse",
        "max_abs_diff_scaled", "mean_abs_diff_scaled",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    exclude = {s.strip() for s in args.exclude.split(",")} if args.exclude else set()

    obo_default, gaf_default = default_paths()
    obo = args.obo or obo_default
    gaf = args.gaf or gaf_default
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover runners (same availability check as the benchmark).
    runners = available_runners(only=only)
    if exclude:
        runners = {n: c for n, c in runners.items() if n not in exclude}
    runners = {n: c for n, c in runners.items() if n in SCORE_FUNCS}
    if not runners:
        raise SystemExit("No tools available for validation.")
    tool_names = sorted(runners)

    # ---- Sample shared pairs ------------------------------------------------
    rng_terms = random.Random(args.seed)
    candidates = select_term_candidates_for_namespace(obo, gaf, args.namespace)
    n_terms = n_for_pair_target(args.term_pairs)
    if n_terms > len(candidates):
        raise SystemExit(
            f"Need {n_terms} terms for {args.term_pairs} pairs, "
            f"but only {len(candidates)} candidates are available."
        )
    term_set = rng_terms.sample(candidates, n_terms)
    term_pairs = all_unique_pairs(term_set)

    aspect = NAMESPACE_TO_ASPECT[args.namespace]
    gene2terms = parse_symbol_to_terms(gaf, namespace_aspect=aspect)
    rng_genes = random.Random(args.seed + 1)
    genes, sel_meta = select_gene_candidates(
        gene2terms, needed_pairs=args.gene_pairs, min_gene_terms=args.min_gene_terms,
    )
    gene_pairs = sample_disjoint_pair_groups(genes, [args.gene_pairs], rng_genes)[args.gene_pairs]

    print(
        f"Validation cohort: {len(term_pairs)} term pairs (closed set of "
        f"{n_terms}), {len(gene_pairs)} gene pairs. "
        f"Tools: {[display_name(t) for t in tool_names]}",
        flush=True,
    )

    # ---- Compute scores per tool -------------------------------------------
    scores: dict[str, dict[str, dict[str, list[float]]]] = {}
    workdir = outdir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    failed: dict[str, str] = {}
    for tool in tool_names:
        print(f"[{tool}] computing scores ...", flush=True)
        try:
            scores[tool] = compute_all_scores(
                tool, obo=obo, gaf=gaf, namespace=args.namespace,
                term_pairs=term_pairs, gene_pairs=gene_pairs,
                gene2terms=gene2terms, workdir=workdir,
            )
        except Exception as exc:
            failed[tool] = f"{type(exc).__name__}: {exc}"
            print(f"[{tool}] FAILED -> {failed[tool]}", flush=True)
    scored_tools = sorted(scores)
    if len(scored_tools) < 2:
        raise SystemExit(
            f"Need at least 2 tools to compute pairwise metrics; got {scored_tools}"
        )

    # ---- Pairwise metrics --------------------------------------------------
    rows: list[dict[str, Any]] = []
    for granularity in GRANULARITIES:
        for method in METHODS:
            for i, ta in enumerate(scored_tools):
                for tb in scored_tools[i + 1:]:
                    a = scores[ta][method][granularity]
                    b = scores[tb][method][granularity]
                    if len(a) != len(b):
                        print(
                            f"  skip {ta} vs {tb} ({granularity}/{method}): "
                            f"length mismatch {len(a)} vs {len(b)}",
                            flush=True,
                        )
                        continue
                    m = compute_metrics(a, b)
                    row = {
                        "granularity": granularity,
                        "method": method,
                        "tool_a": display_name(ta),
                        "tool_b": display_name(tb),
                        **m,
                    }
                    rows.append(row)

    csv_path = args.csv or (outdir / "cross_tool_validation.csv")
    write_csv(csv_path, rows)
    print(f"\nWrote {csv_path}", flush=True)

    # ---- Raw scores + metadata --------------------------------------------
    report: dict[str, Any] = {
        "namespace": args.namespace,
        "seed": args.seed,
        "term_pair_target": args.term_pairs,
        "term_set_size": n_terms,
        "term_pair_count": len(term_pairs),
        "gene_pair_count": len(gene_pairs),
        "candidate_selection": sel_meta,
        "obo": str(obo),
        "gaf": str(gaf),
        "obo_version": parse_obo_version(obo),
        "gaf_version": parse_gaf_version(gaf),
        "system_metadata": collect_system_metadata(),
        "tools_succeeded": [display_name(t) for t in scored_tools],
        "tools_failed": {display_name(k): v for k, v in failed.items()},
        "term_pairs": [list(p) for p in term_pairs],
        "gene_pairs": [list(p) for p in gene_pairs],
        "scores": {display_name(k): v for k, v in scores.items()},
        "metrics": rows,
        "metric_notes": {
            "pearson": "Scale-invariant; strongest headline metric for 'do the tools agree?'.",
            "spearman": "Rank-based; robust to monotonic transforms (e.g., log-base differences in Resnik).",
            "max_abs_diff": "Raw max |a-b|. Directly interpretable for Lin/BMA-Lin (both in [0,1]); "
                            "Resnik values are scale-dependent across tools so interpret with caution.",
            "max_abs_diff_scaled": "Same, but each vector min-max normalised first — useful for Resnik.",
        },
    }
    json_path = args.json or (outdir / "cross_tool_scores.json")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)

    # ---- Plots -------------------------------------------------------------
    if args.plot:
        pivot = args.pivot if args.pivot in scored_tools else scored_tools[0]
        print(f"Plotting scatter grid (pivot={pivot}) ...", flush=True)
        plot_scatter_grid(
            outdir, scored_tools, scores,
            pivot=pivot, paper_ready=args.paper_ready,
        )

    # ---- Console summary ---------------------------------------------------
    print("\nSummary (Pearson r, GO3 vs others) — term Lin / gene Lin:")
    pivot = "go3" if "go3" in scored_tools else scored_tools[0]
    for other in [t for t in scored_tools if t != pivot]:
        t_lin = compute_metrics(scores[pivot]["lin"]["term"], scores[other]["lin"]["term"])
        g_lin = compute_metrics(scores[pivot]["lin"]["gene"], scores[other]["lin"]["gene"])
        print(
            f"  {display_name(pivot)} vs {display_name(other):12s}  "
            f"term r={t_lin['pearson']:.4f}  gene r={g_lin['pearson']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
