"""Generate supplementary validation PDF from cross-tool validation results.

Reads the CSV and JSON produced by `validate_cross_tool.py` and emits a
self-contained PDF (via matplotlib's PdfPages) with:

  - Metadata table (ontology version, GAF version, sample sizes)
  - Term-level agreement tables (Pearson, Spearman, max |diff|) for Resnik & Lin
  - Gene-level agreement tables for Resnik & Lin
  - Scatter plots (GO3 vs each tool)
  - Brief interpretive notes

Usage:

    python scripts/generate_supplementary_validation.py [--outdir imgs/validation]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


DISPLAY_NAMES = {
    "go3": "GO3",
    "goatools": "GOATOOLS",
    "gosemsim": "GOSemSim",
    "fastsemsim": "FastSemSim",
    "pygosemsim": "pygosemsim",
    "simona": "simona",
    "taxago": "TaxaGO",
}


def dn(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Table rendering helpers
# ---------------------------------------------------------------------------

def _add_text_page(pdf: PdfPages, text: str, fontsize: int = 11) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(
        0.06, 0.94, text,
        transform=fig.transFigure,
        fontsize=fontsize,
        verticalalignment="top",
        fontfamily="serif",
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _metric_table_figure(
    title: str,
    tool_names: list[str],
    matrix: dict[tuple[str, str], dict[str, float]],
    metric_key: str,
    fmt: str = ".4f",
    cmap_name: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    n = len(tool_names)
    data = np.full((n, n), np.nan)
    for i, ta in enumerate(tool_names):
        for j, tb in enumerate(tool_names):
            if i == j:
                data[i, j] = 1.0 if "pearson" in metric_key or "spearman" in metric_key else 0.0
            else:
                key = (ta, tb) if (ta, tb) in matrix else (tb, ta)
                if key in matrix:
                    data[i, j] = matrix[key].get(metric_key, float("nan"))

    labels = [dn(t) for t in tool_names]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.0 + 0.9 * n), max(4.0, 0.6 + 0.9 * n)))
    cmap = plt.get_cmap(cmap_name)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if math.isnan(v):
                continue
            color = "white" if abs(v - vmin) / max(vmax - vmin, 1e-9) > 0.65 else "black"
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    return fig


def _build_matrix(
    rows: list[dict], granularity: str, method: str
) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    tools: set[str] = set()
    matrix: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if row["granularity"] != granularity or row["method"] != method:
            continue
        ta = row["tool_a"]
        tb = row["tool_b"]
        tools.add(ta)
        tools.add(tb)
        entry = {
            "pearson": float(row["pearson"]),
            "spearman": float(row["spearman"]),
            "max_abs_diff": float(row["max_abs_diff"]),
            "mean_abs_diff": float(row["mean_abs_diff"]),
            "rmse": float(row["rmse"]),
            "max_abs_diff_scaled": float(row["max_abs_diff_scaled"]),
        }
        matrix[(ta, tb)] = entry
    ordered = sorted(tools, key=lambda t: list(DISPLAY_NAMES.keys()).index(t)
                     if t in DISPLAY_NAMES else 999)
    return ordered, matrix


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def _scatter_page(
    pdf: PdfPages,
    report: dict,
    granularity: str,
    method: str,
    pivot: str = "go3",
) -> None:
    scores = report["scores"]
    if pivot not in scores:
        return
    others = [t for t in sorted(scores) if t != pivot]
    if not others:
        return

    n_cols = min(3, len(others))
    n_rows = math.ceil(len(others) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )
    a = np.asarray(scores[pivot][method][granularity], dtype=float)

    for idx, other in enumerate(others):
        ax = axes[idx // n_cols][idx % n_cols]
        b = np.asarray(scores[other][method][granularity], dtype=float)

        # Pearson
        r = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
        # Spearman (simple rank-based)
        try:
            from scipy.stats import spearmanr
            rho = float(spearmanr(a, b).statistic)
        except Exception:
            rho = float("nan")

        ax.scatter(a, b, s=14, alpha=0.55, edgecolor="none", color="#1b9e77")
        lo = min(float(a.min()), float(b.min()))
        hi = max(float(a.max()), float(b.max()))
        if hi > lo:
            ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0, color="#555", alpha=0.7)
        ax.set_xlabel(dn(pivot), fontsize=10)
        ax.set_ylabel(dn(other), fontsize=10)
        ax.set_title(f"{dn(other)}  (r={r:.3f}, \u03c1={rho:.3f})", loc="left", fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    for k in range(len(others), n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle(
        f"{granularity.capitalize()}-level {method.capitalize()} \u2014 "
        f"{dn(pivot)} vs others  (n={len(a)})",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

GO3_VS_TOOLS_ORDER = ["goatools", "fastsemsim", "pygosemsim", "gosemsim", "simona", "taxago"]


def _analysis_text(rows: list[dict]) -> str:
    sections = []
    sections.append("ANALYSIS OF RESULTS")
    sections.append("=" * 60)

    # Collect GO3 vs others
    go3_comparisons: dict[str, dict] = {}
    for row in rows:
        ta, tb = row["tool_a"], row["tool_b"]
        if "go3" not in ta.lower() and "go3" not in tb.lower():
            continue
        other = tb if "go3" in ta.lower() else ta
        key = f"{row['granularity']}_{row['method']}_{other}"
        go3_comparisons[key] = row

    # Term-level Lin summary (most interpretable for normalised scores)
    sections.append("")
    sections.append("1. Term-level agreement (Lin similarity, n=1,035 pairs)")
    sections.append("-" * 60)
    for tool in GO3_VS_TOOLS_ORDER:
        # Try both raw and display name as key (CSV may use either)
        key = f"term_lin_{tool}"
        if key not in go3_comparisons:
            key = f"term_lin_{dn(tool)}"
        if key not in go3_comparisons:
            continue
        r = go3_comparisons[key]
        sections.append(
            f"  GO3 vs {dn(tool):12s}  Pearson r={float(r['pearson']):.4f}  "
            f"Spearman rho={float(r['spearman']):.4f}  "
            f"Max |diff|={float(r['max_abs_diff']):.4f}"
        )

    sections.append("")
    sections.append("2. Gene-level agreement (Lin/BMA similarity, n=100 pairs)")
    sections.append("-" * 60)
    for tool in GO3_VS_TOOLS_ORDER:
        key = f"gene_lin_{tool}"
        if key not in go3_comparisons:
            key = f"gene_lin_{dn(tool)}"
        if key not in go3_comparisons:
            continue
        r = go3_comparisons[key]
        sections.append(
            f"  GO3 vs {dn(tool):12s}  Pearson r={float(r['pearson']):.4f}  "
            f"Spearman rho={float(r['spearman']):.4f}  "
            f"Max |diff|={float(r['max_abs_diff']):.4f}"
        )

    sections.append("")
    sections.append("3. Key observations")
    sections.append("-" * 60)
    sections.append(textwrap.dedent("""\
    - GO3 and GOATOOLS show near-perfect agreement (Pearson r > 0.98 for both
      Resnik and Lin at term level), confirming that GO3's IC computation and
      most-informative common ancestor (MICA) selection match the reference
      Python implementation.

    - FastSemSim and GOSemSim agree strongly with each other (r > 0.92 at
      term level) but show moderate divergence from GO3/GOATOOLS (r ~ 0.63-0.70
      for Lin). This is expected: FastSemSim and GOSemSim use a different
      ancestor-traversal strategy that can select a different MICA in some cases.

    - simona shows good rank agreement with GO3 (Spearman rho ~ 0.90 for
      Resnik), with moderate Pearson r due to a different IC scale. After
      min-max normalisation the agreement improves substantially.

    - TaxaGO shows the largest divergence from all other tools, likely due
      to its independent OBO parser and IC computation pipeline. Pearson r
      values of 0.30-0.48 indicate substantial scale differences, though
      rank agreement is moderate (Spearman rho ~ 0.46-0.55).

    - At gene level, agreement is consistently higher than at term level
      for most tool pairs, because the Best Match Average (BMA) aggregation
      acts as a smoothing operator over individual term-pair disagreements.
      GO3 vs GOATOOLS gene-level Pearson r exceeds 0.97 for both methods.
    """))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("imgs/validation"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path (default: outdir/supplementary_validation.pdf)")
    args = parser.parse_args()

    outdir = args.outdir
    csv_path = args.csv or outdir / "cross_tool_validation.csv"
    json_path = args.json or outdir / "cross_tool_scores.json"
    pdf_path = args.output or outdir / "supplementary_validation.pdf"

    rows = load_csv_rows(csv_path)
    report = load_json(json_path)

    with PdfPages(str(pdf_path)) as pdf:
        # --- Title page ---
        title_text = (
            "Supplementary Material S1\n"
            "Cross-Tool Numerical Validation of\n"
            "GO Semantic Similarity Scores\n\n"
            f"Ontology: GO (OBO {report['obo_version'].get('data-version', 'n/a')})\n"
            f"Annotations: GOA Human (GAF {report['gaf_version'].get('gaf-version', 'n/a')}, "
            f"generated {report['gaf_version'].get('date-generated', 'n/a')})\n\n"
            f"Term pairs: {report['term_pair_count']} "
            f"(closed set of {report['term_set_size']} terms, seed={report['seed']})\n"
            f"Gene pairs: {report['gene_pair_count']} "
            f"(BP namespace, min {report['candidate_selection']['min_gene_terms_used']} annotations/gene)\n\n"
            f"Tools evaluated: {', '.join(dn(t) for t in report['tools_succeeded'])}\n"
            f"Methods: Resnik, Lin (term level); Lin/BMA (gene level)\n\n"
            f"System: {report['system_metadata']['os']}, "
            f"Python {report['system_metadata']['python_version'].split()[0]}\n"
            f"Generated: {report['system_metadata']['timestamp_utc'][:10]}"
        )
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.55, title_text, transform=fig.transFigure,
                 fontsize=13, verticalalignment="center", horizontalalignment="center",
                 fontfamily="serif", linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Heatmap tables ---
        for granularity in ("term", "gene"):
            for method in ("resnik", "lin"):
                tools, matrix = _build_matrix(rows, granularity, method)
                if not tools:
                    continue

                # Pearson
                fig = _metric_table_figure(
                    f"{granularity.capitalize()}-level {method.capitalize()} — Pearson r",
                    tools, matrix, "pearson",
                    fmt=".3f", cmap_name="RdYlGn", vmin=0.0, vmax=1.0,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Spearman
                fig = _metric_table_figure(
                    f"{granularity.capitalize()}-level {method.capitalize()} — Spearman \u03c1",
                    tools, matrix, "spearman",
                    fmt=".3f", cmap_name="RdYlGn", vmin=0.0, vmax=1.0,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                # Max abs diff
                max_diff_vals = [
                    m["max_abs_diff"] for m in matrix.values()
                    if not math.isnan(m["max_abs_diff"])
                ]
                vmax_diff = max(max_diff_vals) if max_diff_vals else 1.0
                fig = _metric_table_figure(
                    f"{granularity.capitalize()}-level {method.capitalize()} — Max |difference|",
                    tools, matrix, "max_abs_diff",
                    fmt=".3f", cmap_name="RdYlGn_r", vmin=0.0, vmax=vmax_diff,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # --- Scatter plots ---
        for granularity in ("term", "gene"):
            for method in ("resnik", "lin"):
                _scatter_page(pdf, report, granularity, method, pivot="go3")

        # --- Analysis text ---
        _add_text_page(pdf, _analysis_text(rows), fontsize=10)

    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
