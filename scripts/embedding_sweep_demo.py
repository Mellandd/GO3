#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import go3

from demo_utils import auto_n_neighbors, auto_perplexity, default_paths, pick_genes_from_gaf


@dataclass(frozen=True)
class DistanceConfig:
    title: str
    ontology: str
    similarity: str
    groupwise: str
    distance_transform: str


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GO3 embedding sweep demo. Builds gene distance matrices with different settings "
            "and compares the resulting t-SNE/UMAP embeddings side-by-side."
        )
    )
    default_gaf, default_obo = default_paths()

    parser.add_argument("--gaf", type=Path, default=default_gaf, help="Path to GAF file")
    parser.add_argument("--obo", type=Path, default=default_obo, help="Path to GO OBO file")

    genes_group = parser.add_mutually_exclusive_group()
    genes_group.add_argument("--genes", type=str, default=None, help="Comma-separated gene list")
    genes_group.add_argument("--genes-file", type=Path, default=None, help="One gene symbol per line")

    parser.add_argument(
        "--n-genes",
        type=int,
        default=80,
        help="Number of genes to sample from the GAF if no list is provided",
    )

    parser.add_argument(
        "--compare",
        type=str,
        choices=["ontology", "similarity", "both"],
        default="both",
        help="Which axis to sweep",
    )

    parser.add_argument("--sweep-ontologies", type=str, default="BP,MF,CC", help="Comma-separated ontologies")
    parser.add_argument("--sweep-similarities", type=str, default="resnik,lin,wang", help="Comma-separated methods")

    parser.add_argument("--ontology-fixed", type=str, default="BP", help="Fixed ontology for the similarity sweep")
    parser.add_argument("--similarity-fixed", type=str, default="lin", help="Fixed method for the ontology sweep")
    parser.add_argument("--groupwise", type=str, default="bma", help="Groupwise method")
    parser.add_argument("--distance-transform", type=str, default="auto", help="auto, one_minus, max_minus, reciprocal")

    parser.add_argument("--threads", type=int, default=None, help="Limit internal thread pool (rayon)")

    parser.add_argument(
        "--embed",
        type=str,
        choices=["tsne", "umap", "both"],
        default="both",
        help="Which embedding(s) to compute",
    )

    parser.add_argument("--perplexity", type=float, default=None, help="t-SNE perplexity")
    parser.add_argument("--n-iter", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument("--n-neighbors", type=int, default=None, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--align", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-align",
        dest="align",
        action="store_false",
        help="Disable embedding alignment (by default panels are aligned for easier comparisons)",
    )

    parser.add_argument("--annotate", type=str, default="auto", help="auto, all, none")
    parser.add_argument("--max-labels", type=int, default=200, help="Max labels when annotate=auto")
    parser.add_argument("--point-size", type=float, default=18.0, help="Marker size")
    parser.add_argument("--alpha", type=float, default=0.85, help="Marker alpha")

    parser.add_argument("--out-prefix", type=Path, default=Path("embedding_sweep"), help="Output prefix")
    parser.add_argument("--dpi", type=int, default=160, help="Output DPI")

    return parser.parse_args()


def read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            gene = line.strip()
            if not gene or gene.startswith("#"):
                continue
            genes.append(gene)
    return genes


def procrustes_align(source, target):
    import numpy as np

    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.shape != target.shape:
        return source

    src_mean = source.mean(axis=0, keepdims=True)
    tgt_mean = target.mean(axis=0, keepdims=True)
    src = source - src_mean
    tgt = target - tgt_mean

    m = src.T @ tgt
    u, _s, vt = np.linalg.svd(m, full_matrices=False)
    r = u @ vt

    src_rot = src @ r
    denom = float((src_rot**2).sum())
    if denom > 0.0:
        scale = float((tgt * src_rot).sum()) / denom
    else:
        scale = 1.0

    return (scale * src_rot) + tgt_mean


def sweep_configs(args: argparse.Namespace) -> list[list[DistanceConfig]]:
    rows: list[list[DistanceConfig]] = []

    if args.compare in ("ontology", "both"):
        ontologies = parse_csv(args.sweep_ontologies)
        rows.append(
            [
                DistanceConfig(
                    title=f"{ontology} · {args.similarity_fixed}",
                    ontology=ontology,
                    similarity=args.similarity_fixed,
                    groupwise=args.groupwise,
                    distance_transform=args.distance_transform,
                )
                for ontology in ontologies
            ]
        )

    if args.compare in ("similarity", "both"):
        methods = parse_csv(args.sweep_similarities)
        rows.append(
            [
                DistanceConfig(
                    title=f"{args.ontology_fixed} · {method}",
                    ontology=args.ontology_fixed,
                    similarity=method,
                    groupwise=args.groupwise,
                    distance_transform=args.distance_transform,
                )
                for method in methods
            ]
        )

    return rows


def compute_distance_matrices(
    genes: list[str], rows: list[list[DistanceConfig]], counter
) -> tuple[list[str], list[list[tuple[DistanceConfig, list[str], list[list[float]]]]]]:
    # Compute distances for each config, then align genes by intersection so each panel uses the same set.
    raw: list[list[tuple[DistanceConfig, list[str], list[list[float]]]]] = []
    all_gene_sets: list[set[str]] = []

    for row in rows:
        out_row: list[tuple[DistanceConfig, list[str], list[list[float]]]] = []
        for cfg in row:
            t0 = time.perf_counter()
            used_genes, dist = go3.gene_distance_matrix(
                genes=genes,
                ontology=cfg.ontology,
                similarity=cfg.similarity,
                groupwise=cfg.groupwise,
                counter=counter,
                distance_transform=cfg.distance_transform,
            )
            dt = time.perf_counter() - t0
            print(
                f"[dist] {cfg.title}: {len(used_genes)} genes, {dt:.2f}s",
                file=sys.stderr,
            )
            out_row.append((cfg, used_genes, dist))
            all_gene_sets.append(set(used_genes))
        raw.append(out_row)

    if not all_gene_sets:
        return [], raw

    intersection = set.intersection(*all_gene_sets)
    aligned_genes = [g for g in genes if g in intersection]
    dropped = [g for g in genes if g not in intersection]
    if dropped:
        print(
            f"[warn] Dropped {len(dropped)} gene(s) not present in all panels (showing first 12): {dropped[:12]}",
            file=sys.stderr,
        )

    return aligned_genes, raw


def subset_distance_matrix(used_genes: list[str], dist: list[list[float]], genes: list[str]):
    import numpy as np

    index = {g: i for i, g in enumerate(used_genes)}
    indices = [index[g] for g in genes]
    mat = np.asarray(dist, dtype=np.float32)
    return mat[np.ix_(indices, indices)]


def embed_from_distances(
    dist,
    embed: Literal["tsne", "umap"],
    *,
    seed: int,
    perplexity: float,
    n_iter: int,
    n_neighbors: int,
    min_dist: float,
):
    if embed == "tsne":
        from sklearn.manifold import TSNE
        import inspect

        kwargs = dict(
            n_components=2,
            metric="precomputed",
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=seed,
        )
        sig = inspect.signature(TSNE)
        if "max_iter" in sig.parameters:
            kwargs["max_iter"] = n_iter
        else:
            kwargs["n_iter"] = n_iter

        model = TSNE(**kwargs)
        return model.fit_transform(dist)

    import umap

    model = umap.UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return model.fit_transform(dist)


def plot_grid(
    rows: list[list[tuple[DistanceConfig, list[str], list[list[float]]]]],
    genes: list[str],
    *,
    embed: Literal["tsne", "umap"],
    args: argparse.Namespace,
):
    import numpy as np
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = max((len(r) for r in rows), default=1)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    if embed == "tsne":
        perplexity = auto_perplexity(len(genes), args.perplexity)
        if perplexity >= len(genes):
            perplexity = max(1.0, float(len(genes) - 1))
        n_neighbors = 0
    else:
        n_neighbors = auto_n_neighbors(len(genes), args.n_neighbors)
        if n_neighbors >= len(genes):
            n_neighbors = max(2, len(genes) - 1)
        perplexity = 0.0

    embeddings: list[np.ndarray] = []

    for i, row in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i][j]
            if j >= len(row):
                ax.axis("off")
                continue

            cfg, used_genes, dist = row[j]
            dist_sub = subset_distance_matrix(used_genes, dist, genes)

            t0 = time.perf_counter()
            emb = embed_from_distances(
                dist_sub,
                embed,
                seed=args.seed,
                perplexity=perplexity,
                n_iter=args.n_iter,
                n_neighbors=n_neighbors,
                min_dist=args.min_dist,
            )
            dt = time.perf_counter() - t0
            print(f"[{embed}] {cfg.title}: {dt:.2f}s", file=sys.stderr)
            embeddings.append(emb)

            if args.align and embeddings:
                emb = procrustes_align(emb, embeddings[0])

            go3.plot_embedding(
                emb,
                genes=genes,
                title=cfg.title,
                annotate=args.annotate,
                max_labels=args.max_labels,
                s=args.point_size,
                alpha=args.alpha,
                ax=ax,
            )

    # Use shared axis limits to make comparisons easier.
    lims = None
    for axrow in axes:
        for ax in axrow:
            if not ax.has_data():
                continue
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            if lims is None:
                lims = [x0, x1, y0, y1]
            else:
                lims[0] = min(lims[0], x0)
                lims[1] = max(lims[1], x1)
                lims[2] = min(lims[2], y0)
                lims[3] = max(lims[3], y1)
    if lims is not None:
        for axrow in axes:
            for ax in axrow:
                if ax.has_data():
                    ax.set_xlim(lims[0], lims[1])
                    ax.set_ylim(lims[2], lims[3])

    title_bits: list[str] = []
    if args.compare in ("ontology", "both"):
        title_bits.append(f"ontology sweep (fixed={args.similarity_fixed})")
    if args.compare in ("similarity", "both"):
        title_bits.append(f"method sweep (fixed={args.ontology_fixed})")
    fig.suptitle(f"GO3 {embed.upper()} · " + " · ".join(title_bits), fontsize=14)

    return fig


def main() -> int:
    args = parse_args()

    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    elif args.genes_file:
        genes = read_gene_list(args.genes_file)
    else:
        genes = pick_genes_from_gaf(args.gaf, args.n_genes)

    if len(genes) < 3:
        print("Need at least 3 genes to compute embeddings.", file=sys.stderr)
        return 1

    if args.threads is not None:
        go3.set_num_threads(int(args.threads))

    rows_cfg = sweep_configs(args)
    if not rows_cfg or all(not row for row in rows_cfg):
        print("No sweep configs selected.", file=sys.stderr)
        return 1

    go3.load_go_terms(str(args.obo))
    annots = go3.load_gaf(str(args.gaf))
    counter = go3.build_term_counter(annots)

    aligned_genes, raw = compute_distance_matrices(genes, rows_cfg, counter)
    if len(aligned_genes) < 3:
        print("After alignment, fewer than 3 genes remain. Try a different gene list.", file=sys.stderr)
        return 1

    if args.embed in ("tsne", "both"):
        fig = plot_grid(raw, aligned_genes, embed="tsne", args=args)
        out = args.out_prefix.with_name(args.out_prefix.name + "_tsne.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved t-SNE sweep to {out}")

    if args.embed in ("umap", "both"):
        fig = plot_grid(raw, aligned_genes, embed="umap", args=args)
        out = args.out_prefix.with_name(args.out_prefix.name + "_umap.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved UMAP sweep to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
