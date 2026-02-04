#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import go3

from demo_utils import auto_n_neighbors, default_paths, pick_genes_from_gaf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GO3 UMAP demo using precomputed gene distances")
    default_gaf, default_obo = default_paths()
    parser.add_argument("--gaf", type=Path, default=default_gaf, help="Path to GAF file")
    parser.add_argument("--obo", type=Path, default=default_obo, help="Path to GO OBO file")
    parser.add_argument("--genes", type=str, default=None, help="Comma-separated gene list")
    parser.add_argument("--n-genes", type=int, default=120, help="Number of genes to sample if --genes is not provided")
    parser.add_argument("--ontology", type=str, default="BP", help="BP, MF, or CC")
    parser.add_argument("--similarity", type=str, default="lin", help="Similarity method")
    parser.add_argument("--groupwise", type=str, default="bma", help="Groupwise method")
    parser.add_argument("--distance-transform", type=str, default="auto", help="auto, one_minus, max_minus, reciprocal")
    parser.add_argument("--n-neighbors", type=int, default=None, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--annotate", type=str, default="auto", help="auto, all, none")
    parser.add_argument("--max-labels", type=int, default=200, help="Max labels when annotate=auto")
    parser.add_argument("--width", type=float, default=6.5, help="Figure width")
    parser.add_argument("--height", type=float, default=5.5, help="Figure height")
    parser.add_argument("--point-size", type=float, default=18.0, help="Marker size")
    parser.add_argument("--alpha", type=float, default=0.85, help="Marker alpha")
    parser.add_argument("--title", type=str, default="GO3 UMAP", help="Plot title")
    parser.add_argument("--out", type=Path, default=Path("umap_demo.png"), help="Output image path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        genes = pick_genes_from_gaf(args.gaf, args.n_genes)

    if len(genes) < 3:
        print("Need at least 3 genes for UMAP.", file=sys.stderr)
        return 1

    n_neighbors = auto_n_neighbors(len(genes), args.n_neighbors)
    if n_neighbors >= len(genes):
        n_neighbors = max(2, len(genes) - 1)

    go3.load_go_terms(str(args.obo))
    annots = go3.load_gaf(str(args.gaf))
    counter = go3.build_term_counter(annots)

    genes, emb, fig, _ax = go3.plot_umap_genes(
        genes=genes,
        ontology=args.ontology,
        similarity=args.similarity,
        groupwise=args.groupwise,
        counter=counter,
        distance_transform=args.distance_transform,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        annotate=args.annotate,
        max_labels=args.max_labels,
        figsize=(args.width, args.height),
        s=args.point_size,
        alpha=args.alpha,
        title=args.title,
    )

    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved UMAP plot to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
