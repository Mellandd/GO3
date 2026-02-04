#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import go3

from demo_utils import auto_perplexity, default_paths, pick_genes_from_gaf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GO3 t-SNE demo using precomputed gene distances")
    default_gaf, default_obo = default_paths()
    parser.add_argument("--gaf", type=Path, default=default_gaf, help="Path to GAF file")
    parser.add_argument("--obo", type=Path, default=default_obo, help="Path to GO OBO file")
    parser.add_argument("--genes", type=str, default=None, help="Comma-separated gene list")
    parser.add_argument("--n-genes", type=int, default=80, help="Number of genes to sample if --genes is not provided")
    parser.add_argument("--ontology", type=str, default="BP", help="BP, MF, or CC")
    parser.add_argument("--similarity", type=str, default="lin", help="Similarity method")
    parser.add_argument("--groupwise", type=str, default="bma", help="Groupwise method")
    parser.add_argument("--distance-transform", type=str, default="auto", help="auto, one_minus, max_minus, reciprocal")
    parser.add_argument("--perplexity", type=float, default=None, help="t-SNE perplexity")
    parser.add_argument("--n-iter", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--annotate", type=str, default="auto", help="auto, all, none")
    parser.add_argument("--max-labels", type=int, default=200, help="Max labels when annotate=auto")
    parser.add_argument("--width", type=float, default=6.5, help="Figure width")
    parser.add_argument("--height", type=float, default=5.5, help="Figure height")
    parser.add_argument("--point-size", type=float, default=18.0, help="Marker size")
    parser.add_argument("--alpha", type=float, default=0.85, help="Marker alpha")
    parser.add_argument("--title", type=str, default="GO3 t-SNE", help="Plot title")
    parser.add_argument("--out", type=Path, default=Path("tsne_demo.png"), help="Output image path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        genes = pick_genes_from_gaf(args.gaf, args.n_genes)

    if len(genes) < 3:
        print("Need at least 3 genes for t-SNE.", file=sys.stderr)
        return 1

    perplexity = auto_perplexity(len(genes), args.perplexity)
    if perplexity >= len(genes):
        perplexity = max(1.0, float(len(genes) - 1))

    go3.load_go_terms(str(args.obo))
    annots = go3.load_gaf(str(args.gaf))
    counter = go3.build_term_counter(annots)

    genes, emb, fig, _ax = go3.plot_tsne_genes(
        genes=genes,
        ontology=args.ontology,
        similarity=args.similarity,
        groupwise=args.groupwise,
        counter=counter,
        distance_transform=args.distance_transform,
        perplexity=perplexity,
        n_iter=args.n_iter,
        random_state=args.seed,
        annotate=args.annotate,
        max_labels=args.max_labels,
        figsize=(args.width, args.height),
        s=args.point_size,
        alpha=args.alpha,
        title=args.title,
    )

    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved t-SNE plot to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
