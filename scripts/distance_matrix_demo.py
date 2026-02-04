#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import go3

from demo_utils import default_paths, pick_genes_from_gaf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GO3 gene distance matrix demo")
    default_gaf, default_obo = default_paths()
    parser.add_argument("--gaf", type=Path, default=default_gaf, help="Path to GAF file")
    parser.add_argument("--obo", type=Path, default=default_obo, help="Path to GO OBO file")
    parser.add_argument("--genes", type=str, default=None, help="Comma-separated gene list")
    parser.add_argument("--n-genes", type=int, default=50, help="Number of genes to sample if --genes is not provided")
    parser.add_argument("--ontology", type=str, default="BP", help="BP, MF, or CC")
    parser.add_argument("--similarity", type=str, default="lin", help="Similarity method")
    parser.add_argument("--groupwise", type=str, default="bma", help="Groupwise method")
    parser.add_argument("--distance-transform", type=str, default="auto", help="auto, one_minus, max_minus, reciprocal")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.genes:
        genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        genes = pick_genes_from_gaf(args.gaf, args.n_genes)

    if len(genes) < 2:
        print("Need at least 2 genes to build a distance matrix.", file=sys.stderr)
        return 1

    go3.load_go_terms(str(args.obo))
    annots = go3.load_gaf(str(args.gaf))
    counter = go3.build_term_counter(annots)

    genes, dist = go3.gene_distance_matrix(
        genes=genes,
        ontology=args.ontology,
        similarity=args.similarity,
        groupwise=args.groupwise,
        counter=counter,
        distance_transform=args.distance_transform,
    )

    flat = [v for row in dist for v in row]
    print(f"Matrix shape: {len(dist)} x {len(dist)}")
    print(f"Distance min: {min(flat):.6f} max: {max(flat):.6f}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + genes)
            for gene, row in zip(genes, dist):
                writer.writerow([gene] + row)
        print(f"Saved matrix to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
