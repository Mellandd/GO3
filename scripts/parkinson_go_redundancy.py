#!/usr/bin/env python3
"""
End-to-end pipeline: reducing GO enrichment redundancy with GO3
===============================================================

Gene Ontology enrichment analysis on disease gene panels often returns a long
list of significant GO terms.  Because the ontology is hierarchical, many of
those terms overlap semantically — parent/child or sibling terms that describe
essentially the same biology.  This pipeline shows how *go3* can collapse that
redundancy into a compact, non-redundant summary.

Use case
--------
We start from the **Genomics England Parkinson Disease and Complex
Parkinsonism** gene panel (~36 genes).  For each gene we retrieve its GO
Biological Process (BP) annotations, compute pairwise semantic similarity
between all annotated terms, cluster them, and select one representative per
cluster.  The result is a short list of GO terms that captures the functional
landscape of the panel without repetition.

Additionally, we compute gene-level semantic similarity to build a functional
map of the panel genes (t-SNE), illustrating how closely related the genes are
at the pathway level.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import go3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PANEL_TSV = Path(__file__).resolve().parent / "Parkinson Disease and Complex Parkinsonism.tsv"
GAF_PATH  = Path(__file__).resolve().parents[1] / "tests" / "goa_human.gaf"
OBO_PATH  = Path(__file__).resolve().parents[1] / "tests" / "go-basic.obo"

ONTOLOGY   = "BP"       # Biological Process
SIM_METHOD = "lin"      # Lin similarity (normalized, 0-1)
GROUPWISE  = "bma"      # Best Match Average for gene-level comparison
CLUSTER_THRESHOLD = 0.70  # similarity threshold to merge terms


def load_panel_genes(tsv_path: Path) -> list[str]:
    """Read gene symbols from the Genomics England panel TSV."""
    genes = []
    with open(tsv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("Entity type") == "gene":
                symbol = row.get("Gene Symbol", "").strip()
                if symbol:
                    genes.append(symbol)
    return sorted(set(genes))


def collect_gene_go_terms(genes: list[str], annotations: list, ontology_ns: str) -> dict[str, set[str]]:
    """For each gene in the panel, collect its GO term annotations in the
    requested namespace.  Uses the internal gene2go cache populated by
    load_gaf, via get_term_by_id to check the namespace."""
    gene_terms: dict[str, set[str]] = {}
    # Build gene -> terms from the GAF annotations list
    gene2go: dict[str, set[str]] = {}
    for ann in annotations:
        # ann is a GAFAnnotation with db_object_id, go_term, evidence
        # The GAF file uses gene symbols in column 3 (loaded into the cache),
        # but the annotations list has db_object_id.  We need to match by
        # looking at what the cache has.  Instead, we'll iterate all
        # annotations and match by finding the gene symbol.
        pass

    # More robust: read directly from the loaded cache by calling compare_genes
    # on dummy pairs, or use get_term_by_id.  The simplest approach: for each
    # annotation, read the GAF columns to find gene symbols.
    # Actually, the GAF was already loaded and the gene2go cache is populated.
    # We can use the annotations list to build our own mapping.

    # Re-parse the GAF to build gene-symbol -> GO terms (the annotations list
    # stores db_object_id, not gene symbol).
    gene2go_map: dict[str, set[str]] = {}
    with open(GAF_PATH, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            gene_symbol = cols[2]
            qualifier  = cols[3]
            go_id      = cols[4]
            evidence   = cols[6]
            if evidence == "ND" or "NOT" in qualifier:
                continue
            if gene_symbol in genes:
                gene2go_map.setdefault(gene_symbol, set()).add(go_id)

    # Filter by namespace using get_term_by_id
    ns_full = {
        "BP": "biological_process",
        "MF": "molecular_function",
        "CC": "cellular_component",
    }[ontology_ns]

    for gene in genes:
        terms = set()
        for go_id in gene2go_map.get(gene, set()):
            try:
                t = go3.get_term_by_id(go_id)
                if t.namespace == ns_full:
                    terms.add(go_id)
            except Exception:
                continue
        if terms:
            gene_terms[gene] = terms

    return gene_terms


def build_term_sim_matrix(
    term_list: list[str], counter: go3.TermCounter, method: str
) -> np.ndarray:
    """Compute an all-vs-all similarity matrix for a list of GO terms using
    go3.batch_similarity for speed."""
    n = len(term_list)
    sim = np.zeros((n, n))
    np.fill_diagonal(sim, 1.0)

    # Build paired lists for the upper triangle
    list1, list2, indices = [], [], []
    for i, j in combinations(range(n), 2):
        list1.append(term_list[i])
        list2.append(term_list[j])
        indices.append((i, j))

    if list1:
        scores = go3.batch_similarity(list1, list2, method, counter)
        for (i, j), s in zip(indices, scores):
            sim[i][j] = s
            sim[j][i] = s

    return sim


def cluster_terms(
    term_list: list[str],
    sim_matrix: np.ndarray,
    counter: go3.TermCounter,
    threshold: float,
) -> dict[int, list[str]]:
    """Hierarchical clustering of GO terms by semantic similarity.
    Returns a mapping from cluster ID to list of term IDs."""
    dist = 1.0 - sim_matrix
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for idx, cid in enumerate(labels):
        clusters.setdefault(int(cid), []).append(term_list[idx])
    return clusters


def pick_representatives(
    clusters: dict[int, list[str]], counter: go3.TermCounter
) -> list[tuple[str, str, float, int]]:
    """For each cluster, select the term with the highest IC (most specific).
    Returns a list of (go_id, name, ic, cluster_size)."""
    reps = []
    for cid, members in sorted(clusters.items()):
        best_id, best_ic = None, -1.0
        for go_id in members:
            ic = go3.term_ic(go_id, counter)
            if ic > best_ic:
                best_ic = ic
                best_id = go_id
        t = go3.get_term_by_id(best_id)
        reps.append((best_id, t.name, best_ic, len(members)))
    # Sort by IC descending (most specific first)
    reps.sort(key=lambda x: -x[2])
    return reps


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("GO3 Pipeline: Reducing GO Enrichment Redundancy")
    print("Use case: Genomics England Parkinson Disease Gene Panel")
    print("=" * 72)

    # Step 1 — Load ontology and annotations
    print("\n[Step 1] Loading GO ontology and human GAF annotations ...")
    go3.load_go_terms(str(OBO_PATH))
    annotations = go3.load_gaf(str(GAF_PATH))
    counter = go3.build_term_counter(annotations)
    print(f"  Loaded {len(annotations):,} annotations.")

    # Step 2 — Read gene panel
    print("\n[Step 2] Reading Parkinson Disease gene panel ...")
    panel_genes = load_panel_genes(PANEL_TSV)
    print(f"  Panel contains {len(panel_genes)} genes: {', '.join(panel_genes[:10])}{'...' if len(panel_genes) > 10 else ''}")

    # Step 3 — Collect GO BP terms for panel genes
    print(f"\n[Step 3] Collecting GO {ONTOLOGY} annotations for panel genes ...")
    gene_terms = collect_gene_go_terms(panel_genes, annotations, ONTOLOGY)
    genes_with_annot = [g for g in panel_genes if g in gene_terms]
    all_terms = sorted({t for ts in gene_terms.values() for t in ts})
    print(f"  {len(genes_with_annot)}/{len(panel_genes)} genes have {ONTOLOGY} annotations.")
    print(f"  Total unique {ONTOLOGY} terms across all panel genes: {len(all_terms)}")

    if len(all_terms) < 2:
        print("Not enough GO terms to demonstrate redundancy reduction.", file=sys.stderr)
        return 1

    # Step 4 — Show the redundancy problem
    print(f"\n[Step 4] Computing pairwise semantic similarity ({SIM_METHOD}) "
          f"between all {len(all_terms)} terms ...")
    sim_matrix = build_term_sim_matrix(all_terms, counter, SIM_METHOD)
    upper = sim_matrix[np.triu_indices(len(all_terms), k=1)]
    print(f"  Mean pairwise similarity: {upper.mean():.4f}")
    print(f"  Pairs with similarity > 0.5: {(upper > 0.5).sum()} "
          f"out of {len(upper)} ({100 * (upper > 0.5).mean():.1f}%)")
    print(f"  Pairs with similarity > 0.7: {(upper > 0.7).sum()} "
          f"({100 * (upper > 0.7).mean():.1f}%)")
    print("  --> Many terms are semantically redundant!")

    # Step 5 — Cluster and reduce
    print(f"\n[Step 5] Clustering terms (threshold = {CLUSTER_THRESHOLD}) ...")
    clusters = cluster_terms(all_terms, sim_matrix, counter, CLUSTER_THRESHOLD)
    reps = pick_representatives(clusters, counter)
    print(f"  {len(all_terms)} terms collapsed into {len(reps)} non-redundant clusters.\n")

    print("  Non-redundant GO summary (representative per cluster):")
    print(f"  {'GO ID':<14} {'IC':>6}  {'Size':>4}  Name")
    print("  " + "-" * 68)
    for go_id, name, ic, size in reps:
        print(f"  {go_id:<14} {ic:6.3f}  {size:4d}  {name}")

    # Step 6 — Gene-level functional similarity
    print(f"\n[Step 6] Computing gene-level similarity matrix "
          f"({SIM_METHOD}/{GROUPWISE}) for panel genes ...")
    ordered_genes, dist_matrix = go3.gene_distance_matrix(
        genes=genes_with_annot,
        ontology=ONTOLOGY,
        similarity=SIM_METHOD,
        groupwise=GROUPWISE,
        counter=counter,
        distance_transform="one_minus",
    )
    # Convert distance matrix back to similarity for display
    n = len(ordered_genes)
    sims_flat = []
    for i in range(n):
        for j in range(i + 1, n):
            sims_flat.append(1.0 - dist_matrix[i][j])

    print(f"  {n} genes compared.  Mean pairwise similarity: {np.mean(sims_flat):.4f}")

    # Find most similar gene pairs
    pairs_with_sim = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_with_sim.append((ordered_genes[i], ordered_genes[j], 1.0 - dist_matrix[i][j]))
    pairs_with_sim.sort(key=lambda x: -x[2])

    print("\n  Top 10 most functionally similar gene pairs:")
    print(f"  {'Gene 1':<10} {'Gene 2':<10} {'Similarity':>10}")
    print("  " + "-" * 32)
    for g1, g2, s in pairs_with_sim[:10]:
        print(f"  {g1:<10} {g2:<10} {s:10.4f}")

    # Step 7 — Optional: t-SNE visualization
    try:
        import matplotlib
        matplotlib.use("Agg")

        print(f"\n[Step 7] Generating t-SNE visualization ...")
        perplexity = min(8.0, max(2.0, (n - 1) / 3.0))
        ordered_genes, emb, fig, ax = go3.plot_tsne_genes(
            genes=genes_with_annot,
            ontology=ONTOLOGY,
            similarity=SIM_METHOD,
            groupwise=GROUPWISE,
            counter=counter,
            distance_transform="one_minus",
            perplexity=perplexity,
            n_iter=1000,
            random_state=42,
            annotate="all",
            title="Parkinson Gene Panel — Functional Similarity (t-SNE)",
        )
        out_path = Path(__file__).resolve().parent / "parkinson_tsne.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"  Saved t-SNE plot to {out_path}")
    except Exception as e:
        print(f"  [Skipped t-SNE visualization: {e}]")

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Panel genes:             {len(panel_genes)}")
    print(f"  Genes with BP terms:     {len(genes_with_annot)}")
    print(f"  Original BP terms:       {len(all_terms)}")
    print(f"  After redundancy removal: {len(reps)}  "
          f"({100 * (1 - len(reps) / len(all_terms)):.0f}% reduction)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
