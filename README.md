# GO3: Gene Ontology Semantic Similarity (Rust + Python)

![Banner](imgs/readme-banner.svg)

[![PyPI version](https://badge.fury.io/py/GO3.svg)](https://pypi.org/project/GO3/)
[![Documentation](https://readthedocs.org/projects/go3/badge/?version=latest)](https://go3.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/Mellandd/go3)](LICENSE)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Input Data](#input-data)
- [Quick Start](#quick-start)
- [Similarity Methods](#similarity-methods)
- [API Overview](#api-overview)
- [How It Works](#how-it-works)
- [Common Workflows](#common-workflows)
- [Performance and Benchmarks](#performance-and-benchmarks)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Introduction

Gene Ontology (GO) semantic similarity measures how functionally related two GO terms or gene products are, based on their positions and relationships within the GO hierarchy. It is widely used in bioinformatics for gene clustering, functional module detection, disease-gene prioritization, and protein–protein interaction prediction.

GO3 is a high-performance GO semantic similarity library built on a **Rust core** exposed through a **Python API** (via PyO3). It provides 8 term-level similarity methods, 5 groupwise strategies, and parallelized batch operations, all accessible from a simple Python interface.

### Why GO3?

Existing tools like [GOSemSim](https://bioconductor.org/packages/GOSemSim/) (R) and [goatools](https://github.com/tanghaibao/goatools) (Python) cover term-level semantic similarity, but many common operations in GO-based analyses — comparing sets of terms, computing gene-level similarity, building distance matrices, or generating embeddings — require writing ad-hoc glue code or switching between languages and packages. GO3 brings all of these into a single Python library:

- **Term-level similarity** — 8 methods (IC-based, topological, and hybrid) in one place.
- **Term-set and gene-level similarity** — compare two sets of GO terms or two genes directly, with 5 groupwise strategies.
- **Batch operations** — compute thousands of term or gene pairs in a single call, parallelized automatically.
- **All-vs-all distance matrices** — one function call for a full symmetric distance matrix over any gene list.
- **Embeddings and visualization** — built-in t-SNE, UMAP, and plotting helpers, no external pipeline needed.
- **Speed** — 8–25x faster than pure-Python alternatives; the Rust core and Rayon parallelism eliminate interpreter overhead on large workloads.
- **Minimal setup** — load an OBO file (auto-downloadable) and a GAF file, and you're ready to compute.

Preprint:
https://www.biorxiv.org/content/10.1101/2025.09.04.669468v1

## Features

- **8 term-level similarity methods** — Resnik, Lin, Jiang-Conrath, SimRel, ICCoef, GraphIC, Wang, TopoICSim
- **5 groupwise strategies** — BMA, MAX, AVG, Hausdorff, SimGIC
- **Gene-level comparison** with namespace filtering (BP / MF / CC)
- **Batch and all-vs-all operations** parallelized with Rayon
- **Distance matrices** from any similarity/groupwise combination
- **t-SNE and UMAP embeddings** built on top of distance matrices
- **Visualization helpers** (`plot_tsne_genes`, `plot_umap_genes`, `plot_embedding`)
- **Thread control** via `set_num_threads`
- **Auto-download** of the GO OBO file when no path is provided

## Installation

```bash
pip install go3
```

Pre-built wheels are available for common platforms. Requires **Python >= 3.7**.

For visualization support (matplotlib, scikit-learn, umap-learn):

```bash
pip install go3[viz]
```

The `[viz]` extras enable `plot_tsne_genes`, `plot_umap_genes`, `plot_embedding`, and the `tsne_genes` / `umap_genes` embedding functions.

## Input Data

GO3 requires two input files:

### OBO file (Gene Ontology structure)

The OBO file defines the ontology: terms, their names, namespaces, and hierarchical relationships (is_a, part_of).

- **Auto-download**: call `go3.load_go_terms()` with no arguments to download the latest `go-basic.obo` automatically.
- **Manual download**: get it from the [Gene Ontology downloads page](http://purl.obolibrary.org/obo/go/go-basic.obo).

### GAF file (Gene annotations)

A Gene Annotation Format (GAF) file maps gene products to GO terms with evidence codes. GO3 filters obsolete terms automatically (replacing them via `replaced_by` / `consider` fields).

- **Download from UniProt-GOA**: annotation files for many organisms are available at [https://ftp.ebi.ac.uk/pub/databases/GO/goa/](https://ftp.ebi.ac.uk/pub/databases/GO/goa/) (e.g., `goa_human.gaf.gz` for human).

## Quick Start

```python
import go3

# 1) Load the GO ontology (pass no argument to auto-download go-basic.obo)
go3.load_go_terms("go-basic.obo")

# 2) Load gene annotations from a GAF file
annots = go3.load_gaf("goa_human.gaf")

# 3) Build a term counter: computes annotation counts and Information Content (IC)
counter = go3.build_term_counter(annots)

# 4) Term similarity — compare two GO terms using the Lin method
sim = go3.semantic_similarity("GO:0008150", "GO:0009987", "lin", counter)
print(f"Term similarity: {sim:.4f}")

# 5) Gene similarity — compare two genes on Biological Process using Lin + BMA
score = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)
print(f"Gene similarity: {score:.4f}")
```

## Similarity Methods

### Term-level methods

| Method | Key | Description |
|---|---|---|
| Resnik | `resnik` | Information Content (IC) of the Most Informative Common Ancestor (MICA) |
| Lin | `lin` | Normalized Resnik: `2 * IC(MICA) / (IC(t1) + IC(t2))` |
| Jiang-Conrath | `jc` | IC-based distance converted to similarity |
| SimRel | `simrel` | Lin with a relevance correction factor |
| ICCoef | `iccoef` | IC ratio relative to the minimum IC of the two terms |
| GraphIC | `graphic` | IC normalized by graph depth |
| Wang | `wang` | Topological method using weighted ancestor paths (no IC needed) |
| TopoICSim | `topoicsim` | Hybrid topology + IC method using disjunctive common ancestors |

### Groupwise strategies (for term sets and genes)

| Strategy | Key | Description |
|---|---|---|
| Best Match Average | `bma` | Average of best-match similarities in both directions |
| Maximum | `max` | Maximum pairwise similarity across all pairs |
| Average | `avg` | Mean of all pairwise similarities |
| Hausdorff | `hausdorff` | Minimax-based similarity (worst best-match) |
| SimGIC | `simgic` | IC-weighted Jaccard index over ancestor sets |

## API Overview

### Loading and Initialization

| Function | Description |
|---|---|
| `load_go_terms(path=None)` | Load GO terms from an OBO file; auto-downloads if no path given |
| `load_gaf(path)` | Load gene annotations from a GAF file |
| `build_term_counter(annotations)` | Compute annotation counts and IC values for all terms |

### Ontology Traversal

| Function | Description |
|---|---|
| `get_term_by_id(go_id)` | Retrieve a GO term object by its ID |
| `ancestors(go_id)` | Get all ancestor term IDs (via is_a relationships) |
| `common_ancestor(go_id1, go_id2)` | Get all common ancestors of two terms |
| `deepest_common_ancestor(go_id1, go_id2)` | Get the most specific (deepest) common ancestor |

### Term Similarity

| Function | Description |
|---|---|
| `semantic_similarity(id1, id2, method, counter)` | Pairwise similarity between two GO terms |
| `batch_similarity(list1, list2, method, counter)` | Parallel pairwise similarity for lists of term pairs |
| `term_ic(go_id, counter)` | Information Content of a single GO term |

### Set Similarity

| Function | Description |
|---|---|
| `termset_similarity(terms1, terms2, method, groupwise, counter)` | Similarity between two sets of GO terms |

### Gene Similarity

| Function | Description |
|---|---|
| `compare_genes(gene1, gene2, ontology, similarity, groupwise, counter)` | Similarity between two genes |
| `compare_gene_pairs_batch(pairs, ontology, similarity, groupwise, counter)` | Parallel similarity for a list of gene pairs |

### Distance and Embeddings

| Function | Description |
|---|---|
| `gene_distance_matrix(genes, ontology, similarity, groupwise, counter, distance_transform)` | All-vs-all distance matrix for a set of genes |
| `tsne_genes(genes, ontology, similarity, groupwise, counter, ...)` | t-SNE embedding from a gene distance matrix |
| `umap_genes(genes, ontology, similarity, groupwise, counter, ...)` | UMAP embedding from a gene distance matrix |

### Visualization

| Function | Description |
|---|---|
| `plot_embedding(embedding, genes, ...)` | Plot a 2D embedding with matplotlib |
| `plot_tsne_genes(genes, ontology, similarity, groupwise, counter, ...)` | Compute t-SNE and plot in one step |
| `plot_umap_genes(genes, ontology, similarity, groupwise, counter, ...)` | Compute UMAP and plot in one step |

### Configuration

| Function | Description |
|---|---|
| `set_num_threads(n)` | Set the number of threads for parallel operations (0 = all cores) |

## How It Works

GO3's Rust core is compiled into a Python extension module via [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/). The ontology graph, gene-to-GO mappings, ancestor sets, and IC values are stored in global caches (using fast hash maps from `rustc-hash`) so they are computed once and reused across all subsequent calls.

Batch operations (`batch_similarity`, `compare_gene_pairs_batch`, `gene_distance_matrix`) are parallelized with [Rayon](https://github.com/rayon-rs/rayon), which distributes work across threads automatically. Unique pairs are deduplicated before computation to avoid redundant work.

This architecture — native compiled code, global caching, and data-parallel execution — is what gives GO3 its **8–25x speedup** over pure-Python libraries on medium and large workloads.

## Common Workflows

### Batch term similarity

```python
pairs = [("GO:0006397", "GO:0008380"), ("GO:0008150", "GO:0009987")]
scores = go3.batch_similarity([a for a, _ in pairs], [b for _, b in pairs], "lin", counter)
```

### Batch gene similarity

```python
gene_pairs = [("TP53", "BRCA1"), ("EGFR", "AKT1")]
scores = go3.compare_gene_pairs_batch(gene_pairs, "BP", "lin", "bma", counter)
```

### Term-set similarity

```python
# Compare two sets of GO terms directly (without gene lookup)
set_a = ["GO:0006397", "GO:0008380", "GO:0000398"]
set_b = ["GO:0006396", "GO:0010467"]
sim = go3.termset_similarity(set_a, set_b, "lin", "bma", counter)
print(f"Term-set similarity: {sim:.4f}")
```

### Ontology traversal

```python
# Inspect a GO term
term = go3.get_term_by_id("GO:0006397")
print(term.name)        # "mRNA processing"
print(term.namespace)   # "biological_process"
print(term.parents)     # parent term IDs

# Get ancestors and common ancestors
anc = go3.ancestors("GO:0006397")
dca = go3.deepest_common_ancestor("GO:0006397", "GO:0008380")
print(f"Deepest common ancestor: {dca}")
```

### Thread configuration

```python
# Use 4 threads for parallel operations
go3.set_num_threads(4)

# Use all available cores
go3.set_num_threads(0)
```

### Distance matrix + embedding

```python
genes = ["TP53", "BRCA1", "EGFR", "AKT1", "CASP8"]

ordered, dist = go3.gene_distance_matrix(
    genes,
    ontology="BP",
    similarity="lin",
    groupwise="bma",
    counter=counter,
    distance_transform="auto",
)

ordered, emb_tsne = go3.tsne_genes(
    genes, "BP", "lin", "bma", counter, perplexity=2.0, random_state=42
)

ordered, emb_umap = go3.umap_genes(
    genes, "BP", "lin", "bma", counter, n_neighbors=3, random_state=42
)
```

### Plot helpers

```python
ordered, emb_t, fig_t, ax_t = go3.plot_tsne_genes(
    genes, "BP", "lin", "bma", counter, perplexity=2.0, random_state=42, annotate="auto"
)

ordered, emb_u, fig_u, ax_u = go3.plot_umap_genes(
    genes, "BP", "lin", "bma", counter, n_neighbors=3, random_state=42, annotate="auto"
)
```

Example outputs:

![GO3 plot_tsne_genes example](imgs/plot_helper_tsne_example.png)

![GO3 plot_umap_genes example](imgs/plot_helper_umap_example.png)

## Performance and Benchmarks

GO3 was benchmarked against [goatools](https://github.com/tanghaibao/goatools) (Python) and [GOSemSim](https://bioconductor.org/packages/GOSemSim/) (R/Bioconductor) on realistic workloads using the human GO annotation corpus (Biological Process, Lin similarity, BMA groupwise).

| Workload | GO3 vs goatools | GO3 vs GOSemSim |
|---|---|---|
| Loading + IC computation | ~1.6x faster, ~2.9x less memory | — |
| Batch term similarity (up to 20k pairs) | ~8.5x faster | comparable |
| Batch gene similarity (up to 150 pairs) | ~24x faster | ~3x faster |
| All-vs-all genes (up to 16 genes) | ~22x faster | ~3x faster |

The speedup grows with workload size. Exact numbers depend on hardware and dataset versions; see the plots below for detailed scaling behavior.

### Loading and memory

![GO3 vs goatools loading benchmark](imgs/benchmark_loading_time_memory.png)

### Batch GO-term similarity

![GO3 vs goatools batch GO-term benchmark](imgs/benchmark_batch_similarity.png)

### Batch gene similarity

![GO3 vs goatools batch gene benchmark](imgs/benchmark_gene_batch_similarity.png)

### All-vs-all gene similarity

![GO3 vs goatools all-vs-all gene benchmark](imgs/benchmark_all_vs_all_gene_similarity.png)

For full methodology, reproducibility scripts, and raw data, see the [benchmark documentation](https://go3.readthedocs.io/en/latest/benchmarks.html).

## Documentation

Full docs:

- https://go3.readthedocs.io

Main sections:

- `docs/source/introduction.rst`
- `docs/source/examples.rst`
- `docs/source/similarity.rst`
- `docs/source/guide/performance.md`
- `docs/source/guide/visualization.md`
- `docs/source/benchmarks.md`

## Contributing

Contributions are welcome! To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/Mellandd/go3.git
cd go3

# Create a virtual environment and install dev dependencies
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest

# Build the Rust extension in development mode
maturin develop

# Run the tests
pytest tests/
```

Please open an issue or pull request on [GitHub](https://github.com/Mellandd/go3).

## Citation

```text
go3: A Fast and Lightweight Library for Semantic Similarity of GO Terms and Genes
Jose L. Mellina-Andreu, Alejandro Cisterna-Garcia, Juan A. Botia
bioRxiv 2025.09.04.669468; doi: https://doi.org/10.1101/2025.09.04.669468
```

BibTeX:

```bibtex
@article {go3,
  author = {Mellina-Andreu, Jose L. and Cisterna-Garcia, Alejandro and Botia, Juan A.},
  title = {go3: A Fast and Lightweight Library for Semantic Similarity of GO Terms and Genes},
  elocation-id = {2025.09.04.669468},
  year = {2025},
  doi = {10.1101/2025.09.04.669468},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2025/09/04/2025.09.04.669468},
  eprint = {https://www.biorxiv.org/content/early/2025/09/04/2025.09.04.669468.full.pdf},
  journal = {bioRxiv}
}
```

## License

MIT License (see `LICENSE`).
