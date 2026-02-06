# GO3: Gene Ontology Semantic Similarity (Rust + Python)

![Banner](imgs/readme-banner.svg)

[![PyPI version](https://badge.fury.io/py/GO3.svg)](https://pypi.org/project/GO3/)
[![Documentation](https://readthedocs.org/projects/go3/badge/?version=latest)](https://go3.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/Mellandd/go3)](LICENSE)

GO3 is a high-performance Gene Ontology (GO) semantic similarity library with a Rust core and a Python API.

It is designed for bioinformatics workflows that need:

- fast term-to-term similarity
- gene-to-gene similarity from GAF annotations
- batch processing at scale
- distance matrices and t-SNE/UMAP utilities

Preprint:
https://www.biorxiv.org/content/10.1101/2025.09.04.669468v1

## Installation

```bash
pip install go3
```

Optional visualization extras:

```bash
pip install go3[viz]
```

## Quick start

```python
import go3

# 1) Load ontology + annotations
go3.load_go_terms("go-basic.obo")
annots = go3.load_gaf("goa_human.gaf")
counter = go3.build_term_counter(annots)

# 2) Term similarity
sim = go3.semantic_similarity("GO:0008150", "GO:0009987", "lin", counter)
print(f"Term similarity: {sim:.4f}")

# 3) Gene similarity
score = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)
print(f"Gene similarity: {score:.4f}")
```

## Similarity methods

Term-level methods:

- `resnik`
- `lin`
- `jc`
- `simrel`
- `iccoef`
- `graphic`
- `wang`
- `topoicsim`

Groupwise strategies (term sets / genes):

- `bma`
- `max`
- `avg`
- `hausdorff`
- `simgic`

## Common workflows

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

## Performance and benchmarks

GO3 was benchmarked against `goatools` in realistic workloads (BP, Lin, BMA) and shows a large runtime advantage.

Current benchmark snapshot:

- Loading pipeline (`load_go_terms` + `load_gaf` + `build_term_counter`): `~1.59x` faster and `~2.88x` lower peak memory.
- Batch GO-term similarity (1000, 5000, 20000 pairs): median speedup `~7.27x` (`min ~6.65x`, `max ~8.02x`).
- Batch gene similarity (25, 50, 100 pairs): median speedup `~23.10x` (`min ~19.36x`, `max ~23.21x`).
- All-vs-all gene workload (8, 12 genes): median speedup `~21.57x`.

Exact values depend on hardware and dataset versions, but the speed profile is consistently favorable to GO3 in medium/large workloads.

Loading and memory:

![GO3 vs goatools loading benchmark](imgs/benchmark_loading_time_memory.png)

Batch GO-term similarity:

![GO3 vs goatools batch GO-term benchmark](imgs/benchmark_batch_similarity.png)

Batch gene similarity:

![GO3 vs goatools batch gene benchmark](imgs/benchmark_gene_batch_similarity.png)

All-vs-all gene similarity:

![GO3 vs goatools all-vs-all gene benchmark](imgs/benchmark_all_vs_all_gene_similarity.png)

Reproducibility details, methodology, and raw benchmark JSON are documented in `docs/source/benchmarks.md`.

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

## Development

Run tests:

```bash
pytest tests/
```

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
