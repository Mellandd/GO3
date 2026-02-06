# GO3: Gene Ontology Semantic Similarity (Rust + Python)

![Banner](imgs/readme-banner.svg)

[![PyPI version](https://badge.fury.io/py/GO3.svg)](https://pypi.org/project/GO3/)
[![Documentation](https://readthedocs.org/projects/go3/badge/?version=latest)](https://go3.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/Mellandd/go3)](LICENSE)

GO3 is a high-performance semantic similarity library for Gene Ontology (GO), implemented in Rust and exposed through a Python API.

It supports:

- term-to-term similarity
- gene-to-gene similarity
- batch workloads at scale
- visualization-ready distance matrices and embeddings

The preprint is available at:
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

# 1) Load ontology and annotations
go3.load_go_terms("go-basic.obo")
annots = go3.load_gaf("goa_human.gaf")
counter = go3.build_term_counter(annots)

# 2) Term similarity
sim = go3.semantic_similarity("GO:0008150", "GO:0009987", "lin", counter)
print(f"Term similarity: {sim:.4f}")

# 3) Gene similarity
score = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)
print(f"Gene similarity: {score:.4f}")

# 4) Batch gene similarity
gene_pairs = [("TP53", "BRCA1"), ("EGFR", "AKT1")]
scores = go3.compare_gene_pairs_batch(gene_pairs, "BP", "lin", "bma", counter)
print(scores)
```

## Supported term similarity methods

- `resnik`
- `lin`
- `jc`
- `simrel`
- `iccoef`
- `graphic`
- `wang`
- `topoicsim`

Groupwise strategies for term sets / genes:

- `bma`
- `max`
- `avg`
- `hausdorff`
- `simgic`

## Performance-oriented APIs

- `batch_similarity(...)`
- `compare_gene_pairs_batch(...)`
- `gene_distance_matrix(...)`

Use `go3.set_num_threads(n)` to control internal parallelism.

## Embeddings and plotting

```python
import go3

genes = ["TP53", "BRCA1", "EGFR", "AKT1"]

ordered, dist = go3.gene_distance_matrix(
    genes,
    ontology="BP",
    similarity="lin",
    groupwise="bma",
    counter=counter,
    distance_transform="auto",
)

ordered, emb_tsne = go3.tsne_genes(genes, "BP", "lin", "bma", counter, perplexity=2.0, random_state=42)
ordered, emb_umap = go3.umap_genes(genes, "BP", "lin", "bma", counter, n_neighbors=3, random_state=42)
```

## Benchmarks

Benchmark scripts are in `scripts/`.

Main script:

- `scripts/benchmark_go3vsgoatools.py`

Size controls:

- `--term-pair-sizes` for term-level workloads
- `--gene-pair-sizes` for gene-level workloads
- `--pair-sizes` as a legacy shortcut for both

It benchmarks:

1. loading + preprocessing time and memory
2. batch term-pair similarity
3. batch gene-pair similarity
4. all-vs-all gene similarity workloads

Recommended run:

```bash
./venv/bin/python scripts/benchmark_go3vsgoatools.py \
  --namespace BP \
  --term-method lin \
  --gene-method lin \
  --term-pair-sizes 1000,5000,20000 \
  --gene-pair-sizes 25,50,100 \
  --matrix-gene-sizes 8,12 \
  --warmup 1 \
  --repeats 2 \
  --threads 8 \
  --outdir imgs
```

Paper-ready profile:

```bash
./venv/bin/python scripts/benchmark_go3vsgoatools.py \
  --paper-ready \
  --namespace BP \
  --term-method lin \
  --gene-method lin \
  --outdir imgs
```

This mode writes high-resolution PNG figures plus SVG copies and includes system metadata in `benchmark_results.json`.

Generated artifacts:

- `imgs/benchmark_loading_time_memory.png`
- `imgs/benchmark_batch_similarity.png`
- `imgs/benchmark_gene_batch_similarity.png`
- `imgs/benchmark_all_vs_all_gene_similarity.png`
- `imgs/benchmark_results.json`

Optional `GOSemSim` support exists for reference comparisons, but the strict Python ecosystem baseline is `goatools`.

## Documentation

Full docs:

- https://go3.readthedocs.io

Local sources:

- `docs/source/introduction.rst`
- `docs/source/examples.rst`
- `docs/source/similarity.rst`
- `docs/source/guide/performance.md`
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
