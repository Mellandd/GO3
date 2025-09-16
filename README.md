# GO3: High-Performance Gene Ontology Semantic Similarity in Rust + Python

[![PyPI version](https://badge.fury.io/py/GO3.svg)](https://pypi.org/project/GO3/)
[![PyPI Version](https://img.shields.io/pypi/v/go3.svg)](https://pypi.org/project/go3/)
[![Documentation](https://readthedocs.org/projects/go3/badge/?version=latest)](https://go3.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/Mellandd/go3)](LICENSE)

🚀 GO3 is a high-performance Rust library with Python bindings for calculating semantic similarity between Gene Ontology (GO) terms and gene products. Designed to be significantly faster and more memory-efficient than traditional Python libraries like `goatools`, GO3 provides state-of-the-art similarity measures including Resnik, Lin, Jiang-Conrath, SimRel, GraphIC, Information Coefficient, and Wang.

The preprint for this library is: https://www.biorxiv.org/content/10.1101/2025.09.04.669468v1

## Features

✅ Ultra-fast ontology loading (50x faster than Goatools)  
✅ Parallel computation of semantic similarity (Resnik, Lin, Jiang-Conrath, SimRel, GraphIC, IC Coefficient, Wang)  
✅ Both term-to-term and gene-to-gene similarity  
✅ Supports batch processing of large datasets  
✅ Full compatibility with Gene Association Files (GAF)  
✅ Low memory footprint thanks to Rust's performance  

## Installation

### Python (via PyPI)

```bash
pip install go3
```

✅ Requires Python 3.8+
✅ Rust is bundled via maturin, no manual compilation needed

## Quick Start

```python
import go3

# Load Gene Ontology (GO) terms and annotations
go_terms = go3.load_go_terms()
annotations = go3.load_gaf("goa_human.gaf")

# Build IC Counter
counter = go3.build_term_counter(annotations)

# Compute Resnik similarity between two GO terms
sim = go3.semantic_similarity("GO:0008150", "GO:0009987", 'resnik', counter)
print(f"Resnik similarity: {sim:.4f}")

# Compute similarity between two genes using Lin and Best-Match Average (BMA)
score = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)
print(f"Gene similarity (Lin, BMA): {score:.4f}")
```

## Supported Similarity Measures

| Measure         | Type        | Reference                                                                |
|-----------------|------------|---------------------------------------------------------------------------|
| Resnik          | IC-based    | Resnik, 1995                                                              |
| Lin             | IC-based    | Lin, 1998                                                                 |
| Jiang-Conrath   | IC-based    | Jiang & Conrath, 1997                                                     |
| SimRel          | IC-based    | Schlicker et al., 2006                                                    |
| GraphIC         | Hybrid      | Li et al., 2010                                                           |
| IC Coefficient  | Hybrid      | Li et al., 2010                                                           |
| Wang            | Topology    | Wang et al., 2007                                                         |
| TopoICSim       | Hybrid      | Ehsani et al., 2016                                                       |

For the theoretical details behind each measure, see the [Similarity Measures Documentation](https://go3.readthedocs.io/en/latest/similarity.html).

## Batch Processing

GO3 natively supports efficient parallel batch computations for both term and gene similarity.

### Batch GO Term Similarity

```python
pairs = [("GO:0008150", "GO:0009987"), ("GO:0008150", "GO:0003674")]
scores = go3.batch_similarity([a for a, _ in pairs], [b for _, b in pairs], "resnik", counter)
```

### Batch Gene Similarity

```python
gene_pairs = [("TP53", "BRCA1"), ("EGFR", "AKT1")]
scores = go3.compare_gene_pairs_batch(gene_pairs, "BP", "resnik", "bma", counter)
```

Both `resnik` and `lin` (and all other similarity methods) are fully supported in batch mode.

## Benchmark

This library is built as fast, scalable and memory-efficient as possible. Comparing with Goatools, which is the de facto library for manipulating GO in Python

We compare the time and peak memory consumption of go3 vs goatools while loading the ontology and the annotation (.GAF) file, and building the TermCounter.

![Loading time & memory](src/benchmark_loading_time_memory.png)

We also compare the speed of the libraries calculating the similarities between batches of GO Terms of different sizes.

![Batch similarity speed](src/benchmark_batch_similarity.png)

Finally, we compare the gene similarity calculation times. Goatools does not implement natively the groupwise algorithms to compare genes, so we built it for a fair comparison in top of the GO term semantic similarities of the library. 

![Gene similarity speed](src/benchmark_gene_batch_similarity.png)

## Contributing

We welcome contributions!

Steps to contribute:

1. Fork the repository.
2. Create a feature branch:  
   ```bash
   git checkout -b feature/my-feature
   ```
3. Implement your changes with tests.
4. Run tests:  
   ```bash
   pytest tests/
   ```
5. Submit a Pull Request.


## License

MIT License © Jose Luis Mellina Andreu, 2025

📄 Full documentation: https://go3.readthedocs.io

🐞 Report issues: https://github.com/Mellandd/go3/issues

## Citation

```
go3: A Fast and Lightweight Library for Semantic Similarity of GO Terms and Genes
Jose L. Mellina-Andreu, Alejandro Cisterna-Garcia, Juan A. Botia
bioRxiv 2025.09.04.669468; doi: https://doi.org/10.1101/2025.09.04.669468
```

### BibTex

```
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
