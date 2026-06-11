# Architecture

This page describes GO3's internal design and explains why it achieves 8--25x speedups over pure-Python GO semantic similarity libraries.

## Overview

GO3 is a **Rust core** compiled into a **CPython extension module** via [PyO3](https://pyo3.rs/) and [Maturin](https://www.maturin.rs/). All performance-critical code (parsing, graph traversal, IC computation, similarity calculations) runs in compiled Rust, while the Python API provides a familiar interface for scripting and integration.

## Data flow

```
OBO file ──parse──▶ GO graph cache (terms, edges, depths)
                         │
                         ▼
GAF file ──parse──▶ gene-to-GO mapping ──propagate──▶ TermCounter (counts, IC)
                                                           │
                                                           ▼
                                      term, gene, gene-set, batch, and matrix APIs
```

1. **OBO parsing**: the OBO file is parsed into an in-memory directed acyclic graph. Each term stores its parents, children, depth, level, and metadata. Ancestor sets are precomputed and cached.
2. **GAF parsing**: the GAF file is parsed to build a gene-to-GO mapping. Obsolete terms are resolved via `replaced_by` / `consider`. ND and NOT annotations are filtered.
3. **Counter construction**: annotation counts are propagated up the DAG (each annotation increments the term and all its ancestors). IC is computed per term per namespace.
4. **Similarity computation**: pairwise, batch, matrix, and gene-set similarity are computed using cached graphs, ancestor sets, annotation mappings, and IC values.

## Key design decisions

### Global caches

GO3 maintains several global caches within the Python process:

- **Ontology graph**: the full term DAG with parent/child edges.
- **Ancestor sets**: precomputed `is_a` ancestor sets for every term, enabling O(1) lookups for common-ancestor queries.
- **DCA cache**: deepest common ancestor results are cached to avoid redundant DAG traversals.
- **Gene-to-GO mapping**: maps gene symbols to their annotated GO terms per namespace.
- **IC values**: per-term Information Content, computed once by `build_term_counter`.

Caching trades memory for speed. For a typical human analysis, total memory usage is ~100--200 MB.

### Rayon thread pool

Batch and matrix operations (`batch_similarity`, `compare_gene_pairs_batch`, `compare_gene_set_pairs_batch`, `gene_distance_matrix`, `gene_set_distance_matrix`) use [Rayon](https://docs.rs/rayon/) to parallelize pair evaluation across CPU cores. The thread pool is initialized once via `set_num_threads` and reused for all subsequent parallel work.

Direct gene-set comparison has two aggregation layers:

1. each gene pair is compared through its GO-term annotations;
2. the resulting gene-by-gene similarity matrix is aggregated across the two gene sets.

The alternative `compare_gene_set_profiles` path builds weighted GO-term profiles for each gene set and compares those profiles directly.

### FxHashMap

GO3 uses [FxHashMap](https://docs.rs/rustc-hash/) (a fast, non-cryptographic hash map) for internal lookups. FxHashMap is significantly faster than the standard `HashMap` for string keys like GO IDs, reducing lookup overhead in hot loops.

## Why it's fast

The combination of these design choices eliminates the main bottlenecks in pure-Python implementations:

- **No interpreter overhead in hot loops**: similarity computations, ancestor lookups, and IC queries all run in compiled Rust. Python is only involved at the API boundary.
- **Parallel pair evaluation**: Rayon distributes independent pair computations across threads. For N pairs on T threads, wall time scales roughly as N/T.
- **O(1) cached lookups**: precomputed ancestor sets and cached DCA results avoid redundant graph traversals. In pure-Python tools, these traversals are often the dominant cost.
- **Cache-friendly data structures**: Rust's contiguous memory layout and FxHashMap's low overhead reduce cache misses compared to Python's object-heavy, pointer-chasing data structures.
