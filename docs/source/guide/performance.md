# Performance Guide

GO3 is implemented in Rust and exposes Python APIs optimized for high-throughput GO semantic similarity workloads.

This guide focuses on practical performance tuning in real pipelines.

## 1. Load once, reuse many times

The typical high-performance workflow is:

```python
import go3

go3.load_go_terms("go-basic.obo")
annots = go3.load_gaf("goa_human.gaf")
counter = go3.build_term_counter(annots)

# Reuse `counter` and loaded ontology for all subsequent analyses.
```

Avoid repeatedly reloading ontology/GAF or rebuilding `counter` inside loops.

## 2. Configure threads before heavy workloads

```python
import go3
go3.set_num_threads(8)
```

Call `set_num_threads` once at startup, before launching large batch jobs.

**Choosing a thread count**: a good starting point is the number of **physical cores** on your machine (not logical/hyperthreaded cores). For I/O-bound workloads you may benefit from slightly more threads, but for GO3's CPU-bound similarity computations, matching physical cores typically gives the best throughput.

**Important**: the Rayon thread pool is initialized once. Calling `set_num_threads` after the pool has been used (e.g., after a batch call) has no effect. Always set threads before any heavy computation.

## 3. Prefer batch APIs over scalar loops

Use batch/vectorized endpoints whenever possible:

- term pairs: `batch_similarity(...)`
- gene pairs: `compare_gene_pairs_batch(...)`
- gene-set pairs: `compare_gene_set_pairs_batch(...)`
- all-vs-all genes: `gene_distance_matrix(...)`
- all-vs-all named gene sets: `gene_set_distance_matrix(...)`

Python loops over scalar calls (`semantic_similarity`, `compare_genes`, or `compare_gene_sets`) add interpreter overhead and reduce throughput.

## 4. Memory usage

The ontology graph, gene-to-GO mapping, ancestor sets, DCA cache, and IC values are all cached **globally in-process**. This is by design: caching avoids redundant computation and is key to GO3's speed.

Typical memory footprint:

- Ontology (`go-basic.obo`): ~20--40 MB
- Human annotations (`goa_human.gaf`): ~50--100 MB including propagated counts
- Total for a typical human analysis: ~100--200 MB

If memory is a concern, avoid loading multiple large GAF files in the same process. Reloading the ontology (`load_go_terms`) replaces the previous cache.

## 5. Choose realistic workload sizes

For tiny input sizes, fixed overhead can dominate and hide the true performance profile.

To assess production behavior, benchmark with medium/large batches (hundreds to thousands of pairs) and matrix-style workloads.

## 6. Matrix workloads scale quadratically

All-vs-all comparisons on `g` genes produce approximately `g^2 / 2` pairs.

- memory and compute both increase quickly with `g`
- prefer batched pair evaluation and subset/sampling strategies for exploratory phases
- `gene_set_distance_matrix` has the same outer quadratic scaling over the number of named gene sets; each cell also computes pairwise gene similarities between the two sets.

## 7. Distance transforms for embedding pipelines

`gene_distance_matrix` and `gene_set_distance_matrix` support:

- `auto` (recommended default)
- `one_minus`
- `max_minus`
- `reciprocal`

For normalized similarities (for example `lin`, `simrel`, `wang`), `auto` maps to `one_minus`.

## 8. Input quality affects runtime and comparability

Runtime and similarity distributions depend on:

- ontology version
- annotation source/version
- ontology namespace (`BP`, `MF`, `CC`)
- term similarity method (`lin`, `resnik`, `wang`, ...)
- groupwise strategy (`bma`, `max`, `avg`, `hausdorff`, `simgic`)
- for direct gene-set comparison, the outer gene-set strategy (`bma`, `max`, `avg`, `hausdorff`) and optional inner `term_groupwise`

When reporting results, always include these settings.
