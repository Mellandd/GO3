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

## 3. Prefer batch APIs over scalar loops

Use batch/vectorized endpoints whenever possible:

- term pairs: `batch_similarity(...)`
- gene pairs: `compare_gene_pairs_batch(...)`

Python loops over single-pair calls (`semantic_similarity` or `compare_genes`) add interpreter overhead and reduce throughput.

## 4. Benchmark with warmups and medians

For stable measurements:

- include at least one warmup run
- run multiple repeats
- compare median wall time

Use `scripts/benchmark_go3vsgoatools.py` for standardized runs.

## 5. Choose realistic workload sizes

For tiny input sizes, fixed overhead can dominate and hide the true performance profile.

To assess production behavior, benchmark with medium/large batches (hundreds to thousands of pairs) and matrix-style workloads.

## 6. Gene matrix workloads scale quadratically

All-vs-all comparisons on `g` genes produce approximately `g^2 / 2` pairs.

- memory and compute both increase quickly with `g`
- prefer batched pair evaluation and subset/sampling strategies for exploratory phases

## 7. Distance transforms for embedding pipelines

`gene_distance_matrix` supports:

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

When reporting results, always include these settings.

## 9. Suggested benchmark profile

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

This profile usually gives stable and interpretable comparisons for both throughput and memory.

For publication-ready figures and metadata, use:

```bash
./venv/bin/python scripts/benchmark_go3vsgoatools.py \
  --paper-ready \
  --namespace BP \
  --term-method lin \
  --gene-method lin \
  --outdir imgs
```
