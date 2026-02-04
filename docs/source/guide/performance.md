# Performance and parallelism

GO3 is implemented in Rust and uses parallelism for batch computations.

## Control the number of threads

You can limit the maximum number of threads used by GO3's internal thread pool:

```python
import go3

go3.set_num_threads(8)
```

`load_go_terms()` and `load_gaf()` cache data globally in the current process, so you typically call them once.

## Prefer batch APIs for throughput

- Term similarity: `batch_similarity(...)`
- Gene similarity: `compare_gene_pairs_batch(...)`

