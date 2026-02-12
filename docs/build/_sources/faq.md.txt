# FAQ

Common questions and troubleshooting for GO3.

## Gene not found

**Error**: `ValueError: gene "XYZ" not found`

Gene names must exactly match the `db_object_symbol` column in your GAF file (case-sensitive). Common causes:

- Using an alias or old symbol instead of the current one.
- The gene has no GO annotations in your GAF file.
- Typos or extra whitespace in the gene name.

**Fix**: open your GAF file and search for the gene symbol. GAF files use tab-separated columns; `db_object_symbol` is column 3 (0-indexed: column 2).

## Similarity is 0.0

A result of 0.0 typically means:

- The two terms are in **different namespaces** (e.g., one is BP and the other is MF). Cross-namespace terms have no common ancestor.
- One or both GO IDs are **invalid or not in the loaded ontology**.
- The terms have **no common ancestor** with IC > 0 (very rare within the same namespace).
- For gene comparisons: one or both genes have **no annotations** in the selected namespace.

**Fix**: verify that both terms/genes exist and belong to the same namespace. Use `get_term_by_id` to inspect individual terms.

## Which OBO file should I use?

Use **`go-basic.obo`** (recommended). This is the standard filtered version that contains only `is_a` and `part_of` relationships within each namespace.

The full `go.obo` includes cross-ontology links and additional relationship types that may produce unexpected similarity scores. Unless you specifically need those relationships, stick with `go-basic.obo`.

If you call `go3.load_go_terms()` without a path, GO3 downloads `go-basic.obo` automatically.

## Can I use a custom ontology?

Yes. GO3 accepts any OBO-format file. If you have a custom or domain-specific ontology in OBO format, pass its path to `load_go_terms`:

```python
go3.load_go_terms("my_custom_ontology.obo")
```

The same parsing and caching logic applies.

## How do I filter by evidence code?

GO3 automatically filters out `ND` (No biological Data) annotations and annotations with `NOT` qualifiers. For stricter filtering (e.g., excluding `IEA` to use only experimentally curated annotations), pre-filter your GAF file before passing it to `load_gaf`:

```python
# Example: filter GAF to keep only experimental evidence codes
experimental_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

with open("goa_human.gaf") as f:
    with open("goa_human_experimental.gaf", "w") as out:
        for line in f:
            if line.startswith("!"):
                out.write(line)
                continue
            fields = line.split("\t")
            if len(fields) > 6 and fields[6] in experimental_codes:
                out.write(line)

annots = go3.load_gaf("goa_human_experimental.gaf")
```

## set_num_threads doesn't seem to help

The Rayon thread pool is **initialized once**. If you call `set_num_threads` after the pool has already been created (e.g., after a batch similarity call), the new value is ignored.

**Fix**: call `set_num_threads` at the very beginning of your script, before any batch operations:

```python
import go3
go3.set_num_threads(8)  # Must be before any batch calls

go3.load_go_terms("go-basic.obo")
annots = go3.load_gaf("goa_human.gaf")
counter = go3.build_term_counter(annots)
# Now batch calls will use 8 threads
```

## Memory usage is high

GO3 caches the ontology graph, gene-to-GO mapping, ancestor sets, DCA results, and IC values **globally in-process**. This is by design -- caching eliminates redundant computation and is the main reason GO3 is fast.

Typical memory for a human analysis (`go-basic.obo` + `goa_human.gaf`): ~100--200 MB.

If memory is a concern:

- Avoid loading multiple large GAF files in the same process.
- Reloading the ontology (`load_go_terms`) replaces the previous ontology cache.
- For very large all-vs-all analyses, consider working in batches or subsets.

## How do I reproduce results across runs?

For reproducible similarity scores:

- Pin your ontology version (use a specific `go-basic.obo` file, not auto-download).
- Pin your GAF version.
- For embeddings, set `random_state` in `tsne_genes` / `umap_genes`.

Similarity scores are deterministic given the same ontology and annotation files.
