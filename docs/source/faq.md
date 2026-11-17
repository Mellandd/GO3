# FAQ & Troubleshooting

Common questions and quick fixes when working with GO3.

## Troubleshooting Common Errors

| Issue / Error | Cause & Solution |
|---|---|
| `FileNotFoundError` loading OBO/GAF | Check the file path, or call `go3.load_go_terms()` with no arguments to auto-download `go-basic.obo`. Both plain and `.gz` files are supported. |
| `ValueError: Gene 'XYZ' not found in mapping` | Gene symbols must match column 3 (`db_object_symbol`) of your GAF file (case-sensitive). Use `go3.list_genes()` to inspect available symbols. |
| Similarity score is `0.0` | Either the two terms belong to different namespaces (`BP` vs. `MF` vs. `CC`), a GO ID is not in the ontology, or a gene has no annotations in the requested `ontology`. |
| `ValueError: simgic is not valid for direct gene-set aggregation` | In `compare_gene_sets`, the outer gene-matrix aggregator supports `bma`, `max`, `avg`, or `hausdorff`. To use `simgic`, either pass `term_groupwise="simgic"` or use `compare_gene_set_profiles(..., groupwise="simgic")`. |
| `set_num_threads` has no effect | The Rayon thread pool is initialized once per process. Call `go3.set_num_threads(n)` at the top of your script before any batch/matrix call. |

## Frequently Asked Questions

### Which OBO file should I use?

Use **`go-basic.obo`** (or `go-basic.obo.gz`). It is guaranteed to be acyclic across namespaces and is the standard file for semantic similarity and enrichment. Calling `go3.load_go_terms()` without arguments downloads `go-basic.obo` automatically.

### How do I filter out `IEA` (electronic) annotations or keep only experimental evidence?

Pass `GAFParseOptions` to `load_gaf`:

```python
import go3

opts = go3.GAFParseOptions(
    exclude_evidence={"IEA"},
    # Or restrict to experimental codes only:
    # include_evidence={"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"},
    include_aspects={"P", "F", "C"},
    exclude_qualifiers={"NOT"},
)
annots = go3.load_gaf("goa_human.gaf.gz", options=opts)
counter = go3.build_term_counter(annots)
```

### How does `method="scs"` in `go_enrichment` compare to `bh` and `bonferroni`?

Every `EnrichmentResult` returned by `go3.go_enrichment` includes all four p-values (`p_value`, `p_fdr_bh`, `p_scs`, `p_bonferroni`). The `method` argument selects which one is used to filter results (`<= alpha`):
- `method="bh"` (Benjamini-Hochberg FDR) is the most permissive.
- `method="scs"` (g:Profiler-style *Set Counts and Sizes*) accounts for parent-child overlap in the GO DAG and discrete hypergeometric tail probabilities, sitting between `bh` and `bonferroni` (`p_value <= p_scs <= p_bonferroni`).
- `method="bonferroni"` is the most conservative.

### Why are my `BMA` gene-set similarity differences small between different diseases?

Well-annotated human genes are pleiotropic (often annotated to 20–50 GO terms, many of which are broad cellular processes shared across unrelated diseases). With `bma`, each gene pairs with its best match, creating a high baseline similarity (~0.55–0.65). To increase biological contrast:
1. Filter out `IEA` annotations via `GAFParseOptions(exclude_evidence={"IEA"})`.
2. Use `compare_gene_set_profiles(..., groupwise="simgic")` or `similarity="simrel"`.
3. Or compute enriched terms with `go_enrichment(..., method="scs")` and compare the enriched GO IDs via `termset_similarity`.
