# Choosing a Similarity Method

This guide helps you select the right term-level similarity method and groupwise aggregation strategy for your use case.

## Term-level methods

### IC-based methods (Resnik, Lin, JC, SimRel)

Use these when you have a **representative annotation corpus** (e.g., a well-curated GAF file for your organism).

IC-based methods rely on annotation frequencies to determine how informative each term is. They work well when:

- Your annotation corpus is large and representative of the biological domain you're studying.
- You want scores that reflect functional specificity relative to known biology.
- You need well-studied methods with established benchmarks in the literature.

**Trade-offs**: scores depend on the annotation corpus. Different GAF versions or organisms produce different IC distributions and therefore different similarity scores. If your corpus is small or biased toward well-studied genes, IC estimates may be unreliable for less-annotated terms.

| Method   | Range   | Best for                                                   |
|----------|---------|------------------------------------------------------------|
| `resnik` | >= 0    | Raw IC of shared ancestor; good for ranking, not comparing across datasets |
| `lin`    | [0, 1]  | Normalized; most widely used IC-based method               |
| `jc`     | >= 0    | Distance-derived; sensitive to small IC differences        |
| `simrel` | [0, 1]  | Like Lin but penalizes shallow (uninformative) MICAs       |

### Topological methods (Wang)

Use Wang when **annotation data may be biased or unavailable**.

Wang similarity uses only the GO graph structure (edge types and path weights) to compute similarity. It assigns weights of 0.8 for `is_a` edges and 0.6 for `part_of` edges, decaying as paths get longer.

**Best for**:
- Organisms with sparse or biased annotations.
- Comparing results across different annotation versions without IC-induced variation.
- When you want a stable, corpus-independent baseline.

**Trade-off**: does not capture functional specificity from real annotation data.

### Hybrid methods (GraphIC, TopoICSim)

Use these when you want **both IC and topology**.

- **GraphIC**: combines IC values with graph-structural features.
- **TopoICSim**: weights topology-aware paths by IC, producing a normalized score in [0, 1]. It is more discriminating than purely IC-based methods for terms with similar IC but different graph positions.

**Best for**:
- Situations where both graph structure and annotation specificity matter.
- Getting a single method that balances multiple sources of information.

## Quick decision table

| Use case                                    | Recommended method |
|---------------------------------------------|--------------------|
| General-purpose, well-annotated organism     | `lin`              |
| Ranking by shared function (no normalization needed) | `resnik`   |
| Sparse or biased annotations                | `wang`             |
| Maximum discrimination between similar terms | `topoicsim`       |
| Penalize shallow common ancestors            | `simrel`           |
| Corpus-independent baseline                  | `wang`             |

## Groupwise strategy selection

When comparing term sets or genes, individual pairwise term similarities must be aggregated into a single score. The choice of groupwise strategy affects what aspect of functional overlap you measure.

| Strategy    | Key          | What it measures                              | Best for                          |
|-------------|--------------|-----------------------------------------------|-----------------------------------|
| BMA         | `bma`        | Average of best matches from both directions  | Balanced comparison; most common  |
| Maximum     | `max`        | Highest pairwise similarity                   | "Do these genes share any function?" |
| Average     | `avg`        | Mean of all pairwise similarities             | Overall functional overlap        |
| SimGIC      | `simgic`     | IC-weighted Jaccard of shared ancestor sets   | IC-aware set overlap; good for clustering |
| Hausdorff   | `hausdorff`  | Worst-case best-match                         | Worst-case guarantee; conservative |

**General recommendation**: start with **BMA** (`bma`). It is the most widely used strategy and provides a balanced view of functional similarity. Use **SimGIC** when you want IC-weighted GO-term set overlap (especially useful for clustering and enrichment-adjacent tasks). Use **MAX** for a permissive "any shared function" signal.

## Gene-set workflows

GO3 exposes two gene-set modes:

| Workflow | Function | Aggregation | Supports `simgic`? | Best for |
|----------|----------|-------------|--------------------|----------|
| Direct gene-set comparison | `compare_gene_sets` | gene-to-gene similarities, then groupwise across genes | No, outer aggregation is `bma`, `max`, `avg`, or `hausdorff` | Comparing two enrichments while preserving genes as the unit of comparison |
| GO-profile comparison | `compare_gene_set_profiles` | weighted GO-term profiles compared directly | Yes | IC-weighted overlap of functional profiles |

For direct gene-set comparison, `term_groupwise` optionally controls the inner gene-to-gene GO-term aggregation. If omitted, GO3 reuses the outer `groupwise` value for the inner gene comparison.

## Namespace guidance

The Gene Ontology has three namespaces. Choose based on the biological question:

- **BP (Biological Process)**: largest namespace with the most annotations. Best for characterizing overall gene function and pathway involvement. Most commonly used in literature benchmarks.
- **MF (Molecular Function)**: describes biochemical activity (e.g., kinase activity, binding). More specific and smaller than BP. Best for molecular-level questions.
- **CC (Cellular Component)**: describes subcellular localization (e.g., nucleus, membrane). Smallest namespace. Best for localization-related analyses.

If unsure, start with **BP** -- it provides the broadest functional characterization and the most statistical power due to its size.
