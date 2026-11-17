# Choosing Similarity, Aggregation & Enrichment Methods

A quick reference to pick the right parameters for your biological analysis.

## 1. Term-Level Similarity (`method` / `similarity`)

| Use Case | Recommended Key | Why |
|---|---|---|
| **Standard analysis** (well-annotated organism like human/mouse) | `lin` | Normalized to `[0, 1]`, fast, and most widely benchmarked |
| **Penalize broad/shallow common ancestors** | `simrel` | Multiplies Lin by `1 - exp(-IC(MICA))` so generic shared terms score near `0` |
| **Sparse or biased annotations / cross-species baseline** | `wang` | Purely topological (`is_a=0.8`, `part_of=0.6`), independent of GAF corpus frequencies |
| **Fine discrimination among closely related terms** | `topoicsim` | Hybrid metric combining IC with path topology |
| **Raw specificity ranking** | `resnik` | Unbounded `IC(MICA)`; directly reflects the specificity of the shared ancestor |

## 2. Groupwise Aggregation (`groupwise`)

When comparing two genes or two sets of terms, `groupwise` controls how pairwise term similarities are combined:

| Key | Strategy | Best For |
|---|---|---|
| `bma` | **Best-Match Average** | General gene-to-gene comparison (balanced precision + recall) |
| `simgic` | **SimGIC** (IC-weighted Jaccard) | Discriminating pleiotropic genes or gene sets; downweights generic root terms and penalizes non-overlapping terms |
| `max` | **Maximum** | Detecting whether two genes share *at least one* specific function |
| `avg` | **Average** | Measuring global functional homogeneity across all annotated terms |
| `hausdorff` | **Hausdorff** | Conservative worst-case matching |

## 3. Comparing Two Gene Lists (`compare_gene_sets` vs. `compare_gene_set_profiles`)

| Function | How It Works | Best For |
|---|---|---|
| `compare_gene_sets` | Computes the gene-by-gene similarity matrix and aggregates across genes (`bma`, `max`, `avg`, `hausdorff`) | Comparing two gene panels where each gene has equal weight |
| `compare_gene_set_profiles` | Merges each gene list into a weighted GO-term profile and compares profiles directly (supports `simgic`) | Comparing disease/pathway signatures without dilution from ubiquitous housekeeping annotations |

> **Tip — Avoiding score compression with pleiotropic disease genes:**
> Human disease genes often have 20–50 GO annotations including generic processes (apoptosis, stress response, signaling). Using double `bma` (`compare_gene_sets(..., groupwise="bma", term_groupwise="bma")`) with unfiltered `IEA` annotations can compress similarity scores into a narrow band (e.g. `0.55–0.65`). To maximize biological contrast:
> 1. Exclude electronic annotations (`GAFParseOptions(exclude_evidence={"IEA"})`).
> 2. Use `compare_gene_set_profiles(..., groupwise="simgic")` or `similarity="simrel"`.
> 3. Or run `go_enrichment(..., method="scs")` on each gene list first and compare the enriched GO terms with `termset_similarity`.

## 4. Multiple-Testing Correction in `go_enrichment` (`method`)

| Key | Method | Stringency | When to Use |
|---|---|---|---|
| `bh` | Benjamini-Hochberg FDR | Permissive | Exploratory analysis; controls expected proportion of false discoveries |
| `scs` | **g:SCS** (*Set Counts & Sizes*) | **Balanced (Recommended)** | Matches `g:Profiler` philosophy: accounts for GO DAG hierarchy and discrete hypergeometric tail probabilities |
| `bonferroni` | Bonferroni FWER | Strict | High-confidence core terms only (assumes all tested terms are independent) |
| `raw` | Unadjusted `p_value` | None | Custom downstream filtering or ranking |
