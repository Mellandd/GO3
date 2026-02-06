# Visualization (t-SNE / UMAP)

GO3 can build gene-to-gene distance matrices from semantic similarity and use them for embedding.

## Install extras

```bash
pip install go3[viz]
```

## End-to-end example

```python
import go3

go3.load_go_terms("go-basic.obo")
annots = go3.load_gaf("goa_human.gaf")
counter = go3.build_term_counter(annots)

genes = ["TP53", "BRCA1", "EGFR", "AKT1", "CASP8"]

# 1) Distance matrix from GO similarity
ordered_genes, dist = go3.gene_distance_matrix(
    genes,
    ontology="BP",
    similarity="lin",
    groupwise="bma",
    counter=counter,
    distance_transform="auto",
)

# 2) Embeddings (precomputed distance)
ordered_genes, emb_tsne = go3.tsne_genes(
    genes,
    "BP",
    "lin",
    "bma",
    counter,
    distance_transform="auto",
    perplexity=2.0,
    n_iter=500,
    random_state=42,
)

ordered_genes, emb_umap = go3.umap_genes(
    genes,
    "BP",
    "lin",
    "bma",
    counter,
    distance_transform="auto",
    n_neighbors=3,
    min_dist=0.1,
    random_state=42,
)
```

## Plot helpers

```python
ordered_genes, emb, fig, ax = go3.plot_tsne_genes(
    genes,
    "BP",
    "lin",
    "bma",
    counter,
    perplexity=2.0,
    n_iter=500,
    random_state=42,
    annotate="auto",
    title="GO3 t-SNE",
)

ordered_genes, emb_u, fig_u, ax_u = go3.plot_umap_genes(
    genes,
    "BP",
    "lin",
    "bma",
    counter,
    n_neighbors=3,
    min_dist=0.1,
    random_state=42,
    annotate="auto",
    title="GO3 UMAP",
)
```

Example output using plot helpers:

![t-SNE helper example](../../../imgs/plot_helper_tsne_example.png)

![UMAP helper example](../../../imgs/plot_helper_umap_example.png)

## Distance transforms

`gene_distance_matrix` supports:

- `auto`
- `one_minus`
- `max_minus`
- `reciprocal`

`auto` is usually the best choice:

- normalized similarities (`lin`, `wang`, `simrel`, `topoicsim`) use a `1 - sim` style transform
- non-normalized similarities use a max-based transform

## Parameter constraints

- `tsne_genes`: `perplexity < number_of_genes`
- `umap_genes`: `n_neighbors < number_of_genes`
- both require at least 2 genes

## Compare multiple settings

The repository includes a sweep demo script:

```bash
python scripts/embedding_sweep_demo.py --n-genes 80 --embed both
```

Custom sweep:

```bash
python scripts/embedding_sweep_demo.py \
  --compare both \
  --sweep-ontologies BP,MF,CC \
  --sweep-similarities resnik,lin,wang,topoicsim \
  --distance-transform auto \
  --out-prefix embedding_sweep
```
