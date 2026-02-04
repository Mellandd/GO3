# Visualization (t-SNE / UMAP)

GO3 can build gene-to-gene distance matrices and compute t-SNE/UMAP embeddings using precomputed distances.

## Install visualization extras

```bash
pip install go3[viz]
```

## Distance matrix + embedding

```python
import go3

go3.load_go_terms()
annots = go3.load_gaf("goa_human.gaf")
counter = go3.build_term_counter(annots)

genes = ["TP53", "BRCA1", "EGFR", "AKT1"]
genes, dist = go3.gene_distance_matrix(genes, "BP", "lin", "bma", counter)
genes, emb_tsne = go3.tsne_genes(genes, "BP", "lin", "bma", counter, perplexity=30, random_state=42)
genes, emb_umap = go3.umap_genes(genes, "BP", "lin", "bma", counter, n_neighbors=15, random_state=42)
```

## Plot helpers

```python
genes, emb, fig, ax = go3.plot_tsne_genes(
    genes, "BP", "lin", "bma", counter, perplexity=30, random_state=42, annotate="auto"
)

genes, emb, fig, ax = go3.plot_umap_genes(
    genes, "BP", "lin", "bma", counter, n_neighbors=15, random_state=42, annotate="auto"
)
```

