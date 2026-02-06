Utilities and Advanced APIs
===========================

Threading
---------

``set_num_threads(n_threads)``

Controls the maximum number of internal threads used by GO3 batch operations.

.. code-block:: python

   import go3
   go3.set_num_threads(8)

IC lookup
---------

``term_ic(go_id, counter)`` returns the Information Content for one term.

.. code-block:: python

   ic = go3.term_ic("GO:0006397", counter)

Gene distance matrices
----------------------

``gene_distance_matrix(genes=None, ontology="BP", similarity="lin", groupwise="bma", counter=..., distance_transform="auto")``

Returns ``(gene_order, distance_matrix)``.

Distance transforms:

- ``auto``
- ``one_minus``
- ``max_minus``
- ``reciprocal``

Embedding APIs
--------------

- ``tsne_genes(...)``
- ``umap_genes(...)``
- ``plot_tsne_genes(...)``
- ``plot_umap_genes(...)``
- ``plot_embedding(...)``

These helpers build embeddings from precomputed GO similarity-derived distances.

Minimal embedding example
-------------------------

.. code-block:: python

   genes = ["BRCA1", "CASP8", "TP53", "EGFR", "AKT1"]

   genes, emb = go3.tsne_genes(
       genes,
       ontology="BP",
       similarity="lin",
       groupwise="bma",
       counter=counter,
       perplexity=2.0,
       random_state=42,
   )

   fig, ax = go3.plot_embedding(emb, genes=genes, annotate="auto", title="GO embedding")

API reference
-------------

.. automodule:: go3
   :members: set_num_threads, term_ic, termset_similarity, compare_gene_pairs_batch, gene_distance_matrix, tsne_genes, umap_genes, plot_tsne_genes, plot_umap_genes, plot_embedding
   :undoc-members:
   :show-inheritance:
   :no-index:
