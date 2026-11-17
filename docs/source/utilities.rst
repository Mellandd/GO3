Distances, Embeddings & Runtime API
===================================

Functions for converting semantic similarity into distance matrices, computing 2D embeddings (t-SNE and UMAP), plotting, and configuring multi-threading.

Distance Matrices
-----------------

- ``term_distance_matrix(terms, method="lin", counter=None, distance_transform="auto")`` — Returns ``(terms, distance_matrix)`` for a list of GO term IDs.
- ``gene_distance_matrix(genes=None, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto")`` — Returns ``(gene_order, distance_matrix)`` for a list of genes.
- ``gene_set_distance_matrix(gene_sets, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto", term_groupwise=None)`` — Returns ``(set_names, distance_matrix)`` for named gene sets ``[(name, genes), ...]``.

Distance Transforms (``distance_transform``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 27 55

   * - Transform
     - Formula
     - When to use
   * - ``auto``
     - Automatic
     - **Default.** Uses ``one_minus`` for bounded methods (``lin``, ``wang``, ``simrel``, ``topoicsim``, ``simgic``) and ``max_minus`` for unbounded methods (``resnik``, ``jc``).
   * - ``one_minus``
     - :math:`1 - \text{sim}`
     - For normalized similarities in ``[0, 1]``.
   * - ``max_minus``
     - :math:`\max(\text{sim}) - \text{sim}`
     - For unbounded similarities such as ``resnik``.
   * - ``reciprocal``
     - :math:`\frac{1}{1 + \text{sim}}`
     - Non-linear fallback valid for any non-negative similarity.

Embeddings & Plotting (``go3[viz]``)
------------------------------------

Requires optional dependencies: ``pip install 'go3[viz]'``.

- ``tsne_genes(genes, ontology, similarity, groupwise, counter, ...)`` — Returns ``(gene_order, embedding_2d)`` using t-SNE (note: ``perplexity`` must be ``< len(genes)``).
- ``umap_genes(genes, ontology, similarity, groupwise, counter, ...)`` — Returns ``(gene_order, embedding_2d)`` using UMAP (note: ``n_neighbors`` must be ``< len(genes)``).
- ``plot_tsne_genes(...)`` / ``plot_umap_genes(...)`` — Compute the embedding and plot in one call; returns ``(gene_order, embedding, fig, ax)``.
- ``plot_embedding(embedding, genes=..., annotate="auto", title=..., categories=...)`` — Plot any 2D embedding array with matplotlib; returns ``(fig, ax)``.

.. code-block:: python

   genes = ["BRCA1", "CASP8", "TP53", "EGFR", "AKT1"]

   genes, emb, fig, ax = go3.plot_tsne_genes(
       genes,
       ontology="BP",
       similarity="lin",
       groupwise="bma",
       counter=counter,
       perplexity=2.0,
       random_state=42,
       annotate="auto",
       title="GO3 t-SNE",
   )

Thread Pool Configuration
-------------------------

``set_num_threads(n_threads)``

Sets the number of CPU threads used by Rayon for parallel batch and matrix operations (``0`` uses all available logical cores). Must be called **once at the start of the session** before running any parallel operation.

API Reference
-------------

.. automodule:: go3
   :members: set_num_threads, term_distance_matrix, gene_distance_matrix, gene_set_distance_matrix, tsne_genes, umap_genes, plot_tsne_genes, plot_umap_genes, plot_embedding
   :undoc-members:
   :show-inheritance:
   :no-index:
