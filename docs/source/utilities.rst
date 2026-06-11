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

``gene_set_distance_matrix(gene_sets, ontology="BP", similarity="lin", groupwise="bma", counter=..., distance_transform="auto", term_groupwise=None)``

Returns ``(gene_set_names, distance_matrix)`` for named gene sets supplied as ``[(name, genes), ...]``. The matrix uses pairwise gene similarities aggregated across each gene-set pair; ``term_groupwise`` optionally controls the inner gene-to-gene GO-term aggregation.

Notes:

- ``groupwise`` is the outer aggregation across genes and supports ``bma``, ``max``, ``avg``, and ``hausdorff``.
- ``term_groupwise`` controls the inner gene-to-gene GO-term aggregation. If omitted, it defaults to ``groupwise``.
- For IC-weighted GO-profile overlap between gene sets, use ``compare_gene_set_profiles`` instead.

Distance transforms
~~~~~~~~~~~~~~~~~~~

The ``distance_transform`` parameter controls how similarity scores are converted to distances for clustering and embedding algorithms. Available options:

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Transform
     - Formula
     - When to use
   * - ``auto``
     - depends on method
     - **Recommended default.** Automatically selects the best transform: ``one_minus`` for normalized methods (``lin``, ``wang``, ``simrel``, ``topoicsim``), ``max_minus`` for unbounded methods (``resnik``, ``jc``).
   * - ``one_minus``
     - ``1 - sim``
     - Use for normalized methods that produce values in [0, 1]. Self-distance is 0, maximum distance is 1.
   * - ``max_minus``
     - ``max(all_sims) - sim``
     - Use for unbounded methods (e.g., Resnik) where the range is not fixed. Produces distances relative to the highest observed similarity.
   * - ``reciprocal``
     - ``1 / (1 + sim)``
     - Always valid regardless of method range, but produces a non-linear mapping. Useful as a fallback when other transforms are not appropriate.

Embedding APIs
--------------

These helpers build embeddings from precomputed GO similarity-derived distances.

``tsne_genes(genes, ontology, similarity, groupwise, counter, ...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes a t-SNE embedding from gene-level semantic similarity distances.

Key parameters:

- ``perplexity`` (float) -- Controls the balance between local and global structure. Higher values consider more neighbors per point. Must be less than the number of genes. A starting point is ``perplexity ~ sqrt(n_genes)``.
- ``n_iter`` (int) -- Number of optimization iterations. Default is typically 1000; 500 is often sufficient for exploration.
- ``n_components`` (int) -- Dimensionality of the output embedding (usually 2).
- ``random_state`` (int) -- Seed for reproducibility.
- ``distance_transform`` (str) -- How to convert similarity to distance (see above).

``umap_genes(genes, ontology, similarity, groupwise, counter, ...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes a UMAP embedding from gene-level semantic similarity distances.

Key parameters:

- ``n_neighbors`` (int) -- Number of nearest neighbors to consider when constructing the graph. Smaller values emphasize local structure; larger values capture more global patterns. Must be less than the number of genes. A starting point is ``n_neighbors ~ 15`` for exploratory analysis.
- ``min_dist`` (float) -- Minimum distance between embedded points. Smaller values produce tighter clusters; larger values spread points more evenly.
- ``random_state`` (int) -- Seed for reproducibility.
- ``distance_transform`` (str) -- How to convert similarity to distance (see above).

``plot_embedding(embedding, ...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a scatter plot from a 2D embedding array.

- ``genes`` (list[str]) -- Gene labels for each point.
- ``annotate`` -- Label display mode: ``"all"`` labels every point, ``"auto"`` labels only non-overlapping points, ``None`` disables labels.
- ``title`` (str) -- Plot title.
- ``categories`` (list[str], optional) -- Categorical labels for coloring points by group.

Returns ``(fig, ax)`` matplotlib objects.

``plot_tsne_genes(...)`` / ``plot_umap_genes(...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convenience wrappers that combine embedding computation and plotting in one call. They accept the same parameters as ``tsne_genes`` / ``umap_genes`` plus the plotting parameters from ``plot_embedding``.

Return ``(gene_order, embedding, fig, ax)``.

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
   :members: set_num_threads, term_ic, termset_similarity, compare_genes, compare_gene_pairs_batch, compare_gene_sets, compare_gene_set_pairs_batch, compare_gene_set_profiles, gene_distance_matrix, gene_set_distance_matrix, tsne_genes, umap_genes, plot_tsne_genes, plot_umap_genes, plot_embedding
   :undoc-members:
   :show-inheritance:
   :no-index:
