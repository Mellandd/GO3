Examples
========

Basic Term Similarity
---------------------

.. code-block:: python

   import go3
   go3.load_go_terms()
   annots = go3.load_gaf("goa_human.gaf")
   counter = go3.build_term_counter(annots)
   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "resnik", counter)
   print("Resnik similarity:", sim)

Batch Similarity
----------------

.. code-block:: python

   list1 = ["GO:0006397", "GO:0008380"]
   list2 = ["GO:0008380", "GO:0006397"]
   sims = go3.batch_similarity(list1, list2, "lin", counter)
   print(sims)

Gene-to-Gene Similarity
-----------------------

.. code-block:: python

   sim = go3.compare_genes("BRCA1", "CASP8", "BP", "topoicsim", "bma", counter)
   print("TopoICSim gene similarity:", sim)

Gene Distance Matrix and t-SNE
------------------------------

.. code-block:: python

   genes = ["TP53", "BRCA1", "EGFR", "AKT1"]
   genes, dist = go3.gene_distance_matrix(genes, "BP", "lin", "bma", counter)
   genes, emb = go3.tsne_genes(genes, "BP", "lin", "bma", counter, perplexity=30, random_state=42)
   # Or UMAP:
   # genes, emb = go3.umap_genes(genes, "BP", "lin", "bma", counter, n_neighbors=15, random_state=42)

   # Plot (optional)
   import matplotlib.pyplot as plt
   plt.scatter(emb[:, 0], emb[:, 1])
   for (x, y), g in zip(emb, genes):
       plt.text(x, y, g, fontsize=8)
   plt.show()

   # Or use the integrated plot helper
   genes, emb, fig, ax = go3.plot_tsne_genes(
       genes, "BP", "lin", "bma", counter, perplexity=30, random_state=42, annotate="auto"
   )

Error Handling
--------------

.. code-block:: python

   try:
       sim = go3.compare_genes("FAKEGENE", "BRCA1", "BP", "resnik", "bma", counter)
   except ValueError as e:
       print("Error:", e)
