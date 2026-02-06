Examples
========

This page shows practical GO3 workflows from initialization to large-scale comparisons.

Setup
-----

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")
   annots = go3.load_gaf("goa_human.gaf")
   counter = go3.build_term_counter(annots)

Term-to-term similarity
-----------------------

.. code-block:: python

   go1 = "GO:0006397"
   go2 = "GO:0008380"

   # Available methods include: resnik, lin, jc, simrel, iccoef, graphic, wang, topoicsim
   sim = go3.semantic_similarity(go1, go2, "lin", counter)
   print("Lin similarity:", sim)

Batch term similarity
---------------------

.. code-block:: python

   list1 = ["GO:0006397", "GO:0008380", "GO:0008150"]
   list2 = ["GO:0008380", "GO:0006397", "GO:0009987"]

   scores = go3.batch_similarity(list1, list2, "resnik", counter)
   print(scores)

The two lists must have the same length.

Term-set similarity
-------------------

.. code-block:: python

   terms_a = ["GO:0006397", "GO:0008380"]
   terms_b = ["GO:0008380", "GO:0009987"]

   sim_bma = go3.termset_similarity(terms_a, terms_b, term_similarity="lin", groupwise="bma", counter=counter)
   sim_max = go3.termset_similarity(terms_a, terms_b, term_similarity="lin", groupwise="max", counter=counter)
   sim_avg = go3.termset_similarity(terms_a, terms_b, term_similarity="lin", groupwise="avg", counter=counter)
   sim_h   = go3.termset_similarity(terms_a, terms_b, term_similarity="lin", groupwise="hausdorff", counter=counter)
   sim_gic = go3.termset_similarity(terms_a, terms_b, term_similarity="lin", groupwise="simgic", counter=counter)

   print(sim_bma, sim_max, sim_avg, sim_h, sim_gic)

Gene-to-gene similarity
-----------------------

.. code-block:: python

   sim = go3.compare_genes("BRCA1", "CASP8", "BP", "lin", "bma", counter)
   print(sim)

Batch gene similarity
---------------------

.. code-block:: python

   pairs = [("TP53", "BRCA1"), ("EGFR", "AKT1"), ("GSDME", "NLRP1")]
   scores = go3.compare_gene_pairs_batch(pairs, "BP", "lin", "bma", counter)
   print(scores)

Distance matrix for downstream analysis
---------------------------------------

.. code-block:: python

   genes = ["BRCA1", "CASP8", "TP53", "EGFR", "AKT1"]
   ordered_genes, dist = go3.gene_distance_matrix(
       genes,
       ontology="BP",
       similarity="lin",
       groupwise="bma",
       counter=counter,
       distance_transform="auto",
   )

   print(ordered_genes)
   print(len(dist), len(dist[0]))

Embedding helpers (t-SNE / UMAP)
---------------------------------

.. code-block:: python

   # Requires go3[viz] extras
   genes = ["BRCA1", "CASP8", "TP53", "EGFR", "AKT1"]

   genes, emb_tsne = go3.tsne_genes(
       genes,
       "BP",
       "lin",
       "bma",
       counter,
       perplexity=2.0,
       n_iter=500,
       random_state=42,
   )

   genes, emb_umap = go3.umap_genes(
       genes,
       "BP",
       "lin",
       "bma",
       counter,
       n_neighbors=3,
       random_state=42,
   )

Quick plotting helpers
----------------------

.. code-block:: python

   genes, emb, fig, ax = go3.plot_tsne_genes(
       genes=["BRCA1", "CASP8", "TP53", "EGFR", "AKT1"],
       ontology="BP",
       similarity="lin",
       groupwise="bma",
       counter=counter,
       perplexity=2.0,
       n_iter=500,
       random_state=42,
       annotate="auto",
       title="GO3 t-SNE",
   )

   # Reuse generic plotting for custom embeddings
   fig2, ax2 = go3.plot_embedding(emb, genes=genes, annotate="all", title="Custom plot")

Thread control
--------------

.. code-block:: python

   # Call once before heavy batch workloads
   go3.set_num_threads(8)

Basic error handling patterns
-----------------------------

.. code-block:: python

   # Unknown gene in compare_genes -> ValueError
   try:
       go3.compare_genes("FAKE_GENE", "BRCA1", "BP", "lin", "bma", counter)
   except ValueError as exc:
       print("compare_genes error:", exc)

   # Missing term or cross-namespace term pairs return similarity 0.0
   print(go3.semantic_similarity("GO:9999999", "GO:0006397", "lin", counter))
