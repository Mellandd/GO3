Cookbook & Workflows
====================

Practical, copy-pasteable recipes for common GO3 tasks.

1. Setup & Native GAF Filtering
-------------------------------

Load ``.obo``/``.obo.gz`` and ``.gaf``/``.gaf.gz`` files directly. Use :class:`go3.GAFParseOptions` to filter annotations in Rust before building the IC counter:

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")

   # Keep only experimental/curated annotations (exclude IEA) on Biological Process
   opts = go3.GAFParseOptions(
       exclude_evidence={"IEA"},
       include_aspects={"P", "F", "C"},
       exclude_qualifiers={"NOT"},
       allow_obsolete_terms=False,
       allow_nd_root_only=False,
   )
   annots = go3.load_gaf("goa_human.gaf.gz", options=opts)
   counter = go3.build_term_counter(annots)

2. Exploring the Ontology & Gene–Term Lookups
---------------------------------------------

.. code-block:: python

   # Inspect a GO term
   term = go3.get_term_by_id("GO:0006397")
   print(term.name, term.namespace, term.depth, term.parents)

   # Traverse ancestors and descendants (optionally including part_of edges)
   ancs = go3.ancestors("GO:0006397", relations=["part_of"])
   descs = go3.descendants("GO:0006397", relations=["part_of"])
   dca = go3.deepest_common_ancestor("GO:0006397", "GO:0008380")

   # Bidirectional gene <-> GO queries
   bp_genes = go3.list_genes(ontology="BP")
   tp53_terms = go3.get_terms_for_gene("TP53", ontology="BP")
   mrna_genes = go3.get_genes_for_term("GO:0006397", include_descendants=True)

3. Term Similarity, Matrices & Redundancy Reduction
---------------------------------------------------

.. code-block:: python

   # Single pair & batch aligned pairs
   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "lin", counter)
   batch = go3.batch_similarity(
       ["GO:0006397", "GO:0008150"],
       ["GO:0008380", "GO:0009987"],
       "lin",
       counter,
   )

   # Compare two sets of GO terms
   set_sim = go3.termset_similarity(
       ["GO:0006397", "GO:0008380"],
       ["GO:0000398", "GO:0006396"],
       term_similarity="lin",
       groupwise="bma",
       counter=counter,
   )

   # All-vs-all term similarity and distance matrices
   terms = ["GO:0006397", "GO:0008380", "GO:0000398", "GO:0007049"]
   ordered_terms, sim_mat = go3.term_similarity_matrix(terms, method="lin", counter=counter)
   ordered_terms, dist_mat = go3.term_distance_matrix(terms, method="lin", counter=counter)

   # Condense redundant GO terms into non-redundant representatives
   rep_terms = go3.reduce_redundant_terms(terms, threshold=0.7, method="lin", counter=counter)
   clusters = go3.cluster_go_terms(terms, threshold=0.7, method="lin", counter=counter)

4. Gene & Gene-Set Comparisons
------------------------------

.. code-block:: python

   # Single gene pair & batch gene pairs
   g_sim = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)
   g_batch = go3.compare_gene_pairs_batch(
       [("TP53", "BRCA1"), ("EGFR", "AKT1")], "BP", "lin", "bma", counter
   )

   # Compare two gene sets (pairwise gene matrix vs. weighted GO profile)
   dna_repair = ["TP53", "BRCA1", "ATM"]
   apoptosis = ["CASP8", "GSDME", "NLRP1"]

   sim_genes = go3.compare_gene_sets(dna_repair, apoptosis, "BP", "lin", "bma", counter)
   sim_profile = go3.compare_gene_set_profiles(dna_repair, apoptosis, "BP", "lin", "simgic", counter)

   # Batch comparison of multiple gene-set pairs (e.g., permutation/null background)
   set_batch = go3.compare_gene_set_pairs_batch(
       [(dna_repair, apoptosis)], "BP", "lin", "bma", counter
   )

5. GO Over-Representation Enrichment (ORA)
------------------------------------------

Run hypergeometric enrichment with choice of multiple-testing correction:

- ``method="bh"``: Benjamini-Hochberg FDR (default)
- ``method="scs"``: g:Profiler analytical ``g:SCS`` (*Set Counts and Sizes*)
- ``method="bonferroni"``: Bonferroni correction
- ``method="raw"``: unadjusted p-value

.. code-block:: python

   import math

   study_genes = ["PINK1", "PRKN", "PARK7", "SNCA", "LRRK2"]

   results = go3.go_enrichment(
       study_genes,
       ontology="BP",
       alpha=0.05,
       min_overlap=2,
       method="scs",
   )

   for r in results[:5]:
       print(
           r.go_id,
           r.name,
           f"p_scs={r.p_scs:.2e}",
           f"fdr={r.p_fdr_bh:.2e}",
           f"fold={r.fold_enrichment:.1f}",
       )

   # Combine ORA with semantic redundancy reduction
   enriched_ids = [r.go_id for r in results]
   scores = [-math.log10(max(r.p_scs, 1e-300)) for r in results]
   non_redundant = go3.reduce_redundant_terms(
       enriched_ids, threshold=0.7, method="lin", counter=counter, scores=scores
   )

6. Distance Matrices & 2D Embeddings (t-SNE / UMAP)
---------------------------------------------------

.. code-block:: python

   genes = ["BRCA1", "CASP8", "TP53", "EGFR", "AKT1"]

   ordered_genes, dist = go3.gene_distance_matrix(
       genes, ontology="BP", similarity="lin", groupwise="bma", counter=counter
   )

   # Requires: pip install 'go3[viz]'
   ordered_genes, emb, fig, ax = go3.plot_tsne_genes(
       genes,
       ontology="BP",
       similarity="lin",
       groupwise="bma",
       counter=counter,
       perplexity=2.0,
       random_state=42,
       annotate="auto",
       title="Gene Functional Landscape (t-SNE)",
   )
