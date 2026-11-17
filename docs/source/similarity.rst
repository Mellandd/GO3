Similarity, Redundancy & Enrichment API
=======================================

GO3 provides term-level, term-set, gene-level, and gene-set semantic similarity, semantic redundancy reduction, and Over-Representation Analysis (ORA).

Term Similarity Methods (``method`` / ``similarity``)
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 14 14 14 44

   * - Key
     - Name
     - Family
     - Range
     - Formula / Summary
   * - ``lin``
     - Lin
     - IC-based
     - ``[0, 1]``
     - :math:`\frac{2\,IC(\mathrm{MICA})}{IC(t_1)+IC(t_2)}`
   * - ``resnik``
     - Resnik
     - IC-based
     - ``>= 0``
     - :math:`IC(\mathrm{MICA}(t_1, t_2))`
   * - ``simrel``
     - SimRel
     - IC-based
     - ``[0, 1]``
     - :math:`\mathrm{Sim}_{Lin}(t_1, t_2)\cdot\bigl(1-e^{-IC(\mathrm{MICA})}\bigr)`
   * - ``jc``
     - Jiang-Conrath
     - IC-based
     - ``[0, 1]``
     - :math:`\frac{1}{1 + IC(t_1) + IC(t_2) - 2\,IC(\mathrm{MICA})}`
   * - ``iccoef``
     - ICCoef
     - IC-based
     - ``[0, 1]``
     - IC ratio relative to the minimum IC of the two terms
   * - ``wang``
     - Wang
     - Topological
     - ``[0, 1]``
     - Weighted DAG paths (``is_a = 0.8``, ``part_of = 0.6``); no IC needed
   * - ``graphic``
     - GraphIC
     - Hybrid
     - ``[0, 1]``
     - Combines IC with ontology depth
   * - ``topoicsim``
     - TopoICSim
     - Hybrid
     - ``[0, 1]``
     - Disjunctive common-ancestor paths weighted by IC

Groupwise Aggregation Strategies (``groupwise``)
------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Key
     - Name
     - Summary
   * - ``bma``
     - Best-Match Average
     - Symmetrical average of best matches in both directions (recommended default)
   * - ``simgic``
     - SimGIC
     - IC-weighted Jaccard similarity over full ancestor sets
   * - ``max``
     - Maximum
     - Highest similarity across all pairs
   * - ``avg``
     - Average
     - Mean similarity across all pairs
   * - ``hausdorff``
     - Hausdorff
     - Worst-case best match (:math:`\min` of directional maxima)

Function Overview
-----------------

Term & Term-Set Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``term_ic(go_id, counter)`` — Information Content of a single GO term.
- ``semantic_similarity(id1, id2, method, counter)`` — Similarity between two GO terms.
- ``batch_similarity(list1, list2, method, counter)`` — Parallel similarity for aligned lists of GO terms.
- ``term_similarity_matrix(terms, method="lin", counter=None)`` — All-vs-all symmetric similarity matrix ``(terms, matrix)``.
- ``reduce_redundant_terms(terms, threshold=0.7, method="lin", counter=None, scores=None)`` — Greedy semantic deduplication keeping the highest-IC (or highest custom ``scores``) representative per cluster.
- ``cluster_go_terms(terms, threshold=0.7, method="lin", counter=None, scores=None)`` — Group redundant GO terms into ``{representative_id: [member_ids...]}``.
- ``termset_similarity(terms1, terms2, term_similarity="lin", groupwise="bma", counter=None)`` — Similarity between two sets of GO terms.

Gene & Gene-Set Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``compare_genes(gene1, gene2, ontology, similarity, groupwise, counter)`` — Similarity between two genes in ``"BP"``, ``"MF"``, or ``"CC"``.
- ``compare_gene_pairs_batch(pairs, ontology, similarity, groupwise, counter)`` — Parallel similarity for ``(gene1, gene2)`` tuples.
- ``compare_gene_sets(genes1, genes2, ontology="BP", similarity="lin", groupwise="bma", counter=None, term_groupwise=None)`` — Compare two gene lists by aggregating pairwise gene similarities (outer ``groupwise``: ``"bma"``, ``"max"``, ``"avg"``, ``"hausdorff"``).
- ``compare_gene_set_pairs_batch(pairs, ontology="BP", similarity="lin", groupwise="bma", counter=None, term_groupwise=None)`` — Parallel evaluation for a list of ``(genes1, genes2)`` pairs.
- ``compare_gene_set_profiles(genes1, genes2, ontology="BP", similarity="lin", groupwise="bma", counter=None)`` — Compare two gene lists via weighted GO-term profiles (supports ``"simgic"``).

GO Over-Representation Enrichment (ORA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``go_enrichment(study_genes, background_genes=None, ontology=None, alpha=0.05, min_overlap=1, method="bh")``

Runs a one-sided hypergeometric (Fisher's exact) over-representation test with true-path ancestor propagation across all candidate GO terms annotated to ``study_genes``.

Supported ``method`` values for filtering (``<= alpha``):

- ``"bh"`` (or ``"fdr"``): Benjamini-Hochberg step-up False Discovery Rate (default).
- ``"scs"`` (or ``"g:scs"``, ``"g_scs"``): Analytical **g:SCS** (*Set Counts and Sizes*) correction accounting for GO DAG hierarchy and discrete hypergeometric tail probabilities.
- ``"bonferroni"`` (or ``"bonf"``): Family-wise Bonferroni correction.
- ``"raw"`` (or ``"none"``): Unadjusted hypergeometric ``p_value``.

Each returned :class:`go3.EnrichmentResult` object always provides ``p_value``, ``p_fdr_bh``, ``p_scs``, ``p_bonferroni``, ``fold_enrichment``, ``study_genes``, and ``to_dict()``.

API Reference
-------------

.. automodule:: go3
   :members: term_ic, semantic_similarity, batch_similarity, term_similarity_matrix, reduce_redundant_terms, cluster_go_terms, termset_similarity, compare_genes, compare_gene_pairs_batch, compare_gene_sets, compare_gene_set_pairs_batch, compare_gene_set_profiles, go_enrichment, EnrichmentResult
   :undoc-members:
   :show-inheritance:
