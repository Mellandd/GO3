Semantic Similarity
===================

GO3 supports term-level, gene-level, and gene-set semantic similarity across multiple methods.

Background: IC and MICA
------------------------

Most similarity methods in GO3 rely on **Information Content (IC)** and the **Most Informative Common Ancestor (MICA)**. See :doc:`introduction` for full definitions.

In brief:

- **IC(t)** measures how specific a term is: ``IC(t) = -log(count(t) / total(namespace))``. Rare terms have high IC.
- **MICA(t1, t2)** is the common ancestor of *t1* and *t2* with the highest IC -- i.e., the most specific shared concept.

Methods that do **not** require IC (e.g., Wang) use graph topology instead.

Quick reference
---------------

.. list-table:: Similarity methods (``method`` argument)
   :header-rows: 1

   * - Method
     - Key
     - Family
     - Typical range
     - Description
   * - Resnik
     - ``resnik``
     - IC-based
     - ``>= 0``
     - IC of the MICA; unbounded, corpus-dependent
   * - Lin
     - ``lin``
     - IC-based
     - ``[0, 1]``
     - Normalized Resnik: ``2*IC(MICA) / (IC(t1)+IC(t2))``
   * - Jiang-Conrath similarity
     - ``jc``
     - IC-based
     - ``>= 0``
     - Distance-derived: ``1 / (1 + IC(t1)+IC(t2) - 2*IC(MICA))``
   * - SimRel
     - ``simrel``
     - IC-based
     - ``[0, 1]``
     - Lin weighted by MICA relevance: penalizes shallow ancestors
   * - Information Coefficient
     - ``iccoef``
     - IC-based
     - ``>= 0``
     - IC-based coefficient combining ancestor information
   * - GraphIC
     - ``graphic``
     - Hybrid
     - ``>= 0``
     - Combines IC values with graph-structural features
   * - Wang
     - ``wang``
     - Topological
     - ``[0, 1]``
     - Weighted ancestor contributions from graph topology; no IC needed
   * - TopoICSim
     - ``topoicsim``
     - Hybrid
     - ``[0, 1]``
     - Topology-aware paths weighted by IC; bounded and normalized

Groupwise strategies
--------------------

When comparing sets of terms or genes, a **groupwise strategy** aggregates pairwise similarities into a single score.

.. list-table:: Groupwise strategies (``groupwise`` argument)
   :header-rows: 1

   * - Strategy
     - Key
     - Description
   * - Best Match Average
     - ``bma``
     - Average of best matches from both directions; balanced and widely used
   * - Maximum
     - ``max``
     - Highest pairwise similarity; captures strongest shared function
   * - Average
     - ``avg``
     - Mean of all pairwise similarities; measures overall functional overlap
   * - Hausdorff
     - ``hausdorff``
     - Worst-case best-match distance; guarantees a minimum similarity level
   * - SimGIC
     - ``simgic``
     - IC-weighted Jaccard of shared ancestor sets; set-based, not pairwise

Choosing a method
-----------------

See :doc:`guide/choosing_methods` for a dedicated guide on selecting the right similarity method and groupwise strategy for your use case.

Term-level APIs
---------------

``semantic_similarity(id1, id2, method, counter)``

- Computes one score for one term pair.
- Raises ``ValueError`` if ``method`` is unknown.

``batch_similarity(list1, list2, method, counter)``

- Computes one score per aligned pair.
- Requires ``len(list1) == len(list2)``.
- Raises ``ValueError`` if list sizes differ or method is unknown.

Set-level API
-------------

``termset_similarity(terms1, terms2, term_similarity="lin", groupwise="bma", counter=...)``

Groupwise strategies:

- ``bma``
- ``max``
- ``avg``
- ``hausdorff``
- ``simgic``

Notes:

- ``simgic`` is set-based and does not use the pairwise method in the same way as other strategies.
- For empty sets, GO3 returns ``0.0``.

Gene-level APIs
---------------

``compare_genes(gene1, gene2, ontology, similarity, groupwise, counter)``

- Ontology must be one of ``BP``, ``MF``, ``CC``.
- Raises ``ValueError`` if either gene is missing from loaded annotations.

``compare_gene_pairs_batch(pairs, ontology, similarity, groupwise, counter)``

- Fast path for large pair lists.
- Missing/empty per-gene term mappings yield ``0.0`` for those pairs.

Gene-set APIs
-------------

``compare_gene_sets(genes1, genes2, ontology="BP", similarity="lin", groupwise="bma", counter=..., term_groupwise=None)``

- Computes pairwise gene similarities between all genes in both sets.
- Aggregates the gene-by-gene similarity matrix with ``groupwise``.
- Uses the same groupwise method internally for each gene-pair GO-term comparison unless ``term_groupwise`` is supplied.
- Direct gene-set aggregation supports ``bma``, ``max``, ``avg``, and ``hausdorff``.
- Direct gene-set aggregation does not use ``simgic`` as the outer strategy, because ``simgic`` is a GO-term set/profile method rather than a pairwise gene-matrix aggregator.
- Raises ``ValueError`` if any gene in either input set is missing from loaded annotations.
- Genes with no terms in the requested ontology contribute zero pairwise similarity.

``compare_gene_set_pairs_batch(pairs, ontology="BP", similarity="lin", groupwise="bma", counter=..., term_groupwise=None)``

- Fast path for many gene-set pairs.
- ``pairs`` should contain ``(genes1, genes2)`` items, where each side is a list of gene symbols.
- Missing genes are treated as unannotated in batch mode; pairs with no comparable annotated genes yield ``0.0``.

``compare_gene_set_profiles(genes1, genes2, ontology="BP", similarity="lin", groupwise="bma", counter=...)``

- Alternative profile-based method.
- Converts each gene set to a weighted GO-term profile, where each term weight is the number of genes annotated to that term.
- Compares the two weighted GO profiles directly with ``bma``, ``max``, ``avg``, ``hausdorff``, or weighted ``simgic``.

Practical behavior and edge cases
---------------------------------

- Invalid or missing GO IDs in similarity calls generally return ``0.0``.
- Terms from different namespaces produce ``0.0``.
- For normalized methods (for example ``lin`` and ``wang``), self-similarity is typically near 1.0.

Distance-oriented workflow
--------------------------

For clustering/embedding workflows, use:

- ``gene_distance_matrix``
- ``gene_set_distance_matrix``
- ``tsne_genes``
- ``umap_genes``

These functions convert similarity to distance using ``distance_transform`` rules (see :doc:`guide/visualization`).

Mathematical definitions
------------------------

Resnik
~~~~~~

.. math::

   \mathrm{Sim}_{Resnik}(t_1, t_2) = IC(\mathrm{MICA}(t_1, t_2))

The simplest IC-based measure: the similarity between two terms equals the IC of their MICA. The result is unbounded and depends on the annotation corpus.

Lin
~~~

.. math::

   \mathrm{Sim}_{Lin}(t_1, t_2) = \frac{2\,IC(\mathrm{MICA}(t_1, t_2))}{IC(t_1)+IC(t_2)}

Normalizes Resnik by the individual ICs, producing a score in [0, 1].

Jiang-Conrath (distance-derived similarity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   d_{JC} = IC(t_1) + IC(t_2) - 2\,IC(\mathrm{MICA})

.. math::

   \mathrm{Sim}_{JC} = \frac{1}{1 + d_{JC}}

Converts the JC distance into a similarity via the reciprocal transform.

SimRel
~~~~~~

.. math::

   \mathrm{Sim}_{Rel} = \left(\frac{2\,IC(\mathrm{MICA})}{IC(t_1)+IC(t_2)}\right)\left(1-e^{-IC(\mathrm{MICA})}\right)

Combines Lin similarity with a relevance factor that penalizes shallow (low-IC) MICAs.

Wang
~~~~

Wang similarity uses weighted ancestor contributions from GO graph topology and does not require annotation-based IC values.

Each ancestor *a* of a term *t* receives a semantic value *S_A(a)* based on the path from *t* to *a*.  Edge weights decay by relationship type:

- ``is_a`` edges contribute a weight of **0.8**
- ``part_of`` edges contribute a weight of **0.6**

The semantic value of *t* itself is 1. The overall similarity between two terms is the ratio of their shared semantic contributions to their total semantic values. Because it relies only on graph structure, Wang is useful when annotation data may be biased or incomplete.

TopoICSim
~~~~~~~~~

TopoICSim combines topology-aware paths and IC-derived weights to produce a bounded similarity in [0, 1].

It identifies the longest-information-content path between two terms through their common ancestors, weighting path segments by the IC of intermediate nodes. This captures both the structural distance in the DAG and the specificity of the connecting path, making it more discriminating than purely IC-based or purely topological methods.

Bibliography
------------

.. bibliography::
   :style: unsrt

API reference
-------------

.. automodule:: go3
   :members: term_ic, semantic_similarity, batch_similarity, termset_similarity, compare_genes, compare_gene_pairs_batch, compare_gene_sets, compare_gene_set_pairs_batch, compare_gene_set_profiles, gene_set_distance_matrix
   :undoc-members:
   :show-inheritance:
