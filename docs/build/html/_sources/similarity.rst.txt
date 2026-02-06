Semantic Similarity
===================

GO3 supports term-level and gene-level semantic similarity across multiple methods.

Quick reference
---------------

.. list-table:: Similarity methods (`method` argument)
   :header-rows: 1

   * - Method
     - Key
     - Family
     - Typical range
   * - Resnik
     - ``resnik``
     - IC-based
     - ``>= 0``
   * - Lin
     - ``lin``
     - IC-based
     - ``[0, 1]``
   * - Jiang-Conrath similarity
     - ``jc``
     - IC-based
     - ``>= 0``
   * - SimRel
     - ``simrel``
     - IC-based
     - ``[0, 1]``
   * - Information Coefficient
     - ``iccoef``
     - IC-based
     - ``>= 0``
   * - GraphIC
     - ``graphic``
     - Hybrid
     - ``>= 0``
   * - Wang
     - ``wang``
     - Topological
     - ``[0, 1]``
   * - TopoICSim
     - ``topoicsim``
     - Hybrid
     - ``[0, 1]``

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

Practical behavior and edge cases
---------------------------------

- Invalid or missing GO IDs in similarity calls generally return ``0.0``.
- Terms from different namespaces produce ``0.0``.
- For normalized methods (for example ``lin`` and ``wang``), self-similarity is typically near 1.0.

Distance-oriented workflow
--------------------------

For clustering/embedding workflows, use:

- ``gene_distance_matrix``
- ``tsne_genes``
- ``umap_genes``

These functions convert similarity to distance using ``distance_transform`` rules (see :doc:`guide/visualization`).

Mathematical definitions
------------------------

Resnik
~~~~~~

.. math::

   \mathrm{Sim}_{Resnik}(t_1, t_2) = IC(\mathrm{MICA}(t_1, t_2))

Lin
~~~

.. math::

   \mathrm{Sim}_{Lin}(t_1, t_2) = \frac{2\,IC(\mathrm{MICA}(t_1, t_2))}{IC(t_1)+IC(t_2)}

Jiang-Conrath (distance-derived similarity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   d_{JC} = IC(t_1) + IC(t_2) - 2\,IC(\mathrm{MICA})

.. math::

   \mathrm{Sim}_{JC} = \frac{1}{1 + d_{JC}}

SimRel
~~~~~~

.. math::

   \mathrm{Sim}_{Rel} = \left(\frac{2\,IC(\mathrm{MICA})}{IC(t_1)+IC(t_2)}\right)\left(1-e^{-IC(\mathrm{MICA})}\right)

Wang
~~~~

Wang similarity uses weighted ancestor contributions from GO graph topology and does not require annotation IC frequencies in the same way as IC-only methods.

TopoICSim
~~~~~~~~~

TopoICSim combines topology-aware paths and IC-derived weights to produce a bounded similarity.

Bibliography
------------

.. bibliography::
   :style: unsrt

API reference
-------------

.. automodule:: go3
   :members: term_ic, semantic_similarity, batch_similarity, termset_similarity, compare_genes, compare_gene_pairs_batch
   :undoc-members:
   :show-inheritance:
