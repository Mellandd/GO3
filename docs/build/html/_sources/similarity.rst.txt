Semantic Similarity Functions
=============================

Introduction
------------

The `go3` library provides several semantic similarity functions for comparing Gene Ontology (GO) terms. These measures rely on two main principles:

- Information Content (IC) derived from GO annotations.
- Graph-based topological relationships in the GO hierarchy.

Available Similarity Methods
-----------------------------

.. list-table:: **Similarity Methods for the `method` Parameter**
   :header-rows: 1

   * - Method Name
     - String for ``method``
     - Description
   * - Resnik
     - ``resnik``
     - Information content of the most informative common ancestor (MICA)
   * - Lin
     - ``lin``
     - Normalized Resnik similarity
   * - Jiang-Conrath
     - ``jc``
     - Inverse of Jiang-Conrath distance
   * - SimRel
     - ``simrel``
     - Lin similarity with exponential relevance factor
   * - Information Coefficient
     - ``iccoef``
     - Normalized by minimum IC of the two terms
   * - GraphIC
     - ``graphic``
     - IC divided by maximum graph depth
   * - Wang
     - ``wang``
     - Graph-based semantic similarity (Wang et al.)
   * - TopoICSim
     - ``topoicsim``
     - Topological and IC-based hybrid similarity

You can use these strings as the ``method`` parameter in all `go3` similarity functions:

.. code-block:: python

   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "lin", counter)
   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "topoicsim", counter)

Similarity Measures
--------------------

Resnik Similarity
~~~~~~~~~~~~~~~~~

The Resnik similarity :cite:p:`Resnik1995` measures the similarity between two GO terms as the information content (IC) of their Most Informative Common Ancestor (MICA):

.. math::

    \mathrm{Sim}_{Resnik}(t_1, t_2) = IC(\mathrm{MICA}(t_1, t_2))

Lin Similarity
~~~~~~~~~~~~~~

Lin's similarity :cite:p:`Lin1998` normalizes Resnik's similarity by the sum of the ICs of both terms:

.. math::

    \mathrm{Sim}_{Lin}(t_1, t_2) = \frac{2 \times IC(\mathrm{MICA}(t_1, t_2))}{IC(t_1) + IC(t_2)}

Jiang-Conrath Similarity
~~~~~~~~~~~~~~~~~~~~~~~~

Jiang and Conrath define a distance between two GO terms based on IC :cite:p:`JiangConrath1997`:

.. math::

    d_{JC} = IC(t_1) + IC(t_2) - 2 \times IC(\mathrm{MICA})

Similarity is then calculated as:

.. math::

    \mathrm{Sim}_{JC} = \frac{1}{d_{JC}}

SimRel Similarity
~~~~~~~~~~~~~~~~~

The SimRel measure :cite:p:`Schlicker2006` combines Lin's similarity with an exponential relevance factor:

.. math::

    \mathrm{Sim}_{Rel} = \left( \frac{2 \times IC(\mathrm{MICA})}{IC(t_1) + IC(t_2)} \right) \times \left(1 - e^{-IC(\mathrm{MICA})}\right)

Information Coefficient
~~~~~~~~~~~~~~~~~~~~~~~

Li et al. :cite:p:`Li2010` propose a normalization using the minimum IC of the two terms:

.. math::

    \mathrm{IC\_coef} = \frac{IC(\mathrm{MICA})}{\min(IC(t_1), IC(t_2))}

GraphIC Similarity
~~~~~~~~~~~~~~~~~~

The GraphIC measure uses the maximum graph depth of the two terms to scale the similarity:

.. math::

    \mathrm{GraphIC} = \frac{IC(\mathrm{MICA})}{\max(\mathrm{depth}(t_1), \mathrm{depth}(t_2)) + 1}

Wang Similarity
~~~~~~~~~~~~~~~

The Wang similarity :cite:p:`Wang2007` considers the graph structure of GO by propagating weights from each term through its ancestors.

Each ancestor node receives a weight based on the decay factor (usually :math:`w = 0.8`). The similarity is computed as:

.. math::

    \mathrm{Sim}_{Wang}(t_1, t_2) =
    \frac{
        \sum_{x \in A(t_1) \cap A(t_2)} \left( S_{t_1}(x) + S_{t_2}(x) \right)
    }{
        SV(t_1) + SV(t_2)
    }

where

- :math:`A(t)` is the set of ancestors of term :math:`t` (including itself),
- :math:`S_t(x)` is the semantic contribution of ancestor :math:`x` to term :math:`t`,
- :math:`SV(t)` is the total semantic value of term :math:`t`.

The key idea is that ancestors closer to the term contribute more to its meaning than distant ancestors, capturing the hierarchical semantics of the ontology without relying on external annotation statistics.

TopoICSim Similarity
~~~~~~~~~~~~~~~~~~~~

The TopoICSim similarity :cite:p:`Ehsani2016` is a hybrid measure that combines information content and the topology of the GO graph. It is defined as:

.. math::

    \mathrm{Sim}_{TopoICSim}(t_1, t_2) = 1 - \frac{2}{\pi} \arctan \left( \min_{x \in DCA(t_1, t_2)} \frac{wSP(t_1, x) + wSP(t_2, x)}{wLP(x, r)} \right)

where

- :math:`DCA(t_1, t_2)` is the set of disjunctive common ancestors of :math:`t_1` and :math:`t_2`,
- :math:`wSP(t, x)` is the weighted shortest path (sum of inverse ICs) from :math:`t` to ancestor :math:`x`,
- :math:`wLP(x, r)` is the weighted longest path from :math:`x` to a root :math:`r` in the ontology.

This measure captures both the specificity of the common ancestors and the topological distance between terms, providing a robust similarity score.

Batch Computation
-----------------

All these similarity measures are available in efficient batch versions in the `go3` library, taking full advantage of Rust’s parallelism.

Bibliography
------------

.. bibliography::
   :style: unsrt

.. automodule:: go3
   :members: term_ic, semantic_similarity, batch_similarity, compare_genes, compare_gene_pairs_batch
   :undoc-members:
   :show-inheritance: