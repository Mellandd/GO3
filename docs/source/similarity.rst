Semantic Similarity Functions
=============================

Introduction
------------

The `go3` library provides several semantic similarity functions for comparing Gene Ontology (GO) terms. These measures rely on two main principles:

- Information Content (IC) derived from GO annotations.
- Graph-based topological relationships in the GO hierarchy.

In this section, we summarize the main similarity measures provided, along with their theoretical definitions and bibliographic references.

Similarity Measures
-------------------

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

The Wang similarity :cite:p:`Wang2007` considers the graph structure of GO by propagating weights from each term through its ancestors:

- Each ancestor node receives a weight based on the decay factor (usually :math:`w = 0.8`).
- The similarity is computed as:

.. math::

    \text{Sim}_{Wang}(t_1, t_2) = \frac{\sum_{x \in A(t_1) \cap A(t_2)} (S_{t_1}(x) + S_{t_2}(x))}{SV(t_1) + SV(t_2)}

Where:
	•	:math:A(t) is the set of ancestors of term :math:t (including itself).
	•	:math:S_t(x) is the semantic contribution of ancestor :math:x to term :math:t.
	•	:math:SV(t) is the total semantic value of term :math:t.

The key idea is that ancestors that are closer to the term contribute more to its meaning than distant ancestors, capturing the hierarchical semantics of the ontology without relying on external annotation statistics.

Batch Computation
-----------------

All these similarity measures are available in efficient batch versions in the `go3` library, taking full advantage of Rust’s parallelism.

Bibliography
------------

.. bibliography::
   :style: unsrt