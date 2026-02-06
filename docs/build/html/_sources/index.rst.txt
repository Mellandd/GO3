=================
:math:`GO_3`
=================

GO3 is a high-performance Python library (Rust backend) for Gene Ontology semantic similarity.

It supports:

- ontology and annotation loading
- term-level similarity
- term-set and gene-level similarity
- high-throughput batch workflows
- distance-matrix and embedding utilities

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   introduction
   examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Guides

   similarity
   guide/performance
   guide/visualization
   benchmarks

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   ontology
   goterm
   annotations
   utilities

Start Here
==========

- :doc:`introduction`
- :doc:`examples`
- :doc:`similarity`
- :doc:`guide/performance`
- :doc:`guide/visualization`
- :doc:`benchmarks`

Install
=======

.. code-block:: bash

   pip install go3

Optional visualization extras:

.. code-block:: bash

   pip install go3[viz]

Quick example
=============

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")
   annots = go3.load_gaf("goa_human.gaf")
   counter = go3.build_term_counter(annots)

   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "lin", counter)
   print(sim)
