Installation
============

GO3 provides Python bindings for a Rust implementation of Gene Ontology semantic similarity.

Install from PyPI:

.. code-block:: bash

   pip install go3

Optional visualization dependencies:

.. code-block:: bash

   pip install go3[viz]

Requirements
============

GO3 expects:

- a GO ontology file in OBO format (for example ``go-basic.obo``)
- a GO annotation file in GAF format for your organism

If you call ``go3.load_go_terms()`` without a path, GO3 downloads ``go-basic.obo`` automatically.

Minimal workflow
================

.. code-block:: python

   import go3

   # 1) Ontology
   go3.load_go_terms("go-basic.obo")

   # 2) Annotations
   annots = go3.load_gaf("goa_human.gaf")

   # 3) Information Content structures
   counter = go3.build_term_counter(annots)

   # 4) Term similarity
   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "lin", counter)
   print(sim)

   # 5) Gene similarity
   score = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)
   print(score)

Core concepts
=============

- ``load_go_terms`` loads and caches ontology terms in memory.
- ``load_gaf`` parses GAF annotations and builds a gene-to-GO mapping.
- ``build_term_counter`` computes annotation counts and IC values.
- IC-based methods (for example ``resnik`` and ``lin``) require ``counter``.

Namespaces
==========

GO3 uses standard GO sub-ontologies:

- ``BP``: Biological Process
- ``MF``: Molecular Function
- ``CC``: Cellular Component

For gene-level APIs, select namespace explicitly via the ``ontology`` argument.

Next steps
==========

- :doc:`examples` for end-to-end usage patterns
- :doc:`similarity` for available methods and formulas
- :doc:`guide/performance` for throughput-oriented workflows
- :doc:`benchmarks` for reproducible comparisons
