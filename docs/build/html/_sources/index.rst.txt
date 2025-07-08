.. GO3 documentation master file, created by
   sphinx-quickstart on Wed Jun  4 16:14:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
:math:`GO_3`
=================

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction:

   introduction

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples:

   examples

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 📄 API documentation:

   ontology
   goterm
   annotations
   similarity

Table of Contents
=================

* `GO3`_
* :doc:`🖥️ Examples <examples>`

GO3
====

:math:`GO_3`. is a Python library to work with the Gene Ontology (GO). It can calculate similarities between individual terms or between sets of terms.
It also can calculate similarities between genes directly with the associated GO Terms from any given sub-ontology: MF, BP or CC.


Main features
=============


Installation
=============

**go3** is provided as binary wheels for most platforms on PyPI (Linux, Windows and MacOS). You can run

.. code-block:: bash

   pip install go3

**go3** does not ship with any prebuilt GO Ontology by default. If you don't provide any .obo, when you try to load the ontology into memory it automatically downloads the last version of go-basic.obo.

Examples
=============

.. code-block:: Python

   import go3

   # Initialize the ontology
   go3.load_go_terms()

   # Load an specific GO Term
   term_1 = go3.get_term_by_id("GO:0006397")

   print(term_1.name)
   #> mRNA processing


Examples in batch computing
===============================