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

Quick Start
===========

.. code-block:: python

   import go3
   go3.load_go_terms()
   annots = go3.load_gaf("goa_human.gaf")
   counter = go3.build_term_counter(annots)
   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "resnik", counter)
   print("Resnik similarity:", sim)

Main Features
=============

- **Ontology loading**: Download or load any GO OBO file.
- **Annotation parsing**: Read GAF files for any organism.
- **Term and gene similarity**: Compute Resnik, Lin, Wang, TopoICSim, and more.
- **Batch and parallel computation**: Fast, scalable, and memory-efficient.
- **Rich Python API**: All results are Python objects, easy to inspect and use.

See the :doc:`examples` for more!

Benchmark
===========

This library is built as fast, scalable and memory-efficient as possible. Comparing with Goatools, which is the de facto library for manipulating GO in Python

We compare the time and peak memory consumption of go3 vs goatools while loading the ontology and the annotation (.GAF) file, and building the TermCounter.

.. image:: ../../src/benchmark_loading_time_memory.png
  :width: 600
  :alt: Benchmark gene batch similarity

We also compare the speed of the libraries calculating the similarities between batches of GO Terms of different sizes.

.. image:: ../../src/benchmark_batch_similarity.png
  :width: 600
  :alt: Benchmark batch similarity

Finally, we compare the gene similarity calculation times. Goatools does not implement natively the groupwise algorithms to compare genes, so we built it for a fair comparison in top of the GO term semantic similarities of the library. 

.. image:: ../../src/benchmark_gene_batch_similarity.png
  :width: 600
  :alt: Benchmark gene batch similarity

