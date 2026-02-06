Annotations and IC
==================

GO3 parses GAF annotations and builds term-level statistics for IC-based similarity methods.

Functions
---------

``load_gaf(path)``

- Parses a GAF file and caches gene-to-GO mappings.
- Returns a list of ``GAFAnnotation`` objects.

``build_term_counter(annotations)``

- Builds a ``TermCounter`` from parsed annotations.
- Computes counts and Information Content (IC) by namespace.

Filtering rules in ``load_gaf``
-------------------------------

During parsing, GO3 applies key biological filters:

- skips annotations with evidence code ``ND``
- skips annotations with qualifier containing ``NOT``
- handles obsolete GO terms:

  - uses ``replaced_by`` when available
  - otherwise uses first ``consider`` target when available
  - otherwise discards that annotation

These rules affect both downstream scores and benchmark comparability.

Example
-------

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")
   annotations = go3.load_gaf("goa_human.gaf")
   counter = go3.build_term_counter(annotations)

   print("Annotations:", len(annotations))
   print("IC terms:", len(counter.ic))

Inspecting structures
---------------------

.. code-block:: python

   ann = annotations[0]
   print(ann.db_object_id, ann.go_term, ann.evidence)

   print(counter.counts.get("GO:0008150", 0))
   print(counter.total_by_ns)
   print(counter.ic.get("GO:0008150", 0.0))

Class reference
---------------

``GAFAnnotation`` fields:

- ``db_object_id``
- ``go_term``
- ``evidence``

``TermCounter`` fields:

- ``counts``
- ``total_by_ns``
- ``ic``

API reference
-------------

.. automodule:: go3
   :members: load_gaf, build_term_counter, GAFAnnotation, TermCounter
   :undoc-members:
   :show-inheritance:
   :no-index:
