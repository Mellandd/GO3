Annotations
===========

.. automodule:: go3
   :members: load_gaf, build_term_counter, GAFAnnotation, TermCounter
   :undoc-members:
   :show-inheritance:

Annotation Functions
--------------------

.. code-block:: python

   import go3
   go3.load_go_terms()
   annots = go3.load_gaf("goa_human.gaf")
   counter = go3.build_term_counter(annots)

- **load_gaf**: Loads a GAF file and parses gene-to-GO annotations.
- **build_term_counter**: Builds annotation counts and IC values for all terms.
- **GAFAnnotation**: Class representing a single annotation.
- **TermCounter**: Class holding counts and IC values.