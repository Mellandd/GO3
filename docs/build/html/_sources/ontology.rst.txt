Ontology API
============

Ontology loading and traversal functions.

Core functions
--------------

``load_go_terms(path=None)``

- Loads GO terms from an OBO file.
- If ``path`` is omitted, GO3 downloads ``go-basic.obo``.
- Replaces the in-process ontology cache.

``get_term_by_id(go_id)``

- Returns a :class:`go3.PyGOTerm` object.
- Raises ``ValueError`` if the term does not exist.

``ancestors(go_id)``

- Returns all ancestors of ``go_id`` (including the term itself).

``common_ancestor(go_id1, go_id2)``

- Returns common ancestors between two terms.

``deepest_common_ancestor(go_id1, go_id2)``

- Returns the deepest common ancestor (MICA-like depth criterion).
- Raises ``ValueError`` if either GO ID is missing.

Example
-------

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")

   term = go3.get_term_by_id("GO:0006397")
   print(term.name)

   ancs = go3.ancestors("GO:0006397")
   print(len(ancs))

   common = go3.common_ancestor("GO:0006397", "GO:0008380")
   print(len(common))

   dca = go3.deepest_common_ancestor("GO:0006397", "GO:0008380")
   print(dca)

Notes for developers
--------------------

- Ontology data is cached globally within the Python process.
- Loading a new ontology refreshes dependent internal caches.
- For reproducible analyses, keep ontology version fixed across runs.

API reference
-------------

.. automodule:: go3
   :members: load_go_terms, get_term_by_id, ancestors, common_ancestor, deepest_common_ancestor
   :undoc-members:
   :show-inheritance:
   :no-index:
