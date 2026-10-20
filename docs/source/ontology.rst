Ontology API
============

Ontology loading and traversal functions.

Core functions
--------------

``load_go_terms(path=None)``

- Loads GO terms from an OBO file (plain ``.obo`` or compressed ``.obo.gz``).
- If ``path`` is omitted, GO3 downloads ``go-basic.obo``.
- Replaces the in-process ontology cache.

``get_term_by_id(go_id)``

- Returns a :class:`go3.PyGOTerm` object.
- Raises ``ValueError`` if the term does not exist.

``ancestors(go_id, relations=None)``

- Returns all ancestors of ``go_id`` (including the term itself).
- By default (``relations=None``), traversal follows ``is_a`` relationships using the precomputed cache.
- Pass additional relationship types (e.g. ``relations=["part_of"]``) to also traverse those edges.

.. note::

   The returned set **includes the query term itself**. For example,
   ``ancestors("GO:0006397")`` will contain ``"GO:0006397"`` plus all
   its ``is_a`` parents up to the root.

``descendants(go_id, relations=None)``

- Returns a sorted list of all descendant GO term IDs of ``go_id`` (including the term itself).
- Optionally traverses additional relationship types such as ``relations=["part_of"]``.
- Raises ``ValueError`` if ``go_id`` is not in the ontology.

``common_ancestor(go_id1, go_id2)``

- Returns common ancestors between two terms.

``deepest_common_ancestor(go_id1, go_id2)``

- Returns the deepest common ancestor (MICA-like depth criterion).
- Raises ``ValueError`` if either GO ID is missing.

``list_genes(ontology=None)``

- Returns a sorted list of all gene symbols loaded from the GAF file.
- Optionally filters to genes with annotations in ``"BP"``, ``"MF"``, or ``"CC"``.

``get_terms_for_gene(gene, ontology=None)``

- Returns the unique sorted GO term IDs annotated to ``gene`` (optionally filtered by subontology).
- Raises ``ValueError`` if ``gene`` is not found in the loaded mapping.

``get_genes_for_term(go_id, include_descendants=True)``

- Returns a sorted list of gene symbols annotated to ``go_id``.
- When ``include_descendants=True`` (default), includes genes annotated to any descendant of ``go_id`` following the GO true-path rule.

Depth vs. level
---------------

GO terms have two distance-to-root metrics:

- **depth** -- the *maximum* distance (longest path) from the term to the root of its namespace. A higher depth means the term is located deeper in the DAG along at least one path.
- **level** -- the *minimum* distance (shortest path) from the term to the root. This is always less than or equal to depth.

The distinction matters because the GO graph is a DAG, not a tree: a term can have multiple parents and therefore multiple paths to the root.

Obsolete terms
--------------

GO3 loads obsolete terms from the OBO file but flags them with ``is_obsolete = True``.
When loading annotations via ``load_gaf``, GO3 automatically resolves obsolete terms:

1. If ``replaced_by`` is set, the annotation is remapped to the replacement term.
2. Otherwise, if ``consider`` targets exist, the first one is used.
3. If neither is available, the annotation is discarded.

You can inspect obsolete terms directly:

.. code-block:: python

   term = go3.get_term_by_id("GO:0005765")  # example obsolete term
   if term.is_obsolete:
       print("Obsolete! Replaced by:", term.replaced_by)
       print("Consider:", term.consider)

Example
-------

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")

   term = go3.get_term_by_id("GO:0006397")
   print(term.name)

   ancs = go3.ancestors("GO:0006397")
   print(len(ancs), "ancestors (including self)")

   common = go3.common_ancestor("GO:0006397", "GO:0008380")
   print(len(common), "common ancestors")

   dca = go3.deepest_common_ancestor("GO:0006397", "GO:0008380")
   print("Deepest common ancestor:", dca)

Notes for developers
--------------------

- Ontology data is cached globally within the Python process.
- Loading a new ontology refreshes dependent internal caches.
- For reproducible analyses, keep ontology version fixed across runs.

API reference
-------------

.. automodule:: go3
   :members: load_go_terms, get_term_by_id, ancestors, descendants, common_ancestor, deepest_common_ancestor, list_genes, get_terms_for_gene, get_genes_for_term
   :undoc-members:
   :show-inheritance:
   :no-index:
