Ontology
========

.. automodule:: go3
   :members: load_go_terms, get_term_by_id, ancestors, common_ancestor, deepest_common_ancestor
   :undoc-members:
   :show-inheritance:

Ontology Functions
------------------

.. code-block:: python

   import go3
   go3.load_go_terms()
   term = go3.get_term_by_id("GO:0006397")
   print(term.name)

- **load_go_terms**: Loads the GO ontology from an OBO file (downloads if not present).
- **get_term_by_id**: Retrieve a GO term by its ID.
- **ancestors**: List all ancestors of a GO term.
- **common_ancestor**: List all common ancestors of two terms.
- **deepest_common_ancestor**: Find the deepest common ancestor (MICA) of two terms.