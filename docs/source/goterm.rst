GOTerm Object
=============

A ``PyGOTerm`` represents one term in the loaded Gene Ontology.

Main fields
-----------

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``id``
     - ``str``
     - GO identifier (for example ``GO:0006397``)
   * - ``name``
     - ``str``
     - Human-readable label
   * - ``namespace``
     - ``str``
     - One of ``biological_process``, ``molecular_function``, ``cellular_component``
   * - ``definition``
     - ``str``
     - GO definition text
   * - ``parents``
     - ``list[str]``
     - Immediate ``is_a`` parents
   * - ``children``
     - ``list[str]``
     - Immediate ``is_a`` children
   * - ``depth``
     - ``int | None``
     - Maximum distance to root
   * - ``level``
     - ``int | None``
     - Minimum distance to root
   * - ``is_obsolete``
     - ``bool``
     - Obsolescence flag
   * - ``alt_ids``
     - ``list[str]``
     - Alternate GO IDs
   * - ``replaced_by``
     - ``str | None``
     - Suggested canonical replacement for obsolete terms
   * - ``consider``
     - ``list[str]``
     - Alternative replacements
   * - ``synonyms``
     - ``list[str]``
     - Synonym strings
   * - ``xrefs``
     - ``list[str]``
     - Cross-references
   * - ``relationships``
     - ``list[tuple[str, str]]``
     - Additional relationships (for example ``part_of``)
   * - ``comment``
     - ``str | None``
     - GO comment field

Example
-------

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")
   term = go3.get_term_by_id("GO:0006397")

   print(term.id)
   print(term.name)
   print(term.namespace)
   print(term.parents[:5])

API reference
-------------

.. autoclass:: go3.PyGOTerm
   :members:
   :undoc-members:
   :show-inheritance:
