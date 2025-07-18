GOTerm
======

A `GOTerm` object represents a single term in the Gene Ontology.

**Fields:**

=================  ===================  ===========================================================
Field              Type                  Description
=================  ===================  ===========================================================
id                 str                   GO term identifier (e.g., ``GO:0006397``)
name               str                   Human-readable name (e.g., ``mRNA processing``)
namespace          str                   Ontology namespace (``biological_process``, ``molecular_function``, or ``cellular_component``)
definition         str                   Textual definition of the term
parents            list[str]             List of parent GO term IDs (is_a relationships)
children           list[str]             List of child GO term IDs (is_a relationships)
depth              int or None           Maximum distance to a root term (None if not computed)
level              int or None           Minimum distance to a root term (None if not computed)
is_obsolete        bool                  True if the term is obsolete
alt_ids            list[str]             Alternative GO IDs for this term
replaced_by        str or None           If obsolete, the term that replaces this one
consider           list[str]             Suggested replacement terms if obsolete
synonyms           list[str]             List of synonyms
xrefs              list[str]             Cross-references to other databases
relationships      list[(str, str)]      Other relationships (e.g., ``part_of``)
comment            str or None           Additional comments
=================  ===================  ===========================================================


.. autoclass:: go3.PyGOTerm
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   import go3
   go3.load_go_terms()
   term = go3.get_term_by_id("GO:0006397")
   print(term.name)
   print(term.parents)