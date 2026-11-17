Getting Started & Core Concepts
===============================

Requirements & Installation
---------------------------

- **Python >= 3.9** (pre-built wheels available for Linux, macOS, and Windows).
- An ontology file in OBO format (``go-basic.obo`` or ``go-basic.obo.gz``). Calling ``go3.load_go_terms()`` with no arguments downloads ``go-basic.obo`` automatically.
- An organism annotation file in GAF 2.2 format (plain ``.gaf`` or compressed ``.gaf.gz``).

.. code-block:: bash

   pip install go3
   pip install 'go3[viz]'  # Optional: t-SNE, UMAP, and matplotlib helpers

The 3-Step GO3 Pipeline
-----------------------

Every GO3 analysis starts with three steps executed once per Python session:

.. code-block:: python

   import go3

   # Step 1: Load the ontology DAG into memory
   go3.load_go_terms("go-basic.obo")

   # Step 2: Load GAF 2.2 annotations (optionally filtered via GAFParseOptions)
   annots = go3.load_gaf("goa_human.gaf.gz")

   # Step 3: Propagate counts up the DAG and compute Information Content (IC)
   counter = go3.build_term_counter(annots)

1. **Ontology graph (``load_go_terms``)**: Parses the OBO file and caches terms, ``is_a``/``part_of`` relationships, depths, and ancestor sets in process memory.
2. **Gene annotations (``load_gaf``)**: Maps gene symbols (``db_object_symbol``) to GO terms and returns ``GAFAnnotation`` objects. By default, ``ND`` and ``NOT`` annotations are excluded; custom filters (such as ``exclude_evidence={"IEA"}``) can be passed via :class:`go3.GAFParseOptions`.
3. **Term counter (``build_term_counter``)**: Propagates annotation counts to ancestor terms (*true-path rule*) and computes Information Content (IC) for each term.

Core Concepts
-------------

Namespaces (Sub-ontologies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gene Ontology is partitioned into three independent namespaces:

- ``"BP"`` — **Biological Process** (e.g., *mitophagy*, *mRNA processing*)
- ``"MF"`` — **Molecular Function** (e.g., *ATP binding*, *protein kinase activity*)
- ``"CC"`` — **Cellular Component** (e.g., *mitochondrion*, *spliceosomal complex*)

Terms from different namespaces do not share ancestors and always have similarity ``0.0``.

Information Content (IC) & MICA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Information Content quantifies how specific a GO term *t* is within its namespace:

.. math::

   IC(t) = -\log\!\left(\frac{\text{count}(t)}{\text{total}(\text{namespace})}\right)

Broad terms near the root have many propagated annotations and low IC (~0), whereas specific leaf terms have few annotations and high IC.

The **Most Informative Common Ancestor (MICA)** of two terms :math:`t_1, t_2` is their shared ancestor with the highest IC. Most IC-based similarity metrics (Resnik, Lin, Jiang-Conrath, SimRel) are functions of :math:`IC(\text{MICA}(t_1, t_2))`, :math:`IC(t_1)`, and :math:`IC(t_2)`.

Comparing Genes vs. Gene Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Single gene pair (``compare_genes``)**: Looks up the GO terms of each gene in the chosen namespace (``BP``, ``MF``, or ``CC``) and aggregates pairwise term similarities using a groupwise strategy (``bma``, ``simgic``, ``max``, ``avg``, or ``hausdorff``).
- **Gene lists (``compare_gene_sets`` vs. ``compare_gene_set_profiles``)**:
  - ``compare_gene_sets`` computes the gene-by-gene similarity matrix between the two lists and aggregates across genes (``bma``, ``max``, ``avg``, ``hausdorff``).
  - ``compare_gene_set_profiles`` merges each gene list into a weighted GO-term profile and compares the two profiles directly (supports weighted ``simgic``).
