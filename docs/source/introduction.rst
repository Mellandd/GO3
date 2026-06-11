Installation
============

GO3 provides Python bindings for a Rust implementation of Gene Ontology semantic similarity.

Install from PyPI:

.. code-block:: bash

   pip install go3

Optional visualization dependencies:

.. code-block:: bash

   pip install 'go3[viz]'

Quote the extras spec to avoid shell glob expansion in shells such as zsh.

Requirements
============

- **Python >= 3.7** (pre-built wheels are available for Linux, macOS, and Windows on common architectures).
- A GO ontology file in OBO format (for example ``go-basic.obo``).
- A GO annotation file in GAF format for your organism.

If you call ``go3.load_go_terms()`` without a path, GO3 downloads ``go-basic.obo`` automatically.

Minimal workflow
================

.. code-block:: python

   import go3

   # 1) Load the ontology graph into memory
   go3.load_go_terms("go-basic.obo")

   # 2) Parse annotations and build a gene-to-GO mapping
   annots = go3.load_gaf("goa_human.gaf")

   # 3) Compute annotation counts and IC values for every term
   counter = go3.build_term_counter(annots)

   # 4) Compute term-to-term similarity
   sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "lin", counter)
   print(sim)

   # 5) Compute gene-to-gene similarity (using BP namespace)
   score = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)
   print(score)

   # 6) Compare gene sets directly, e.g. enrichment or pathway gene lists.
   # GO3 computes pairwise gene similarities first, then aggregates across genes.
   gene_set_score = go3.compare_gene_sets(
       ["TP53", "BRCA1", "ATM"],
       ["CASP8", "GSDME", "NLRP1"],
       "BP",
       "lin",
       "bma",
       counter,
   )
   print(gene_set_score)

Core concepts
=============

The typical GO3 pipeline has three stages:

1. **Load the ontology** -- ``load_go_terms`` parses an OBO file and caches the full GO directed acyclic graph (DAG) in memory, including parent/child edges, depths, and ancestor sets.
2. **Load annotations** -- ``load_gaf`` parses a GAF file to build a mapping from genes (``db_object_symbol``) to their annotated GO terms. Obsolete terms are automatically remapped via ``replaced_by`` or ``consider`` fields when possible.
3. **Build the counter** -- ``build_term_counter`` walks the annotations and computes per-term counts with ancestor propagation, then derives IC values for every term. The resulting ``TermCounter`` is passed to similarity functions.

Gene sets
---------

GO3 supports two gene-set workflows:

- ``compare_gene_sets`` compares all pairwise gene similarities between two gene lists, then applies a groupwise strategy across the gene-by-gene similarity matrix.
- ``compare_gene_set_profiles`` builds one weighted GO-term profile per gene set and compares those profiles directly. This is the profile-based alternative and supports weighted ``simgic``.

Information Content (IC)
------------------------

Information Content quantifies how specific a GO term is within its namespace.  It is defined as:

.. math::

   IC(t) = -\log\!\left(\frac{\text{count}(t)}{\text{total}(\text{namespace})}\right)

where ``count(t)`` is the number of annotations for term *t* (including propagated annotations from descendant terms) and ``total(namespace)`` is the total annotation count for that namespace.

Intuitively, **rare terms have high IC** (they are more informative), while the root term -- which is an ancestor of every term -- has IC close to zero.

MICA (Most Informative Common Ancestor)
----------------------------------------

Many IC-based methods compare two terms by finding their **Most Informative Common Ancestor (MICA)**: the common ancestor with the highest IC.  Since IC grows with specificity, the MICA is the most specific term that subsumes both query terms.

For example, the Resnik similarity of two terms is simply ``IC(MICA)``, and Lin similarity normalizes it by the sum of individual ICs.

Namespaces
==========

GO3 uses standard GO sub-ontologies:

- ``BP``: Biological Process
- ``MF``: Molecular Function
- ``CC``: Cellular Component

For gene-level APIs, select namespace explicitly via the ``ontology`` argument.

Troubleshooting
===============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Problem
     - Solution
   * - ``FileNotFoundError`` when loading OBO/GAF
     - Verify the file path is correct. Omit the path in ``load_go_terms()`` to auto-download ``go-basic.obo``.
   * - ``ValueError``: gene not found
     - Gene names must match the ``db_object_symbol`` column in your GAF file exactly (case-sensitive). Check your GAF for the correct symbol.
   * - Similarity is always 0.0
     - Terms may be in different namespaces, one or both IDs may be invalid, or the terms may have no common ancestor. Verify both terms belong to the same namespace.
   * - Wrong namespace in ``compare_genes``
     - The ``ontology`` argument (``BP``, ``MF``, ``CC``) filters which annotations are used. If a gene has no annotations in the chosen namespace, the result will be 0.0.
   * - ``simgic`` in ``compare_gene_sets``
     - Direct gene-set comparison aggregates pairwise gene similarities and supports ``bma``, ``max``, ``avg``, and ``hausdorff`` as the outer strategy. Use ``compare_gene_set_profiles(..., groupwise="simgic")`` for IC-weighted GO-profile overlap.
   * - Which OBO file to use?
     - Use ``go-basic.obo`` (recommended). The full ``go.obo`` includes cross-ontology links that may produce unexpected results.

Next steps
==========

- :doc:`examples` for end-to-end usage patterns
- :doc:`similarity` for available methods and formulas
- :doc:`guide/choosing_methods` for picking the right method
- :doc:`guide/performance` for throughput-oriented workflows
- :doc:`benchmarks` for reproducible comparisons
