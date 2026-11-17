=================
:math:`GO_3`
=================

**GO3** is a high-performance Python library with a compiled **Rust core** for Gene Ontology (GO) semantic similarity, GAF 2.2 annotation filtering, semantic redundancy reduction, and Over-Representation Analysis (ORA).

Key Capabilities
----------------

- **8 Term Similarity Methods**: Resnik, Lin, Jiang-Conrath (`jc`), SimRel, ICCoef, GraphIC, Wang, and TopoICSim.
- **5 Groupwise Strategies**: Best-Match Average (`bma`), `simgic`, `max`, `avg`, and `hausdorff`.
- **Term, Gene, and Gene-Set Comparisons**: Compare individual terms, sets of terms, gene pairs, or entire gene lists in parallel across CPU cores.
- **GO Enrichment Analysis (ORA)**: Fast hypergeometric over-representation (`go_enrichment`) with true-path propagation and choice of **Benjamini-Hochberg (`bh`)**, **g:Profiler `g:SCS` (`scs`)**, or **Bonferroni (`bonferroni`)** correction.
- **Redundancy Reduction & Clustering**: Condense redundant GO terms into representative clusters (`reduce_redundant_terms`, `cluster_go_terms`).
- **Native GAF 2.2 Filtering & `.gz` Support**: Parse `.obo`/`.obo.gz` and `.gaf`/`.gaf.gz` directly with structured filters (`GAFParseOptions`).
- **Distances & Embeddings**: All-vs-all distance matrices and built-in t-SNE / UMAP visualization helpers.

Installation
------------

.. code-block:: bash

   pip install go3

   # Optional visualization dependencies (scikit-learn, umap-learn, matplotlib):
   pip install 'go3[viz]'

Quick Start
-----------

.. code-block:: python

   import go3

   # 1. Load ontology (auto-downloads go-basic.obo if omitted) and GAF annotations
   go3.load_go_terms("go-basic.obo")
   annots = go3.load_gaf("goa_human.gaf.gz")
   counter = go3.build_term_counter(annots)

   # 2. Term and gene semantic similarity
   term_sim = go3.semantic_similarity("GO:0006397", "GO:0008380", "lin", counter)
   gene_sim = go3.compare_genes("TP53", "BRCA1", "BP", "lin", "bma", counter)

   # 3. GO enrichment with g:SCS or BH correction
   enriched = go3.go_enrichment(["PINK1", "PRKN", "PARK7", "SNCA", "LRRK2"], ontology="BP", method="scs")

Documentation Map
-----------------

- **Getting Started**: :doc:`introduction` · :doc:`examples` · :doc:`faq`
- **Method & Performance Guides**: :doc:`guide/choosing_methods` · :doc:`guide/visualization` · :doc:`guide/performance` · :doc:`guide/architecture` · :doc:`benchmarks`
- **API Reference**: :doc:`ontology` · :doc:`goterm` · :doc:`annotations` · :doc:`similarity` · :doc:`utilities`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   introduction
   examples
   faq

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Guides

   guide/choosing_methods
   guide/visualization
   guide/performance
   guide/architecture
   benchmarks

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   ontology
   goterm
   annotations
   similarity
   utilities
