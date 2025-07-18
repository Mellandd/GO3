Installation
===============

**go3** is provided as binary wheels for most platforms on PyPI (Linux, Windows and MacOS). You can run

.. code-block:: bash

   pip install go3

Initializating the Ontology
============================

**go3** does not ship with any prebuilt GO Ontology by default. If you don't provide any .obo, when you try to load the ontology into memory it automatically downloads the last version of go-basic.obo.

.. code-block:: python

   import go3

   # Initialize the ontology. Downloads the last go-basic.obo version of the oficial website
   go3.load_go_terms()

   # Load an specific GO Term
   term_1 = go3.get_term_by_id("GO:0006397")

   print(term_1.name)
   #> mRNA processing

Instead, you can pass to the ``load_go_term`` function the path to any .obo version of the ontology that you already have downloaded.

Initializating the annotations
===============================

In the Gene Ontology, the annotations comes in the GO Association File (GAF) format.

A standard GO annotation is a statement that links a gene product and a GO term via a relation from the Relations Ontology (RO). It minimally contains:

- a gene product: may be a protein, an miRNA, a tRNA, etc.
- a GO term
- a reference, usually a PMID, but DOIs and GO Reference (GO_REF) are also use
- an evidence code, using a GO Evidence Code, which describes the type of evidence: experimental evidence, sequence similarity or phylogenetic relation, as well as whether the evidence was reviewed by an expert biocurator. If not manually reviewed, the annotation is described as ‘automated’.

The Gene Ontology contains annotations for almost any organism. Depending on your choice, you must download the corresponding GAF file from the oficial website to use it in this library:
`Website to download the annotations <https://current.geneontology.org/products/pages/downloads.html>`_

.. code-block:: python
    
    import go3
    # Initialize the ontology
    go3.load_go_terms()

    # Initialize the annotations from the downloaded gaf file.
    annots = go3.load_gaf("goa_human.gaf")

    # Build the Term Counter that counts the annotations for every GO Term. 
    # Needed to calculate the Information Content (IC) of a GO Term, and to calculate similarities 
    # using Resnik, Lin or any other measure between the terms.
    term_counts = go3.build_term_counter(annots)