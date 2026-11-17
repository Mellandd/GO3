Annotations and IC
==================

GO3 parses GAF annotations and builds term-level statistics for IC-based similarity methods.

Functions
---------

``load_gaf(path, options=None)``

- Parses a GAF 2.2 file (plain ``.gaf`` or gzip-compressed ``.gaf.gz``) and caches gene-to-GO mappings.
- Optionally accepts a :class:`go3.GAFParseOptions` instance to filter annotations during parsing.
- Returns a list of :class:`go3.GAFAnnotation` objects with all 17 GAF 2.2 fields.

``build_term_counter(annotations)``

- Builds a ``TermCounter`` from parsed annotations.
- Computes counts and Information Content (IC) by namespace.

Information Content formula
---------------------------

IC quantifies how specific a term is within its namespace:

.. math::

   IC(t) = -\log\!\left(\frac{\text{count}(t)}{\text{total}(\text{namespace})}\right)

- ``count(t)`` is the number of annotations for term *t*, **including annotations propagated from descendant terms**. If a gene is annotated to a child term, all of its ancestors also receive that count.
- ``total(namespace)`` is the total annotation count for the namespace (the value stored in ``counter.total_by_ns``).

The ``total_by_ns`` dictionary maps each namespace (``biological_process``, ``molecular_function``, ``cellular_component``) to its total propagated annotation count. This is the denominator in the IC formula.

Filtering rules in ``load_gaf`` and ``GAFParseOptions``
-------------------------------------------------------

During parsing, GO3 applies biological filters that can be customized via :class:`go3.GAFParseOptions`:

- **Evidence codes (``include_evidence``, ``exclude_evidence``)**: include only a specific set of evidence codes or exclude unwanted codes (e.g. ``exclude_evidence={"IEA"}``).
- **Aspects (``include_aspects``)**: restrict parsing to selected GO aspects (``{"P", "F", "C"}`` or ``{"BP", "MF", "CC"}``).
- **Qualifiers (``exclude_qualifiers``)**: defaults to ``{"NOT"}`` so negated annotations are skipped. Pass ``exclude_qualifiers=set()`` to keep negated annotations (which will have ``ann.is_negated == True``).
- **ND (No biological Data) & ``allow_nd_root_only``**: by default (``allow_nd_root_only=False``), all ``ND`` annotations are skipped. When ``allow_nd_root_only=True``, ``ND`` annotations on the root terms (``GO:0008150``, ``GO:0003674``, ``GO:0005575``) are retained while non-root ``ND`` annotations are excluded.
- **Obsolete GO terms (``allow_obsolete_terms``)**: by default (``allow_obsolete_terms=False``), obsolete terms are skipped. When ``allow_obsolete_terms=True``, obsolete terms are automatically remapped via ``replaced_by`` or ``consider`` targets when available.
- **Duplicate annotations (``remove_duplicates``)**: GAF files frequently annotate the same gene to the same GO term several times, with different evidence codes or references (about 60% of the rows in ``goa_human.gaf``). By default (``remove_duplicates=False``) all of them are kept. When ``remove_duplicates=True``, only the first annotation of each (gene, GO term) pair is returned and stored in the gene-to-GO mapping. The pair is checked after obsolete-term remapping, so two IDs that resolve to the same term also count as duplicates.

.. code-block:: python

   opts = go3.GAFParseOptions(
       include_evidence=None,
       exclude_evidence={"IEA"},
       include_aspects={"P", "F", "C"},
       exclude_qualifiers={"NOT"},
       allow_obsolete_terms=False,
       allow_nd_root_only=True,
       remove_duplicates=True,
   )
   annotations = go3.load_gaf("goa_human.gaf.gz", options=opts)

These rules affect both downstream similarity scores and Information Content (IC) deterministically.

Duplicates do not change IC, because ``build_term_counter`` counts each gene once per term. They do change gene-level similarity: a term repeated three times in a gene's annotation list counts three times in ``bma`` or ``avg`` aggregation.

Filtering annotations by Information Content
--------------------------------------------

``filter_annotations_by_ic(counter, ic_threshold)``

- Keeps a gene's GO term only if ``IC(term) > ic_threshold``. This removes very generic terms (roots, ``binding``, ``cellular process``, ...) that carry little information and make unrelated genes look similar.
- Changes the gene-to-GO mapping loaded by ``load_gaf``, so every gene-level function uses the filtered annotations afterwards: ``compare_genes``, ``compare_gene_sets``, ``gene_distance_matrix``, ``go_enrichment``, ``get_terms_for_gene``, ``list_genes``, embeddings, etc.
- Genes left with no annotations are removed from the mapping and listed in the result.
- Always starts from the mapping produced by ``load_gaf``: calling it again with another threshold replaces the previous filter instead of stacking on it, and ``ic_threshold=None`` restores the unfiltered mapping. Loading a new GAF also resets it.
- Does not change the ``TermCounter``. IC needs a ``TermCounter``, which is built after parsing, so this filter is a separate step rather than a ``GAFParseOptions`` field. Build the counter from the full set of annotations so IC reflects the whole corpus.
- Returns a dictionary with ``annotations_before``, ``annotations_after``, ``genes_before``, ``genes_after`` and ``removed_genes``.

.. code-block:: python

   annotations = go3.load_gaf("goa_human.gaf.gz", go3.GAFParseOptions(remove_duplicates=True))
   counter = go3.build_term_counter(annotations)

   summary = go3.filter_annotations_by_ic(counter, ic_threshold=2.0)
   print(summary["annotations_before"], "->", summary["annotations_after"])
   print(len(summary["removed_genes"]), "genes lost all their annotations")

   score = go3.compare_genes("TP53", "MDM2", "BP", "lin", "bma", counter)

   go3.filter_annotations_by_ic(counter, ic_threshold=None)  # back to all annotations

``ic_threshold=0.0`` removes only terms with IC 0, i.e. terms annotated (directly or by propagation) to every gene in their namespace, such as the root terms. Inspect ``counter.ic`` to choose a threshold for your data.

Annotation propagation
----------------------

When ``build_term_counter`` computes counts, each annotation is propagated up the DAG: if a gene is annotated to term *t*, then *t* and all ancestors of *t* receive a count increment. This means that general (high-level) terms accumulate many counts and have low IC, while specific (leaf-level) terms have fewer counts and high IC.

Common evidence codes
---------------------

GAF files record **evidence codes** that describe how an annotation was determined. GO3 does not filter by evidence code (other than ``ND``), but knowing what they mean helps interpret annotation quality:

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Code
     - Name
     - Description
   * - ``IEA``
     - Inferred from Electronic Annotation
     - Computationally assigned, not curator-reviewed; most abundant code
   * - ``EXP``
     - Inferred from Experiment
     - Supported by a published experiment
   * - ``IDA``
     - Inferred from Direct Assay
     - Based on a direct experimental assay
   * - ``IPI``
     - Inferred from Physical Interaction
     - Based on a physical interaction experiment
   * - ``IMP``
     - Inferred from Mutant Phenotype
     - Based on observed mutant phenotype
   * - ``IGI``
     - Inferred from Genetic Interaction
     - Based on a genetic interaction experiment
   * - ``IEP``
     - Inferred from Expression Pattern
     - Based on gene expression data
   * - ``TAS``
     - Traceable Author Statement
     - Directly cited from a published paper
   * - ``NAS``
     - Non-traceable Author Statement
     - Author statement without a specific citation
   * - ``ND``
     - No biological Data
     - Placeholder when no annotation exists; **filtered out by GO3**

For stricter filtering (e.g., excluding ``IEA`` to use only curated annotations), pass ``GAFParseOptions(exclude_evidence={"IEA"})`` directly to ``load_gaf``.

Example
-------

.. code-block:: python

   import go3

   go3.load_go_terms("go-basic.obo")
   annotations = go3.load_gaf("goa_human.gaf.gz")
   counter = go3.build_term_counter(annotations)

   print("Annotations:", len(annotations))
   print("IC terms:", len(counter.ic))

Inspecting structures
---------------------

.. code-block:: python

   ann = annotations[0]
   print(ann.db, ann.db_object_id, ann.db_object_symbol, ann.go_id, ann.evidence)
   print(ann.qualifiers, ann.references, ann.taxon, ann.date, ann.assigned_by)
   print(ann.annotation_extensions, ann.gene_product_form_id, ann.is_negated)

   # Raw annotation count for a term (after propagation)
   print(counter.counts.get("GO:0008150", 0))

   # Total annotations per namespace (denominator for IC)
   print(counter.total_by_ns)

   # IC value for a specific term
   print(counter.ic.get("GO:0008150", 0.0))

Class reference
---------------

``GAFAnnotation`` fields (full GAF 2.2 model):

- ``db`` -- database source (e.g. ``"UniProtKB"``)
- ``db_object_id`` -- database identifier of the gene product
- ``db_object_symbol`` -- gene symbol (e.g. ``"TP53"``)
- ``qualifiers`` -- list of qualifier strings (e.g. ``["enables"]``)
- ``go_id`` (and alias ``go_term``) -- GO term identifier
- ``references`` -- list of supporting reference IDs
- ``evidence`` -- evidence code (e.g. ``"IDA"``, ``"IEA"``)
- ``with_from`` -- list of with/from identifiers
- ``aspect`` -- ``"P"``, ``"F"``, or ``"C"``
- ``db_object_name`` -- full name of the gene product
- ``db_object_synonyms`` -- list of gene product synonyms
- ``db_object_type`` -- product type (e.g. ``"protein"``)
- ``taxon`` -- list of taxon IDs (e.g. ``["taxon:9606"]``)
- ``date`` -- annotation date (``YYYYMMDD``)
- ``assigned_by`` -- assigning authority
- ``annotation_extensions`` -- list of cross-link extensions
- ``gene_product_form_id`` -- isoform or modified form ID
- ``is_negated`` -- ``True`` if ``"NOT"`` is present in ``qualifiers``

``GAFParseOptions`` fields:

- ``include_evidence`` -- optional set of evidence codes to include
- ``exclude_evidence`` -- set of evidence codes to exclude
- ``include_aspects`` -- optional set of aspects (``{"P", "F", "C"}``)
- ``exclude_qualifiers`` -- set of qualifiers to exclude (default ``{"NOT"}``)
- ``allow_obsolete_terms`` -- whether to allow and remap obsolete GO terms (default ``False``)
- ``allow_nd_root_only`` -- whether to allow ``ND`` annotations on root GO terms (default ``False``)
- ``remove_duplicates`` -- keep only the first annotation of each (gene, GO term) pair (default ``False``)

``TermCounter`` fields:

- ``counts`` -- per-term annotation count (after ancestor propagation)
- ``total_by_ns`` -- total annotation count per namespace (used as IC denominator)
- ``ic`` -- per-term Information Content values

API reference
-------------

.. automodule:: go3
   :members: load_gaf, build_term_counter, filter_annotations_by_ic, GAFAnnotation, GAFParseOptions, TermCounter
   :undoc-members:
   :show-inheritance:
   :no-index:
