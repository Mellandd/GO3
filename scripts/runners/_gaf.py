"""GAF parsing helpers shared by runners.

Avoids depending on goatools' GafReader for runners that don't need it,
and centralises the namespace + UniProt remapping logic.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

NAMESPACE_TO_FULL = {
    "BP": "biological_process",
    "MF": "molecular_function",
    "CC": "cellular_component",
}
NAMESPACE_TO_ASPECT = {"BP": "P", "MF": "F", "CC": "C"}


def iter_gaf_rows(gaf_path: Path) -> Iterable[list[str]]:
    """Yield non-comment, non-NOT, non-ND rows from a GAF file."""
    with open(gaf_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            qualifier = cols[3]
            evidence = cols[6]
            if evidence == "ND" or "NOT" in qualifier:
                continue
            yield cols


def parse_symbol_to_terms(
    gaf_path: Path,
    *,
    namespace_aspect: str | None = None,
    keep_obsolete: bool = False,
    obsolete_filter: set[str] | None = None,
) -> dict[str, list[str]]:
    """Map gene symbol (GAF col 3) to its GO terms.

    `namespace_aspect` is one of {"P", "F", "C"} to filter to a sub-ontology;
    pass None to keep everything.
    """
    gene2gos: dict[str, set[str]] = defaultdict(set)
    for cols in iter_gaf_rows(gaf_path):
        aspect = cols[8].strip()
        if namespace_aspect is not None and aspect != namespace_aspect:
            continue
        gene_symbol = cols[2].strip()
        go_id = cols[4].strip()
        if not gene_symbol or not go_id:
            continue
        if obsolete_filter is not None and go_id in obsolete_filter:
            continue
        gene2gos[gene_symbol].add(go_id)
    return {gene: sorted(terms) for gene, terms in gene2gos.items() if terms}


def parse_uniprot_to_symbol(gaf_path: Path) -> dict[str, str]:
    """Map UniProt accession (col 2) to gene symbol (col 3)."""
    out: dict[str, str] = {}
    for cols in iter_gaf_rows(gaf_path):
        accession = cols[1].strip()
        symbol = cols[2].strip()
        if accession and symbol and accession not in out:
            out[accession] = symbol
    return out


def parse_symbol_to_uniprot(gaf_path: Path) -> dict[str, list[str]]:
    """Map gene symbol -> list of UniProt accessions seen for it."""
    out: dict[str, set[str]] = defaultdict(set)
    for cols in iter_gaf_rows(gaf_path):
        accession = cols[1].strip()
        symbol = cols[2].strip()
        if accession and symbol:
            out[symbol].add(accession)
    return {sym: sorted(accs) for sym, accs in out.items()}
