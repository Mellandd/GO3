"""pygosemsim runner.

pygosemsim's `from_resource` API only reads files from its packaged
`_resources/` folder. We work around it by:

  * temporarily placing symlinks (or copies) of our OBO and GAF into the
    package's resources directory, named so that
    `graph.from_resource("go-basic")` and `annotation.from_resource("goa_human")`
    pick them up;
  * relying on the GAF's UniProt accession column (col 2) to key the
    annotation dict, and pre-computing a symbol -> accession map so the
    benchmark can hand the runner gene symbols just like the others.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any

from ._base import (
    Runner,
    RunnerCaps,
    RunResult,
    loading_finalize,
    loading_step,
    make_point,
    python_module_available,
    register,
    time_runs,
)
from ._gaf import parse_symbol_to_uniprot
from ._subproc import run_python_runner_child


def _resource_dir() -> Path:
    spec = importlib.util.find_spec("pygosemsim")
    if spec is None or spec.origin is None:
        raise RuntimeError("pygosemsim is not importable")
    return Path(spec.origin).parent / "_resources"


def _stage_resources(obo: Path, gaf: Path) -> None:
    """Place the user's OBO/GAF in the spot pygosemsim expects."""
    resdir = _resource_dir()
    resdir.mkdir(parents=True, exist_ok=True)
    target_obo = resdir / "go-basic.obo"
    target_gaf = resdir / "goa_human.gaf"
    if not target_obo.exists() or target_obo.stat().st_size != obo.stat().st_size:
        shutil.copyfile(obo, target_obo)
    if not target_gaf.exists() or target_gaf.stat().st_size != gaf.stat().st_size:
        shutil.copyfile(gaf, target_gaf)


def _load_graph():
    from pygosemsim import graph

    G = graph.from_resource("go-basic")
    graph.precalc_lower_bounds(G)
    return G


def _load_annotation():
    from pygosemsim import annotation

    return annotation.from_resource("goa_human")


def _term_sim_func(method: str):
    from pygosemsim import similarity

    method = method.lower()
    if method == "resnik":
        return similarity.resnik
    if method == "lin":
        return similarity.lin
    if method == "wang":
        return similarity.wang
    raise ValueError(f"Unsupported method: {method}")


def _resolve_to_uniprot(
    symbol: str,
    annot: dict,
    sym2uni: dict[str, list[str]],
) -> str | None:
    """Pick a UniProt accession for `symbol` that exists in pygosemsim's annot."""
    for acc in sym2uni.get(symbol, []):
        if acc in annot:
            return acc
    return None


def _terms_for_symbol(symbol: str, annot: dict, sym2uni: dict[str, list[str]]) -> list[str]:
    acc = _resolve_to_uniprot(symbol, annot, sym2uni)
    if acc is None:
        return []
    entry = annot.get(acc)
    if entry is None:
        return []
    ann = entry.get("annotation", {})
    return list(ann.keys())


@register
class PygosemsimRunner(Runner):
    name = "pygosemsim"
    display_name = "pygosemsim"

    @classmethod
    def is_available(cls) -> bool:
        return python_module_available("pygosemsim")

    @classmethod
    def caps(cls) -> RunnerCaps:
        return RunnerCaps(
            loading=True,
            term_pair_methods={"resnik", "lin"},
            gene_pair_methods={"resnik", "lin"},
            notes="Pure Python; UniProt-keyed annotation, remapped from gene symbols via GAF.",
        )

    # ------------------------------------------------------------------
    @classmethod
    def loading_in_process(cls, obo: Path, gaf: Path, namespace: str) -> dict[str, Any]:
        _stage_resources(obo, gaf)

        graph_holder: list[Any] = [None]
        annot_holder: list[Any] = [None]

        def _load_obo() -> dict[str, Any]:
            graph_holder[0] = _load_graph()
            return {"n_nodes": graph_holder[0].number_of_nodes()}

        def _load_gaf() -> dict[str, Any]:
            annot_holder[0] = _load_annotation()
            return {"n_objects": len(annot_holder[0])}

        def _build_ic() -> dict[str, Any]:
            # Touch a similarity that uses IC so the lazy pre-compute runs
            # inside the loading region rather than the first timed call.
            from pygosemsim import similarity

            G = graph_holder[0]
            try:
                # Pick any two nodes that exist.
                nodes = list(G.nodes())
                if len(nodes) >= 2:
                    similarity.lin(G, nodes[0], nodes[1])
            except Exception:
                pass
            return {"primed": True}

        steps = [
            loading_step("Load ontology", _load_obo),
            loading_step("Load annotations", _load_gaf),
            loading_step("Prime IC", _build_ic),
        ]
        return loading_finalize(steps, cls.name)

    @classmethod
    def loading(cls, obo: Path, gaf: Path, namespace: str, *, python_executable: str | None = None) -> dict[str, Any]:
        return run_python_runner_child(
            runner=cls.name,
            task="loading",
            obo=obo,
            gaf=gaf,
            namespace=namespace,
            workdir=Path("/tmp/go3_bench") / cls.name,
            python_executable=python_executable,
        )

    # ------------------------------------------------------------------
    @classmethod
    def term_pairs(
        cls,
        *,
        obo: Path,
        gaf: Path,
        namespace: str,
        method: str,
        pairs_by_size: dict[int, list[tuple[str, str]]],
        warmup: int,
        repeats: int,
        threads: int | None = None,
        workdir: Path | None = None,
    ) -> dict[int, RunResult]:
        _stage_resources(obo, gaf)
        G = _load_graph()
        # Annotation needed for IC-based methods (resnik, lin)
        if method.lower() in {"resnik", "lin"}:
            _load_annotation()
        sim = _term_sim_func(method)

        out: dict[int, RunResult] = {}
        first_size = next(iter(pairs_by_size))
        if pairs_by_size[first_size]:
            a, b = pairs_by_size[first_size][0]
            try:
                sim(G, a, b)
            except Exception:
                pass

        for size in sorted(pairs_by_size, reverse=True):
            pairs = pairs_by_size[size]

            def _fn() -> None:
                for a, b in pairs:
                    try:
                        sim(G, a, b)
                    except Exception:
                        pass

            timing = time_runs(_fn, warmup=warmup, repeats=repeats)
            out[size] = make_point(size, timing, len(pairs))
        return out

    # ------------------------------------------------------------------
    @classmethod
    def gene_pairs(
        cls,
        *,
        obo: Path,
        gaf: Path,
        namespace: str,
        method: str,
        gene_pairs_by_size: dict[int, list[tuple[str, str]]],
        gene2terms: dict[str, list[str]],
        warmup: int,
        repeats: int,
        threads: int | None = None,
        workdir: Path | None = None,
    ) -> dict[int, RunResult]:
        from pygosemsim import term_set

        _stage_resources(obo, gaf)
        G = _load_graph()
        annot = _load_annotation()
        sim = _term_sim_func(method)

        # Symbol -> UniProt mapping from the GAF; we use pygosemsim's own
        # annotation (UniProt-keyed) as the source of truth for term lists,
        # so its IC and topology are self-consistent.
        sym2uni = parse_symbol_to_uniprot(gaf)
        sf = functools.partial(term_set.sim_func, G, sim)

        out: dict[int, RunResult] = {}
        first_size = next(iter(gene_pairs_by_size))
        if gene_pairs_by_size[first_size]:
            g1, g2 = gene_pairs_by_size[first_size][0]
            t1 = _terms_for_symbol(g1, annot, sym2uni)
            t2 = _terms_for_symbol(g2, annot, sym2uni)
            if t1 and t2:
                try:
                    term_set.sim_bma(t1, t2, sf)
                except Exception:
                    pass

        for size in sorted(gene_pairs_by_size, reverse=True):
            pairs = gene_pairs_by_size[size]

            def _fn() -> None:
                for g1, g2 in pairs:
                    t1 = _terms_for_symbol(g1, annot, sym2uni)
                    t2 = _terms_for_symbol(g2, annot, sym2uni)
                    if not t1 or not t2:
                        continue
                    try:
                        term_set.sim_bma(t1, t2, sf)
                    except Exception:
                        pass

            timing = time_runs(_fn, warmup=warmup, repeats=repeats)
            out[size] = make_point(size, timing, len(pairs))
        return out
