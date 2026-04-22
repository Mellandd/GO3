"""fastsemsim runner.

fastsemsim ships Resnik / Lin / Jiang-Conrath / SimGIC but **no Wang** —
caps() reflects this. The orchestrator skips the Wang column for this lib.
"""

from __future__ import annotations

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
from ._gaf import NAMESPACE_TO_FULL
from ._subproc import run_python_runner_child


def _load_ontology(obo: Path):
    import fastsemsim

    return fastsemsim.load_ontology(
        source_file=str(obo),
        ontology_type="GeneOntology",
        file_type="obo",
    )


def _load_ac(ontology, gaf: Path):
    import fastsemsim

    return fastsemsim.load_ac(
        ontology,
        source_file=str(gaf),
        file_type="gaf-2.0",
        species="human",
    )


def _make_termsim(ontology, ac, util, method: str):
    method = method.lower()
    if method == "resnik":
        from fastsemsim.semsim import ResnikSemSim

        return ResnikSemSim(ontology, ac, util)
    if method == "lin":
        from fastsemsim.semsim import LinSemSim

        return LinSemSim(ontology, ac, util)
    if method == "wang":
        raise ValueError("fastsemsim does not implement Wang")
    raise ValueError(f"Unsupported method: {method}")


@register
class FastsemsimRunner(Runner):
    name = "fastsemsim"
    display_name = "fastsemsim"

    @classmethod
    def is_available(cls) -> bool:
        return python_module_available("fastsemsim")

    @classmethod
    def caps(cls) -> RunnerCaps:
        # No Wang.
        return RunnerCaps(
            loading=True,
            term_pair_methods={"resnik", "lin"},
            gene_pair_methods={"resnik", "lin"},
            notes="Wang not implemented; uses BMASemSim for groupwise BMA.",
        )

    # ------------------------------------------------------------------
    @classmethod
    def loading_in_process(cls, obo: Path, gaf: Path, namespace: str) -> dict[str, Any]:
        ontology_holder: list[Any] = [None]
        ac_holder: list[Any] = [None]

        def _load_obo() -> dict[str, Any]:
            ontology_holder[0] = _load_ontology(obo)
            return {"n_terms": len(getattr(ontology_holder[0], "nodes", []) or [])}

        def _load_gaf() -> dict[str, Any]:
            ac_holder[0] = _load_ac(ontology_holder[0], gaf)
            n = len(getattr(ac_holder[0], "annotations", {}) or {})
            return {"n_objects": n}

        def _build_util() -> dict[str, Any]:
            from fastsemsim.semsim import SemSimUtils

            util = SemSimUtils(ontology_holder[0], ac_holder[0])
            # Force the lazy IC computation, which would otherwise hide
            # in the first term-pair call.
            try:
                util.det_IC_table()
            except Exception:
                pass
            return {"ic_ready": True}

        steps = [
            loading_step("Load ontology", _load_obo),
            loading_step("Load annotations", _load_gaf),
            loading_step("Build SemSimUtils (IC)", _build_util),
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
        from fastsemsim.semsim import SemSimUtils

        ontology = _load_ontology(obo)
        ac = _load_ac(ontology, gaf)
        util = SemSimUtils(ontology, ac)
        try:
            util.det_IC_table()
        except Exception:
            pass

        sim = _make_termsim(ontology, ac, util, method)

        out: dict[int, RunResult] = {}
        first_size = next(iter(pairs_by_size))
        if pairs_by_size[first_size]:
            a, b = pairs_by_size[first_size][0]
            try:
                sim.SemSim(a, b)
            except Exception:
                pass

        for size in sorted(pairs_by_size, reverse=True):
            pairs = pairs_by_size[size]

            def _fn() -> None:
                for a, b in pairs:
                    sim.SemSim(a, b)

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
        from fastsemsim.semsim import BMASemSim, SemSimUtils

        ontology = _load_ontology(obo)
        ac = _load_ac(ontology, gaf)
        util = SemSimUtils(ontology, ac)
        try:
            util.det_IC_table()
        except Exception:
            pass

        term_sim = _make_termsim(ontology, ac, util, method)
        bma = BMASemSim(ontology, ac, util)

        out: dict[int, RunResult] = {}
        first_size = next(iter(gene_pairs_by_size))
        if gene_pairs_by_size[first_size]:
            g1, g2 = gene_pairs_by_size[first_size][0]
            t1 = gene2terms.get(g1, [])
            t2 = gene2terms.get(g2, [])
            if t1 and t2:
                try:
                    bma.SemSim(t1, t2, term_sim)
                except Exception:
                    pass

        for size in sorted(gene_pairs_by_size, reverse=True):
            pairs = gene_pairs_by_size[size]

            def _fn() -> None:
                for g1, g2 in pairs:
                    t1 = gene2terms.get(g1, [])
                    t2 = gene2terms.get(g2, [])
                    if not t1 or not t2:
                        continue
                    bma.SemSim(t1, t2, term_sim)

            timing = time_runs(_fn, warmup=warmup, repeats=repeats)
            out[size] = make_point(size, timing, len(pairs))
        return out
