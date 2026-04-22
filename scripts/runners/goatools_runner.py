"""goatools runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
from ._subproc import run_python_runner_child


def _bma_one_pass(
    go_terms1: list[str],
    go_terms2: list[str],
    sim: Callable[[str, str], float | None],
) -> float:
    if not go_terms1 or not go_terms2:
        return 0.0
    total = float(len(go_terms1) + len(go_terms2))
    col_max = [0.0] * len(go_terms2)
    sum_row_max = 0.0
    for t1 in go_terms1:
        row_max = 0.0
        for j, t2 in enumerate(go_terms2):
            value = sim(t1, t2) or 0.0
            if value > row_max:
                row_max = value
            if value > col_max[j]:
                col_max[j] = value
        sum_row_max += row_max
    return (sum_row_max + sum(col_max)) / total


def _build_godag(obo: Path):
    from goatools.obo_parser import GODag

    return GODag(str(obo), optional_attrs={"relationship"}, prt=None)


def _build_termcounts(godag, gaf: Path):
    from goatools.anno.gaf_reader import GafReader
    from goatools.semantic import TermCounts

    reader = GafReader(str(gaf), godag=godag, prt=None)
    id2gos = reader.get_id2gos_nss(prt=None)
    return TermCounts(godag, id2gos, prt=None), id2gos


def _term_sim_factory(method: str, godag, termcounts, all_terms: set[str] | None):
    from goatools.semantic import lin_sim, resnik_sim
    from goatools.semsim.termwise.wang import SsWang

    method = method.lower()
    if method == "resnik":
        return lambda a, b: resnik_sim(a, b, godag, termcounts)
    if method == "lin":
        return lambda a, b: lin_sim(a, b, godag, termcounts, dfltval=0.0)
    if method == "wang":
        wang = SsWang(all_terms or set(), godag, {"part_of"})
        return lambda a, b: wang.get_sim(a, b)
    raise ValueError(f"Unsupported method: {method}")


@register
class GoatoolsRunner(Runner):
    name = "goatools"
    display_name = "goatools"

    @classmethod
    def is_available(cls) -> bool:
        return python_module_available("goatools")

    @classmethod
    def caps(cls) -> RunnerCaps:
        return RunnerCaps(
            loading=True,
            term_pair_methods={"resnik", "lin"},
            gene_pair_methods={"resnik", "lin"},
        )

    # ------------------------------------------------------------------
    @classmethod
    def loading_in_process(cls, obo: Path, gaf: Path, namespace: str) -> dict[str, Any]:
        godag_holder: list[Any] = [None]
        id2gos_holder: list[Any] = [None]

        def _load_obo() -> dict[str, Any]:
            godag_holder[0] = _build_godag(obo)
            return {"n_terms": len(godag_holder[0])}

        def _load_gaf() -> dict[str, Any]:
            from goatools.anno.gaf_reader import GafReader

            reader = GafReader(str(gaf), godag=godag_holder[0], prt=None)
            id2gos_holder[0] = reader.get_id2gos_nss(prt=None)
            return {"n_objects": len(id2gos_holder[0])}

        def _build_counter() -> dict[str, Any]:
            from goatools.semantic import TermCounts

            tc = TermCounts(godag_holder[0], id2gos_holder[0], prt=None)
            return {"n_goids": len(tc.goids)}

        steps = [
            loading_step("Load ontology", _load_obo),
            loading_step("Load annotations", _load_gaf),
            loading_step("Build counter", _build_counter),
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
        godag = _build_godag(obo)
        termcounts, _ = _build_termcounts(godag, gaf)

        all_terms = {t for ps in pairs_by_size.values() for pair in ps for t in pair}
        sim = _term_sim_factory(method, godag, termcounts, all_terms)

        out: dict[int, RunResult] = {}
        first_size = next(iter(pairs_by_size))
        if pairs_by_size[first_size]:
            a, b = pairs_by_size[first_size][0]
            sim(a, b)

        for size in sorted(pairs_by_size, reverse=True):
            pairs = pairs_by_size[size]

            def _fn() -> None:
                for a, b in pairs:
                    sim(a, b)

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
        godag = _build_godag(obo)
        termcounts, _ = _build_termcounts(godag, gaf)

        all_terms: set[str] = set()
        if method.lower() == "wang":
            for ps in gene_pairs_by_size.values():
                for g1, g2 in ps:
                    all_terms.update(gene2terms.get(g1, []))
                    all_terms.update(gene2terms.get(g2, []))
        sim = _term_sim_factory(method, godag, termcounts, all_terms)

        out: dict[int, RunResult] = {}
        first_size = next(iter(gene_pairs_by_size))
        if gene_pairs_by_size[first_size]:
            g1, g2 = gene_pairs_by_size[first_size][0]
            _bma_one_pass(gene2terms.get(g1, []), gene2terms.get(g2, []), sim)

        for size in sorted(gene_pairs_by_size, reverse=True):
            pairs = gene_pairs_by_size[size]

            def _fn() -> None:
                for g1, g2 in pairs:
                    _bma_one_pass(gene2terms.get(g1, []), gene2terms.get(g2, []), sim)

            timing = time_runs(_fn, warmup=warmup, repeats=repeats)
            out[size] = make_point(size, timing, len(pairs))
        return out
