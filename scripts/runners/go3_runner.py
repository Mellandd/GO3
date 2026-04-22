"""go3 (this project) runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _base
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


@register
class Go3Runner(Runner):
    name = "go3"
    display_name = "go3"

    @classmethod
    def is_available(cls) -> bool:
        return python_module_available("go3")

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
        import go3

        annotations: list[Any] | None = None
        steps: list[dict[str, Any]] = []

        steps.append(loading_step(
            "Load ontology",
            lambda: {"n_terms": len(go3.load_go_terms(str(obo)))},
        ))

        def _load_gaf() -> dict[str, Any]:
            nonlocal annotations
            annotations = go3.load_gaf(str(gaf))
            return {"n_annotations": len(annotations)}

        steps.append(loading_step("Load annotations", _load_gaf))

        def _build_counter() -> dict[str, Any]:
            assert annotations is not None
            counter = go3.build_term_counter(annotations)
            return {"n_ic_terms": len(counter.ic)}

        steps.append(loading_step("Build counter", _build_counter))
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
        import go3

        go3.load_go_terms(str(obo))
        annotations = go3.load_gaf(str(gaf))
        counter = go3.build_term_counter(annotations)
        if threads is not None:
            go3.set_num_threads(int(threads))

        out: dict[int, RunResult] = {}
        # Prime once.
        first_size = next(iter(pairs_by_size))
        if pairs_by_size[first_size]:
            a, b = pairs_by_size[first_size][0]
            go3.batch_similarity([a], [b], method, counter)

        for size in sorted(pairs_by_size, reverse=True):
            pairs = pairs_by_size[size]
            list1 = [a for a, _ in pairs]
            list2 = [b for _, b in pairs]

            def _fn() -> None:
                go3.batch_similarity(list1, list2, method, counter)

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
        import go3

        go3.load_go_terms(str(obo))
        annotations = go3.load_gaf(str(gaf))
        counter = go3.build_term_counter(annotations)
        if threads is not None:
            go3.set_num_threads(int(threads))

        out: dict[int, RunResult] = {}
        first_size = next(iter(gene_pairs_by_size))
        if gene_pairs_by_size[first_size]:
            first_pair = gene_pairs_by_size[first_size][0]
            go3.compare_gene_pairs_batch([first_pair], namespace, method, "bma", counter)

        for size in sorted(gene_pairs_by_size, reverse=True):
            pairs = gene_pairs_by_size[size]

            def _fn() -> None:
                go3.compare_gene_pairs_batch(pairs, namespace, method, "bma", counter)

            timing = time_runs(_fn, warmup=warmup, repeats=repeats)
            out[size] = make_point(size, timing, len(pairs))
        return out
