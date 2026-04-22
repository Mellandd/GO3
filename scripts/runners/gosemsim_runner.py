"""GOSemSim (R) runner.

Wraps `scripts/benchmark_gosemsim.R`. Annotation comes from a TSV the
orchestrator writes from the GAF (gene<TAB>GO<TAB>ONTOLOGY) so that the
underlying IC matches every other library's view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._base import (
    Runner,
    RunnerCaps,
    RunResult,
    register,
)
from ._r import (
    measure_for_method,
    rscript_available,
    run_rscript_loading,
    run_rscript_pairs,
    write_anno_tsv,
)
from ._subproc import write_pairs_tsv

R_HELPER = Path(__file__).resolve().parents[1] / "benchmark_gosemsim.R"


def _gene_pairs_tsv(path: Path, gene_pairs_by_size, gene2terms):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for size in sorted(gene_pairs_by_size):
            for g1, g2 in gene_pairs_by_size[size]:
                t1 = ",".join(gene2terms.get(g1, []))
                t2 = ",".join(gene2terms.get(g2, []))
                out.write(f"{size}\t{t1}\t{t2}\n")


@register
class GosemsimRunner(Runner):
    name = "gosemsim"
    display_name = "GOSemSim"

    @classmethod
    def is_available(cls) -> bool:
        return rscript_available() and R_HELPER.exists()

    @classmethod
    def caps(cls) -> RunnerCaps:
        return RunnerCaps(
            loading=True,
            term_pair_methods={"resnik", "lin"},
            gene_pair_methods={"resnik", "lin"},
            notes="Subprocess Rscript; annotation built from GAF -> TSV.",
        )

    # ------------------------------------------------------------------
    @classmethod
    def loading(cls, obo: Path, gaf: Path, namespace: str, *, python_executable: str | None = None) -> dict[str, Any]:
        workdir = Path("/tmp/go3_bench") / cls.name
        workdir.mkdir(parents=True, exist_ok=True)
        anno_tsv = workdir / "anno.tsv"
        write_anno_tsv(gaf, anno_tsv)
        return run_rscript_loading(
            helper=R_HELPER,
            namespace=namespace,
            measure="Resnik",
            extra_args=["--anno-tsv", str(anno_tsv)],
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
        wd = workdir or (Path("/tmp/go3_bench") / cls.name)
        wd.mkdir(parents=True, exist_ok=True)
        anno_tsv = wd / "anno.tsv"
        if not anno_tsv.exists():
            write_anno_tsv(gaf, anno_tsv)
        pairs_tsv = wd / f"term_pairs_{namespace}_{method}.tsv"
        write_pairs_tsv(pairs_tsv, pairs_by_size)

        rows = run_rscript_pairs(
            helper=R_HELPER,
            mode="term",
            namespace=namespace,
            measure=measure_for_method(method),
            pairs_tsv=pairs_tsv,
            warmup=warmup,
            repeats=repeats,
            seed=42,
            extra_args=["--anno-tsv", str(anno_tsv)],
        )
        return {
            int(p["n"]): RunResult(
                n=int(p["n"]),
                median_s=p["median_s"],
                min_s=p["min_s"],
                max_s=p["max_s"],
                runs_s=p["runs_s"],
                throughput_per_s=p["throughput_per_s"],
            )
            for p in rows
        }

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
        wd = workdir or (Path("/tmp/go3_bench") / cls.name)
        wd.mkdir(parents=True, exist_ok=True)
        anno_tsv = wd / "anno.tsv"
        if not anno_tsv.exists():
            write_anno_tsv(gaf, anno_tsv)
        pairs_tsv = wd / f"gene_pairs_{namespace}_{method}.tsv"
        _gene_pairs_tsv(pairs_tsv, gene_pairs_by_size, gene2terms)

        rows = run_rscript_pairs(
            helper=R_HELPER,
            mode="gene",
            namespace=namespace,
            measure=measure_for_method(method),
            pairs_tsv=pairs_tsv,
            warmup=warmup,
            repeats=repeats,
            seed=42,
            extra_args=["--anno-tsv", str(anno_tsv)],
        )
        return {
            int(p["n"]): RunResult(
                n=int(p["n"]),
                median_s=p["median_s"],
                min_s=p["min_s"],
                max_s=p["max_s"],
                runs_s=p["runs_s"],
                throughput_per_s=p["throughput_per_s"],
            )
            for p in rows
        }
