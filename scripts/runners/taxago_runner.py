"""TaxaGO (Rust CLI) runner.

TaxaGO ships two separate binaries:

  * `taxago` — enrichment analysis (not used here)
  * `semantic-similarity` — term-pair similarity, operating on a closed
    term set (given N terms it produces the full N×N similarity matrix).

We invoke the `semantic-similarity` binary directly. The term-pair
battery feeds it a closed term set (same workload every runner sees).

Gene-level BMA is not exposed by the TaxaGO CLI, so this runner
implements BMA on top: for each size group, all unique GO terms across
the gene pairs are sent to `semantic-similarity` in a single call, the
N×N matrix is read back, and BMA is computed in Python. The timing
includes the full end-to-end pipeline (write-terms → run binary → parse
matrix → BMA), which is what a user would pay for doing gene-level
similarity through TaxaGO.

Discovery: the binary is expected on PATH as `semantic-similarity`.
Override with `TAXAGO_SEMSIM_BIN` env var.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from statistics import median
from typing import Any

from ._base import (
    Runner,
    RunnerCaps,
    RunResult,
    is_executable_on_path,
    register,
)
from ._proc import run_with_rusage


METHOD_FLAG = {"resnik": "resnik", "lin": "lin", "wang": "wang"}


def _binary() -> str | None:
    env = os.environ.get("TAXAGO_SEMSIM_BIN") or os.environ.get("TAXAGO_BIN")
    if env:
        return env
    if is_executable_on_path("semantic-similarity"):
        return "semantic-similarity"
    return None


def _terms_for_size(pairs: list[tuple[str, str]]) -> list[str]:
    seen: dict[str, None] = {}
    for a, b in pairs:
        seen[a] = None
        seen[b] = None
    return list(seen.keys())


def _matrix_path(outdir: Path, method: str) -> Path:
    # semantic-similarity writes `similarity_<method>_taxon_<id>.tsv`.
    return outdir / f"similarity_{method}_taxon_9606.tsv"


def _parse_matrix(path: Path) -> dict[tuple[str, str], float]:
    """Read a TaxaGO similarity matrix TSV into a {(a,b): score} dict.

    The first row is column labels (with an empty leading cell); each
    subsequent row starts with the row label followed by floats.
    """
    scores: dict[tuple[str, str], float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        cols = header[1:]
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            row = parts[0]
            for col, value in zip(cols, parts[1:]):
                try:
                    scores[(row, col)] = float(value)
                except ValueError:
                    continue
    return scores


def _bma(t1: list[str], t2: list[str], scores: dict[tuple[str, str], float]) -> float:
    if not t1 or not t2:
        return 0.0
    best_a: list[float] = []
    for a in t1:
        m = 0.0
        for b in t2:
            v = scores.get((a, b))
            if v is None:
                v = scores.get((b, a), 0.0)
            if v > m:
                m = v
        best_a.append(m)
    best_b: list[float] = []
    for b in t2:
        m = 0.0
        for a in t1:
            v = scores.get((a, b))
            if v is None:
                v = scores.get((b, a), 0.0)
            if v > m:
                m = v
        best_b.append(m)
    return 0.5 * (sum(best_a) / len(best_a) + sum(best_b) / len(best_b))


@register
class TaxagoRunner(Runner):
    name = "taxago"
    display_name = "TaxaGO"

    @classmethod
    def is_available(cls) -> bool:
        return _binary() is not None

    @classmethod
    def caps(cls) -> RunnerCaps:
        return RunnerCaps(
            loading=True,
            term_pair_methods={"resnik", "lin"},
            gene_pair_methods={"resnik", "lin"},
            notes=(
                "Term-pair: N×N matrix over the closed term set supplied per "
                "size. Gene-pair: matrix over the union of terms across gene "
                "pairs + Python-side BMA (TaxaGO's CLI has no native BMA)."
            ),
        )

    # ------------------------------------------------------------------
    @classmethod
    def loading(cls, obo: Path, gaf: Path, namespace: str, *, python_executable: str | None = None) -> dict[str, Any]:
        binary = _binary()
        if binary is None:
            raise RuntimeError("taxago binary not found (set TAXAGO_BIN)")
        wd = Path("/tmp/go3_bench") / cls.name
        wd.mkdir(parents=True, exist_ok=True)
        terms_file = wd / "loading_terms.txt"
        # Single root-term call as a "cold load" proxy, like the SML runner.
        terms_file.write_text("GO:0008150\n", encoding="utf-8")
        cmd = [
            binary,
            "-o", str(obo),
            "-t", str(terms_file),
            "-m", "resnik",
            "-i", "9606",
            "-d", str(wd / "loading_out"),
        ]
        t0 = time.perf_counter()
        proc = run_with_rusage(cmd)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(
                f"TaxaGO loading invocation failed.\nstderr:\n{proc.stderr}"
            )
        peak_mb = proc.peak_rss_mb
        return {
            "lib": cls.name,
            "display_name": cls.display_name,
            "steps": [{"name": "Cold load + 1 term", "time_s": elapsed,
                       "rss_mb": peak_mb, "peak_rss_mb": peak_mb,
                       "details": {}}],
            "total_time_s": elapsed,
            "peak_rss_mb": peak_mb,
            "final_rss_mb": peak_mb,
            "child_user_time_s": proc.user_time_s,
            "child_sys_time_s": proc.sys_time_s,
            "platform": "subprocess-rust",
            "notes": "Wall-clock of one cold taxago invocation (loads OBO + background).",
        }

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
        binary = _binary()
        if binary is None:
            raise RuntimeError("taxago binary not found (set TAXAGO_BIN)")
        wd = workdir or (Path("/tmp/go3_bench") / cls.name)
        wd.mkdir(parents=True, exist_ok=True)

        method_flag = METHOD_FLAG[method.lower()]
        out: dict[int, RunResult] = {}

        for size in sorted(pairs_by_size, reverse=True):
            pairs = pairs_by_size[size]
            terms = _terms_for_size(pairs)
            terms_file = wd / f"terms_{size}.txt"
            terms_file.write_text("\n".join(terms) + "\n", encoding="utf-8")
            outdir = wd / f"out_{size}"
            outdir.mkdir(parents=True, exist_ok=True)

            cmd = [
                binary,
                "-o", str(obo),
                "-t", str(terms_file),
                "-m", method_flag,
                "-i", "9606",
                "-d", str(outdir),
            ]
            for _ in range(max(0, warmup)):
                run_with_rusage(cmd)

            runs: list[float] = []
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                proc = run_with_rusage(cmd)
                runs.append(time.perf_counter() - t0)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"TaxaGO term-pair run failed (n={size}).\n"
                        f"stderr:\n{proc.stderr}"
                    )
            med = median(runs)
            out[size] = RunResult(
                n=size,
                median_s=med, min_s=min(runs), max_s=max(runs), runs_s=runs,
                throughput_per_s=float(len(pairs)) / max(med, 1e-12),
            )
        return out

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
        binary = _binary()
        if binary is None:
            raise RuntimeError("taxago binary not found (set TAXAGO_SEMSIM_BIN)")
        wd = workdir or (Path("/tmp/go3_bench") / cls.name)
        wd.mkdir(parents=True, exist_ok=True)

        method_flag = METHOD_FLAG[method.lower()]
        out: dict[int, RunResult] = {}

        for size in sorted(gene_pairs_by_size, reverse=True):
            pairs = gene_pairs_by_size[size]
            term_lists: list[tuple[list[str], list[str]]] = []
            all_terms: dict[str, None] = {}
            for g1, g2 in pairs:
                t1 = gene2terms.get(g1, [])
                t2 = gene2terms.get(g2, [])
                term_lists.append((t1, t2))
                for t in t1:
                    all_terms[t] = None
                for t in t2:
                    all_terms[t] = None
            terms = list(all_terms)
            terms_file = wd / f"gene_terms_{method_flag}_{size}.txt"
            terms_file.write_text("\n".join(terms) + "\n", encoding="utf-8")
            outdir = wd / f"gene_out_{method_flag}_{size}"
            outdir.mkdir(parents=True, exist_ok=True)

            cmd = [
                binary,
                "-o", str(obo),
                "-t", str(terms_file),
                "-m", method_flag,
                "-i", "9606",
                "-d", str(outdir),
            ]
            matrix_file = _matrix_path(outdir, method_flag)

            def _one_run() -> None:
                proc = run_with_rusage(cmd)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"TaxaGO gene-pair run failed (n={size}).\n"
                        f"stderr:\n{proc.stderr}"
                    )
                scores = _parse_matrix(matrix_file)
                for t1, t2 in term_lists:
                    _bma(t1, t2, scores)

            for _ in range(max(0, warmup)):
                _one_run()

            runs: list[float] = []
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                _one_run()
                runs.append(time.perf_counter() - t0)
            med = median(runs)
            out[size] = RunResult(
                n=size,
                median_s=med, min_s=min(runs), max_s=max(runs), runs_s=runs,
                throughput_per_s=float(len(pairs)) / max(med, 1e-12),
            )
        return out
