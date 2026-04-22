"""Shared runner interface and helpers.

A runner is a thin adapter that knows how to drive a single GO semantic
similarity library through three benchmarks:

* `loading(obo, gaf)`         -> dict (time + peak RSS for ontology+annotation+IC build)
* `term_pairs(...)`           -> dict {n_pairs: timing point}
* `gene_pairs(...)`           -> dict {n_pairs: timing point}

Runners that only support a subset of methods/tasks declare it in `caps()`.
The orchestrator uses caps to skip combinations the runner cannot handle
(e.g. fastsemsim has no Wang, TaxaGO has no gene BMA).

Heavy library imports MUST live inside the runner methods, not at module
import time, so that `available_runners()` does not pay for libraries that
are not installed.
"""

from __future__ import annotations

import importlib
import os
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

VALID_METHODS = {"resnik", "lin"}


@dataclass
class RunnerCaps:
    """What a runner can do.

    `loading`, `term_pairs`, `gene_pairs` are the three benchmark batteries.
    Each entry, when non-empty, lists the methods (subset of VALID_METHODS)
    that the runner supports for that battery.
    """

    loading: bool
    term_pair_methods: set[str] = field(default_factory=set)
    gene_pair_methods: set[str] = field(default_factory=set)
    notes: str = ""

    def supports_term(self, method: str) -> bool:
        return method.lower() in self.term_pair_methods

    def supports_gene(self, method: str) -> bool:
        return method.lower() in self.gene_pair_methods


@dataclass
class RunResult:
    """Per-size timing point produced by a runner."""

    n: int
    median_s: float
    min_s: float
    max_s: float
    runs_s: list[float]
    throughput_per_s: float

    def to_dict(self) -> dict[str, Any]:
        lo, hi = bootstrap_ci_median(self.runs_s)
        return {
            "n": int(self.n),
            "median_s": float(self.median_s),
            "min_s": float(self.min_s),
            "max_s": float(self.max_s),
            "runs_s": [float(v) for v in self.runs_s],
            "throughput_per_s": float(self.throughput_per_s),
            "ci_low_s": float(lo),
            "ci_high_s": float(hi),
            "n_runs": len(self.runs_s),
        }


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
# Percentile bootstrap on the median. For the small sample sizes we use
# (repeats=5), this is more honest than a t-based interval (no normality
# assumption) and still gives a visible, reviewer-grade uncertainty band.
# Seeded so the CI is reproducible across runs.


def bootstrap_ci_median(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    iters: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    resamples: list[float] = []
    data = list(values)
    for _ in range(iters):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        resamples.append(median(sample))
    resamples.sort()
    lo_idx = max(0, int((alpha / 2.0) * iters))
    hi_idx = min(iters - 1, int((1.0 - alpha / 2.0) * iters) - 1)
    return (float(resamples[lo_idx]), float(resamples[hi_idx]))


# ---------------------------------------------------------------------------
# Memory helpers (Linux + macOS)
# ---------------------------------------------------------------------------


def _linux_mem_field_mb(field_name: str) -> float | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field_name):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except FileNotFoundError:
        return None
    return None


def rss_mb() -> float:
    if sys.platform.startswith("linux"):
        value = _linux_mem_field_mb("VmRSS:")
        if value is not None:
            return value
    # Fall back to ru_maxrss (only "high water mark", not current RSS,
    # but useful on macOS where /proc is unavailable).
    return peak_rss_mb()


def peak_rss_mb() -> float:
    if sys.platform.startswith("linux"):
        value = _linux_mem_field_mb("VmHWM:")
        if value is not None:
            return value
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(peak) / (1024.0 * 1024.0)
    return float(peak) / 1024.0


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_runs(fn: Callable[[], Any], *, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        fn()
    runs: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    return {
        "runs_s": runs,
        "median_s": median(runs),
        "min_s": min(runs),
        "max_s": max(runs),
    }


def make_point(n: int, timing: dict[str, Any], work_items: int) -> RunResult:
    med = max(float(timing["median_s"]), 1e-12)
    return RunResult(
        n=int(n),
        median_s=float(timing["median_s"]),
        min_s=float(timing["min_s"]),
        max_s=float(timing["max_s"]),
        runs_s=[float(v) for v in timing["runs_s"]],
        throughput_per_s=float(work_items) / med,
    )


def loading_step(name: str, fn: Callable[[], Any]) -> dict[str, Any]:
    """Run a single loading step, capturing time and post-step memory.

    Designed to be called by Python runners from inside an isolated child
    process — that way peak_rss reflects only this runner's footprint.
    """
    start = time.perf_counter()
    details = fn()
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "time_s": elapsed,
        "rss_mb": rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "details": details if isinstance(details, dict) else {"value": details},
    }


def loading_finalize(steps: list[dict[str, Any]], lib: str) -> dict[str, Any]:
    return {
        "lib": lib,
        "steps": steps,
        "total_time_s": sum(s["time_s"] for s in steps),
        "peak_rss_mb": max((s["peak_rss_mb"] for s in steps), default=0.0),
        "final_rss_mb": steps[-1]["rss_mb"] if steps else rss_mb(),
        "platform": sys.platform,
        "python": sys.version,
    }


# ---------------------------------------------------------------------------
# CLI presence
# ---------------------------------------------------------------------------


def is_executable_on_path(name: str) -> bool:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return True
    return False


def python_module_available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Runner base + registry
# ---------------------------------------------------------------------------


class Runner:
    """Subclass and register one per supported library."""

    name: str = ""
    display_name: str = ""

    @classmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def caps(cls) -> RunnerCaps:
        raise NotImplementedError

    # ---- loading ---------------------------------------------------------
    # Run the loading sequence in-process (typically called inside a child
    # process so peak RSS is isolated). Python runners override this; non-
    # Python runners should override `loading_external` instead and leave
    # this as NotImplementedError.
    @classmethod
    def loading_in_process(cls, obo: Path, gaf: Path, namespace: str) -> dict[str, Any]:
        raise NotImplementedError

    # Subprocess-style loading: spawn whatever is needed and return the same
    # dict shape as `loading_in_process`.
    @classmethod
    def loading(
        cls,
        obo: Path,
        gaf: Path,
        namespace: str,
        *,
        python_executable: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    # ---- term-pair benchmark --------------------------------------------
    # Both batteries take a list of "size groups" already sampled by the
    # orchestrator (so every runner sees the same workload).
    #   pairs_by_size: {n_pairs: [(go_a, go_b), ...]}
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
        raise NotImplementedError

    # ---- gene-pair benchmark --------------------------------------------
    #   gene_pairs_by_size: {n_pairs: [(gene_a, gene_b), ...]}
    #   gene2terms: {gene_symbol: [GO ids]}  (BP/MF/CC-filtered upstream)
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
        raise NotImplementedError


_REGISTRY: dict[str, type[Runner]] = {}


def register(cls: type[Runner]) -> type[Runner]:
    if not cls.name:
        raise ValueError(f"Runner {cls!r} has empty name")
    _REGISTRY[cls.name] = cls
    return cls


def get_runner(name: str) -> type[Runner]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown runner '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_runners(only: list[str] | None = None) -> dict[str, type[Runner]]:
    """Return registered runners that pass `is_available()`.

    If `only` is provided, restrict to those names (still filtered by
    availability).
    """
    # Force-import every runner module so they self-register.
    from . import (  # noqa: F401  (side-effect imports)
        go3_runner,
        goatools_runner,
        fastsemsim_runner,
        pygosemsim_runner,
        gosemsim_runner,
        simona_runner,
        taxago_runner,
    )

    items = _REGISTRY
    if only:
        items = {n: items[n] for n in only if n in items}
    return {name: cls for name, cls in items.items() if cls.is_available()}
