"""Benchmark runners for GO semantic similarity libraries.

Each runner module exposes a small interface (see `_base.Runner`) so the
orchestrator in `scripts/benchmark_all.py` can drive every library through
the same loading / term-pair / gene-pair workflow.

Importing this package only registers the runner names; the heavy library
imports happen lazily inside each runner.
"""

from __future__ import annotations

from ._base import Runner, RunnerCaps, RunResult, available_runners, register, get_runner

__all__ = [
    "Runner",
    "RunnerCaps",
    "RunResult",
    "available_runners",
    "register",
    "get_runner",
]
