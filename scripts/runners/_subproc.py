"""Subprocess helpers shared by Python runners.

Spawns a child python interpreter that imports a runner via the package
`runners` (so we add `scripts/` to PYTHONPATH).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ._proc import run_with_rusage

SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(SCRIPTS_DIR)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def write_pairs_tsv(path: Path, pairs_by_size: dict[int, list[tuple[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for size in sorted(pairs_by_size):
            for a, b in pairs_by_size[size]:
                handle.write(f"{size}\t{a}\t{b}\n")


def write_gene2terms_tsv(path: Path, gene2terms: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for symbol, terms in gene2terms.items():
            handle.write(f"{symbol}\t{','.join(terms)}\n")


def run_python_runner_child(
    *,
    runner: str,
    task: str,
    obo: Path,
    gaf: Path,
    namespace: str,
    workdir: Path,
    method: str | None = None,
    pairs_tsv: Path | None = None,
    gene2terms_tsv: Path | None = None,
    warmup: int | None = None,
    repeats: int | None = None,
    threads: int | None = None,
    python_executable: str | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    out_json = workdir / f"{runner}_{task}.json"

    cmd = [
        python_executable or sys.executable,
        "-u",
        "-m",
        "runners._child",
        "--runner",
        runner,
        "--task",
        task,
        "--obo",
        str(obo),
        "--gaf",
        str(gaf),
        "--namespace",
        namespace,
        "--workdir",
        str(workdir),
        "--json",
        str(out_json),
    ]
    if method is not None:
        cmd.extend(["--method", method])
    if pairs_tsv is not None:
        cmd.extend(["--pairs-tsv", str(pairs_tsv)])
    if gene2terms_tsv is not None:
        cmd.extend(["--gene2terms-tsv", str(gene2terms_tsv)])
    if warmup is not None:
        cmd.extend(["--warmup", str(warmup)])
    if repeats is not None:
        cmd.extend(["--repeats", str(repeats)])
    if threads is not None:
        cmd.extend(["--threads", str(threads)])

    proc = run_with_rusage(
        cmd,
        env=_child_env(extra_env),
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Runner '{runner}' subprocess failed (task={task}).\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    if not out_json.exists():
        raise RuntimeError(
            f"Runner '{runner}' produced no JSON output.\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
        )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    # Parent-captured peak RSS is authoritative (works on Linux + macOS,
    # covers the whole child lifetime including import overhead). Overwrite
    # whatever the child self-reported.
    if isinstance(payload, dict) and task == "loading":
        payload["peak_rss_mb"] = proc.peak_rss_mb
        payload["child_user_time_s"] = proc.user_time_s
        payload["child_sys_time_s"] = proc.sys_time_s
    return payload
