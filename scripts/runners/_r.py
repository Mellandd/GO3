"""Shared helpers for R-based runners (GOSemSim, simona)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._base import is_executable_on_path
from ._gaf import NAMESPACE_TO_ASPECT, iter_gaf_rows
from ._proc import run_with_rusage


def rscript_available() -> bool:
    return is_executable_on_path("Rscript")


def write_anno_tsv(gaf_path: Path, out_path: Path) -> None:
    """Write a `gene\tGO\tONTOLOGY` TSV from a GAF.

    Used as input by both GOSemSim (`godata(annoDb=...)`) and simona's
    helper script.
    """
    aspect_to_ns = {"P": "BP", "F": "MF", "C": "CC"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("gene\tGO\tONTOLOGY\n")
        for cols in iter_gaf_rows(gaf_path):
            aspect = cols[8].strip()
            ontology = aspect_to_ns.get(aspect)
            if not ontology:
                continue
            symbol = cols[2].strip()
            go_id = cols[4].strip()
            if not symbol or not go_id:
                continue
            out.write(f"{symbol}\t{go_id}\t{ontology}\n")


def build_r_env(r_libs_user: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if r_libs_user:
        env["R_LIBS_USER"] = r_libs_user
    return env


def run_rscript_loading(
    *,
    helper: Path,
    namespace: str,
    measure: str,
    extra_args: list[str],
    r_libs_user: str | None = None,
) -> dict[str, Any]:
    cmd = [
        "Rscript",
        str(helper),
        "--mode",
        "loading",
        "--ontology",
        namespace,
        "--measure",
        measure,
    ] + extra_args
    proc = run_with_rusage(
        cmd,
        env=build_r_env(r_libs_user),
        check=True,
    )
    payload = json.loads(proc.stdout.strip())
    # Parent-captured peak RSS is authoritative on both Linux and macOS.
    # R's own /proc-based self-report returns NA on macOS, so always
    # overwrite it with the rusage value from wait4.
    if isinstance(payload, dict):
        payload["peak_rss_mb"] = proc.peak_rss_mb
        payload["child_user_time_s"] = proc.user_time_s
        payload["child_sys_time_s"] = proc.sys_time_s
    return payload


def run_rscript_pairs(
    *,
    helper: Path,
    mode: str,
    namespace: str,
    measure: str,
    pairs_tsv: Path,
    warmup: int,
    repeats: int,
    seed: int,
    extra_args: list[str],
    r_libs_user: str | None = None,
) -> list[dict[str, Any]]:
    cmd = [
        "Rscript",
        str(helper),
        "--mode",
        mode,
        "--ontology",
        namespace,
        "--measure",
        measure,
        "--pairs-tsv",
        str(pairs_tsv),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
    ] + extra_args
    proc = run_with_rusage(
        cmd,
        env=build_r_env(r_libs_user),
        check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected R helper output:\n{proc.stdout}\n{proc.stderr}")
    header = lines[0].split("\t")
    if header[:2] != ["size", "median_s"]:
        raise RuntimeError(f"Unexpected R header: {header}")
    out: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        n = int(parts[0])
        med = float(parts[1])
        out.append(
            {
                "n": n,
                "median_s": med,
                "min_s": med,
                "max_s": med,
                "runs_s": [med],
                "throughput_per_s": float(n) / max(med, 1e-12),
            }
        )
    out.sort(key=lambda item: item["n"])
    return out


def measure_for_method(method: str) -> str:
    return {"resnik": "Resnik", "lin": "Lin", "wang": "Wang"}[method.lower()]
