"""Subprocess helpers that capture per-child peak RSS via `os.wait4`.

POSIX-only. Both Linux and macOS are supported; on Linux `ru_maxrss` is
kilobytes, on macOS it is bytes. `run_with_rusage` normalises both to MB.

Why `os.wait4` instead of `resource.getrusage(RUSAGE_CHILDREN)`:
`RUSAGE_CHILDREN.ru_maxrss` is monotonic-max over all reaped children,
so you can't isolate a single child once another one has used more
memory in the same parent. `os.wait4` gives you rusage of the specific
child that just exited — exact per-invocation peak RSS.

Why not `subprocess.run` directly: its internal `wait()` consumes the
SIGCHLD and rusage is lost. We run Popen + drain pipes in threads + call
`os.wait4` ourselves.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    peak_rss_mb: float
    user_time_s: float
    sys_time_s: float


def _drain(pipe, sink: list) -> None:
    if pipe is None:
        return
    try:
        for chunk in iter(lambda: pipe.read(8192), ""):
            if not chunk:
                break
            sink.append(chunk)
    finally:
        pipe.close()


def _rusage_to_mb(ru_maxrss: int) -> float:
    if sys.platform == "darwin":
        return float(ru_maxrss) / (1024.0 * 1024.0)
    return float(ru_maxrss) / 1024.0


def run_with_rusage(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    capture_output: bool = True,
    check: bool = False,
    text: bool = True,
    stdin: Any = None,
) -> RunResult:
    """Run a subprocess and return its return code, captured output,
    and peak resident-set-size (MB) observed for that specific child.
    """
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        text=text,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    threads: list[threading.Thread] = []
    if capture_output:
        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_chunks))
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks))
        t_out.start()
        t_err.start()
        threads.extend([t_out, t_err])

    try:
        pid, status, rusage = os.wait4(proc.pid, 0)
    except ChildProcessError:
        # Fallback if the child was already reaped elsewhere.
        proc.wait(timeout=timeout)
        rusage = None
        status = (proc.returncode or 0) << 8

    for t in threads:
        t.join()

    # Sync Popen's bookkeeping so it doesn't try to reap again.
    if hasattr(os, "waitstatus_to_exitcode"):
        rc = os.waitstatus_to_exitcode(status)
    else:  # pragma: no cover
        if os.WIFEXITED(status):
            rc = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            rc = -os.WTERMSIG(status)
        else:
            rc = -1
    proc.returncode = rc

    peak_mb = float("nan")
    user_s = float("nan")
    sys_s = float("nan")
    if rusage is not None:
        peak_mb = _rusage_to_mb(rusage.ru_maxrss)
        user_s = float(rusage.ru_utime)
        sys_s = float(rusage.ru_stime)

    result = RunResult(
        returncode=rc,
        stdout="".join(stdout_chunks) if capture_output else "",
        stderr="".join(stderr_chunks) if capture_output else "",
        peak_rss_mb=peak_mb,
        user_time_s=user_s,
        sys_time_s=sys_s,
    )

    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, result.stdout, result.stderr)

    return result
