from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Backward-compatible wrapper for the benchmark script.

    This file used to contain an older benchmark implementation. The canonical
    benchmark script is now `scripts/benchmark_go3vsgoatools.py`.
    """

    target = Path(__file__).with_name("benchmark_go3vsgoatools.py")
    return subprocess.call([sys.executable, str(target), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

