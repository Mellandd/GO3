from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple


def default_paths() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "tests" / "goa_human.gaf", root / "tests" / "go-basic.obo"


def pick_genes_from_gaf(gaf_path: Path, n_genes: int) -> List[str]:
    counts = Counter()
    with open(gaf_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue
            qualifier = cols[3]
            evidence = cols[6]
            if evidence == "ND":
                continue
            if "NOT" in qualifier:
                continue
            gene = cols[2]
            counts[gene] += 1
    return [gene for gene, _ in counts.most_common(n_genes)]


def auto_perplexity(n: int, user_value: float | None) -> float:
    if user_value is not None:
        return user_value
    return min(30.0, max(2.0, (n - 1) / 3.0))


def auto_n_neighbors(n: int, user_value: int | None) -> int:
    if user_value is not None:
        return user_value
    return min(15, max(2, n // 3))
