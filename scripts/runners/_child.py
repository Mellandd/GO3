"""Subprocess entrypoint for in-process Python runners.

Used so that loading metrics (peak RSS) stay isolated per library and so
that timed runs aren't contaminated by other libraries already loaded into
the parent process.

Invoked as:

    python -m scripts.runners._child --runner go3 --task loading \
        --obo path --gaf path --namespace BP --json out.json

    python -m scripts.runners._child --runner fastsemsim --task term_pairs \
        --obo ... --gaf ... --namespace BP --method lin \
        --pairs-tsv pairs.tsv --warmup 2 --repeats 5 --json out.json

The pairs TSV format is shared with the R helpers:
    size<TAB>a<TAB>b      (each row contributes one pair to group `size`)

For gene_pairs, an extra TSV is supplied via --gene2terms-tsv with rows
    symbol<TAB>GO:xxx,GO:yyy,...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import _base
from ._base import Runner, get_runner


def _read_pairs_tsv(path: Path) -> dict[int, list[tuple[str, str]]]:
    out: dict[int, list[tuple[str, str]]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            size = int(parts[0])
            out.setdefault(size, []).append((parts[1], parts[2]))
    return out


def _read_gene2terms_tsv(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            symbol = parts[0]
            terms = [t for t in parts[1].split(",") if t]
            out[symbol] = terms
    return out


def _serialize(points: dict[int, _base.RunResult]) -> dict[str, object]:
    return {str(n): p.to_dict() for n, p in points.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--task", required=True, choices=["loading", "term_pairs", "gene_pairs"])
    parser.add_argument("--obo", required=True, type=Path)
    parser.add_argument("--gaf", required=True, type=Path)
    parser.add_argument("--namespace", default="BP")
    parser.add_argument("--method", default=None)
    parser.add_argument("--pairs-tsv", type=Path, default=None)
    parser.add_argument("--gene2terms-tsv", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    # Force-import so the registry is populated.
    _base.available_runners()
    runner_cls: type[Runner] = get_runner(args.runner)

    if args.task == "loading":
        out = runner_cls.loading_in_process(args.obo, args.gaf, args.namespace)
    elif args.task == "term_pairs":
        if args.method is None or args.pairs_tsv is None:
            raise SystemExit("--method and --pairs-tsv required for term_pairs")
        pairs_by_size = _read_pairs_tsv(args.pairs_tsv)
        points = runner_cls.term_pairs(
            obo=args.obo,
            gaf=args.gaf,
            namespace=args.namespace,
            method=args.method,
            pairs_by_size=pairs_by_size,
            warmup=args.warmup,
            repeats=args.repeats,
            threads=args.threads,
            workdir=args.workdir,
        )
        out = _serialize(points)
    elif args.task == "gene_pairs":
        if args.method is None or args.pairs_tsv is None or args.gene2terms_tsv is None:
            raise SystemExit("--method, --pairs-tsv and --gene2terms-tsv required for gene_pairs")
        pairs_by_size = _read_pairs_tsv(args.pairs_tsv)
        gene2terms = _read_gene2terms_tsv(args.gene2terms_tsv)
        points = runner_cls.gene_pairs(
            obo=args.obo,
            gaf=args.gaf,
            namespace=args.namespace,
            method=args.method,
            gene_pairs_by_size=pairs_by_size,
            gene2terms=gene2terms,
            warmup=args.warmup,
            repeats=args.repeats,
            threads=args.threads,
            workdir=args.workdir,
        )
        out = _serialize(points)
    else:
        raise SystemExit(f"Unknown task: {args.task}")

    payload = json.dumps(out)
    if args.json is not None:
        args.json.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
