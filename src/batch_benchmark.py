import time
import numpy as np
import go3
from goatools.obo_parser import GODag
from goatools.semantic import TermCounts, resnik_sim, lin_sim
from goatools.associations import dnld_assc
import os

go3.load_go_terms()
counter = go3.build_term_counter(go3.load_gaf("goa_human.gaf"))
godag = GODag("go-basic.obo")
assoc = dnld_assc("goa_human.gaf", godag)
tc = TermCounts(godag, assoc)

# GO terms válidos
go_terms = list(counter.ic.keys())
pairs = [(go_terms[i], go_terms[i + 1]) for i in range(0, len(go_terms) - 1, 2)]

batch_sizes = [10, 100, 500, 1000, 5000, 10000, 50000, 100000]
print(f"{'Batch Size':>10} | {'go3 (s)':>10} | {'goatools (s)':>14}")
print("-" * 40)

for size in batch_sizes:
    sub1 = [a for a, _ in pairs[:size]]
    sub2 = [b for _, b in pairs[:size]]

    # 🦀 go3
    t0 = time.perf_counter()
    go3.batch_resnik(sub1, sub2, counter)
    t1 = time.perf_counter()
    go3_time = t1 - t0

    # 🐍 goatools
    t0 = time.perf_counter()
    _ = [resnik_sim(a, b, godag, tc) or 0.0 for a, b in zip(sub1, sub2)]
    t1 = time.perf_counter()
    goatools_time = t1 - t0

    print(f"{size:>10} | {go3_time:>10.6f} | {goatools_time:>14.6f}")