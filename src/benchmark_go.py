import time
import os
import psutil
import numpy as np
from goatools.obo_parser import GODag
from goatools.associations import dnld_assc
from goatools.semantic import TermCounts, resnik_sim, lin_sim
import go3

# --- Configuración ---
obo_path = "go-basic.obo"
gaf_path = "goa_human.gaf"
godag = GODag(obo_path)
assocs = dnld_assc(gaf_path, godag)
termcounts = TermCounts(godag, assocs)

go3.load_go_terms()
anns = go3.load_gaf(gaf_path)
counter = go3.build_term_counter(anns)

go_terms = list(counter.ic.keys())
pairs = [(go_terms[i], go_terms[i + 1]) for i in range(0, len(go_terms) - 1, 2)]
batch_sizes = [10]
proc = psutil.Process(os.getpid())

print(f"{'Batch':>6} | {'go3 time':>9} | {'goatools':>9} | {'Diff?':>6} | {'go3 MB':>8} | {'Peak MB':>8}")
print("-" * 60)

for size in batch_sizes:
    a = [p[0] for p in pairs[:size]]
    b = [p[1] for p in pairs[:size]]

    # Medida memoria antes
    mem_before = proc.memory_info().rss / 1024 / 1024

    # ⏱ go3
    t0 = time.perf_counter()
    res_go3 = go3.batch_resnik(a, b, counter)
    print(res_go3)
    t1 = time.perf_counter()
    time_go3 = t1 - t0

    # ⏱ goatools
    t0 = time.perf_counter()
    res_gt = [resnik_sim(x, y, godag, termcounts) or 0.0 for x, y in zip(a, b)]
    print(res_gt)
    t1 = time.perf_counter()
    time_gt = t1 - t0

    # Validación de resultados
    mismatch = not np.allclose(res_go3, res_gt, rtol=1e-6, atol=1e-6)

    # Medida memoria después
    mem_after = proc.memory_info().rss / 1024 / 1024
    mem_diff = mem_after - mem_before

    print(f"{size:6} | {time_go3:9.5f} | {time_gt:9.5f} | {'❌' if mismatch else '✅':^6} | {mem_after:8.2f} | {mem_diff:8.2f}")