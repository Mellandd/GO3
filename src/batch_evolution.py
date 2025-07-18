import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import go3
from goatools.obo_parser import GODag
from goatools.semantic import TermCounts, resnik_sim, lin_sim
from goatools.associations import dnld_assc

# --- Setup ---
go_terms = go3.load_go_terms()
gaf_path = "goa_human.gaf"
annotations = go3.load_gaf(gaf_path)
counter = go3.build_term_counter(annotations)

go_dag = GODag("go-basic.obo")
from goatools.gosubdag.gosubdag import GoSubDag
go_subdag = GoSubDag(go_dag.keys(), go_dag)
gene2go = dnld_assc(gaf_path, go_dag)
termcounts = TermCounts(go_dag, gene2go)

# Fixed GO term pairs for fair test
pairs = [("GO:0008150", "GO:0009987"), ("GO:0003674", "GO:0005488")]

batch_sizes = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000]
goatools_times = []
go3_times = []

for size in batch_sizes:
    test_pairs = pairs * (size // len(pairs))

    # Goatools Resnik batch
    start = time.perf_counter()
    _ = [resnik_sim(a, b, go_dag, termcounts) for a, b in test_pairs]
    goatools_times.append(time.perf_counter() - start)

    # go3 Resnik batch
    start = time.perf_counter()
    _ = go3.batch_similarity([a for a, _ in test_pairs], [b for _, b in test_pairs], 'resnik', counter)
    go3_times.append(time.perf_counter() - start)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, goatools_times, label="Goatools (Python)", marker='o')
plt.plot(batch_sizes, go3_times, label="go3 (Rust)", marker='s')
plt.xlabel('Batch Size (# GO term pairs)')
plt.ylabel('Time (seconds)')
plt.title('🚀 Batch Resnik Similarity: Time vs Batch Size')
plt.legend()
plt.xscale('log')
#plt.yscale('log')  # Optional: if times grow too much
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("benchmark_batch_evolution.png")
plt.show()