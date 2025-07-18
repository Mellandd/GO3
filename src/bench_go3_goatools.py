import time
import matplotlib.pyplot as plt
import numpy as np
import go3
from goatools.base import get_godag
from goatools.semantic import TermCounts, resnik_sim, lin_sim
from goatools.associations import dnld_assc
import random

def sample_unique_pairs(go_ids, n_pairs):
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = random.sample(go_ids, 2)
        if a != b:
            pairs.add((a, b))
    return list(pairs)

# --- Setup ---
go_terms = go3.load_go_terms()
gaf_path = "goa_human.gaf"
annotations = go3.load_gaf(gaf_path)
counter = go3.build_term_counter(annotations)
from goatools.semsim.termwise.wang import SsWang

go_dag = get_godag("go-basic.obo", optional_attrs={'relationship'})
gene2go = dnld_assc(gaf_path, go_dag)
termcounts = TermCounts(go_dag, gene2go)


def goatools_wang(a, b):
    return wang.get_sim(a, b)

# Fixed GO term pairs for fair test
bp_terms = [t.id for t in go_terms if t.namespace == "biological_process"]

batch_sizes = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 500000]

# --- Similarity methods to test ---
methods = [
    ("resnik", lambda a, b: resnik_sim(a, b, go_dag, termcounts), "resnik"),
    ("lin", lambda a, b: lin_sim(a, b, go_dag, termcounts), "lin"),
    ("wang", goatools_wang, "wang"),
    # Add more goatools methods if available, e.g. Wang if you have an implementation
]

go3_methods = ["resnik", "lin", "wang"]  # Add more if implemented in go3

# --- Benchmark term-to-term batch ---
results = {}
for method, goatools_func, go3_method in methods:
    goatools_times = []
    go3_times = []
    for size in batch_sizes:
        test_pairs = sample_unique_pairs(bp_terms, size)
        list1, list2 = zip(*test_pairs)
        # Goatools
        start = time.perf_counter()
        if method == 'wang':
            all_goids = set([a for a, _ in test_pairs] + [b for _, b in test_pairs])
            wang = SsWang(all_goids, go_dag, {'part_of'})
        _ = [goatools_func(a, b) for a, b in test_pairs]
        goatools_times.append(time.perf_counter() - start)
        # go3
        start = time.perf_counter()
        _ = go3.batch_similarity(list(list1), list(list2), go3_method, counter)
        go3_times.append(time.perf_counter() - start)

    results[method] = (goatools_times, go3_times)

# --- Plot term-to-term batch ---
plt.figure(figsize=(10, 6))
for method in results:
    goatools_times, go3_times = results[method]
    plt.plot(batch_sizes, goatools_times, label=f"Goatools {method}", marker='o', linestyle='--')
for method in results:
    goatools_times, go3_times = results[method]
    plt.plot(batch_sizes, go3_times, label=f"go3 {method}", marker='s')
plt.xlabel('Batch Size (# GO term pairs)')
plt.ylabel('Time (seconds)')
plt.title('Batch GO Term Similarity: Time vs Batch Size')
plt.legend()
plt.xscale('log')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("benchmark_batch_similarity.png")
plt.show()

gene_results = {}
gene2go = go3.load_gaf(gaf_path)  # or however you get a dict: gene -> [GO terms]
# If go3.load_gaf returns a list of annotations, build the mapping:
from collections import defaultdict
gene2go_map = defaultdict(list)
for ann in annotations:
    gene2go_map[ann.db_object_id].append(ann.go_term)
genes = list(gene2go_map.keys())
print(genes)

def sample_unique_gene_pairs(genes, n_pairs):
    pairs = set()
    while len(pairs) < n_pairs:
        a, b = random.sample(genes, 2)
        if a != b:
            pairs.add((a, b))
    return list(pairs)

def bma_groupwise(go_terms1, go_terms2, sim_func):
    if not go_terms1 or not go_terms2:
        return 0.0
    max1 = []
    for a in go_terms1:
        sims = [sim_func(a, b) for b in go_terms2]
        sims = [s for s in sims if s is not None]
        if sims:
            max1.append(max(sims))
    max2 = []
    for b in go_terms2:
        sims = [sim_func(b, a) for a in go_terms1]
        sims = [s for s in sims if s is not None]
        if sims:
            max2.append(max(sims))
    if not max1 and not max2:
        return 0.0
    return (sum(max1) + sum(max2)) / (len(max1) + len(max2))

gene_batch_sizes = [10, 50, 100, 500, 1000]
go3_gene_times = []
goatools_gene_times = []

# Choose a similarity function, e.g., Resnik
def goatools_resnik(a, b):
    return resnik_sim(a, b, go_dag, termcounts)
def goatools_lin(a, b):
    return lin_sim(a, b, go_dag, termcounts)
def goatools_wang(a, b):
    return wang.get_sim(a, b)

sim_funcs = {
    "resnik": goatools_resnik,
    "lin": goatools_lin,
    "wang": goatools_wang,
}

for sim_name, sim_func in sim_funcs.items():
    go3_gene_times = []
    goatools_gene_times = []
    for size in gene_batch_sizes:
        test_gene_pairs = sample_unique_gene_pairs(genes, size)
        # Goatools BMA
        start = time.perf_counter()
        for g1, g2 in test_gene_pairs:
            terms1 = gene2go_map[g1]
            terms2 = gene2go_map[g2]
            if sim_name == 'wang':
                all_goids = set([a for a in terms1] + [b for b in terms2])
                wang = SsWang(all_goids, go_dag, {'part_of'})
            _ = bma_groupwise(terms1, terms2, sim_func)
        goatools_gene_times.append(time.perf_counter() - start)
        # go3 BMA
        start = time.perf_counter()
        _ = go3.compare_gene_pairs_batch(test_gene_pairs, "BP", sim_name, "bma", counter)
        go3_gene_times.append(time.perf_counter() - start)
    gene_results[sim_name] = (goatools_gene_times, go3_gene_times)

# --- Plot gene-to-gene batch ---
plt.figure(figsize=(10, 6))
for sim_name in sim_funcs:
    goatools_gene_times, go3_gene_times = gene_results[sim_name]
    plt.plot(gene_batch_sizes, goatools_gene_times, label=f"Goatools {sim_name}", marker='o', linestyle='--')
for sim_name in sim_funcs:
    goatools_gene_times, go3_gene_times = gene_results[sim_name]
    plt.plot(gene_batch_sizes, go3_gene_times, label=f"go3 {sim_name}", marker='s')
plt.xlabel('Batch Size (# gene pairs)')
plt.ylabel('Time (seconds)')
plt.title('Batch Gene Similarity (BMA): Time vs Batch Size')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("benchmark_gene_batch_similarity.png")
plt.show()