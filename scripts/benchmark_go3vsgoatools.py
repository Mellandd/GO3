import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import go3
import goatools
from goatools.obo_parser import GODag
from goatools.associations import dnld_assc
from goatools.semantic import TermCounts, resnik_sim, lin_sim
import os

sns.set_theme(style="whitegrid")

OBO_FILE = "go-basic.obo"
GAF_FILE = "goa_human.gaf"

def measure_time_and_memory(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_used = mem_after - mem_before
    return result, elapsed, max(mem_used, 0)

def benchmark_goatools():
    print("Running Goatools Benchmark...")
    data = {}

    godag, t1, m1 = measure_time_and_memory(GODag, OBO_FILE)
    reader, t2, m2 = measure_time_and_memory(lambda: dnld_assc(GAF_FILE, godag))
    termcounts, t3, m3 = measure_time_and_memory(lambda: TermCounts(godag, reader))

    # Single similarity
    _, t4, m4 = measure_time_and_memory(lambda: resnik_sim('GO:0008150', 'GO:0009987', godag, termcounts))
    _, t5, m5 = measure_time_and_memory(lambda: lin_sim('GO:0008150', 'GO:0009987', godag, termcounts))

    # Batch similarity (100 pairs)
    pairs = [('GO:0008150', 'GO:0009987')] * 1000000
    start = time.perf_counter()
    for a, b in pairs:
        resnik_sim(a, b, godag, termcounts)
    t6 = time.perf_counter() - start

    data['Goatools'] = {
        'Load Ontology': (t1, m1),
        'Load Annotations': (t2, m2),
        'Build Counter': (t3, m3),
        'Resnik': (t4, m4),
        'Lin': (t5, m5),
        'Batch 100 Resnik': (t6, 0),  # Memory not measured here
    }

    return data

def benchmark_go3():
    print("Running GO3 Benchmark...")
    data = {}

    terms, t1, m1 = measure_time_and_memory(go3.load_go_terms)
    gaf, t2, m2 = measure_time_and_memory(go3.load_gaf, GAF_FILE)
    counter, t3, m3 = measure_time_and_memory(go3.build_term_counter, gaf)

    # Single similarity
    _, t4, m4 = measure_time_and_memory(go3.semantic_similarity, 'GO:0008150', 'GO:0009987', 'resnik', counter)
    _, t5, m5 = measure_time_and_memory(go3.semantic_similarity, 'GO:0008150', 'GO:0009987', 'resnik', counter)

    # Batch similarity (100 pairs)
    lista1 = ['GO:0008150'] * 1000000
    lista2 = ['GO:0009987'] * 1000000
    _, t6, m6 = measure_time_and_memory(go3.batch_similarity, lista1, lista2, 'resnik', counter)

    data['GO3'] = {
        'Load Ontology': (t1, m1),
        'Load Annotations': (t2, m2),
        'Build Counter': (t3, m3),
        'Resnik': (t4, m4),
        'Lin': (t5, m5),
        'Batch 100 Resnik': (t6, 0),
    }

    return data

def plot_results(results):
    # Sum up time and memory for each tool
    totals = {}
    for tool, metrics in results.items():
        total_mem = max([metrics[step][1] for step in ['Load Ontology', 'Load Annotations', 'Build Counter']])
        total_time = sum([metrics[step][0] for step in ['Load Ontology', 'Load Annotations', 'Build Counter']])
        totals[tool] = {'Time (s)': total_time, 'Memory (MB)': total_mem}

    # Ensure order: go3, goatools
    tools = ['GO3', 'Goatools']
    time_vals = [totals[tool]['Time (s)'] for tool in tools]
    mem_vals = [totals[tool]['Memory (MB)'] for tool in tools]

    # X-axis: two groups: Time, Memory
    groups = ['Time', 'Memory']
    x = np.arange(len(groups))  # [0, 1]
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot time bars (left y-axis)
    bars_time = ax1.bar(x[0] - width/2, time_vals[0], width, label='go3', color='tab:blue')
    bars_time2 = ax1.bar(x[0] + width/2, time_vals[1], width, label='goatools', color='tab:orange')
    ax1.set_ylabel('Total Time (s)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_xlabel('Metric')
    ax1.set_title('GO3 vs Goatools: Total Loading Time and Peak Memory')

    # Annotate time bars
    for bar in [bars_time, bars_time2]:
        for rect in bar:
            height = rect.get_height()
            ax1.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='tab:blue', fontsize=10)

    # Plot memory bars (right y-axis)
    ax2 = ax1.twinx()
    bars_mem = ax2.bar(x[1] - width/2, mem_vals[0], width, label='go3', color='tab:blue')
    bars_mem2 = ax2.bar(x[1] + width/2, mem_vals[1], width, label='goatools', color='tab:orange')
    ax2.set_ylabel('Total Memory (MB)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Annotate memory bars
    for bar in [bars_mem, bars_mem2]:
        for rect in bar:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', color='tab:orange', fontsize=10)

    # Custom legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color='tab:blue', label='go3'),
        Patch(color='tab:orange', label='goatools')
    ]
    ax1.legend(handles=legend_patches, loc='upper left')

    plt.tight_layout()
    plt.savefig("benchmark_loading_time_memory.png")
    plt.show()

if __name__ == "__main__":
    goatools_data = benchmark_goatools()
    go3_data = benchmark_go3()
    combined_results = {**goatools_data, **go3_data}
    plot_results(combined_results)