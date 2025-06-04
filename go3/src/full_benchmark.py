import os
import time
import tracemalloc
import go3
from goatools.obo_parser import GODag
from goatools.associations import dnld_assc
from goatools.semantic import TermCounts, resnik_sim, lin_sim, semantic_similarity, get_info_content
from goatools.base import download_go_basic_obo

# --- Configuración
GAF_PATH = "goa_human.gaf"
OBO_PATH = "go-basic.obo"
go_id_1, go_id_2 = "GO:0006397", "GO:0008380"
gene_1, gene_2 = "P12345", "Q9Y6K9"  # IDs comunes en GAF
BATCH_SIZE = 100000

def benchmark(name, func):
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.6f}s")
    return result

def mem_usage(func):
    tracemalloc.start()
    result = func()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, current / 1024 / 1024, peak / 1024 / 1024  # MB

# ========== 🐍 Benchmark: goatools ==========
print("\n🐍 Benchmark: goatools (Python)")
download_go_basic_obo()

# GO DAG
go_dag, mem_cur, mem_peak = mem_usage(lambda: benchmark("Goatools Load GO", lambda: GODag(OBO_PATH)))

# GAF associations
associations = benchmark("Goatools Load GAF + Assoc", lambda: dnld_assc(GAF_PATH, go_dag))

# Term Counts
termcounts = benchmark("Goatools Build Counter", lambda: TermCounts(go_dag, associations))

# Individual GO
benchmark("Goatools Access Term", lambda: go_dag[go_id_1])
benchmark("Goatools IC", lambda: get_info_content(go_id_1, termcounts))
benchmark("Goatools Resnik", lambda: resnik_sim(go_id_1, go_id_2, go_dag, termcounts))
benchmark("Goatools Lin", lambda: lin_sim(go_id_1, go_id_2, go_dag, termcounts))

# Batch
benchmark("Goatools Batch Resnik", lambda: [resnik_sim(go_id_1, go_id_2, go_dag, termcounts) for _ in range(BATCH_SIZE)])
benchmark("Goatools Batch Lin", lambda: [lin_sim(go_id_1, go_id_2, go_dag, termcounts) for _ in range(BATCH_SIZE)])

# # Gene similarity
# gene1_go = associations.get(gene_1, set())
# gene2_go = associations.get(gene_2, set())
# benchmark("Goatools Gene Resnik", lambda: semantic_similarity(gene1_go, gene2_go, go_dag, termcounts, 'resnik'))
# benchmark("Goatools Gene Lin", lambda: semantic_similarity(gene1_go, gene2_go, go_dag, termcounts, 'lin'))

print(f"🧠 Goatools Mem (Current): {mem_cur:.2f} MB | Peak: {mem_peak:.2f} MB")

print("\n🔧 Benchmark: go3 (Rust)")
terms, mem_cur, mem_peak = mem_usage(lambda: benchmark("GO3 Load GO", go3.load_go_terms))
annotations = benchmark("GO3 Load GAF", lambda: go3.load_gaf(GAF_PATH))
counter = benchmark("GO3 Build Counter", lambda: go3.build_term_counter(annotations))

benchmark("GO3 Access Term", lambda: go3.get_term_by_id(go_id_1))
benchmark("GO3 Ancestors", lambda: go3.ancestors(go_id_1))
benchmark("GO3 IC", lambda: go3.term_ic(go_id_1, counter))
benchmark("GO3 Resnik", lambda: go3.resnik_similarity(go_id_1, go_id_2, counter))
benchmark("GO3 Lin", lambda: go3.lin_similarity(go_id_1, go_id_2, counter))

# Batch
pairs_1 = [go_id_1] * BATCH_SIZE
pairs_2 = [go_id_2] * BATCH_SIZE
benchmark("GO3 Batch Resnik", lambda: go3.batch_resnik(pairs_1, pairs_2, counter))
benchmark("GO3 Batch Lin", lambda: go3.batch_lin(pairs_1, pairs_2, counter))

# # Gene similarity
# gene1_terms = go3.annotations_for_gene(gene_1)
# gene2_terms = go3.annotations_for_gene(gene_2)
# benchmark("GO3 Gene Resnik", lambda: go3.groupwise_resnik(gene1_terms, gene2_terms, counter))
# benchmark("GO3 Gene Lin", lambda: go3.groupwise_lin(gene1_terms, gene2_terms, counter))

print(f"🧠 GO3 Mem (Current): {mem_cur:.2f} MB | Peak: {mem_peak:.2f} MB")