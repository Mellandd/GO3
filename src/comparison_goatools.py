from goatools.obo_parser import GODag
from goatools.associations import dnld_assc
from goatools.semantic import TermCounts, resnik_sim, get_info_content
import go3
import numpy as np

# --- Carga común ---
go_dag = GODag("go-basic.obo")
assocs = dnld_assc("goa_human.gaf", go_dag)
tc = TermCounts(go_dag, assocs)
go3.load_go_terms()
annots = go3.load_gaf("goa_human.gaf")
tc_rust = go3.build_term_counter(annots)

# --- Términos a comparar ---
t1 = "GO:0006397"
t2 = "GO:0008380"

# --- Funciones útiles ---
def get_go3_props(go_id):
    term = go3.get_term_by_id(go_id)
    parents = term.parents
    children = term.children
    depth = term.depth
    level = term.level
    ancestors = set(go3.ancestors(go_id))
    ic = go3.term_ic(go_id, tc_rust)
    return parents, children, depth, level, ancestors, ic

def get_gt_props(go_id):
    term = go_dag[go_id]
    parents = [p.id for p in term.parents]
    children = [c.id for c in term.children]
    depth = term.depth
    level = term.level
    ancestors = set(term.get_all_parents())
    ic = get_info_content(go_id, tc)
    return parents, children, depth, level, ancestors, ic

# --- Comparación ---
p1_r, c1_r, d1_r, l1_r, a1_r, ic1_r = get_go3_props(t1)
p1_g, c1_g, d1_g, l1_g, a1_g, ic1_g = get_gt_props(t1)

print(f"\n🧬 {t1}")
print("Parents (go3 vs goatools):", set(p1_r) == set(p1_g), "\n  ", p1_r, "\n  ", p1_g)
print("Children:", set(c1_r) == set(c1_g))
print("Depth:", d1_r, d1_g)
print("Level:", l1_r, l1_g)
print("IC:", ic1_r, ic1_g)
print("Ancestors:", a1_r == a1_g)

# Segundo término
p2_r, c2_r, d2_r, l2_r, a2_r, ic2_r = get_go3_props(t2)
p2_g, c2_g, d2_g, l2_g, a2_g, ic2_g = get_gt_props(t2)

print(f"\n🧬 {t2}")
print("Parents:", set(p2_r) == set(p2_g))
print("Children:", set(c2_r) == set(c2_g))
print("Depth:", d2_r, d2_g)
print("Level:", l2_r, l2_g)
print("IC:", ic2_r, ic2_g)
print("Ancestors:", a2_r == a2_g)

# Comparación de similitud
res_r = go3.resnik_similarity(t1, t2, tc_rust)
res_g = resnik_sim(t1, t2, go_dag, tc)

print(f"\n🔁 Resnik go3: {res_r:.6f}")
print(f"🔁 Resnik goatools: {res_g:.6f}")
print("✅ Igual?" , np.isclose(res_r, res_g, atol=1e-6))