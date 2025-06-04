from go3 import load_go_terms, ancestors
from goatools.obo_parser import GODag

def compare_ancestors(go_id):
    load_go_terms()
    go_terms = ancestors(go_id)
    go_terms_set = set(go_terms)

    godag = GODag("go-basic.obo")
    goatools_terms_set = godag[go_id].get_all_parents()

    only_in_go3 = go_terms_set - goatools_terms_set
    only_in_goatools = goatools_terms_set - go_terms_set

    print(f"🔍 Comparando ancestros de {go_id}:")
    print(f"✅ Total comunes: {len(go_terms_set & goatools_terms_set)}")
    if only_in_go3:
        print(f"⚠️ Solo en go3: {sorted(only_in_go3)}")
    if only_in_goatools:
        print(f"⚠️ Solo en Goatools: {sorted(only_in_goatools)}")
    if not only_in_go3 and not only_in_goatools:
        print("✔️ ¡Ancestros coinciden completamente!")

# Ejemplo de prueba
compare_ancestors("GO:0008380")