# test_go.py
import go3

# --- Test carga GO ---
print("🌀 Cargando ontología GO...")
terms = go3.load_go_terms()
print(f"✔️ Se cargaron {len(terms)} términos GO")

# # Elegimos un término para probar (asegúrate de que existe)
go_id = "GO:0090728"  # biological_process

print(f"\n📘 Probando funciones con el término {go_id}...\n")

term = go3.get_term_by_id(go_id)
if term:
    print("➡️ Término encontrado:")
    print(term)
else:
    print("❌ Término no encontrado.")

ancs = go3.ancestors(go_id)
print(f"\n🔗 Ancestros de {go_id} ({len(ancs)}):")
print(f"{go_id} depth: {term.depth}")
print(f"{go_id} level: {term.level}")
print(ancs[:10])  # Solo mostramos 10 para abreviar

common = go3.common_ancestor("GO:0045852", "GO:0090728")
print(f"\n🤝 Ancestros comunes entre GO:0045852 y GO:0090728:")
print(common)

# --- Test carga GAF ---
import os
import gzip
import requests

GAF_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
GAF_LOCAL_GZ = "goa_human.gaf.gz"
GAF_LOCAL_TXT = "goa_human.gaf"

if not os.path.exists(GAF_LOCAL_TXT):
    print(f"\n⬇️ Descargando archivo GAF desde:\n{GAF_URL}")
    r = requests.get(GAF_URL)
    with open(GAF_LOCAL_GZ, "wb") as f:
        f.write(r.content)

    print(f"🗜️ Descomprimiendo...")
    with gzip.open(GAF_LOCAL_GZ, "rt") as f_in, open(GAF_LOCAL_TXT, "w") as f_out:
        f_out.write(f_in.read())

print("\n📂 Cargando anotaciones GAF...")
annotations = go3.load_gaf(GAF_LOCAL_TXT)
print(f"✔️ {len(annotations)} anotaciones cargadas.")

print("\n🔍 Primeras 5 anotaciones:")
for ann in annotations[:5]:
    print(f"🧬 {ann.db_object_id} - {ann.go_term} ({ann.evidence})")
    
counter = go3.build_term_counter(annotations)

print("IC GO:0006397 =", go3.term_ic("GO:0006397", counter))
print("Resnik:", go3.resnik_similarity("GO:0006397", "GO:0008380", counter))
print("Lin:", go3.lin_similarity("GO:0006397", "GO:0008380", counter))