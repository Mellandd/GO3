import go3
import math

def test_similarity_resnik_and_lin():
    # Carga de datos
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)

    # Términos GO conocidos (puedes cambiarlos por otros)
    go1 = "GO:0006397"  # mRNA processing
    go2 = "GO:0008380"  # RNA splicing

    # IC no nulo
    ic1 = go3.term_ic(go1, counter)
    ic2 = go3.term_ic(go2, counter)
    assert ic1 > 0
    assert ic2 > 0

    # Resnik
    resnik = go3.semantic_similarity(go1, go2, 'resnik', counter)
    assert resnik > 0

    # Lin
    lin = go3.semantic_similarity(go1, go2, 'lin', counter)
    assert lin > 0 and lin <= 1.0
    
    # Wang
    wang = go3.semantic_similarity(go1, go2, 'wang', counter)
    assert wang > 0 and wang <= 1.0
    

def test_similarity_batch():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)

    list1 = ["GO:0006397", "GO:0008380"]
    list2 = ["GO:0008380", "GO:0006397"]

    resniks = go3.batch_similarity(list1, list2, 'resnik', counter)
    lins = go3.batch_similarity(list1, list2, 'lin', counter)

    assert len(resniks) == 2
    assert len(lins) == 2
    for sim in resniks + lins:
        assert sim >= 0

def test_compare_genes():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    sim = go3.compare_genes("BRCA1", "CASP8", "BP", "resnik", "bma", counter)
    assert sim >= 0.0

def test_compare_genes_batch():
    terms = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    pairs = [("BRCA1", "CASP8"), ("GSDME", "NLRP1")]
    sims = go3.compare_gene_pairs_batch(pairs, "BP", "resnik", "max", counter)
    assert len(sims) == len(pairs)
    assert all(s >= 0.0 for s in sims)