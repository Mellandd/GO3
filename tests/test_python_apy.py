def test_similarity():
    import go3
    go3.load_go_terms()
    counter = go3.build_term_counter(go3.load_gaf("goa_human.gaf"))
    t1 = "GO:0006397"
    t2 = "GO:0008380"
    sim = go3.resnik_similarity(t1, t2, counter)
    assert 0.0 <= sim