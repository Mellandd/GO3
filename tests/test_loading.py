import go3

def test_load_go_terms():
    terms = go3.load_go_terms()
    assert isinstance(terms, list)
    assert len(terms) > 10000  # Número típico de términos GO
    assert any(t.id.startswith("GO:") for t in terms)

def test_load_gaf():
    gaf = go3.load_gaf("tests/goa_human.gaf")  # Debe estar descargado
    assert isinstance(gaf, list)
    assert len(gaf) > 100000  # Número típico de anotaciones
    assert hasattr(gaf[0], "db_object_id")
    assert gaf[0].go_term.startswith("GO:")