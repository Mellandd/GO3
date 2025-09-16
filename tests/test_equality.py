from goatools.semantic import TermCounts, resnik_sim, lin_sim, get_info_content
from goatools.obo_parser import GODag
from goatools.associations import dnld_assc

import go3


# -------------------------------
# CONFIG
# -------------------------------

def test_equal_size():    
    terms_go3 = {t.id for t in go3.load_go_terms()}
    terms_goatools = {t for t in GODag("go-basic.obo")}
    assert(len(terms_go3) == len(terms_goatools))

