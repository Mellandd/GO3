import go3
import math
import pytest

# If you get AttributeError, ensure go3 is built and exposes all functions as module-level attributes.
# If not, you may need to recompile or check the Rust #[pymodule] and #[pyfunction] exports.

def test_similarity_all_methods():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)

    go1 = "GO:0006397"  # mRNA processing
    go2 = "GO:0008380"  # RNA splicing
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]

    ic1 = go3.term_ic(go1, counter)
    ic2 = go3.term_ic(go2, counter)
    assert ic1 > 0
    assert ic2 > 0

    for method in methods:
        sim = go3.semantic_similarity(go1, go2, method, counter)
        assert isinstance(sim, float)
        if method in ("lin", "wang", "simrel", "topoicsim"):
            assert 0.0 <= sim <= 1.0 + 1e-9
        else:
            assert sim >= 0.0


def test_similarity_batch_all_methods():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)

    list1 = ["GO:0006397", "GO:0008380"]
    list2 = ["GO:0008380", "GO:0006397"]
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]

    for method in methods:
        sims = go3.batch_similarity(list1, list2, method, counter)
        assert len(sims) == 2
        for sim in sims:
            assert isinstance(sim, float)
            if method in ("lin", "wang", "simrel", "topoicsim"):
                assert 0.0 <= sim <= 1.0 + 1e-9
            else:
                assert sim >= 0.0


def test_compare_genes_all_methods():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    gene1, gene2 = "BRCA1", "CASP8"
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        sim = go3.compare_genes(gene1, gene2, "BP", method, "bma", counter)
        assert isinstance(sim, float)
        if method in ("lin", "wang", "simrel", "topoicsim"):
            assert 0.0 <= sim <= 1.0 + 1e-9
        else:
            assert sim >= 0.0


def test_compare_genes_batch_all_methods():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    pairs = [("BRCA1", "CASP8"), ("GSDME", "NLRP1")]
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        sims = go3.compare_gene_pairs_batch(pairs, "BP", method, "max", counter)
        assert len(sims) == len(pairs)
        for sim in sims:
            assert isinstance(sim, float)
            if method in ("lin", "wang", "simrel", "topoicsim"):
                assert 0.0 <= sim <= 1.0 + 1e-9
            else:
                assert sim >= 0.0
                
                
def test_self_similarity_all_methods():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    go_id = "GO:0006397"
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        self_sim = go3.semantic_similarity(go_id, go_id, method, counter)
        other_sim = go3.semantic_similarity(go_id, "GO:0008380", method, counter)
        assert isinstance(self_sim, float)
        # Self-similarity should be at least as high as similarity to another term
        assert self_sim >= other_sim
        # For normalized methods, should be in [0, 1 + epsilon]
        if method in ("lin", "wang", "simrel", "topoicsim"):
            assert 0.0 <= self_sim <= 1.0 + 1e-9
            
            
def test_unrelated_terms_similarity():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    go1 = "GO:0006397"  # BP
    go2 = "GO:0003674"  # MF
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        sim = go3.semantic_similarity(go1, go2, method, counter)
        assert sim == 0.0
        
def test_invalid_terms():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    invalid = "GO:9999999"
    valid = "GO:0006397"
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        sim = go3.semantic_similarity(invalid, valid, method, counter)
        assert sim == 0.0
        sim = go3.semantic_similarity(valid, invalid, method, counter)
        assert sim == 0.0
        sim = go3.semantic_similarity(invalid, invalid, method, counter)
        assert sim == 0.0
        
def test_batch_vs_single_consistency():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    go1 = "GO:0006397"
    go2 = "GO:0008380"
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        single = go3.semantic_similarity(go1, go2, method, counter)
        batch = go3.batch_similarity([go1], [go2], method, counter)[0]
        assert math.isclose(single, batch, rel_tol=1e-9)
        
def test_gene_not_found():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        with pytest.raises(ValueError):
            go3.compare_genes("FAKEGENE1", "BRCA1", "BP", method, "bma", counter)
        with pytest.raises(ValueError):
            go3.compare_genes("BRCA1", "FAKEGENE2", "BP", method, "bma", counter)
        with pytest.raises(ValueError):
            go3.compare_genes("FAKEGENE1", "FAKEGENE2", "BP", method, "bma", counter)
        
def test_groupwise_strategies():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    gene1, gene2 = "BRCA1", "CASP8"
    methods = ["resnik", "lin", "jc", "simrel", "iccoef", "graphic", "wang", "topoicsim"]
    for method in methods:
        for groupwise in ["bma", "max"]:
            sim = go3.compare_genes(gene1, gene2, "BP", method, groupwise, counter)
            assert isinstance(sim, float)
            assert sim >= 0.0


def test_compare_gene_sets_groupwise_methods():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    groupwise_methods = ["bma", "max", "avg", "hausdorff"]
    for groupwise in groupwise_methods:
        sim = go3.compare_gene_sets(["BRCA1"], ["CASP8"], "BP", "lin", groupwise, counter)
        gene_sim = go3.compare_genes("BRCA1", "CASP8", "BP", "lin", groupwise, counter)
        self_sim = go3.compare_gene_sets(["BRCA1"], ["BRCA1"], "BP", "lin", groupwise, counter)
        assert isinstance(sim, float)
        assert math.isclose(sim, gene_sim, rel_tol=1e-9)
        assert 0.0 <= sim <= 1.0 + 1e-9
        if groupwise in ("bma", "max", "hausdorff"):
            assert math.isclose(self_sim, 1.0, rel_tol=1e-9)


def test_compare_gene_sets_uses_pairwise_gene_bma():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    left = ["BRCA1", "CASP8"]
    right = ["GSDME", "NLRP1"]
    matrix = [
        [go3.compare_genes(g1, g2, "BP", "lin", "bma", counter) for g2 in right]
        for g1 in left
    ]
    expected = (
        sum(max(row) for row in matrix)
        + sum(max(matrix[i][j] for i in range(len(left))) for j in range(len(right)))
    ) / (len(left) + len(right))
    got = go3.compare_gene_sets(left, right, "BP", "lin", "bma", counter)
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_compare_gene_set_profiles_supports_simgic():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    sim = go3.compare_gene_set_profiles(
        ["BRCA1", "CASP8"], ["GSDME", "NLRP1"], "BP", "lin", "simgic", counter
    )
    assert isinstance(sim, float)
    assert 0.0 <= sim <= 1.0 + 1e-9


def test_compare_gene_sets_rejects_simgic_outer_groupwise():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    with pytest.raises(ValueError):
        go3.compare_gene_sets(["BRCA1"], ["CASP8"], "BP", "lin", "simgic", counter)


def test_compare_gene_set_pairs_batch_consistency():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    pairs = [
        (["BRCA1", "CASP8"], ["GSDME", "NLRP1"]),
        (["BRCA1"], ["CASP8", "GSDME"]),
    ]
    batch = go3.compare_gene_set_pairs_batch(pairs, "BP", "lin", "bma", counter)
    expected = [
        go3.compare_gene_sets(left, right, "BP", "lin", "bma", counter)
        for left, right in pairs
    ]
    assert len(batch) == len(expected)
    for got, want in zip(batch, expected):
        assert math.isclose(got, want, rel_tol=1e-9)


def test_gene_distance_matrix():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    genes = ["BRCA1", "CASP8", "GSDME"]
    out_genes, dist = go3.gene_distance_matrix(genes, "BP", "lin", "bma", counter)
    assert out_genes == genes
    n = len(genes)
    assert len(dist) == n
    assert all(len(row) == n for row in dist)
    for i in range(n):
        assert dist[i][i] == 0.0
        for j in range(n):
            assert dist[i][j] >= 0.0
            assert abs(dist[i][j] - dist[j][i]) < 1e-9


def test_gene_set_distance_matrix():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    gene_sets = [
        ("dna_repair", ["BRCA1"]),
        ("death_signaling", ["CASP8", "GSDME"]),
        ("inflammasome", ["NLRP1"]),
    ]
    names, dist = go3.gene_set_distance_matrix(gene_sets, "BP", "lin", "bma", counter)
    assert names == [name for name, _ in gene_sets]
    n = len(gene_sets)
    assert len(dist) == n
    assert all(len(row) == n for row in dist)
    for i in range(n):
        assert dist[i][i] == 0.0
        for j in range(n):
            assert dist[i][j] >= 0.0
            assert abs(dist[i][j] - dist[j][i]) < 1e-9


def test_compare_gene_sets_missing_gene():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    with pytest.raises(ValueError):
        go3.compare_gene_sets(["BRCA1", "FAKEGENE1"], ["CASP8"], "BP", "lin", "bma", counter)
