import go3
import pytest

def test_termset_similarity_methods():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)

    # Use some real terms
    # GO:0006397: mRNA processing
    # GO:0008380: RNA splicing
    t1 = ["GO:0006397"] 
    t2 = ["GO:0008380"] 
    
    # Existing methods
    sim_bma = go3.termset_similarity(t1, t2, "lin", "bma", counter)
    sim_max = go3.termset_similarity(t1, t2, "lin", "max", counter)
    
    # New methods
    sim_avg = go3.termset_similarity(t1, t2, "lin", "avg", counter)
    sim_hausdorff = go3.termset_similarity(t1, t2, "lin", "hausdorff", counter)
    sim_simgic = go3.termset_similarity(t1, t2, "lin", "simgic", counter)

    assert 0.0 <= sim_bma <= 1.0
    assert 0.0 <= sim_max <= 1.0
    assert 0.0 <= sim_avg <= 1.0
    assert 0.0 <= sim_hausdorff <= 1.0
    assert 0.0 <= sim_simgic <= 1.0
    
    print(f"BMA: {sim_bma}, MAX: {sim_max}, AVG: {sim_avg}, HAUSDORFF: {sim_hausdorff}, SIMGIC: {sim_simgic}")

def test_simgic_identity():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    
    t1 = ["GO:0006397", "GO:0008380"]
    
    # Simgic of a set with itself should be 1.0
    sim = go3.termset_similarity(t1, t1, "lin", "simgic", counter)
    assert sim == 1.0

def test_simgic_disjoint():
    # Find two terms with no shared info (root IC is 0 usually, but let's check)
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    
    # BP vs MF
    t1 = ["GO:0008150"] # biological_process
    t2 = ["GO:0003674"] # molecular_function
    
    sim = go3.termset_similarity(t1, t2, "lin", "simgic", counter)
    # They share no ancestors except maybe owl:Thing equivalent or nothing if separate trees
    # Usually they are disjoint in standard GO.
    assert sim == 0.0

def test_hausdorff_simple():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)
    
    t1 = ["GO:0006397"]
    t2 = ["GO:0008380"]
    
    # For single terms, hausdorff == pairwise similarity
    sim_h = go3.termset_similarity(t1, t2, "lin", "hausdorff", counter)
    sim_p = go3.semantic_similarity("GO:0006397", "GO:0008380", "lin", counter)
    
    assert abs(sim_h - sim_p) < 1e-6

def test_termset_multiple_terms():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    counter = go3.build_term_counter(gaf)

    # Set 1: mRNA processing, RNA splicing
    t1 = ["GO:0006397", "GO:0008380"]
    # Set 2: RNA splicing, Molecular Function (root)
    # Using a mix of overlapping and disparate terms
    t2 = ["GO:0008380", "GO:0003674"]

    # BMA: linear time, should be average of best matches
    sim_bma = go3.termset_similarity(t1, t2, "lin", "bma", counter)
    
    # Max: Should be 1.0 because GO:0008380 is in both
    sim_max = go3.termset_similarity(t1, t2, "lin", "max", counter)
    assert sim_max == 1.0

    # Avg: Average of all 2x2 = 4 pairs
    sim_avg = go3.termset_similarity(t1, t2, "lin", "avg", counter)
    
    # Simgic: Setwise Jaccard
    sim_simgic = go3.termset_similarity(t1, t2, "lin", "simgic", counter)

    print(f"Multi-term BMA: {sim_bma}")
    print(f"Multi-term MAX: {sim_max}")
    print(f"Multi-term AVG: {sim_avg}")
    print(f"Multi-term SIMGIC: {sim_simgic}")

    assert 0.0 < sim_bma < 1.0  # Should be high but not 1.0 (since not identical sets)
    assert 0.0 < sim_avg < 1.0
    assert 0.0 < sim_simgic < 1.0
