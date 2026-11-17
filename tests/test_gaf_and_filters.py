import go3
import pytest


def test_gaf_2_2_full_model_on_real_goa():
    go3.load_go_terms("tests/go-basic.obo")
    annots = go3.load_gaf("tests/goa_human.gaf")
    assert len(annots) > 100000

    first = annots[0]
    assert first.db == "UniProtKB"
    assert isinstance(first.db_object_id, str) and len(first.db_object_id) > 0
    assert isinstance(first.db_object_symbol, str) and len(first.db_object_symbol) > 0
    assert isinstance(first.qualifiers, list) and len(first.qualifiers) >= 1
    assert first.go_id.startswith("GO:")
    assert first.go_term == first.go_id  # backwards-compatibility alias
    assert isinstance(first.references, list) and len(first.references) >= 1
    assert isinstance(first.evidence, str) and len(first.evidence) > 0
    assert isinstance(first.with_from, list)
    assert first.aspect in {"P", "F", "C"}
    assert isinstance(first.taxon, list) and any("taxon:" in t for t in first.taxon)
    assert isinstance(first.date, str) and len(first.date) == 8
    assert isinstance(first.assigned_by, str) and len(first.assigned_by) > 0
    assert isinstance(first.annotation_extensions, list)
    assert isinstance(first.gene_product_form_id, str)
    assert first.is_negated is False


def test_gaf_parse_options_synthetic_and_ic(tmp_path):
    go3.load_go_terms("tests/go-basic.obo")

    # Create a controlled GAF 2.2 fixture including:
    # - normal experimental & IEA annotations
    # - NOT negated annotation
    # - ND root annotation (GO:0008150) and ND non-root annotation (GO:0006397)
    # - Obsolete term annotation (GO:0000108 has replaced_by: GO:0000723 in go-basic.obo)
    lines = [
        "!gaf-version: 2.2",
        "UniProtKB\tP00001\tGENE1\tinvolved_in\tGO:0006397\tPMID:1\tIDA\t\tP\tGene 1\tG1\tprotein\ttaxon:9606\t20260101\tUniProt\tpart_of(GO:0005634)\tUniProtKB:P00001-1",
        "UniProtKB\tP00002\tGENE2\tinvolved_in\tGO:0008380\tPMID:2\tIEA\tInterPro:IPR001\tP\tGene 2\tG2\tprotein\ttaxon:9606|taxon:10090\t20260102\tInterPro\t\t",
        "UniProtKB\tP00003\tGENE3\tNOT|involved_in\tGO:0006397\tPMID:3\tIMP\t\tP\tGene 3\tG3\tprotein\ttaxon:9606\t20260103\tUniProt\t\t",
        "UniProtKB\tP00004\tGENE4\tinvolved_in\tGO:0008150\tGO_REF:0000015\tND\t\tP\tGene 4\tG4\tprotein\ttaxon:9606\t20260104\tGOC\t\t",
        "UniProtKB\tP00005\tGENE5\tinvolved_in\tGO:0006397\tGO_REF:0000015\tND\t\tP\tGene 5\tG5\tprotein\ttaxon:9606\t20260105\tGOC\t\t",
        "UniProtKB\tP00006\tGENE6\tinvolved_in\tGO:0000108\tPMID:6\tIDA\t\tP\tGene 6\tG6\tprotein\ttaxon:9606\t20260106\tUniProt\t\t",
        "UniProtKB\tP00001\tGENE1\tenables\tGO:0005515\tPMID:7\tIPI\tUniProtKB:P00002\tF\tGene 1\tG1\tprotein\ttaxon:9606\t20260107\tIntAct\t\t",
    ]
    gaf_file = tmp_path / "test.gaf"
    gaf_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 1. Default options: excludes NOT, excludes all ND, excludes obsolete
    ann_default = go3.load_gaf(str(gaf_file))
    assert len(ann_default) == 3  # P00001 (BP), P00002 (BP), P00001 (MF)
    assert ann_default[0].annotation_extensions == ["part_of(GO:0005634)"]
    assert ann_default[0].gene_product_form_id == "UniProtKB:P00001-1"
    assert ann_default[1].taxon == ["taxon:9606", "taxon:10090"]

    counter_default = go3.build_term_counter(ann_default)
    ic_default = go3.term_ic("GO:0006397", counter_default)

    # 2. allow_nd_root_only=True: includes ND on root GO:0008150 (P00004), excludes ND on non-root GO:0006397 (P00005)
    opts_nd = go3.GAFParseOptions(allow_nd_root_only=True)
    ann_nd = go3.load_gaf(str(gaf_file), opts_nd)
    assert len(ann_nd) == 4
    nd_objs = [a.db_object_id for a in ann_nd if a.evidence == "ND"]
    assert nd_objs == ["P00004"]
    counter_nd = go3.build_term_counter(ann_nd)
    ic_nd = go3.term_ic("GO:0006397", counter_nd)
    # Adding P00004 increases total BP genes from 2 to 3, so IC of GO:0006397 increases deterministically!
    assert ic_nd > ic_default

    # 3. exclude_evidence={"IEA"}: removes P00002
    opts_no_iea = go3.GAFParseOptions(exclude_evidence={"IEA"})
    ann_no_iea = go3.load_gaf(str(gaf_file), opts_no_iea)
    assert len(ann_no_iea) == 2
    assert all(a.evidence != "IEA" for a in ann_no_iea)

    # 4. include_evidence={"IDA"} and include_aspects={"P"}
    opts_ida_p = go3.GAFParseOptions(include_evidence={"IDA"}, include_aspects={"P"})
    ann_ida_p = go3.load_gaf(str(gaf_file), opts_ida_p)
    assert len(ann_ida_p) == 1
    assert ann_ida_p[0].db_object_id == "P00001"
    assert ann_ida_p[0].aspect == "P"

    # 5. exclude_qualifiers=set(): keeps NOT|involved_in and marks is_negated=True
    opts_keep_not = go3.GAFParseOptions(exclude_qualifiers=set())
    ann_keep_not = go3.load_gaf(str(gaf_file), opts_keep_not)
    negated = [a for a in ann_keep_not if a.is_negated]
    assert len(negated) == 1
    assert negated[0].db_object_id == "P00003"
    assert negated[0].qualifiers == ["NOT", "involved_in"]

    # 6. allow_obsolete_terms=True: resolves GO:0000108 -> GO:0000109 (via consider)
    opts_obs = go3.GAFParseOptions(allow_obsolete_terms=True)
    ann_obs = go3.load_gaf(str(gaf_file), opts_obs)
    p6 = [a for a in ann_obs if a.db_object_id == "P00006"]
    assert len(p6) == 1
    assert p6[0].go_id == "GO:0000109"


def test_gzip_gaf_loading(tmp_path):
    import gzip

    go3.load_go_terms("tests/go-basic.obo")
    lines = [
        "!gaf-version: 2.2",
        "UniProtKB\tP00001\tGENE1\tinvolved_in\tGO:0006397\tPMID:1\tIDA\t\tP\tGene 1\tG1\tprotein\ttaxon:9606\t20260101\tUniProt\t\t",
        "UniProtKB\tP00002\tGENE2\tinvolved_in\tGO:0008380\tPMID:2\tIEA\t\tP\tGene 2\tG2\tprotein\ttaxon:9606\t20260102\tInterPro\t\t",
    ]
    plain_path = tmp_path / "sample.gaf"
    gz_path = tmp_path / "sample.gaf.gz"
    content = "\n".join(lines) + "\n"
    plain_path.write_text(content, encoding="utf-8")
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(content)

    ann_plain = go3.load_gaf(str(plain_path))
    ann_gz = go3.load_gaf(str(gz_path))
    assert len(ann_plain) == len(ann_gz) == 2
    assert [a.go_id for a in ann_plain] == [a.go_id for a in ann_gz]
    c_plain = go3.build_term_counter(ann_plain)
    c_gz = go3.build_term_counter(ann_gz)
    assert go3.term_ic("GO:0006397", c_plain) == go3.term_ic("GO:0006397", c_gz)


def test_gene_term_queries(tmp_path):
    go3.load_go_terms("tests/go-basic.obo")
    lines = [
        "!gaf-version: 2.2",
        "UniProtKB\tP00001\tGENE1\tinvolved_in\tGO:0006397\tPMID:1\tIDA\t\tP\tGene 1\tG1\tprotein\ttaxon:9606\t20260101\tUniProt\t\t",
        "UniProtKB\tP00001\tGENE1\tenables\tGO:0005515\tPMID:2\tIPI\t\tF\tGene 1\tG1\tprotein\ttaxon:9606\t20260101\tUniProt\t\t",
        "UniProtKB\tP00002\tGENE2\tinvolved_in\tGO:0008380\tPMID:3\tIEA\t\tP\tGene 2\tG2\tprotein\ttaxon:9606\t20260102\tInterPro\t\t",
    ]
    gaf_path = tmp_path / "queries.gaf"
    gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    go3.load_gaf(str(gaf_path))

    assert go3.list_genes() == ["GENE1", "GENE2"]
    assert go3.list_genes("MF") == ["GENE1"]
    assert go3.list_genes("CC") == []

    assert go3.get_terms_for_gene("GENE1") == ["GO:0005515", "GO:0006397"]
    assert go3.get_terms_for_gene("GENE1", "BP") == ["GO:0006397"]
    assert go3.get_terms_for_gene("GENE1", "MF") == ["GO:0005515"]

    # GO:0006396 (RNA processing) is an ancestor of both GO:0006397 and GO:0008380
    assert go3.get_genes_for_term("GO:0006397", include_descendants=False) == ["GENE1"]
    assert go3.get_genes_for_term("GO:0006396", include_descendants=False) == []
    assert go3.get_genes_for_term("GO:0006396", include_descendants=True) == ["GENE1", "GENE2"]


def test_descendants_and_relations():
    go3.load_go_terms("tests/go-basic.obo")
    desc_isa = set(go3.descendants("GO:0006396"))
    assert "GO:0006397" in desc_isa
    assert "GO:0008380" in desc_isa
    assert "GO:0006396" in desc_isa

    desc_partof = set(go3.descendants("GO:0006396", relations=["part_of"]))
    assert desc_isa.issubset(desc_partof)
    assert len(desc_partof) >= len(desc_isa)

    anc_isa = set(go3.ancestors("GO:0006397"))
    anc_partof = set(go3.ancestors("GO:0006397", relations=["part_of"]))
    assert anc_isa.issubset(anc_partof)


def test_term_matrices(tmp_path):
    go3.load_go_terms("tests/go-basic.obo")
    lines = [
        "!gaf-version: 2.2",
        "UniProtKB\tP00001\tGENE1\tinvolved_in\tGO:0006397\tPMID:1\tIDA\t\tP\tGene 1\tG1\tprotein\ttaxon:9606\t20260101\tUniProt\t\t",
        "UniProtKB\tP00002\tGENE2\tinvolved_in\tGO:0008380\tPMID:2\tIEA\t\tP\tGene 2\tG2\tprotein\ttaxon:9606\t20260102\tInterPro\t\t",
        "UniProtKB\tP00004\tGENE4\tinvolved_in\tGO:0007049\tPMID:4\tIDA\t\tP\tGene 4\tG4\tprotein\ttaxon:9606\t20260104\tUniProt\t\t",
        "UniProtKB\tP00003\tGENE3\tenables\tGO:0005515\tPMID:3\tIPI\t\tF\tGene 3\tG3\tprotein\ttaxon:9606\t20260103\tUniProt\t\t",
    ]
    gaf_path = tmp_path / "mat.gaf"
    gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ann = go3.load_gaf(str(gaf_path))
    counter = go3.build_term_counter(ann)

    terms = ["GO:0006397", "GO:0008380", "GO:0005515"]
    out_terms, sim_mat = go3.term_similarity_matrix(terms, "lin", counter)
    assert out_terms == terms
    assert len(sim_mat) == 3 and all(len(r) == 3 for r in sim_mat)
    assert abs(sim_mat[0][0] - 1.0) < 1e-9
    assert sim_mat[0][1] > 0.0
    assert sim_mat[0][2] == 0.0  # BP vs MF

    out_terms_d, dist_mat = go3.term_distance_matrix(terms, "lin", counter)
    assert out_terms_d == terms
    for i in range(3):
        assert dist_mat[i][i] == 0.0
        for j in range(3):
            assert abs(dist_mat[i][j] - dist_mat[j][i]) < 1e-9
    assert abs(dist_mat[0][1] - (1.0 - sim_mat[0][1])) < 1e-9


def test_reduce_redundant_terms(tmp_path):
    go3.load_go_terms("tests/go-basic.obo")
    lines = [
        "!gaf-version: 2.2",
        "UniProtKB\tP00001\tGENE1\tinvolved_in\tGO:0000398\tPMID:1\tIDA\t\tP\tGene 1\tG1\tprotein\ttaxon:9606\t20260101\tUniProt\t\t",
        "UniProtKB\tP00002\tGENE2\tinvolved_in\tGO:0006397\tPMID:2\tIEA\t\tP\tGene 2\tG2\tprotein\ttaxon:9606\t20260102\tInterPro\t\t",
        "UniProtKB\tP00003\tGENE3\tinvolved_in\tGO:0007049\tPMID:3\tIDA\t\tP\tGene 3\tG3\tprotein\ttaxon:9606\t20260103\tUniProt\t\t",
    ]
    gaf_path = tmp_path / "red.gaf"
    gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ann = go3.load_gaf(str(gaf_path))
    counter = go3.build_term_counter(ann)

    # GO:0000398 (mRNA splicing, via spliceosome) is a child of GO:0006397 (mRNA processing)
    # and therefore has higher IC and high similarity to GO:0006397, whereas GO:0007049 (cell cycle) is unrelated.
    terms = ["GO:0006397", "GO:0000398", "GO:0007049"]
    reduced = go3.reduce_redundant_terms(terms, threshold=0.5, method="lin", counter=counter)
    assert "GO:0000398" in reduced
    assert "GO:0007049" in reduced
    assert "GO:0006397" not in reduced

    clusters = go3.cluster_go_terms(terms, threshold=0.5, method="lin", counter=counter)
    assert set(clusters["GO:0000398"]) == {"GO:0000398", "GO:0006397"}
    assert clusters["GO:0007049"] == ["GO:0007049"]


def test_go_enrichment(tmp_path):
    go3.load_go_terms("tests/go-basic.obo")
    lines = ["!gaf-version: 2.2"]
    # 5 study genes annotated to GO:0006397 (mRNA processing)
    for i in range(1, 6):
        lines.append(
            f"UniProtKB\tP0000{i}\tSRSF{i}\tinvolved_in\tGO:0006397\tPMID:1\tIDA\t\tP\tGene {i}\tS{i}\tprotein\ttaxon:9606\t20260101\tUniProt\t\t"
        )
    # 20 background genes annotated to GO:0007049 (cell cycle)
    for i in range(6, 26):
        lines.append(
            f"UniProtKB\tP000{i:02d}\tBG{i}\tinvolved_in\tGO:0007049\tPMID:2\tIDA\t\tP\tBG {i}\tB{i}\tprotein\ttaxon:9606\t20260101\tUniProt\t\t"
        )
    gaf_path = tmp_path / "enrich.gaf"
    gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    go3.load_gaf(str(gaf_path))

    study = [f"SRSF{i}" for i in range(1, 6)]
    results = go3.go_enrichment(study, ontology="BP", alpha=0.05)
    assert len(results) > 0
    top = results[0]
    assert top.go_id == "GO:0006397"
    assert top.study_count == 5
    assert top.study_total == 5
    assert top.pop_count == 5
    assert top.pop_total == 25
    assert top.fold_enrichment == 5.0
    assert top.p_value < 1e-4
    assert top.p_fdr_bh < 1e-2
    assert top.p_value <= top.p_scs <= top.p_bonferroni
    assert top.study_genes == sorted(study)
    d = top.to_dict()
    assert d["go_id"] == "GO:0006397"
    assert "p_scs" in d and "p_fdr_bh" in d and "p_bonferroni" in d

    res_scs = go3.go_enrichment(study, ontology="BP", alpha=0.05, method="scs")
    assert len(res_scs) > 0 and res_scs[0].go_id == "GO:0006397"

    res_gscs = go3.go_enrichment(study, ontology="BP", alpha=0.05, method="g:SCS")
    assert len(res_gscs) == len(res_scs)

    res_bonf = go3.go_enrichment(study, ontology="BP", alpha=0.05, method="bonferroni")
    assert len(res_bonf) > 0 and res_bonf[0].go_id == "GO:0006397"
    assert len(res_bonf) <= len(res_scs)

    with pytest.raises(ValueError, match="Unknown multiple-testing correction method"):
        go3.go_enrichment(study, ontology="BP", method="invalid_method")









def _gaf_line(obj_id, symbol, go_id, evidence="IDA", ref="PMID:1", aspect="P"):
    return (
        f"UniProtKB\t{obj_id}\t{symbol}\tinvolved_in\t{go_id}\t{ref}\t{evidence}\t\t{aspect}"
        f"\t{symbol}\t{symbol}\tprotein\ttaxon:9606\t20260101\tUniProt\t\t"
    )


def test_remove_duplicates(tmp_path):
    go3.load_go_terms("tests/go-basic.obo")
    lines = [
        "!gaf-version: 2.2",
        # GENE1 -> GO:0006397 three times (different evidence / references)
        _gaf_line("P00001", "GENE1", "GO:0006397", "IDA", "PMID:1"),
        _gaf_line("P00001", "GENE1", "GO:0006397", "IMP", "PMID:2"),
        _gaf_line("P00001", "GENE1", "GO:0006397", "IEA", "GO_REF:0000002"),
        _gaf_line("P00001", "GENE1", "GO:0007049", "IDA", "PMID:3"),
        _gaf_line("P00002", "GENE2", "GO:0008380", "IDA", "PMID:4"),
        _gaf_line("P00002", "GENE2", "GO:0008380", "IDA", "PMID:5"),
        # Extra gene so the shared ancestor of GENE1/GENE2 terms has IC > 0
        _gaf_line("P00003", "GENE3", "GO:0007049", "IDA", "PMID:6"),
    ]
    gaf_path = tmp_path / "dups.gaf"
    gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert go3.GAFParseOptions().remove_duplicates is False

    ann_all = go3.load_gaf(str(gaf_path))
    assert len(ann_all) == 7
    counter = go3.build_term_counter(ann_all)
    sim_with_dups = go3.compare_genes("GENE1", "GENE2", "BP", "lin", "bma", counter)

    opts = go3.GAFParseOptions(remove_duplicates=True)
    assert "remove_duplicates=true" in repr(opts)
    ann_unique = go3.load_gaf(str(gaf_path), opts)
    assert [(a.db_object_symbol, a.go_id) for a in ann_unique] == [
        ("GENE1", "GO:0006397"),
        ("GENE1", "GO:0007049"),
        ("GENE2", "GO:0008380"),
        ("GENE3", "GO:0007049"),
    ]
    # The first occurrence of each (gene, term) pair is the one kept
    assert ann_unique[0].evidence == "IDA" and ann_unique[0].references == ["PMID:1"]

    # IC does not depend on duplicates (counts are per gene)
    counter_unique = go3.build_term_counter(ann_unique)
    assert counter_unique.ic == counter.ic

    # Gene similarity now weights each GO term once
    sim_unique = go3.compare_genes("GENE1", "GENE2", "BP", "lin", "bma", counter)
    expected = go3.termset_similarity(
        ["GO:0006397", "GO:0007049"], ["GO:0008380"], "lin", "bma", counter
    )
    assert abs(sim_unique - expected) < 1e-12
    assert abs(sim_unique - sim_with_dups) > 1e-6


def test_filter_annotations_by_ic(tmp_path):
    go3.load_go_terms("tests/go-basic.obo")
    lines = [
        "!gaf-version: 2.2",
        _gaf_line("P00001", "GENE1", "GO:0006397"),  # mRNA processing
        _gaf_line("P00002", "GENE2", "GO:0008380"),  # RNA splicing
        _gaf_line("P00003", "GENE3", "GO:0006396"),  # RNA processing (parent of both above)
        _gaf_line("P00003", "GENE3", "GO:0007049"),  # cell cycle
        _gaf_line("P00004", "GENE4", "GO:0008150"),  # biological_process (root)
    ]
    gaf_path = tmp_path / "ic.gaf"
    gaf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ann = go3.load_gaf(str(gaf_path))
    counter = go3.build_term_counter(ann)

    # 4 BP genes: root IC = 0, RNA processing IC = ln(4/3), leaves IC = ln(4)
    assert go3.term_ic("GO:0008150", counter) == 0.0
    assert 0.0 < go3.term_ic("GO:0006396", counter) < 0.5
    assert go3.term_ic("GO:0006397", counter) > 0.5

    # Threshold 0 drops root-only annotations (IC must be strictly greater)
    summary = go3.filter_annotations_by_ic(counter, 0.0)
    assert summary == {
        "annotations_before": 5,
        "annotations_after": 4,
        "genes_before": 4,
        "genes_after": 3,
        "removed_genes": ["GENE4"],
    }
    assert go3.list_genes() == ["GENE1", "GENE2", "GENE3"]
    assert go3.get_terms_for_gene("GENE3") == ["GO:0006396", "GO:0007049"]

    # A higher threshold also removes the generic "RNA processing" term from GENE3.
    # Filters are applied to the original mapping, so they never stack.
    summary = go3.filter_annotations_by_ic(counter, 0.5)
    assert summary["annotations_before"] == 5
    assert summary["annotations_after"] == 3
    assert summary["removed_genes"] == ["GENE4"]
    assert go3.get_terms_for_gene("GENE3") == ["GO:0007049"]
    with pytest.raises(ValueError, match="not found"):
        go3.get_terms_for_gene("GENE4")

    # Lowering the threshold again brings GO:0006396 back
    go3.filter_annotations_by_ic(counter, 0.1)
    assert go3.get_terms_for_gene("GENE3") == ["GO:0006396", "GO:0007049"]

    # None restores the unfiltered mapping
    summary = go3.filter_annotations_by_ic(counter, None)
    assert summary["annotations_after"] == 5 and summary["removed_genes"] == []
    assert go3.list_genes() == ["GENE1", "GENE2", "GENE3", "GENE4"]
    assert go3.get_terms_for_gene("GENE4") == ["GO:0008150"]

    # Downstream analyses see the filtered mapping
    go3.filter_annotations_by_ic(counter, 0.5)
    sim = go3.compare_genes("GENE1", "GENE3", "BP", "lin", "bma", counter)
    expected = go3.termset_similarity(["GO:0006397"], ["GO:0007049"], "lin", "bma", counter)
    assert abs(sim - expected) < 1e-12

    # Reloading the GAF resets the filter
    go3.load_gaf(str(gaf_path))
    assert go3.list_genes() == ["GENE1", "GENE2", "GENE3", "GENE4"]
    summary = go3.filter_annotations_by_ic(counter, 0.0)
    assert summary["annotations_before"] == 5

    with pytest.raises(ValueError, match="NaN"):
        go3.filter_annotations_by_ic(counter, float("nan"))
