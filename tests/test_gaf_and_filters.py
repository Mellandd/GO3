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
    assert top.study_genes == sorted(study)






