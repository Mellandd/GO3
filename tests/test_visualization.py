import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import go3


GENES = ["BRCA1", "CASP8", "GSDME", "NLRP1"]


@pytest.fixture(scope="module")
def counter():
    _ = go3.load_go_terms()
    gaf = go3.load_gaf("tests/goa_human.gaf")
    return go3.build_term_counter(gaf)


def test_gene_distance_matrix_small(counter):
    genes, dist = go3.gene_distance_matrix(GENES, "BP", "lin", "bma", counter)
    assert genes == GENES
    n = len(genes)
    assert len(dist) == n
    assert all(len(row) == n for row in dist)
    for i in range(n):
        assert dist[i][i] == 0.0
        for j in range(n):
            assert dist[i][j] >= 0.0
            assert abs(dist[i][j] - dist[j][i]) < 1e-9
    flat = [v for row in dist for v in row]
    assert max(flat) <= 1.0 + 1e-9


def test_tsne_genes_shape(counter):
    pytest.importorskip("sklearn")
    np = pytest.importorskip("numpy")
    genes, embedding = go3.tsne_genes(
        GENES,
        "BP",
        "lin",
        "bma",
        counter,
        perplexity=2.0,
        n_iter=250,
        random_state=0,
    )
    assert genes == GENES
    emb = np.asarray(embedding)
    assert emb.shape == (len(GENES), 2)


def test_umap_genes_shape(counter):
    pytest.importorskip("umap")
    np = pytest.importorskip("numpy")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="using precomputed metric; inverse_transform will be unavailable",
        )
        genes, embedding = go3.umap_genes(
            GENES,
            "BP",
            "lin",
            "bma",
            counter,
            n_neighbors=2,
            min_dist=0.1,
            random_state=0,
        )
    assert genes == GENES
    emb = np.asarray(embedding)
    assert emb.shape == (len(GENES), 2)


def test_plot_embedding_basic():
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    emb = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
    labels = ["A", "A", "B", "B"]
    fig, ax = go3.plot_embedding(emb, genes=GENES, labels=labels, annotate="all", title="demo")
    assert hasattr(fig, "savefig")
    assert hasattr(ax, "scatter")
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_tsne_genes(counter):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    genes, emb, fig, ax = go3.plot_tsne_genes(
        GENES,
        "BP",
        "lin",
        "bma",
        counter,
        perplexity=2.0,
        n_iter=250,
        random_state=0,
        annotate="none",
        title="tsne",
    )
    assert genes == GENES
    emb = np.asarray(emb)
    assert emb.shape == (len(GENES), 2)
    assert hasattr(fig, "savefig")
    assert hasattr(ax, "scatter")
    import matplotlib.pyplot as plt
    plt.close(fig)
