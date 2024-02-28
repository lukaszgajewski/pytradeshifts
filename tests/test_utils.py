from numpy.random import sample
import src.utils as utils
import numpy as np
import networkx as nx
import pytest
import pandas as pd


def test_all_equal() -> None:
    random_numbers, all_ones = np.random.random(20), np.ones(20)
    assert utils.all_equal(random_numbers) is False
    assert utils.all_equal(all_ones) is True


def test_jaccard_index() -> None:
    iterable_a = "konstantynopolitanczykowianeczka"
    iterable_b = "xqr"
    iterable_c = "xqrz"
    assert np.isclose(utils.jaccard_index(iterable_a, iterable_a), 1.0)
    assert np.isclose(utils.jaccard_index(iterable_a, iterable_b), 0.0)
    assert np.isclose(utils.jaccard_index(iterable_b, iterable_c), 3 / 4)


def test_get_right_stochastic_matrix() -> None:
    G = nx.erdos_renyi_graph(25, p=0.5)
    RSM = utils.get_right_stochastic_matrix(G)
    assert np.isclose(RSM.sum(axis=1).sum(), 25)


def test_get_stationary_probability_vector() -> None:
    RMS = np.array([[0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [0, 1 / 2, 1 / 2]])
    SPV = utils.get_stationary_probability_vector(RMS)
    for _ in range(100):
        RMS = RMS @ RMS
    assert np.isclose(RMS, SPV).all()

    RMS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    SPV = utils.get_stationary_probability_vector(RMS)
    assert np.isclose(np.real(SPV), 1 / 3).all()


def test_entropy_rate() -> None:
    RMS = np.array([[0, 1 / 2, 1 / 2], [1 / 2, 0, 1 / 2], [0, 1 / 2, 1 / 2]])
    SPV = utils.get_stationary_probability_vector(RMS)
    assert utils._compute_entropy_rate(RMS, SPV) > 0

    RMS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    SPV = utils.get_stationary_probability_vector(RMS)
    assert np.real(utils._compute_entropy_rate(RMS, SPV)) == 0


def test_graph_efficiency() -> None:
    adj_mat = np.random.random((100, 100))
    G = nx.DiGraph(adj_mat)
    weak = utils.get_graph_efficiency(G, "weak")
    strong = utils.get_graph_efficiency(G, "strong")
    none = utils.get_graph_efficiency(G, None)
    wrong = utils.get_graph_efficiency(G, "jibjib")
    # normalised should all be in range [0, 1]
    assert weak >= 0 and weak <= 1
    assert strong >= 0 and weak <= 1
    # unnormalised should be biggest
    assert none >= weak and none >= strong
    # when wrong norm param is given it should default to weak
    assert wrong == weak


@pytest.fixture
def sample_graph() -> nx.Graph:
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=3)
    G.add_edge(1, 2, weight=1)
    G.add_edge(0, 3, weight=2)
    return G


def test_degree_centrality(sample_graph) -> None:
    outs = utils.get_degree_centrality(sample_graph, out=True)
    ins = utils.get_degree_centrality(sample_graph, out=False)
    assert np.isclose(outs[0], 5 / 6)
    assert np.isclose(outs[1], 1 / 6)
    assert np.isclose(ins[1], 3 / 6)
    assert np.isclose(ins[2], 1 / 6)
    assert np.isclose(ins[3], 2 / 6)


def test_entropic_degree(sample_graph) -> None:
    outs = utils.get_entropic_degree(sample_graph, out=True)
    ins = utils.get_entropic_degree(sample_graph, out=False)
    assert outs[1] == 1
    assert ins[1] == 3


def test_dict_min_max() -> None:
    d = {0: 1, 1: 2, 3: 4}
    mink, minv, maxk, maxv = utils.get_dict_min_max(d)
    assert mink == 0
    assert maxk == 3
    assert minv == 1
    assert maxv == 4


def test_stability_index() -> None:
    r = utils.get_stability_index()
    assert isinstance(r, dict)


def test_distance_matrix() -> None:
    d = utils.get_distance_matrix(
        pd.Index(["China", "Ukraine", "Australia"]),
        pd.Index(["China", "Ukraine", "Australia"]),
    )
    assert d is not None
    assert d.values.shape == (3, 3)
    # all non diagonal elements should be non-zero
    assert np.all(d.values[~np.eye(3, dtype=bool)] != 0)
    d = utils.get_distance_matrix(
        pd.Index(["Narnia", "Ukraine", "Australia"]),
        pd.Index(["Narnia", "Ukraine", "Australia"]),
    )
    assert d is None


@pytest.fixture
def percolated_sample_graph(sample_graph) -> nx.Graph:
    sample_graph.add_edge(2, 3)
    sample_graph.add_edge(3, 0)
    return sample_graph


def test_percolation_eigenvalue(percolated_sample_graph) -> None:
    eigv = utils.get_percolation_eigenvalue(
        nx.to_numpy_array(percolated_sample_graph), np.array([1, 0, 0, 0])
    )
    assert eigv == 0
    eigv = utils.get_percolation_eigenvalue(
        nx.to_numpy_array(percolated_sample_graph), np.array([0, 0, 0, 0])
    )
    assert eigv > 1


def test_percolation_threshold(percolated_sample_graph) -> None:
    t, rn, eigs = utils.get_percolation_threshold(
        nx.to_numpy_array(percolated_sample_graph),
        list(range(len(percolated_sample_graph))),
    )
    assert t == 1
    assert rn == [1, 2, 3, 4]
    assert not all(eigs)
