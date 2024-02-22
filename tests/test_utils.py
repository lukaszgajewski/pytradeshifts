import src.utils as utils
import numpy as np
import networkx as nx


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
