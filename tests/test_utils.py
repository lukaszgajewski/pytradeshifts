import src.utils as utils
import numpy as np
import networkx as nx


def test_all_equal() -> None:
    random_numbers, all_ones = np.random.random(20), np.ones(20)
    assert utils.all_equal(random_numbers) == False
    assert utils.all_equal(all_ones) == True


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


if __name__ == "__main__":
    test_graph_efficiency()
