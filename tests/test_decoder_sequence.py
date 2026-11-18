"""Sequence tests with independently known answers, before final evaluation."""
import itertools
import numpy as np
from pathsnap.io import Trip
from pathsnap.geo import RoadNetwork
from pathsnap.candidates import Candidates, FEATURES
from pathsnap.decoder import viterbi, DecoderConfig, transition_log_probabilities
from pathsnap.reconstruct import reconstruct


def test_viterbi_chooses_connected_branch_over_alternating_local_winners():
    nodes = np.array([[8, 50], [8.00015, 50], [8.00030, 50],
                      [8, 50.0001], [8.00015, 50.0001], [8.00030, 50.0001]])
    arcs = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    track = np.array([[8.00003, 50, 1], [8.00012, 50, 2],
                      [8.00018, 50, 3], [8.00027, 50, 4]])
    net = RoadNetwork(Trip('00000001', track, nodes, arcs))
    ids = np.array([[0, 2], [0, 2], [1, 3], [1, 3]])
    fraction = np.array([[.2, .2], [.8, .8], [.2, .2], [.8, .8]])
    candidates = Candidates(ids, fraction, np.zeros((4, 2, len(FEATURES))), ids >= 0)
    probabilities = np.array([[.9, .1], [.4, .6], [.9, .1], [.4, .6]])
    output, resets = viterbi(net, candidates, probabilities)
    assert output.tolist() == [0, 0, 1, 1]
    assert not resets
    # Exhaustive path scoring independently verifies the maximum for this tiny case.
    scores = []
    for states in itertools.product(range(2), repeat=4):
        value = np.log(probabilities[0, states[0]])
        for t in range(1, 4):
            transition = transition_log_probabilities(net, ids[t-1], fraction[t-1],
                ids[t], fraction[t], 1., np.linalg.norm(net.gps[t]-net.gps[t-1]), DecoderConfig())
            value += transition[states[t-1], states[t]] + np.log(probabilities[t, states[t]])
        scores.append((value, states))
    best = max(scores)[1]
    assert np.array_equal(output, ids[np.arange(4), best])


def test_reconstruction_preserves_a_loop_returning_to_the_same_arc():
    nodes = np.array([[8, 50], [8.0002, 50], [8.0002, 50.0002]])
    arcs = np.array([[0, 1], [1, 2], [2, 0]])
    track = np.array([[8.00016, 50, 1], [8.00004, 50, 2]])
    net = RoadNetwork(Trip('00000001', track, nodes, arcs))
    assert reconstruct(net, np.array([0, 0])) == [[0, 1, 2, 0]]
