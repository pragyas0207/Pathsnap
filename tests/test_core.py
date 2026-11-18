from pathlib import Path
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from pathsnap.io import Trip, load_trip
from pathsnap.geo import RoadNetwork
from pathsnap.candidates import generate, CandidateConfig, candidate_probabilities, independent_predictions
from pathsnap.reference import align_reference
from pathsnap.decoder import viterbi, DecoderConfig
from pathsnap.metrics import point_metrics
from pathsnap.reconstruct import reconstruct


def simple_net(track=None):
    nodes = np.array([[8., 50.], [8.01, 50.], [8.02, 50.], [8.03, 50.], [8.04, 50.01]])
    arcs = np.array([[0, 1], [1, 2], [2, 3], [4, 3]])
    if track is None:
        track = np.array([[8.005, 50., 1.], [8.015, 50., 2.]])
    return RoadNetwork(Trip('00000001', track, nodes, arcs))


def test_segment_query_finds_long_edge_midpoint():
    net = simple_net()
    c = generate(net, CandidateConfig(radius_m=3, max_candidates=2))
    assert c.ids[0, 0] == 0  # endpoints are hundreds of metres away
    assert c.X[0, 0, 0] < 1


def test_empty_candidates_are_unmatched():
    net = simple_net(np.array([[9., 51., 1.], [9.001, 51., 2.]]))
    c = generate(net, CandidateConfig(radius_m=1))
    model = DummyClassifier(strategy='prior').fit([[0] * 12, [1] * 12], [0, 1])
    probabilities = candidate_probabilities(model, c)
    assert (independent_predictions(c, probabilities) == -1).all()
    decoded, resets = viterbi(net, c, probabilities)
    assert (decoded == -1).all()
    assert len(resets) == 2


def test_candidate_misses_stay_in_accuracy_denominator():
    scores = point_metrics(np.array([3, -1, 5]), np.array([3, 4, 9]), np.array([[3], [-1], [5]]))
    assert scores['top1_accuracy_proxy'] == pytest.approx(1 / 3)
    assert scores['candidate_miss_rate_proxy'] == pytest.approx(2 / 3)


def test_alignment_positions_cannot_run_backwards_at_loops():
    net = simple_net(np.array([[8.015, 50., 1.], [8.005, 50., 2.], [8.025, 50., 3.]]))
    labels, residual, positions = align_reference(net, np.array([0, 1, 2, 0, 1, 2]))
    assert (np.diff(positions) >= 0).all()
    assert set(labels).issubset({-1, 0, 1, 2})


def test_directed_graph_cannot_travel_reverse_without_return_arc():
    net = simple_net()
    distances = net.transition_distances(np.array([2]), np.array([.5]), np.array([0]), np.array([.5]), 10000)
    assert np.isinf(distances[0, 0])


def test_gap_resets_even_if_edges_are_disconnected():
    net = simple_net(np.array([[8.005, 50., 1.], [8.035, 50.005, 100.]]))
    c = generate(net, CandidateConfig(radius_m=50))
    p = c.valid.astype(float); p /= np.maximum(1, p.sum(axis=1, keepdims=True))
    prediction, resets = viterbi(net, c, p)
    assert len(prediction) == 2
    assert any(r['reason'] == 'gps_gap' for r in resets)


def test_reconstruction_fills_intermediate_arcs_without_fake_bridge():
    net = simple_net(np.array([[8.009, 50., 1.], [8.021, 50., 100.]]))
    parts = reconstruct(net, np.array([0, 2]), config=DecoderConfig(gap_seconds=200))
    assert parts == [[0, 1, 2]]
    assert reconstruct(net, np.array([0, 2])) == [[0], [2]]


def test_prediction_loader_never_reads_reference(tmp_path):
    path = tmp_path / '00000001.track'
    np.savetxt(path, [[8., 50., 1.], [8.001, 50., 2.]])
    np.savetxt(path.with_suffix('.nodes'), [[8., 50.], [8.001, 50.]])
    np.savetxt(path.with_suffix('.arcs'), [[0, 1]], fmt='%d')
    path.with_suffix('.route').write_text('THIS IS NOT A VALID REFERENCE')
    assert len(load_trip(path).track) == 2


def test_bad_timestamps_are_rejected(tmp_path):
    path = tmp_path / '00000001.track'
    np.savetxt(path, [[8., 50., 2.], [8.001, 50., 1.]])
    np.savetxt(path.with_suffix('.nodes'), [[8., 50.], [8.001, 50.]])
    np.savetxt(path.with_suffix('.arcs'), [[0, 1]], fmt='%d')
    with pytest.raises(ValueError, match='strictly increasing'):
        load_trip(path)
