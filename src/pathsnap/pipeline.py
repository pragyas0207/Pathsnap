from __future__ import annotations
from dataclasses import asdict
from time import perf_counter
from threadpoolctl import threadpool_limits
from .geo import RoadNetwork
from .candidates import generate, CandidateConfig, candidate_probabilities, independent_predictions
from .decoder import viterbi, DecoderConfig
from .io import Trip
from .reconstruct import reconstruct


def predict(trip: Trip, bundle: dict):
    """Reference-blind inference. The caller supplies GPS, nodes and arcs only."""
    start = perf_counter()
    net = RoadNetwork(trip)
    c = generate(net, CandidateConfig(**bundle['candidate_config']))
    features_seconds = perf_counter() - start
    start = perf_counter()
    with threadpool_limits(limits=1):
        probability = candidate_probabilities(bundle['model'], c)
    raw = independent_predictions(c, probability)
    classifier_seconds = perf_counter() - start
    start = perf_counter()
    hmm, resets = viterbi(net, c, probability, DecoderConfig(**bundle['decoder_config']))
    route_parts = reconstruct(net, hmm, [r['observation'] for r in resets], DecoderConfig(**bundle['decoder_config']))
    decoder_seconds = perf_counter() - start
    timing = {'features_seconds': features_seconds, 'classifier_seconds': classifier_seconds,
              'decoder_seconds': decoder_seconds,
              'prediction_seconds': features_seconds + classifier_seconds + decoder_seconds}
    return net, c, raw, hmm, route_parts, resets, timing
