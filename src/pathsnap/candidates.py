from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import shapely
from .geo import RoadNetwork

FEATURES = ['distance_m', 'heading_cos', 'heading_known', 'log_length_m',
            'fraction', 'speed_mps', 'dt_prev_s', 'gps_step_m', 'distance_rank',
            'candidate_count', 'from_outdegree', 'to_outdegree']


@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 60.0
    max_candidates: int = 8
    gap_seconds: float = 30.0


@dataclass
class Candidates:
    ids: np.ndarray
    fraction: np.ndarray
    X: np.ndarray
    valid: np.ndarray


def generate(net: RoadNetwork, config: CandidateConfig) -> Candidates:
    """Exact distance-to-segment radius query; no reference information is used."""
    n, k = len(net.gps), config.max_candidates
    if k < 1 or config.radius_m <= 0:
        raise ValueError('Positive radius and candidate count required')
    pairs = net.tree.query(shapely.points(net.gps), predicate='dwithin', distance=config.radius_m)
    obs, arc = pairs
    fraction, distance, _ = net.projections(net.gps[obs], arc)
    order = np.lexsort((arc, distance, obs))
    obs, arc, fraction, distance = [x[order] for x in (obs, arc, fraction, distance)]
    counts = np.bincount(obs, minlength=n)
    starts = np.cumsum(counts) - counts
    rank = np.arange(len(obs)) - np.repeat(starts, counts)
    keep = rank < k
    obs, arc, fraction, distance, rank = [x[keep] for x in (obs, arc, fraction, distance, rank)]
    ids = np.full((n, k), -1, dtype=np.int64)
    fractions = np.zeros((n, k), dtype=np.float32)
    X = np.zeros((n, k, len(FEATURES)), dtype=np.float32)
    ids[obs, rank], fractions[obs, rank] = arc, fraction
    step_vector = np.zeros_like(net.gps)
    step_vector[1:] = np.diff(net.gps, axis=0)
    dt = np.r_[1., np.diff(net.trip.track[:, 2])]
    step = np.linalg.norm(step_vector, axis=1)
    speed = step / dt
    known = (step >= 3.) & (dt <= config.gap_seconds)
    dot = np.sum(step_vector[obs] * net.vector[arc], axis=1)
    cosine = dot / np.maximum(step[obs] * net.length[arc], 1e-12)
    cosine[~known[obs]] = 0.
    X[obs, rank] = np.column_stack([
        distance, np.clip(cosine, -1, 1), known[obs], np.log1p(net.length[arc]),
        fraction, np.minimum(speed[obs], 100), np.minimum(dt[obs], 300),
        np.minimum(step[obs], 10000), rank, np.minimum(counts[obs], 100),
        net.outdegree[net.trip.arcs[arc, 0]], net.outdegree[net.trip.arcs[arc, 1]]])
    return Candidates(ids, fractions, X, ids >= 0)


def candidate_probabilities(model, candidates: Candidates):
    scores = np.zeros(candidates.ids.shape)
    if candidates.valid.any():
        scores[candidates.valid] = model.predict_proba(candidates.X[candidates.valid])[:, 1]
    scores[candidates.valid] = np.maximum(scores[candidates.valid], 1e-9)
    return scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-9)


def independent_predictions(candidates: Candidates, probabilities):
    result = candidates.ids[np.arange(len(candidates.ids)), probabilities.argmax(axis=1)].copy()
    result[~candidates.valid.any(axis=1)] = -1
    return result
