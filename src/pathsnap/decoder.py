from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .candidates import Candidates
from .geo import RoadNetwork


@dataclass(frozen=True)
class DecoderConfig:
    beta_m: float = 30.
    max_speed_mps: float = 60.
    slack_m: float = 30.
    gap_seconds: float = 30.


def transition_log_probabilities(net, a, fa, b, fb, dt, displacement, config):
    limit = config.max_speed_mps * dt + config.slack_m
    distances = net.transition_distances(a, fa, b, fb, limit)
    # Permit up to 5 m of projection jitter while staying on a directed edge.
    for i, arc in enumerate(a):
        same = (b == arc) & ((fb - fa[i]) * net.length[arc] >= -5)
        distances[i, same] = np.maximum(0, (fb[same] - fa[i]) * net.length[arc])
    valid = np.isfinite(distances) & (distances <= limit)
    logp = np.full(distances.shape, -np.inf)
    logp[valid] = -np.abs(distances[valid] - displacement) / config.beta_m
    # Each source state's transition probabilities sum to one over feasible targets.
    for row in logp:
        mask = np.isfinite(row)
        if mask.any():
            peak = row[mask].max()
            row[mask] -= peak + np.log(np.exp(row[mask] - peak).sum())
    return logp


def viterbi(net: RoadNetwork, c: Candidates, probabilities, config=DecoderConfig()):
    """Log-space Viterbi on variable candidate sets with explicit segment resets."""
    if config.beta_m <= 0:
        raise ValueError('beta_m must be positive')
    n, k = c.ids.shape
    output = np.full(n, -1, dtype=np.int64)
    back = np.full((n, k), -1, dtype=np.int32)
    scores = np.full(k, -np.inf)
    start = None
    resets = []

    def finish(end):
        if start is None:
            return
        state = int(scores.argmax())
        for t in range(end, start - 1, -1):
            output[t] = c.ids[t, state]
            if t > start:
                state = int(back[t, state])

    for t in range(n):
        current = np.flatnonzero(c.valid[t])
        if len(current) == 0:
            finish(t - 1)
            start = None
            scores[:] = -np.inf
            resets.append({'observation': t, 'reason': 'no_candidates'})
            continue
        emission = np.log(np.maximum(probabilities[t, current], 1e-12))
        dt = 0 if t == 0 else net.trip.track[t, 2] - net.trip.track[t - 1, 2]
        reset = start is None or dt > config.gap_seconds
        if reset:
            if start is not None:
                finish(t - 1)
                resets.append({'observation': t, 'reason': 'gps_gap'})
            start = t
            scores[:] = -np.inf
            scores[current] = emission
            continue
        previous = np.flatnonzero(c.valid[t - 1])
        displacement = float(np.linalg.norm(net.gps[t] - net.gps[t - 1]))
        transitions = transition_log_probabilities(net, c.ids[t - 1, previous],
            c.fraction[t - 1, previous], c.ids[t, current], c.fraction[t, current],
            dt, displacement, config)
        combined = scores[previous, None] + transitions
        parents = combined.argmax(axis=0)
        new_scores = combined[parents, np.arange(len(current))] + emission
        if not np.isfinite(new_scores).any():
            finish(t - 1)
            start = t
            new_scores = emission
            resets.append({'observation': t, 'reason': 'no_feasible_transition'})
        else:
            back[t, current] = previous[parents]
        scores[:] = -np.inf
        scores[current] = new_scores - new_scores.max()
    finish(n - 1)
    return output, resets
