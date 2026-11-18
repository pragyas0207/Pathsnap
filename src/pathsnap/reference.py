"""Offline/scorer-only reference alignment. Never import into the prediction path."""
from __future__ import annotations
import numpy as np
from .geo import RoadNetwork


def align_reference(net: RoadNetwork, route: np.ndarray, max_distance_m=100.):
    """Monotone minimum-distance DP over occurrences in the supplied route.

    States are route *positions*, so repeated arc IDs retain their order. A state
    can stay or advance arbitrarily. This is weak supervision, not measured
    pointwise ground truth. No candidate set/model predictions enter alignment.
    """
    n, r = len(net.gps), len(route)
    back = np.zeros((n, r), dtype=np.int32)
    costs = np.zeros(r)
    index = np.arange(r)
    for t, point in enumerate(net.gps):
        _, distances, _ = net.projections(point, route)
        emission = np.minimum(distances, 1000.) ** 2
        if t == 0:
            costs = emission
        else:
            minimum = np.minimum.accumulate(costs)
            # Deterministic latest minimizer of each prefix.
            argmin = np.maximum.accumulate(np.where(costs == minimum, index, 0))
            back[t] = argmin
            costs = minimum + emission
        costs -= costs.min()
    positions = np.zeros(n, dtype=int)
    positions[-1] = int(costs.argmin())
    for t in range(n - 1, 0, -1):
        positions[t - 1] = back[t, positions[t]]
    labels = route[positions].copy()
    _, residual, _ = net.projections(net.gps, labels)
    labels[residual > max_distance_m] = -1
    return labels, residual, positions
