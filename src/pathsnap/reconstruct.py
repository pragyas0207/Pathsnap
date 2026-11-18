"""Recover intermediate directed road arcs, preserving explicit route breaks."""
from __future__ import annotations
import numpy as np
from scipy.sparse.csgraph import dijkstra
from .decoder import DecoderConfig


def reconstruct(net, prediction, reset_observations=(), config=DecoderConfig()):
    n_nodes = len(net.trip.nodes)
    keys = net.trip.arcs[:, 0] * n_nodes + net.trip.arcs[:, 1]
    # Stable ties choose the first ID; parallel straight-endpoint arcs have equal geometry.
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    resets = set(reset_observations)
    parts, current, previous = [], [], None
    fractions = np.zeros(len(prediction))
    valid = np.asarray(prediction) >= 0
    if valid.any():
        fractions[valid], _, _ = net.projections(net.gps[valid], np.asarray(prediction)[valid])
    for t, value in enumerate(prediction):
        value = int(value)
        gap = t > 0 and net.trip.track[t, 2] - net.trip.track[t - 1, 2] > config.gap_seconds
        split = gap or t in resets or value < 0 or previous is None
        bridge = []
        loop = (not split and value == previous and
                (fractions[t] - fractions[t-1]) * net.length[value] < -5.)
        if not split and (value != previous or loop):
            source, target = int(net.trip.arcs[previous, 1]), int(net.trip.arcs[value, 0])
            dt = net.trip.track[t, 2] - net.trip.track[t - 1, 2]
            limit = config.max_speed_mps * dt + config.slack_m
            offsets = (1-fractions[t-1])*net.length[previous] + fractions[t]*net.length[value]
            if offsets > limit:
                split = True
            if source != target:
                distance, parents = dijkstra(net.graph, directed=True, indices=source,
                                             limit=limit, return_predecessors=True)
                if not np.isfinite(distance[target]) or offsets + distance[target] > limit:
                    split = True
                else:
                    cursor = target
                    while cursor != source:
                        parent = int(parents[cursor])
                        if parent < 0:
                            split = True
                            break
                        key = parent * n_nodes + cursor
                        bridge.append(int(order[np.searchsorted(sorted_keys, key)]))
                        cursor = parent
                    bridge.reverse()
        if split:
            if current:
                parts.append(current)
            current = []
            bridge = []
        if value >= 0 and (not current or value != current[-1] or loop):
            current.extend(bridge)
            current.append(value)
        previous = value if value >= 0 else None
    if current:
        parts.append(current)
    return parts
