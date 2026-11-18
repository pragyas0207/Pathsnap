from __future__ import annotations
import numpy as np
from pyproj import Transformer
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .io import Trip


class RoadNetwork:
    def __init__(self, trip: Trip):
        self.trip = trip
        lon, lat = np.median(trip.track[:, :2], axis=0)
        if not -80 < lat < 84:
            raise ValueError('UTM requires a latitude between -80 and 84 degrees')
        zone = min(60, max(1, int((lon + 180) // 6) + 1))
        self.epsg = (32600 if lat >= 0 else 32700) + zone
        self.forward = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self.inverse = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        self.xy = np.column_stack(self.forward.transform(*trip.nodes.T))
        self.gps = np.column_stack(self.forward.transform(*trip.track[:, :2].T))
        self.start = self.xy[trip.arcs[:, 0]]
        self.vector = self.xy[trip.arcs[:, 1]] - self.start
        self.length = np.linalg.norm(self.vector, axis=1)
        self.lines = shapely.linestrings(np.stack([self.start, self.start + self.vector], axis=1))
        self.tree = shapely.STRtree(self.lines)
        self.outdegree = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
        # Collapse duplicate directed endpoints by minimum length, never by summation.
        pairs, indices = np.unique(trip.arcs, axis=0, return_index=True)
        self.graph = coo_matrix((np.maximum(self.length[indices], 1e-6),
                                 (pairs[:, 0], pairs[:, 1])),
                                shape=(len(trip.nodes), len(trip.nodes))).tocsr()
        self._shortest_cache = {}

    def projections(self, points, arc_ids):
        delta = points - self.start[arc_ids]
        fraction = np.clip(np.sum(delta * self.vector[arc_ids], axis=-1) /
                           np.maximum(self.length[arc_ids] ** 2, 1e-12), 0, 1)
        projected = self.start[arc_ids] + fraction[..., None] * self.vector[arc_ids]
        distance = np.linalg.norm(points - projected, axis=-1)
        return fraction, distance, projected

    def node_distances(self, source: int, limit: float):
        # Bounded Dijkstra. Quantized limits allow reuse, with a bounded cache.
        bound = float(max(100, np.ceil(limit / 100) * 100))
        key = (int(source), bound)
        if key not in self._shortest_cache:
            if len(self._shortest_cache) >= 128:
                self._shortest_cache.clear()
            self._shortest_cache[key] = dijkstra(self.graph, directed=True, indices=int(source), limit=bound)
        return self._shortest_cache[key]

    def transition_distances(self, prev_ids, prev_fraction, ids, fraction, limit):
        result = np.full((len(prev_ids), len(ids)), np.inf)
        arcs = self.trip.arcs
        for i, (a, fa) in enumerate(zip(prev_ids, prev_fraction)):
            nodes = self.node_distances(arcs[a, 1], limit)
            result[i] = ((1 - fa) * self.length[a] + nodes[arcs[ids, 0]] +
                         fraction * self.length[ids])
            # Same directed segment can continue forward without traversing a loop.
            same = (ids == a) & (fraction >= fa - 1e-8)
            result[i, same] = (fraction[same] - fa) * self.length[a]
        return result

    def latlon_segments(self, ids):
        return [self.trip.nodes[self.trip.arcs[int(a)]][:, ::-1].tolist()
                for a in ids if a >= 0]
