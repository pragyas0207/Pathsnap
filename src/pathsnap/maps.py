from __future__ import annotations
from pathlib import Path
import folium
import numpy as np
import shapely
from .geo import RoadNetwork


def make_map(net: RoadNetwork, prediction, reference=None, classifier=None, expose_reference=False,
             asset_base=None, tiles=True):
    """Reference layer is explicit and only enabled for offline development maps."""
    center = net.trip.track[:, :2].mean(axis=0)[::-1].tolist()
    result = folium.Map(location=center, zoom_start=13, tiles=None, control_scale=True,
                        prefer_canvas=True)
    if tiles:
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap basemap (internet)', show=True).add_to(result)
    if asset_base:
        # The geometry remains usable without internet; only optional basemap tiles are remote.
        result.default_js = [('leaflet', asset_base.rstrip('/') + '/leaflet.js')]
        result.default_css = [('leaflet_css', asset_base.rstrip('/') + '/leaflet.css')]
    points = net.trip.track[:, :2][:, ::-1].tolist()
    gps_layer = folium.FeatureGroup('GPS observations (blue)', show=True).add_to(result)
    # No straight line across known GPS gaps.
    for i, point in enumerate(points):
        folium.CircleMarker(point, radius=2, color='#2563eb', weight=1,
                            tooltip=f'GPS observation {i}').add_to(gps_layer)
    lo, hi = net.gps.min(axis=0) - 120, net.gps.max(axis=0) + 120
    visible = net.tree.query(shapely.box(*lo, *hi))
    roads = folium.FeatureGroup('Road network (grey)', show=True).add_to(result)
    folium.PolyLine(net.latlon_segments(visible), color='#64748b', weight=1, opacity=.45).add_to(roads)
    for name, ids, color, show in [('Classifier arcs (orange)', classifier, '#e76f00', False),
                                   ('HMM matched road arcs (purple)', prediction, '#7c3aed', True)]:
        if ids is not None:
            layer = folium.FeatureGroup(name, show=show).add_to(result)
            selected = np.unique(np.asarray(ids)[np.asarray(ids) >= 0])
            folium.PolyLine(net.latlon_segments(selected), color=color, weight=4, opacity=.8).add_to(layer)
    if expose_reference and reference is not None:
        layer = folium.FeatureGroup('Reference route - development only (green)').add_to(result)
        folium.PolyLine(net.latlon_segments(reference), color='#16a34a', weight=3, opacity=.7).add_to(layer)
    result.fit_bounds([np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()])
    folium.LayerControl(collapsed=False).add_to(result)
    note = ('GPS: blue dots | HMM arcs: purple | classifier: orange | roads: grey. '
            'Selected road arcs are drawn individually; gaps are not fabricated as connecting lines.')
    result.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:12px;right:12px;z-index:9999;background:white;'
        'padding:8px;color:#0f172a;font:12px sans-serif;border-radius:6px">' + note + '</div>'))
    return result
