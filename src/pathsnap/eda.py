from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from .io import trip_files, load_trip, load_reference, digest
from .geo import RoadNetwork


def explore(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows, intervals = [], {}
    for partition in ('train', 'student_test'):
        intervals[partition] = []
        for path in trip_files(root, partition):
            trip = load_trip(path)
            net = RoadNetwork(trip)
            route = load_reference(path, len(trip.arcs))
            dt = np.diff(trip.track[:, 2])
            intervals[partition].extend(dt.tolist())
            degree = np.bincount(trip.arcs.ravel(), minlength=len(trip.nodes))
            weak, _ = connected_components(net.graph, directed=True, connection='weak')
            strong, _ = connected_components(net.graph, directed=True, connection='strong')
            row = {'partition': partition, 'trip_id': trip.trip_id, 'gps_rows': len(trip.track),
                   'nodes': len(trip.nodes), 'arcs': len(trip.arcs), 'route_rows': len(route),
                   'missing_numeric_cells': int((~np.isfinite(trip.track)).sum()),
                   'duplicate_coordinate_steps': int((np.diff(trip.track[:, :2], axis=0) == 0).all(axis=1).sum()),
                   'timestamp_nonpositive': int((dt <= 0).sum()), 'dt_max_s': float(dt.max()) if len(dt) else None,
                   'gaps_gt_30s': int((dt > 30).sum()), 'one_second_steps': int(np.isclose(dt, 1).sum()),
                   'dt_steps': len(dt), 'weak_components': weak, 'strong_components': strong,
                   'isolated_nodes': int((degree == 0).sum()), 'zero_length_arcs': int((net.length == 0).sum()),
                   'reference_breaks': int((trip.arcs[route[:-1], 1] != trip.arcs[route[1:], 0]).sum()),
                   'lon_min': float(trip.track[:, 0].min()), 'lon_max': float(trip.track[:, 0].max()),
                   'lat_min': float(trip.track[:, 1].min()), 'lat_max': float(trip.track[:, 1].max()),
                   'gps_path_length_m': float(np.linalg.norm(np.diff(net.gps, axis=0), axis=1).sum()),
                   'epsg': net.epsg}
            for suffix in ('track', 'nodes', 'arcs', 'route'):
                p = path.with_suffix('.' + suffix)
                row[suffix + '_sha256'] = digest(p)
                row[suffix + '_bytes'] = p.stat().st_size
            rows.append(row)
            print('EDA', partition, trip.trip_id, len(trip.track), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'trip_inventory.csv', index=False)
    summary = {}
    for partition in ('train', 'student_test'):
        sub = frame[frame.partition == partition]
        dt = np.array(intervals[partition])
        summary[partition] = {'trips': len(sub), 'gps_observations': int(sub.gps_rows.sum()),
            'gps_rows_min': int(sub.gps_rows.min()), 'gps_rows_max': int(sub.gps_rows.max()),
            'missing_numeric_cells': int(sub.missing_numeric_cells.sum()),
            'one_second_step_fraction': float(np.isclose(dt, 1).mean()),
            'sampling_quantiles_s': dict(zip(('min', 'median', 'p95', 'p99', 'max'), np.quantile(dt, [0, .5, .95, .99, 1]).tolist())),
            'gaps_gt_30s': int(sub.gaps_gt_30s.sum()), 'reference_breaks': int(sub.reference_breaks.sum()),
            'zero_length_arcs': int(sub.zero_length_arcs.sum()),
            'isolated_nodes_sum_per_trip': int(sub.isolated_nodes.sum())}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for partition, color in [('train', '#2563eb'), ('student_test', '#e76f00')]:
        sub = frame[frame.partition == partition]
        axes[0].hist(sub.gps_rows, bins=20, alpha=.6, label=partition, color=color)
        axes[1].scatter(sub.lon_min, sub.lat_min, label=partition, color=color, s=20)
        axes[2].scatter(sub.nodes, sub.arcs, label=partition, color=color, s=20)
    for ax, title, xlabel, ylabel in zip(axes, ['Observations per trip', 'Trip geographic extent (minimum corner)', 'Network size'],
        ['GPS observations', 'Longitude (degrees)', 'Nodes'], ['Trips', 'Latitude (degrees)', 'Directed arcs']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'eda.png', dpi=150)
    plt.close(fig)
    return summary
