from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

DEFAULT_DATA = Path('data/raw/Student folder/Student folder')
PARTITIONS = ('train', 'student_test')


def digest(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def trip_files(root: Path, partition: str) -> list[Path]:
    if partition not in PARTITIONS:
        raise ValueError('Development commands only accept train or student_test')
    return sorted((root / partition).glob('*/*.track'))


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray


def load_trip(track_path: Path) -> Trip:
    """Prediction inputs only. This function must never open a .route file."""
    track = np.loadtxt(track_path, ndmin=2)
    nodes = np.loadtxt(track_path.with_suffix('.nodes'), ndmin=2)
    arc_raw = np.loadtxt(track_path.with_suffix('.arcs'), ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arc_raw.shape[1] != 2:
        raise ValueError('Expected track[lon,lat,time], nodes[lon,lat], arcs[from,to]')
    if not all(np.isfinite(x).all() for x in (track, nodes, arc_raw)):
        raise ValueError('Non-finite data: repair explicitly before prediction')
    for coordinates in (track[:, :2], nodes):
        if (np.abs(coordinates[:, 0]) > 180).any() or (np.abs(coordinates[:, 1]) > 90).any():
            raise ValueError('Invalid longitude/latitude')
    if not np.equal(arc_raw, np.floor(arc_raw)).all():
        raise ValueError('Non-integer node ID')
    arcs = arc_raw.astype(np.int64)
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError('Arc endpoint outside node table')
    if len(track) < 1 or (np.diff(track[:, 2]) <= 0).any():
        raise ValueError('Timestamps must be strictly increasing; do not silently reorder')
    return Trip(track_path.stem, track, nodes, arcs)


def load_reference(track_path: Path, n_arcs: int) -> np.ndarray:
    raw = np.loadtxt(track_path.with_suffix('.route'), ndmin=1)
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError('Invalid route IDs')
    route = raw.astype(np.int64)
    if len(route) == 0 or route.min() < 0 or route.max() >= n_arcs:
        raise ValueError('Reference IDs outside arc table')
    return route
