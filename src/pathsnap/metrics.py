from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, log_loss)


def binary_metrics(y, probabilities):
    if len(y) == 0:
        return {'accuracy': None, 'macro_f1': None, 'log_loss': None, 'n_candidates': 0}
    prediction = (probabilities >= .5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, labels=[0, 1], average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(y, prediction)), 'macro_precision': float(precision),
            'macro_recall': float(recall), 'macro_f1': float(f1),
            'log_loss': float(log_loss(y, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])),
            'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            'n_candidates': len(y)}


def collapse(sequence):
    result = []
    previous = None
    for x in sequence:
        x = int(x)
        if x >= 0 and x != previous:
            result.append(x)
        previous = x
    return result


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def point_metrics(prediction, labels, candidate_ids):
    eligible = labels >= 0
    n = int(eligible.sum())
    covered = (candidate_ids == labels[:, None]).any(axis=1) & eligible
    return {'n_observations': len(labels), 'n_reference_aligned': n,
            'alignment_coverage': n / max(len(labels), 1),
            'top1_accuracy_proxy': float((prediction[eligible] == labels[eligible]).mean()) if n else None,
            'candidate_miss_rate_proxy': float((~covered[eligible]).mean()) if n else None,
            'no_candidate_rate': float((candidate_ids < 0).all(axis=1).mean())}


def sequence_metrics(net, prediction, route, gap_seconds=30.):
    """Exact adjacency diagnostic plus order-aware sequence comparison.

    Arc-sequence edit score can penalize unobserved intermediate edges; this is
    deliberately named a sampled-arc score rather than full-route accuracy.
    """
    pairs = 0
    connected = 0
    for i, (a, b) in enumerate(zip(prediction[:-1], prediction[1:])):
        if net.trip.track[i + 1, 2] - net.trip.track[i, 2] > gap_seconds:
            continue
        pairs += 1
        if a >= 0 and b >= 0 and (a == b or net.trip.arcs[a, 1] == net.trip.arcs[b, 0]):
            connected += 1
    sampled = collapse(prediction)
    reference = collapse(route)
    overlap = len(set(sampled) & set(reference))
    p = overlap / max(len(set(sampled)), 1)
    r = overlap / max(len(set(reference)), 1)
    return {'strict_adjacency_rate': connected / pairs if pairs else None,
            'adjacency_pairs': pairs,
            'arc_set_precision': p, 'arc_set_recall': r,
            'arc_set_f1': 2 * p * r / (p + r) if p + r else 0.,
            'sampled_arc_edit_similarity': 1 - edit_distance(sampled, reference) / max(len(sampled), len(reference), 1),
            'sampled_arc_count': len(sampled), 'reference_arc_count': len(reference)}
