from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import threading
import uuid
from time import perf_counter
from datetime import datetime, timezone
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .io import DEFAULT_DATA, load_trip, load_reference, digest
from .candidates import FEATURES
from .pipeline import predict
from .reference import align_reference
from .metrics import point_metrics, sequence_metrics
from .maps import make_map

log = logging.getLogger('uvicorn.error')
if os.getenv('PATHSNAP_BACKEND_LOG'):
    log_path = Path(os.environ['PATHSNAP_BACKEND_LOG']).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in log.handlers):
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        log.addHandler(handler)


class MatchRequest(BaseModel):
    trip_id: str = Field(pattern=r'^\d{8}$')


def create_app(data_dir=None, artifact_path=None, partition=None):
    app = FastAPI(title='PathSnap', version='0.1.0')
    origins = [o.strip() for o in os.getenv('PATHSNAP_CORS_ORIGINS', 'http://localhost:5500,http://127.0.0.1:5500').split(',') if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
    data_dir = Path(data_dir or os.getenv('PATHSNAP_DATA_DIR', str(DEFAULT_DATA)))
    artifact_path = Path(artifact_path or os.getenv('PATHSNAP_MODEL', 'artifacts/selected.joblib'))
    partition = partition or os.getenv('PATHSNAP_PARTITION', 'student_test')
    if partition not in ('student_test', 'evaluator'):
        raise ValueError('Serve only student_test or evaluator, never train')
    # Evaluator import folder is distinct: data_dir/evaluator/<id>/<id>.track etc.
    lock = threading.Lock()
    maps = {}
    loaded_bundle = None
    model_hash = None
    frozen = None
    project = Path(__file__).resolve().parents[2]
    frontend = project / 'frontend'
    asset_dir = frontend / 'vendor/leaflet'
    require_frozen = partition == 'evaluator' or os.getenv('PATHSNAP_REQUIRE_FROZEN') == '1'
    receipt_dir = Path(os.getenv('PATHSNAP_REQUEST_LOG_DIR', str(project / 'artifacts/server_runs')))
    offline_maps = os.getenv('PATHSNAP_OFFLINE_MAPS') == '1'

    if frontend.exists():
        app.mount('/ui', StaticFiles(directory=str(frontend), html=True), name='frontend')
    if asset_dir.exists():
        app.mount('/assets/leaflet', StaticFiles(directory=str(asset_dir)), name='leaflet')

    @app.get('/', include_in_schema=False)
    def home():
        return RedirectResponse('/ui/')

    def model_bundle():
        nonlocal loaded_bundle, model_hash, frozen
        if loaded_bundle is not None:
            return loaded_bundle
        if not artifact_path.exists():
            raise HTTPException(503, 'No trained artifact. Complete the development workflow first.')
        manifest_path = artifact_path.with_name('model_manifest.json')
        model_hash = digest(artifact_path)
        if require_frozen:
            if not manifest_path.exists():
                raise HTTPException(503, 'Evaluator inference requires a frozen model manifest')
            frozen = json.loads(manifest_path.read_text(encoding='utf-8'))
            if frozen.get('model_sha256') != model_hash:
                raise HTTPException(503, 'Frozen model checksum mismatch')
            for relative, expected in frozen.get('inference_source_sha256', {}).items():
                source = project / relative
                if not source.is_file() or digest(source) != expected:
                    raise HTTPException(503, 'Inference source differs from the frozen manifest: ' + relative)
        bundle = joblib.load(artifact_path)
        if bundle.get('features', None if require_frozen else FEATURES) != FEATURES:
            raise HTTPException(503, 'Model feature schema does not match this implementation')
        loaded_bundle = bundle
        return bundle

    def paths():
        return {p.stem: p for p in sorted((data_dir / partition).glob('*/*.track'))
                if len(p.stem) == 8 and p.stem.isdigit()}

    @app.get('/health')
    def health():
        error = None
        try:
            bundle = model_bundle()
        except HTTPException as exc:
            bundle = None
            error = exc.detail
        return {'status': 'ok' if bundle else 'not_ready', 'ready': bundle is not None,
                'model_available': artifact_path.exists(), 'model': bundle.get('model_name') if bundle else None,
                'model_sha256': model_hash, 'frozen': frozen is not None,
                'partition': partition, 'trip_count': len(paths()), 'error': error}

    @app.get('/trips')
    def trips():
        return {'partition': partition, 'trips': list(paths()),
                'notice': 'Development trips' if partition == 'student_test' else 'Reserved evaluator trips'}

    @app.post('/match')
    def match(request: MatchRequest):
        nonlocal loaded_bundle
        available = paths()
        if request.trip_id not in available:
            raise HTTPException(404, 'Unknown trip ID')
        if not lock.acquire(blocking=False):
            raise HTTPException(429, 'An inference request is running; retry after it completes')
        try:
            request_started = perf_counter()
            loaded_bundle = model_bundle()
            path = available[request.trip_id]
            request_id = uuid.uuid4().hex
            log.info('PATHSNAP START request_id=%s trip=%s partition=%s model_sha256=%s',
                     request_id, request.trip_id, partition, model_hash)
            # Predict to completion BEFORE loading any reference data.
            trip = load_trip(path)
            net, candidates, raw, hmm, route_parts, resets, timing = predict(trip, loaded_bundle)
            metrics = None
            scoring_error = None
            if path.with_suffix('.route').exists():
                try:
                    reference = load_reference(path, len(trip.arcs))
                    labels, _, _ = align_reference(net, reference)
                    metrics = {
                        'label_status': 'reference-derived monotone alignment proxy; not measured point labels',
                        'classifier': {**point_metrics(raw, labels, candidates.ids), **sequence_metrics(net, raw, reference)},
                        'hmm': {**point_metrics(hmm, labels, candidates.ids), **sequence_metrics(net, hmm, reference)}}
                except ValueError as exc:
                    scoring_error = str(exc)
                    log.warning('PATHSNAP SCORING request_id=%s error=%s', request_id, scoring_error)
            # Response maps contain GPS/network/predictions, never a reference layer.
            route_arcs = np.array([a for part in route_parts for a in part], dtype=int)
            html = make_map(net, route_arcs, classifier=raw,
                            asset_base='/assets/leaflet' if asset_dir.exists() else None,
                            tiles=not offline_maps).get_root().render()
            timing['server_processing_seconds'] = perf_counter() - request_started
            if len(maps) >= 10:
                maps.pop(next(iter(maps)))
            maps[request_id] = html
            result = {'request_id': request_id, 'trip_id': request.trip_id, 'partition': partition,
                'model': loaded_bundle['model_name'], 'prediction_arc_ids': hmm.tolist(),
                'model_sha256': model_hash,
                'classifier_arc_ids': raw.tolist(), 'observations': len(trip.track),
                'reconstructed_route_parts': route_parts,
                'unmatched_observations': int((hmm < 0).sum()), 'decoder_resets': resets,
                'timing': timing, 'metrics': metrics, 'scoring_error': scoring_error,
                'map_path': '/maps/' + request_id}
            receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'request_id': request_id,
                'trip_id': trip.trip_id, 'partition': partition, 'model_sha256': model_hash,
                'observations': len(trip.track), 'metrics': metrics, 'scoring_error': scoring_error,
                'timing': timing, 'unmatched': result['unmatched_observations'], 'resets': len(resets)}
            receipt_dir.mkdir(parents=True, exist_ok=True)
            with (receipt_dir / (partition + '_requests.jsonl')).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(receipt) + '\n')
            log.info('PATHSNAP END request_id=%s trip=%s observations=%d model=%s seconds=%.3f classifier_proxy=%s hmm_proxy=%s coverage=%s',
                     request_id, trip.trip_id, len(trip.track), loaded_bundle['model_name'], timing['prediction_seconds'],
                     metrics['classifier']['top1_accuracy_proxy'] if metrics else None,
                     metrics['hmm']['top1_accuracy_proxy'] if metrics else None,
                     metrics['hmm']['alignment_coverage'] if metrics else None)
            return result
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            lock.release()

    @app.get('/maps/{request_id}', response_class=HTMLResponse)
    def map_view(request_id: str):
        if request_id not in maps:
            raise HTTPException(404, 'Map expired; run inference again')
        return maps[request_id]

    return app


app = create_app()
