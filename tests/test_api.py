import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from pathsnap.api import create_app
from pathsnap.candidates import CandidateConfig, FEATURES
from pathsnap.decoder import DecoderConfig
from dataclasses import asdict


def test_api_reference_never_enters_response_or_prediction(tmp_path):
    folder = tmp_path / 'student_test' / '00000001'
    folder.mkdir(parents=True)
    stem = folder / '00000001'
    np.savetxt(stem.with_suffix('.track'), [[8., 50., 1.], [8.0001, 50., 2.]])
    np.savetxt(stem.with_suffix('.nodes'), [[8., 50.], [8.001, 50.]])
    np.savetxt(stem.with_suffix('.arcs'), [[0, 1]], fmt='%d')
    np.savetxt(stem.with_suffix('.route'), [0], fmt='%d')
    model = DummyClassifier(strategy='prior').fit(np.array([[0] * len(FEATURES), [1] * len(FEATURES)]), [0, 1])
    artifact = tmp_path / 'model.joblib'
    joblib.dump({'model': model, 'model_name': 'unit_fixture', 'candidate_config': asdict(CandidateConfig()),
                 'decoder_config': asdict(DecoderConfig())}, artifact)
    client = TestClient(create_app(tmp_path, artifact))
    assert client.get('/trips').json()['trips'] == ['00000001']
    assert client.get('/health').json()['ready'] is True
    assert client.post('/match', json={'trip_id': '../bad'}).status_code == 422
    assert client.post('/match', json={'trip_id': '99999999'}).status_code == 404
    response = client.post('/match', json={'trip_id': '00000001'})
    assert response.status_code == 200
    data = response.json()
    assert data['prediction_arc_ids'] == [0, 0]
    assert 'reference_arc_ids' not in data and 'labels' not in data
    html = client.get(data['map_path']).text
    assert 'development only' not in html and 'Reference route' not in html
    assert '/assets/leaflet/leaflet.js' in html
    assert client.get('/assets/leaflet/leaflet.js').status_code == 200
    assert client.get('/data/evaluation_server/anything.route').status_code == 404
    allowed = client.options('/match', headers={'Origin': 'http://localhost:5500',
        'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'})
    assert allowed.headers.get('access-control-allow-origin') == 'http://localhost:5500'
    # Without any reference file the very same prediction still works.
    stem.with_suffix('.route').unlink()
    second = client.post('/match', json={'trip_id': '00000001'}).json()
    assert second['prediction_arc_ids'] == data['prediction_arc_ids']
    assert second['metrics'] is None
    stem.with_suffix('.route').write_text('999999999')
    third = client.post('/match', json={'trip_id': '00000001'}).json()
    assert third['prediction_arc_ids'] == data['prediction_arc_ids']
    assert third['metrics'] is None and third['scoring_error']


def test_missing_model_is_explicit(tmp_path):
    folder = tmp_path / 'student_test' / '00000001'; folder.mkdir(parents=True)
    (folder / '00000001.track').write_text('8 50 1')
    client = TestClient(create_app(tmp_path, tmp_path / 'absent.joblib'))
    assert client.post('/match', json={'trip_id': '00000001'}).status_code == 503


def test_evaluator_requires_frozen_manifest(tmp_path):
    folder = tmp_path / 'evaluator' / '00000001'; folder.mkdir(parents=True)
    (folder / '00000001.track').write_text('8 50 1')
    artifact = tmp_path / 'model.joblib'; artifact.write_bytes(b'not loaded before manifest validation')
    client = TestClient(create_app(tmp_path, artifact, partition='evaluator'))
    health = client.get('/health').json()
    assert health['ready'] is False and 'frozen' in health['error']
    assert client.post('/match', json={'trip_id': '00000001'}).status_code == 503
