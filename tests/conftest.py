import pytest


@pytest.fixture(autouse=True)
def isolate_api_request_receipts(tmp_path, monkeypatch):
    monkeypatch.setenv('PATHSNAP_REQUEST_LOG_DIR', str(tmp_path / 'request_receipts'))
