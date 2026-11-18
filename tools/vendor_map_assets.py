"""Download the pinned Leaflet distribution for self-hosted Folium maps."""
from pathlib import Path
import hashlib
import json
import urllib.request

root = Path(__file__).resolve().parents[1]
target = root / 'frontend/vendor/leaflet'
files = ['leaflet.js', 'leaflet.css', 'images/layers.png', 'images/layers-2x.png',
         'images/marker-icon.png', 'images/marker-icon-2x.png', 'images/marker-shadow.png']
files.append('LICENSE')
records = []
for member in files:
    url = 'https://unpkg.com/leaflet@1.9.3/' + (member if member == 'LICENSE' else 'dist/' + member)
    path = target / member
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read()
    path.write_bytes(data)
    records.append({'path': member, 'source': url, 'sha256': hashlib.sha256(data).hexdigest()})
    print(member, len(data))
(target / 'provenance.json').write_text(json.dumps({'library': 'Leaflet', 'version': '1.9.3',
    'license': 'BSD-2-Clause', 'files': records}, indent=2))
