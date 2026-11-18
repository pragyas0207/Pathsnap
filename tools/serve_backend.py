"""Run one backend on an exclusively bound TCP socket (important on Windows)."""
from pathlib import Path
import argparse
import os
import socket
import sys
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    os.environ.setdefault('PATHSNAP_PARTITION', 'evaluator')
    os.environ.setdefault('PATHSNAP_DATA_DIR', str(ROOT / 'data/evaluation_server'))
    os.environ.setdefault('PATHSNAP_MODEL', str(ROOT / 'artifacts/final/model.joblib'))
    os.environ.setdefault('PATHSNAP_REQUIRE_FROZEN', '1')
    os.environ.setdefault('PATHSNAP_BACKEND_LOG', str(ROOT / 'artifacts/backend.log'))
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name == 'nt':
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        server_socket.bind((args.host, args.port))
        server_socket.listen(128)
        server_socket.setblocking(False)
    except OSError as exc:
        server_socket.close()
        raise SystemExit(f'Cannot bind backend port {args.port}: {exc}. Another server may already be running.')
    print(f'PathSnap backend: http://127.0.0.1:{args.port}/ui/', flush=True)
    print('Open the same port at this laptop\'s Wi-Fi IPv4 address from Laptop 2.', flush=True)
    try:
        uvicorn.Server(uvicorn.Config('pathsnap.api:app', host=args.host, port=args.port,
            log_level='info', access_log=True)).run(sockets=[server_socket])
    finally:
        server_socket.close()


if __name__ == '__main__':
    main()
