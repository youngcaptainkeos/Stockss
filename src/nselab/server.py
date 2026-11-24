"""Loopback-only read-only static server; no trade or account endpoints."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from .common import ROOT


def serve(port=8765):
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "ui"))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(
        f"NSE Research Lab: http://127.0.0.1:{port} (historical paper replay only)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
