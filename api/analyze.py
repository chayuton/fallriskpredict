"""
POST /api/analyze  body: JSON {frames, trials, preds (CSV text), settings}; may be gzip-compressed
                   (header X-Content-Encoding: gzip). The browser sends only the columns needed.
GET  /api/analyze?demo=1  runs the same analysis on synthetic demo data.
Nothing is stored: files live only in memory for the duration of the request.
"""
import gzip
import io
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MAX_BODY = 4_400_000          # Vercel request body limit is 4.5 MB
MAX_INFLATED = 120_000_000


def _send(h, code, body):
    data = json.dumps(body, ensure_ascii=False, allow_nan=False, default=str).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(data)


def _error(h, code, msg, detail=None):
    import re
    msg = re.sub(r"<_io\.\w+ object at 0x[0-9a-f]+>:\s*", "", msg)
    _send(h, code, {"status": "error", "error": msg, **({"detail": detail} if detail else {})})


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from webapp.analysis import analyze, demo_inputs
        q = parse_qs(urlparse(self.path).query)
        if q.get("demo", ["0"])[0] != "1":
            return _error(self, 405, "POST your files to /api/analyze, or GET /api/analyze?demo=1")
        try:
            n = max(1, min(int(q.get("subjects", ["4"])[0]), 6))
            settings = {k: v[0] for k, v in q.items() if k not in ("demo", "subjects")}
            fr, tr, pr = demo_inputs(n)
            _send(self, 200, {"status": "ok", "demo": True, **analyze(fr, tr, pr, settings)})
        except ValueError as e:
            _error(self, 400, str(e))
        except Exception as e:  # noqa: BLE001
            _error(self, 500, repr(e), traceback.format_exc()[-1500:])

    def do_POST(self):
        from webapp.analysis import analyze
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return _error(self, 400, "empty request body")
            if n > MAX_BODY:
                return _error(self, 413, "request too large (Vercel limit 4.5 MB after compression)")
            raw = self.rfile.read(n)
            if (self.headers.get("X-Content-Encoding") or "").lower() == "gzip":
                raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(MAX_INFLATED + 1)
                if len(raw) > MAX_INFLATED:
                    return _error(self, 413, "uncompressed data too large")
            body = json.loads(raw)
            for k in ("frames", "trials", "preds"):
                if not isinstance(body.get(k), str) or not body[k].strip():
                    return _error(self, 400, f"missing '{k}' CSV text")
            res = analyze(io.StringIO(body["frames"]), io.StringIO(body["trials"]), io.StringIO(body["preds"]),
                          body.get("settings") or {}, include_grid=bool(body.get("grid", True)))
            _send(self, 200, {"status": "ok", "demo": False, **res})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            _error(self, 400, str(e))
        except Exception as e:  # noqa: BLE001
            _error(self, 500, repr(e), traceback.format_exc()[-1500:])
