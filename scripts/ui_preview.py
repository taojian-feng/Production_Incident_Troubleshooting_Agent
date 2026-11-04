"""Preview the Analyst UI locally, with sample data and no sign-in.

    uv run python scripts/ui_preview.py

Serves on http://localhost:5173 only. The sample data includes markup in log text, to
show that the page displays it as plain text.
"""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "ui"
PORT = 5173

_ALERT = {
    "service": "orders-api",
    "severity": "SEV2",
    "title": "ia-orders-api-5xx",
    "description": "Threshold Crossed: 1 datapoint [6.5] was above the threshold (2.0).",
}
REPORT = {
    "report_id": "rpt-sample1",
    "alert": _ALERT,
    "status": "needs_review",
    "created_at": "2026-11-04T16:52:00Z",
    "latency_ms": 12600.0,
    "cost": {"usd": 0.031},
    "steps": [
        {
            "number": 1,
            "kind": "summary",
            "content": "Checkout is failing with ERR_CACHE_TIMEOUT, and writes reach a replica.",
            "citations": ["E1", "E2"],
            "verified": True,
            "review_reason": None,
        },
        {
            "number": 2,
            "kind": "precedent",
            "content": "INC0050101 is the closest match. A rolling restart cleared it quickly.",
            "citations": ["E2"],
            "verified": False,
            "review_reason": "sources do not support this: the cited ticket has no recovery time",
        },
        {
            "number": 3,
            "kind": "recommendation",
            "content": "Confirm whether orders-cache recently failed over.",
            "citations": ["E1"],
            "verified": True,
            "review_reason": None,
        },
        {
            "number": 4,
            "kind": "recommendation",
            "content": "If it did, restart orders-api with a rolling deployment.",
            "citations": ["E2"],
            "verified": True,
            "review_reason": None,
        },
    ],
    "evidence": [
        {
            "evidence_id": "E1",
            "source": "logs",
            "title": "40x in /app/orders-api",
            "snippet": "ERROR READONLY <img src=x onerror=alert(1)> host=orders-cache-002",
        },
        {
            "evidence_id": "E2",
            "source": "tickets",
            "title": "INC0050101 [SEV2] Checkout 503s after orders-cache failover",
            "snippet": "Resolution: restarted orders-api with a rolling deployment.",
        },
    ],
}
SUMMARY = {
    "report_id": REPORT["report_id"],
    "service": _ALERT["service"],
    "severity": _ALERT["severity"],
    "title": _ALERT["title"],
    "status": REPORT["status"],
    "created_at": REPORT["created_at"],
}


class PreviewHandler(SimpleHTTPRequestHandler):
    """Static files, plus a stand-in for the API."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(UI), **kwargs)  # type: ignore[arg-type]

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve sample config and API responses, otherwise static files."""
        if self.path == "/config.js":
            self._send(200, "text/javascript", b"window.IA_CONFIG={preview:true};")
        elif self.path == "/api/reports":
            self._send(200, "application/json", json.dumps({"reports": [SUMMARY]}).encode())
        elif self.path.startswith("/api/reports/"):
            feedback = [{"user": "sample@example.com", "rating": "up", "correction": None}]
            body = json.dumps({"report": REPORT, "feedback": feedback}).encode()
            self._send(200, "application/json", body)
        else:
            super().do_GET()

    def do_POST(self) -> None:
        """Accept feedback without storing it."""
        self._send(201, "application/json", b'{"ok": true}')


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), PreviewHandler).serve_forever()
