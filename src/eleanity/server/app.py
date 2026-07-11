from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from eleanity.core.runs_index import list_runs
from eleanity.utils.logging import get_logger, log_event

logger = get_logger("eleanity.server")


class ReportHandler(SimpleHTTPRequestHandler):
    """Serves run artifacts from a local runs directory. Never phones home."""

    runs_dir: Path = Path(".eleanity/runs")

    def __init__(self, *args, runs_dir: Path | None = None, **kwargs):
        if runs_dir is not None:
            self.runs_dir = Path(runs_dir)
        super().__init__(*args, directory=str(self.runs_dir), **kwargs)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html", "/api/runs"}:
            if parsed.path == "/api/runs":
                return self._json(self._runs_payload())
            return self._html(self._index_html())
        if parsed.path.startswith("/api/runs/"):
            run_id = parsed.path.split("/api/runs/", 1)[-1].strip("/")
            path = self.runs_dir / run_id / "result.json"
            if not path.is_file():
                self.send_error(404, "run not found")
                return
            return self._json(json.loads(path.read_text(encoding="utf-8")))
        return super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        log_event(logger, "http", message=format % args)

    def _runs_payload(self) -> dict[str, Any]:
        runs = list_runs(self.runs_dir)
        return {
            "runs": [
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "scenario": r.scenario,
                    "model": r.model,
                    "backends": r.backends,
                    "first_divergence": r.first_divergence,
                    "report_html": f"/{r.run_id}/report.html",
                    "result_json": f"/{r.run_id}/result.json",
                }
                for r in runs
            ]
        }

    def _index_html(self) -> str:
        runs = list_runs(self.runs_dir)
        rows = []
        for r in runs[:100]:
            rows.append(
                "<tr>"
                f"<td><code>{r.run_id[:8]}</code></td>"
                f"<td>{r.status}</td>"
                f"<td>{r.scenario}</td>"
                f"<td><code>{r.model}</code></td>"
                f"<td><a href='/{r.run_id}/report.html'>html</a> · "
                f"<a href='/{r.run_id}/result.json'>json</a></td>"
                "</tr>"
            )
        body = "\n".join(rows) or "<tr><td colspan=5>No runs yet</td></tr>"
        return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Eleanity reports</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#0d1117;color:#e6edf3}}
a{{color:#58a6ff}} table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #30363d;padding:.5rem;text-align:left;font-size:14px}}
code{{font-family:ui-monospace,monospace}}
.banner{{padding:1rem;border:1px solid #30363d;background:#161b22;margin-bottom:1rem}}
</style></head><body>
<div class='banner'><strong>Eleanity</strong> self-hosted report index · local files only · no prompt upload</div>
<table><thead><tr><th>run</th><th>status</th><th>scenario</th><th>model</th><th>links</th></tr></thead>
<tbody>{body}</tbody></table>
</body></html>"""

    def _json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve_reports(
    host: str = "127.0.0.1",
    port: int = 8787,
    runs_dir: Path | str = ".eleanity/runs",
) -> None:
    """Start a local-only report browser for existing runs."""

    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    handler = partial(ReportHandler, runs_dir=runs_path)
    server = ThreadingHTTPServer((host, port), handler)
    log_event(logger, "server_start", host=host, port=port, runs_dir=str(runs_path))
    print(f"Eleanity report server on http://{host}:{port} (runs_dir={runs_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\nstopped")
    finally:
        server.server_close()
