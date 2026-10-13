"""The console server: security checks and a full demo run through its API."""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pipefix.ui import server


@pytest.fixture(scope="module")
def base():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def req(url, body=None, token=server.TOKEN, host=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Pipefix-Token"] = token
    if host:
        headers["Host"] = host
    r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers,
                               method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(r, timeout=20) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_page_carries_session_token_and_fonts_are_local(base):
    code, html = req(base + "/")
    assert code == 200 and server.TOKEN.encode() in html
    assert b"fonts.googleapis" not in html                        # no third-party requests
    assert req(base + "/fonts/instrument-sans-var.woff2")[0] == 200


def test_rejects_other_hosts_and_missing_token(base):
    assert req(base + "/api/status", host="evil.example")[0] == 403          # DNS rebinding
    assert req(base + "/api/investigate", {"mode": "demo"}, token=None)[0] == 403
    assert req(base + "/api/investigate", {"mode": "demo"}, token="wrong")[0] == 403


def test_demo_run_end_to_end(base):
    code, out = req(base + "/api/investigate", {"mode": "demo"})
    assert code == 202
    jid = json.loads(out)["id"]
    for _ in range(100):
        job = json.loads(req(base + f"/api/jobs/{jid}")[1])
        if job["status"] != "running":
            break
        time.sleep(0.1)
    assert job["status"] == "done", job.get("error")
    r = job["result"]
    assert [f["name"] for f in r["failures"] if f["fixed"]] == ["GATK4_HAPLOTYPECALLER", "STAR_ALIGN"]
    gatk = r["failures"][0]
    assert gatk["memory_gb"] == 16.0 and gatk["new_memory_gb"] == 32.0 and gatk["peak_gb"] == 15.4
    assert any(t["where"] == "conf/base.config:16" and t["wins"] for t in gatk["trace"])
    assert "123456789012" not in json.dumps(r)                                # account ids redacted
    assert all(s["state"] == "done" for s in job["stages"])
    code, out = req(base + f"/api/jobs/{jid}/apply", {})
    assert code == 200 and "conf/pipefix.config" in json.loads(out)["written"]
    # the demo repo has no remote, so a merge request is refused with a clear message, nothing pushed
    code, out = req(base + f"/api/jobs/{jid}/open-mr", {})
    assert code == 400 and json.loads(out)["error"]
