"""The console server: security checks and a full demo run through its API."""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from strandline.ui import server


@pytest.fixture(scope="module")
def base():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def req(url, body=None, token=server.TOKEN, host=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Strandline-Token"] = token
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


FAKE = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


@pytest.fixture
def fake_github(monkeypatch):
    from strandline import vcs
    seen = []

    def who(provider, token, host):
        seen.append((provider, host))
        if token != FAKE:
            raise ValueError("GitHub rejected this token (401).")
        return "octo-cat"
    monkeypatch.setattr(vcs, "_who", who)
    monkeypatch.setattr(vcs.set_token, "__defaults__", (None, who))
    for var in ("GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    yield seen
    vcs.clear_token("github"), vcs.clear_token("gitlab")


def test_token_set_in_console_is_checked_kept_and_never_returned(base, fake_github):
    from strandline import vcs
    st = json.loads(req(base + "/api/status")[1])
    assert st["tokens"]["github"] == {"set": False}
    assert req(base + "/api/token", {"provider": "github", "token": FAKE}, token=None)[0] == 403   # needs session token
    code, out = req(base + "/api/token", {"provider": "github", "token": "ghp_wrong" + "x" * 30})
    assert code == 400 and "rejected" in json.loads(out)["error"] and "ghp_wrong" not in out.decode()
    code, out = req(base + "/api/token", {"provider": "github", "token": "  " + FAKE + "\n"})
    assert code == 200 and fake_github[-1] == ("github", "github.com")
    for raw in (out, req(base + "/api/status")[1]):
        assert FAKE.encode() not in raw                                         # never sent back to the page
    st = json.loads(req(base + "/api/status")[1])["tokens"]["github"]
    assert st == {"set": True, "source": "console", "hint": FAKE[-4:], "login": "octo-cat", "host": "github.com"}
    assert vcs.get_token("github") == (FAKE, "console")
    from strandline.report import redact
    assert FAKE not in redact(f"push failed: https://x:{FAKE}@github.com")      # never leaks into errors or MRs
    assert json.loads(req(base + "/api/token/clear", {"provider": "github"})[1]) == {"set": False}
    assert vcs.get_token("github") == (None, None)


def test_token_input_validation_and_env_fallback(base, fake_github, monkeypatch):
    from strandline import vcs
    assert req(base + "/api/token", {"provider": "bitbucket", "token": FAKE})[0] == 400
    assert req(base + "/api/token", {"provider": "github", "token": "two words"})[0] == 400
    assert req(base + "/api/token", {"provider": "gitlab", "token": FAKE, "host": "evil.com/x?y"})[0] == 400
    assert fake_github == []                                                     # nothing was sent anywhere
    monkeypatch.setenv("GITHUB_TOKEN", FAKE)
    st = json.loads(req(base + "/api/status")[1])["tokens"]["github"]
    assert st["set"] and st["source"] == "GITHUB_TOKEN" and FAKE not in json.dumps(st)


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
    assert code == 200 and "conf/strandline.config" in json.loads(out)["written"]
    # the demo repo has no remote, so a merge request is refused with a clear message, nothing pushed
    code, out = req(base + f"/api/jobs/{jid}/open-mr", {})
    assert code == 400 and json.loads(out)["error"]
