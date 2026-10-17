"""API keys for the opt-in AI fix research: Serper (Google search) and an OpenAI-compatible LLM.

Same rules as the GitHub/GitLab tokens: a key pasted into the console is checked, then kept in this process's
memory only (never written to disk, never sent back to the page: only "set", where from, and its last four
characters). Keys can also be read from a .env file (`--env-file`, e.g. the review crew's, so the same keys are
reused), or from environment variables. Order: console, then .env file, then environment:
  SERPER_API_KEY
  PIPEFIX_LLM_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY), PIPEFIX_LLM_BASE_URL, PIPEFIX_LLM_MODEL
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request

SERPER_URL = "https://google.serper.dev/search"
_ENV = {"serper": ("SERPER_API_KEY",), "llm": ("PIPEFIX_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")}
_DEFAULTS = {  # which endpoint and model each env var implies
    "DEEPSEEK_API_KEY": ("https://api.deepseek.com", "deepseek-chat"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "PIPEFIX_LLM_API_KEY": ("https://api.deepseek.com", "deepseek-chat"),
}
_KEYS: dict[str, dict] = {}
_FILE: dict[str, tuple[str, str]] = {}     # env var name -> (value, file name), read with load_env_file
_LOCK = threading.Lock()
_FILE_VARS = ("SERPER_API_KEY", "PIPEFIX_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
              "PIPEFIX_LLM_BASE_URL", "PIPEFIX_LLM_MODEL", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "OPENAI_MODEL")


def load_env_file(path: str) -> list[str]:
    """Read the keys pipefix uses (and nothing else) from a .env file, e.g. the review crew's, so the same keys
    can be reused without copying them anywhere. Values stay in memory. Returns the variable names found."""
    from pathlib import Path
    found = []
    for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
        if m and m.group(1) in _FILE_VARS:
            value = m.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].strip()
            if value:
                with _LOCK:
                    _FILE[m.group(1)] = (value, Path(path).name if Path(path).name != ".env"
                                         else f"{Path(path).resolve().parent.name}/.env")
                found.append(m.group(1))
    return found


def _var(name: str) -> tuple[str | None, str | None]:
    """A setting from a loaded .env file first, then the environment."""
    with _LOCK:
        if name in _FILE:
            return _FILE[name][0], f"{name} in {_FILE[name][1]}"
    return (os.environ[name], name) if os.environ.get(name) else (None, None)


def get(kind: str) -> tuple[str | None, str | None]:
    """(key, source) where source is 'console' or the env var name."""
    with _LOCK:
        k = _KEYS.get(kind)
    if k:
        return k["key"], "console"
    for var in _ENV.get(kind, ()):
        value, source = _var(var)
        if value:
            return value, source
    return None, None


def llm_endpoint() -> tuple[str | None, str, str]:
    """(api key, base url, model) for the LLM."""
    key, source = get("llm")
    with _LOCK:
        k = dict(_KEYS.get("llm") or {})
    var = (source or "").split(" in ")[0]
    base, model = _DEFAULTS.get(var, _DEFAULTS["PIPEFIX_LLM_API_KEY"])
    vendor = {"DEEPSEEK_API_KEY": "DEEPSEEK", "OPENAI_API_KEY": "OPENAI"}.get(var, "")
    base = k.get("base_url") or _var("PIPEFIX_LLM_BASE_URL")[0] or (vendor and _var(f"{vendor}_BASE_URL")[0]) or base
    model = k.get("model") or _var("PIPEFIX_LLM_MODEL")[0] or (vendor and _var(f"{vendor}_MODEL")[0]) or model
    return key, base.rstrip("/"), model


def status() -> dict:
    out = {}
    for kind in _ENV:
        key, source = get(kind)
        out[kind] = {"set": bool(key)} if not key else \
            {"set": True, "source": source, "hint": key[-4:] if len(key) >= 12 else ""}
    _, base, model = llm_endpoint()
    out["llm"].update({"base_url": base, "model": model})
    return out


def _verify(kind: str, key: str, base_url: str, model: str) -> None:
    if kind == "serper":
        req = urllib.request.Request(SERPER_URL, data=json.dumps({"q": "nextflow", "num": 1}).encode(),
                                     headers={"X-API-KEY": key, "Content-Type": "application/json"})
        what = "Serper"
    else:
        req = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"})
        what = "The LLM provider"
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ValueError(f"{what} rejected this key ({e.code}). Check it was copied in full.") from None
        raise ValueError(f"Couldn't check the key: {what} answered {e.code}.") from None
    except urllib.error.URLError as e:
        raise ValueError(f"Couldn't reach {what} to check the key ({e.reason}).") from None


def set_key(kind: str, key: str, base_url: str = "", model: str = "", verify=_verify) -> dict:
    if kind not in _ENV:
        raise ValueError("Unknown key type.")
    key = (key or "").strip()
    if not key or len(key) > 512 or re.search(r"\s", key):
        raise ValueError("That doesn't look like an API key. Paste it in full, without spaces.")
    base_url = (base_url or "").strip().rstrip("/")
    if base_url and not re.fullmatch(r"https://[A-Za-z0-9.-]+(:\d+)?(/[\w./-]*)?", base_url):
        raise ValueError("The LLM base URL must be an https:// address, e.g. https://api.deepseek.com")
    model = (model or "").strip()
    if model and not re.fullmatch(r"[\w.:/-]{1,100}", model):
        raise ValueError("That model name has characters a model name can't have.")
    _, dflt_base, dflt_model = llm_endpoint()
    verify(kind, key, base_url or dflt_base, model or dflt_model)
    with _LOCK:
        _KEYS[kind] = {"key": key, "base_url": base_url, "model": model}
    return status()


def clear(kind: str) -> dict:
    with _LOCK:
        _KEYS.pop(kind, None)
    return status()


def known_keys() -> list[str]:
    with _LOCK:
        return [k["key"] for k in _KEYS.values()] + [v for name, (v, _) in _FILE.items() if "KEY" in name]
