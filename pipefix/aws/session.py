"""boto3 session handling, with throttling retries and a record of every AWS call made.

pipefix only ever calls read-only APIs (Describe*, Get*, List*, LookupEvents, s3:GetObject). The call log is
written into the investigation report so a reviewer can see exactly what was read.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

READ_ONLY_PREFIXES = ("describe_", "get_", "list_", "lookup_", "head_", "filter_", "batch_get_")


class AwsUnavailable(RuntimeError):
    pass


@dataclass
class Clients:
    """Lazily created boto3 clients. `factory(service)` returns a client; tests pass a fake factory."""
    factory: Callable[[str], Any]
    region: str | None = None
    calls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)

    def client(self, service: str):
        if service not in self._cache:
            self._cache[service] = self.factory(service)
        return self._cache[service]

    def call(self, service: str, op: str, **kwargs) -> dict:
        """Call one read-only API with retries on throttling. Returns {} (and records why) on access errors."""
        if not op.startswith(READ_ONLY_PREFIXES):
            raise ValueError(f"pipefix only makes read-only AWS calls, refused {service}:{op}")
        self.calls.append(f"{service}:{op}")
        delay = 0.5
        for attempt in range(5):
            try:
                return getattr(self.client(service), op)(**kwargs)
            except Exception as e:  # botocore.exceptions.ClientError and friends
                code = _error_code(e)
                if code in ("ThrottlingException", "Throttling", "TooManyRequestsException",
                            "RequestLimitExceeded", "SlowDown") and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if code in SETUP_ERRORS:
                    raise AwsUnavailable(f"Cannot reach AWS ({code}): set a region with --region or AWS_REGION, "
                                         "and check the network.") from e
                if code in CREDENTIAL_ERRORS:
                    raise AwsUnavailable(f"AWS rejected the credentials ({code}). Refresh them "
                                         "(`aws sso login` or new keys) and try again.") from e
                # Anything else (denied, not found, not offered in this region...) is recorded and skipped,
                # so one unreadable resource never stops the investigation.
                self.errors.append(f"{service}:{op} -> {code}")
                return {"__error__": code, "message": str(e)[:400]}
        return {}


SETUP_ERRORS = ("NoRegionError", "EndpointConnectionError", "ConnectTimeoutError", "ProxyConnectionError")
CREDENTIAL_ERRORS = ("ExpiredToken", "ExpiredTokenException", "UnrecognizedClientException", "InvalidClientTokenId",
                     "InvalidSignatureException", "SignatureDoesNotMatch", "NoCredentialsError",
                     "TokenRetrievalError", "SSOTokenLoadError", "UnauthorizedSSOTokenError")


def _error_code(e: Exception) -> str:
    resp = getattr(e, "response", None) or {}
    return str((resp.get("Error") or {}).get("Code") or type(e).__name__)


def boto3_clients(profile: str | None = None, region: str | None = None) -> Clients:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:  # pragma: no cover - depends on the install
        raise AwsUnavailable("AWS access needs boto3: pip install 'pipefix[aws]'") from e
    import os
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    session = boto3.Session(profile_name=profile, region_name=region)
    if session.get_credentials() is None:
        raise AwsUnavailable("No AWS credentials found. Run `aws configure` or `aws sso login`, or pass --profile.")
    if not session.region_name:
        raise AwsUnavailable("No AWS region set. Pass --region (e.g. us-east-1) or set AWS_REGION.")
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"}, user_agent_extra="pipefix")
    return Clients(lambda svc: session.client(svc, config=cfg), region=session.region_name)


def paginate(clients: Clients, service: str, op: str, items_key: str, token_in: str = "nextToken",
             token_out: str = "nextToken", limit: int = 1000, **kwargs) -> list[dict]:
    """Manual pagination (works with fakes and with every API's own token names)."""
    out: list[dict] = []
    token = None
    while len(out) < limit:
        args = dict(kwargs)
        if token:
            args[token_in] = token
        page = clients.call(service, op, **args)
        out += page.get(items_key, []) or []
        token = page.get(token_out)
        if not token:
            break
    return out[:limit]
