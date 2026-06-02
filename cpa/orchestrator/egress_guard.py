"""Egress guard — enforces the sandbox boundary (SAFETY.md).

Asserted at orchestrator start-up. Blocks any outbound connection to known real-world
CTI publishing platforms so a misconfiguration cannot leak generated fake CTI.
"""
from __future__ import annotations

import socket
from typing import Iterable

# Hostnames/domains we must never connect to from this harness.
BLOCKED_HOST_SUBSTRINGS = (
    "otx.alienvault.com",
    "alienvault",
    "github.com",
    "githubusercontent",
    "gist.github",
    "pastebin.com",
    "virustotal.com",
    "misp",                 # MISP threat-sharing instances
    "threatfox",
)


class EgressViolation(RuntimeError):
    """Raised when code attempts to reach a blocked CTI platform."""


def _host_is_blocked(host: str) -> bool:
    h = (host or "").lower()
    return any(bad in h for bad in BLOCKED_HOST_SUBSTRINGS)


def install_guard() -> None:
    """Monkeypatch socket.getaddrinfo to refuse blocked hosts. Idempotent."""
    if getattr(socket, "_cpa_guard_installed", False):
        return
    original = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        if _host_is_blocked(str(host)):
            raise EgressViolation(
                f"Blocked egress to CTI platform '{host}'. "
                f"This harness is sandbox-only; see SAFETY.md."
            )
        return original(host, *args, **kwargs)

    socket.getaddrinfo = guarded  # type: ignore[assignment]
    socket._cpa_guard_installed = True  # type: ignore[attr-defined]


def assert_no_real_publish(allow_real_publish: bool) -> None:
    if allow_real_publish:
        raise EgressViolation(
            "config.sandbox.allow_real_publish=true is not permitted. "
            "Real-world CTI publishing is out of scope (SAFETY.md)."
        )


def audit_urls(urls: Iterable[str]) -> None:
    """Fail fast if any configured URL points at a blocked platform."""
    for u in urls:
        if _host_is_blocked(u):
            raise EgressViolation(f"Configured URL targets a blocked CTI platform: {u}")
