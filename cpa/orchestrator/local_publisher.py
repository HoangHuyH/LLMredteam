"""LocalPublisher — replaces the proposal's `Publisher` node.

It writes fake CTI ONLY to the local MockCTIStore. It has no network client and asserts
the egress guard is installed. This is the single chokepoint that keeps the harness
sandbox-only (SAFETY.md).
"""
from __future__ import annotations

import socket

from cpa.cti_store import MockCTIStore
from data.schema import CTIRecord
from cpa.orchestrator.egress_guard import install_guard, EgressViolation


class LocalPublisher:
    def __init__(self, store: MockCTIStore) -> None:
        install_guard()
        if not getattr(socket, "_cpa_guard_installed", False):
            raise EgressViolation("Egress guard not installed; refusing to operate.")
        self.store = store
        self._count = 0

    def publish(self, record: CTIRecord, channel_id: int = 0) -> None:
        """Insert a poison record into the LOCAL store only. No external I/O, ever."""
        record.is_poison = True
        self.store.add(record)
        self._count += 1

    @property
    def published_count(self) -> int:
        return self._count
