"""Thin logging wrapper: records everything the victim emits, nothing it hides.

Persists per-turn JSONL so episodes are fully reproducible and auditable.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from cpa.victim.base import VictimTurn


class LoggingWrapper:
    def __init__(self, log_dir: str, episode_id: str) -> None:
        self.dir = Path(log_dir) / episode_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fp = (self.dir / "turns.jsonl").open("a", encoding="utf-8")

    def record(self, turn: VictimTurn) -> VictimTurn:
        self._fp.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        self._fp.flush()
        return turn

    def close(self) -> None:
        self._fp.close()
