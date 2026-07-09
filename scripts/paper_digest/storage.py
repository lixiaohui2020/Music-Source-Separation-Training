from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class SentPaperStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"papers": {}}
        with self.path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, ensure_ascii=False, indent=2)

    def has_been_sent(self, stable_id: str) -> bool:
        return stable_id in self._data.get("papers", {})

    def mark_sent(self, stable_id: str, title: str) -> None:
        self._data.setdefault("papers", {})[stable_id] = {
            "title": title,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }

    def filter_unsent(self, stable_ids: list[str]) -> list[str]:
        return [sid for sid in stable_ids if not self.has_been_sent(sid)]
