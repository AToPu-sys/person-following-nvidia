from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import threading


class JsonlEventSink:
    """Append aggregate metadata; never stores frames, source URLs, or tracker IDs."""
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, *, tenant_id: str, region: str, site_id: str, camera_id: str,
             event: str, zone_id: str | None = None, people_visible: int | None = None):
        record = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "region": region,
            "site_id": site_id,
            "camera_id": camera_id,
            "zone_id": zone_id,
            "event": event,
            "people_visible": people_visible,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
