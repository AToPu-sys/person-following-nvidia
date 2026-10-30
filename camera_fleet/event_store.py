from __future__ import annotations
from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path
import threading
import time


class EventStore:
    """Thread-safe metadata store; never persists frames or person tracker IDs."""
    def __init__(self, path: str | Path, retention_days: int = 30):
        self.path = str(Path(path).expanduser())
        self.retention_days = max(1, retention_days)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_cleanup = 0.0
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                region TEXT NOT NULL,
                site_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                zone_id TEXT,
                event TEXT NOT NULL,
                people_visible INTEGER
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(events)")}
            # Migration drops historical per-track IDs while preserving aggregate event records.
            if "track_id" in columns or "zone_id" not in columns:
                zone_expr = "zone_id" if "zone_id" in columns else "NULL"
                db.execute("""CREATE TABLE events_privacy_migration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    site_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    zone_id TEXT,
                    event TEXT NOT NULL,
                    people_visible INTEGER
                )""")
                db.execute(f"""INSERT INTO events_privacy_migration
                    (id, timestamp, tenant_id, region, site_id, camera_id, zone_id, event, people_visible)
                    SELECT id, timestamp, tenant_id, region, site_id, camera_id, {zone_expr}, event, people_visible
                    FROM events""")
                db.execute("DROP TABLE events")
                db.execute("ALTER TABLE events_privacy_migration RENAME TO events")
            db.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp DESC)")
            self._cleanup(db)

    def _cleanup(self, db):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        self._last_cleanup = time.monotonic()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def emit(self, *, tenant_id: str, region: str, site_id: str, camera_id: str,
             event: str, zone_id: str | None = None, people_visible: int | None = None):
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as db:
            if time.monotonic() - self._last_cleanup > 86400:
                self._cleanup(db)
            db.execute("""INSERT INTO events
                (timestamp, tenant_id, region, site_id, camera_id, zone_id, event, people_visible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, tenant_id, region, site_id, camera_id, zone_id, event, people_visible))

    def recent(self, limit: int = 100):
        with self._lock, self._connect() as db:
            if time.monotonic() - self._last_cleanup > 86400:
                self._cleanup(db)
            rows = db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]
