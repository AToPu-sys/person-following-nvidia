from __future__ import annotations
from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")


@dataclass(frozen=True)
class ZoneConfig:
    zone_id: str
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source: int | str | None
    site_id: str = "default-site"
    zones: tuple[ZoneConfig, ...] = ()
    authorized: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class FleetConfig:
    tenant_id: str
    region: str
    model: str
    tracker: str
    cameras: tuple[CameraConfig, ...]
    confidence: float = 0.3
    image_size: int = 640


def _resolve_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise ValueError("camera source must be an integer index or a string")
    match = _ENV.match(value)
    if match:
        name, fallback = match.groups()
        value = os.getenv(name, "").strip()
        if not value and fallback is not None:
            value = fallback.strip()
        if not value:
            raise ValueError(f"camera source environment variable {name} is not set; add it to .env or provide a default in the YAML")
    if value.isdigit():
        return int(value)
    return value


def load_config(path: str | Path) -> FleetConfig:
    with Path(path).expanduser().open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("version") != 1:
        raise ValueError("configuration version must be 1")
    cameras = []
    seen = set()
    for entry in raw.get("cameras", []):
        camera_id = str(entry.get("id", "")).strip()
        if not camera_id or camera_id in seen:
            raise ValueError("each camera needs a unique, non-empty id")
        seen.add(camera_id)
        enabled = bool(entry.get("enabled", True))
        authorized = bool(entry.get("authorized", False))
        if enabled and not authorized:
            raise ValueError(f"camera {camera_id} must explicitly set authorized: true")
        source = _resolve_source(entry.get("source")) if enabled else None
        zones = []
        zone_ids = set()
        for raw_zone in entry.get("zones", []):
            zone_id = str(raw_zone.get("id", "")).strip()
            if not zone_id or zone_id in zone_ids:
                raise ValueError(f"camera {camera_id} needs unique, non-empty zone IDs")
            zone_ids.add(zone_id)
            polygon = raw_zone.get("polygon", [])
            if len(polygon) < 3:
                raise ValueError(f"zone {zone_id} on camera {camera_id} needs at least 3 polygon points")
            normalized = []
            for point in polygon:
                if len(point) != 2:
                    raise ValueError(f"zone {zone_id} points must be [x, y]")
                x, y = float(point[0]), float(point[1])
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError(f"zone {zone_id} points must use normalized 0..1 coordinates")
                normalized.append((x, y))
            zones.append(ZoneConfig(zone_id, tuple(normalized)))
        cameras.append(CameraConfig(camera_id=camera_id, source=source,
                                    site_id=str(entry.get("site_id", "default-site")),
                                    zones=tuple(zones), authorized=authorized, enabled=enabled))
    if not any(camera.enabled for camera in cameras):
        raise ValueError("enable at least one camera")
    return FleetConfig(
        tenant_id=str(raw.get("tenant_id", "local-demo")),
        region=str(raw.get("region", "local")),
        model=str(raw.get("model", "yolo11n-pose.pt")),
        tracker=str(raw.get("tracker", "botsort.yaml")),
        cameras=tuple(c for c in cameras if c.enabled),
        confidence=float(raw.get("confidence", 0.3)),
        image_size=int(raw.get("image_size", 640)),
    )
