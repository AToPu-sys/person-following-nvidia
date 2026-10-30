from __future__ import annotations
from dataclasses import dataclass
import time

import cv2
import numpy as np

from .config import ZoneConfig


@dataclass(frozen=True)
class ZoneTransition:
    zone_id: str
    event: str
    occupancy: int


class ZoneEngine:
    """Transient per-camera zone state; tracker IDs are never returned or persisted."""
    def __init__(self, exit_grace_seconds: float = 0.8):
        self.exit_grace_seconds = exit_grace_seconds
        self._active: dict[str, set[int]] = {}
        self._last_inside: dict[str, dict[int, float]] = {}

    def update(self, frame_shape, ids: list[int], boxes: list[list[float]],
               zones: tuple[ZoneConfig, ...], now: float | None = None):
        now = now if now is not None else time.monotonic()
        height, width = frame_shape[:2]
        detections = {}
        for track_id, box in zip(ids, boxes):
            x1, _y1, x2, y2 = box
            # Bottom-center is more stable for floor zones than the box center.
            detections[int(track_id)] = (int((x1+x2)/2), int(y2))
        counts: dict[str, int] = {}
        transitions: list[ZoneTransition] = []
        for zone in zones:
            polygon = np.array([[round(x*width), round(y*height)] for x, y in zone.polygon], dtype=np.int32)
            previous = self._active.setdefault(zone.zone_id, set())
            last_inside = self._last_inside.setdefault(zone.zone_id, {})
            inside = set()
            for track_id, point in detections.items():
                if cv2.pointPolygonTest(polygon, point, False) >= 0:
                    inside.add(track_id)
                    last_inside[track_id] = now
            entered = inside - previous
            previous.update(inside)
            exited = {tid for tid in previous - inside if now - last_inside.get(tid, now) > self.exit_grace_seconds}
            previous.difference_update(exited)
            for tid in exited:
                last_inside.pop(tid, None)
            occupancy = len(previous)
            counts[zone.zone_id] = occupancy
            if entered:
                transitions.append(ZoneTransition(zone.zone_id, "zone_entered", occupancy))
            if exited:
                transitions.append(ZoneTransition(zone.zone_id, "zone_cleared" if occupancy == 0 else "zone_occupancy_changed", occupancy))
        return counts, transitions


def draw_zones(frame, zones: tuple[ZoneConfig, ...], counts: dict[str, int]):
    height, width = frame.shape[:2]
    for zone in zones:
        polygon = np.array([[round(x*width), round(y*height)] for x, y in zone.polygon], dtype=np.int32)
        cv2.polylines(frame, [polygon], True, (0, 215, 255), 2, cv2.LINE_AA)
        x, y = polygon[0]
        cv2.putText(frame, f"{zone.zone_id}: {counts.get(zone.zone_id, 0)}", (int(x), max(45, int(y)-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 215, 255), 2, cv2.LINE_AA)
