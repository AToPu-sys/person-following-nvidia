from __future__ import annotations
from dataclasses import dataclass
import threading
import time
from typing import Any

import cv2
from ultralytics import YOLO

from .config import CameraConfig
from .event_store import EventStore
from .zones import ZoneEngine, draw_zones

# COCO-17 connections used by the local 2D preview.
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
         (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]


@dataclass
class CameraSnapshot:
    camera_id: str
    frame: Any = None
    jpeg: bytes | None = None
    status: str = "starting"
    people_visible: int = 0
    zone_counts: dict[str, int] | None = None
    fps: float = 0.0
    updated_at: float = 0.0


class LatestFrameHub:
    """One latest frame per camera: slow dashboards drop old frames, never backlog."""
    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._items: dict[str, CameraSnapshot] = {}

    def publish(self, item: CameraSnapshot):
        with self._condition:
            self._items[item.camera_id] = item
            self._condition.notify_all()

    def snapshots(self) -> dict[str, CameraSnapshot]:
        with self._lock:
            return dict(self._items)

    def wait_for_update(self, camera_id: str, after: float, timeout: float = 1.0):
        with self._condition:
            self._condition.wait_for(
                lambda: camera_id in self._items and self._items[camera_id].updated_at > after,
                timeout=timeout,
            )
            return self._items.get(camera_id)


class CameraWorker(threading.Thread):
    def __init__(self, *, camera: CameraConfig, tenant_id: str, region: str, model_path: str,
                 tracker: str, confidence: float, image_size: int,
                 hub: LatestFrameHub, events: EventStore):
        super().__init__(name=f"camera-{camera.camera_id}", daemon=True)
        self.camera = camera
        self.tenant_id = tenant_id
        self.region = region
        self.model_path = model_path
        self.tracker = tracker
        self.confidence = confidence
        self.image_size = image_size
        self.hub = hub
        self.events = events
        self.stop_event = threading.Event()
        self.zone_engine = ZoneEngine()
        self.zone_counts: dict[str, int] = {}

    def stop(self):
        self.stop_event.set()

    def _open(self):
        source = self.camera.source
        if isinstance(source, int) and hasattr(cv2, "CAP_AVFOUNDATION"):
            return cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        return cv2.VideoCapture(source)

    def _emit(self, event: str, people_visible=None, zone_id=None):
        self.events.emit(tenant_id=self.tenant_id, region=self.region, site_id=self.camera.site_id,
                         camera_id=self.camera.camera_id, event=event,
                         people_visible=people_visible, zone_id=zone_id)

    def _status(self, state: str, frame=None, count=0, fps=0.0, zone_counts=None):
        jpeg = None
        if frame is not None:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                jpeg = encoded.tobytes()
        self.hub.publish(CameraSnapshot(camera_id=self.camera.camera_id, frame=frame, jpeg=jpeg,
                                        status=state, people_visible=count,
                                        zone_counts=zone_counts or {}, fps=fps, updated_at=time.monotonic()))

    def _draw(self, frame, result):
        boxes = result.boxes
        ids = boxes.id.int().cpu().tolist() if boxes is not None and boxes.id is not None else []
        coords = boxes.xyxy.cpu().tolist() if boxes is not None else []
        keypoints = result.keypoints.xy.cpu().tolist() if result.keypoints is not None else []
        scores = result.keypoints.conf.cpu().tolist() if result.keypoints is not None and result.keypoints.conf is not None else []
        for i, (track_id, box) in enumerate(zip(ids, coords)):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 225, 90), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, max(22, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .56, (40, 225, 90), 2, cv2.LINE_AA)
            if i >= len(keypoints):
                continue
            pts = keypoints[i]
            conf = scores[i] if i < len(scores) else [1.0] * len(pts)
            def visible(j):
                return j < len(pts) and j < len(conf) and conf[j] >= .25
            for a, b in EDGES:
                if visible(a) and visible(b):
                    cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])),
                             (35, 45, 245), 3, cv2.LINE_AA)
            for j, pt in enumerate(pts):
                if visible(j):
                    cv2.circle(frame, tuple(map(int, pt)), 3, (35, 45, 245), -1, cv2.LINE_AA)
        return ids, coords

    def run(self):
        try:
            model = YOLO(self.model_path)
        except Exception as exc:
            # Only log the exception class; error strings can include camera URLs/secrets.
            self._status(f"model_error:{type(exc).__name__}")
            self._emit("model_error")
            return

        retry_seconds = 1.0
        last_people_count = None
        fps = 0.0
        last_tick = time.monotonic()
        while not self.stop_event.is_set():
            cap = self._open()
            if not cap.isOpened():
                self._status("camera_unavailable")
                self._emit("camera_unavailable")
                cap.release()
                self.stop_event.wait(retry_seconds)
                retry_seconds = min(30.0, retry_seconds * 2)
                continue
            if isinstance(self.camera.source, int):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            retry_seconds = 1.0
            self._emit("camera_online")
            self._status("connecting")
            last_frame_at = time.monotonic()
            try:
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    now = time.monotonic()
                    if not ok or frame is None:
                        if now - last_frame_at > 5.0:
                            self._status("reconnecting")
                            break
                        self.stop_event.wait(.05)
                        continue
                    last_frame_at = now
                    result = model.track(frame, persist=True, tracker=self.tracker, classes=[0],
                                         conf=self.confidence, imgsz=self.image_size, verbose=False)[0]
                    ids, coords = self._draw(frame, result)
                    self.zone_counts, transitions = self.zone_engine.update(
                        frame.shape, ids, coords, self.camera.zones, now=now
                    )
                    draw_zones(frame, self.camera.zones, self.zone_counts)
                    if last_people_count != len(ids):
                        self._emit("people_count_changed", people_visible=len(ids))
                        last_people_count = len(ids)
                    for transition in transitions:
                        self._emit(transition.event, people_visible=transition.occupancy,
                                   zone_id=transition.zone_id)
                    dt = max(now-last_tick, 1e-6)
                    fps = .85*fps + .15/dt if fps else 1/dt
                    last_tick = now
                    self._status("active", frame, len(ids), fps, self.zone_counts)
            except Exception as exc:
                self._status(f"inference_error:{type(exc).__name__}")
                self._emit("inference_error")
            finally:
                cap.release()
            if not self.stop_event.is_set():
                self._emit("camera_reconnecting")
                self.stop_event.wait(retry_seconds)
                retry_seconds = min(30.0, retry_seconds * 2)
        self._emit("camera_stopped")
        self._status("stopped")
