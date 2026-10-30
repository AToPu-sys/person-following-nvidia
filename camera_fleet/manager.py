from __future__ import annotations
from .config import FleetConfig
from .event_store import EventStore
from .worker import CameraWorker, LatestFrameHub


class CameraManager:
    def __init__(self, config: FleetConfig, store: EventStore):
        self.config = config
        self.store = store
        self.hub = LatestFrameHub()
        self.workers = [CameraWorker(
            camera=camera, tenant_id=config.tenant_id, region=config.region,
            model_path=config.model, tracker=config.tracker, confidence=config.confidence,
            image_size=config.image_size, hub=self.hub, events=store,
        ) for camera in config.cameras]

    def start(self):
        for worker in self.workers:
            worker.start()

    def stop(self):
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=8)

    def camera_ids(self):
        return [worker.camera.camera_id for worker in self.workers]
