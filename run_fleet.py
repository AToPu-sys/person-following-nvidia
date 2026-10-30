"""Start the modular local camera-fleet prototype."""
from __future__ import annotations
import argparse
import signal
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from camera_fleet.config import load_config
from camera_fleet.dashboard import FleetDashboard
from camera_fleet.events import JsonlEventSink
from camera_fleet.worker import CameraWorker, LatestFrameHub


def main():
    ap = argparse.ArgumentParser(description="Run authorized camera feeds in one local fleet dashboard")
    ap.add_argument("--config", default="cameras.yaml")
    ap.add_argument("--events", default="events/camera-events.jsonl")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    load_dotenv(root / ".env.cameras", override=True)
    config = load_config(args.config)
    hub = LatestFrameHub()
    sink = JsonlEventSink(args.events)
    workers = [CameraWorker(camera=camera, tenant_id=config.tenant_id, region=config.region, model_path=config.model,
                            tracker=config.tracker, confidence=config.confidence,
                            image_size=config.image_size, hub=hub, events=sink)
               for camera in config.cameras]
    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    for worker in workers:
        worker.start()
    dashboard = FleetDashboard()
    print(f"Starting {len(workers)} authorized camera feed(s) in region '{config.region}'. Press Q to stop.")
    try:
        while not stopping.is_set():
            dashboard.render(hub.snapshots())
            if dashboard.is_quit_pressed():
                stopping.set()
            time.sleep(.01)
    finally:
        stopping.set()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=8)
        dashboard.close()
        print(f"Stopped. Metadata events are in {Path(args.events).expanduser()}.")

if __name__ == "__main__":
    main()
