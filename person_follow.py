"""Visible multi-person pose and movement tracking for an authorized camera feed."""
from __future__ import annotations
import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

# COCO-17 2D pose graph. NVIDIA NIM's Nova-77 skeleton is a separate 3D model.
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
WINDOW = "Person Follow | camera tracking"
RED = (35, 45, 245)
DIM = (70, 105, 180)
WHITE = (240, 240, 240)


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def open_camera(source):
    if isinstance(source, int) and hasattr(cv2, "CAP_AVFOUNDATION"):
        return cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(source)


def draw_skeleton(frame, points, confidences, threshold, color, thickness=3):
    def visible(i):
        return i < len(points) and i < len(confidences) and confidences[i] >= threshold
    for a, b in EDGES:
        if visible(a) and visible(b):
            cv2.line(frame, tuple(map(int, points[a])), tuple(map(int, points[b])),
                     color, thickness, cv2.LINE_AA)
    for i, point in enumerate(points):
        if visible(i):
            cv2.circle(frame, tuple(map(int, point)), 4 if thickness > 2 else 3,
                       color, -1, cv2.LINE_AA)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Visible multi-person pose/movement tracker")
    ap.add_argument("--source", default="0", help="webcam number or authorized RTSP/HTTP URL")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose model")
    ap.add_argument("--tracker", default="botsort.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    ap.add_argument("--target-id", type=int, help="initial person tracker ID")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--keypoint-conf", type=float, default=0.25)
    ap.add_argument("--lost-seconds", type=float, default=3.0)
    ap.add_argument("--imgsz", type=int, default=640, help="use 512 for higher speed")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--trail-frames", type=int, default=50, help="movement path length")
    args = ap.parse_args()

    source = parse_source(args.source)
    model = YOLO(args.model)
    cap = open_camera(source)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open camera source: {args.source}\n"
            "On macOS, enable camera access for the app running Python in "
            "System Settings > Privacy & Security > Camera, then quit and reopen it."
        )
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ui = {"click": None}
    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ui["click"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1100, 700)
    cv2.setMouseCallback(WINDOW, on_mouse)
    target_id = args.target_id
    last_seen = 0.0
    histories = defaultdict(lambda: deque(maxlen=max(5, args.trail_frames)))
    previous_centers = {}
    previous_side = {}
    crossings = {"left_to_right": 0, "right_to_left": 0}
    line_enabled = False
    boxes_enabled = False
    trails_enabled = True
    all_poses = True
    recording = None
    recording_path = None
    fps = 0.0
    last_tick = time.monotonic()
    banner_message, banner_until = "Click person to select | N: next | K: crossing line | V: record", time.monotonic() + 8
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                cv2.waitKey(250)
                continue
            now = time.monotonic()
            result = model.track(frame, persist=True, tracker=args.tracker, classes=[0],
                                 conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            ids = result.boxes.id.int().cpu().tolist() if result.boxes is not None and result.boxes.id is not None else []
            boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
            poses = result.keypoints.xy.cpu().tolist() if result.keypoints is not None else []
            confs = result.keypoints.conf.cpu().tolist() if result.keypoints is not None and result.keypoints.conf is not None else []
            centers = {}
            for tid, box in zip(ids, boxes):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                centers[tid] = (cx, cy)
                histories[tid].append((int(cx), int(cy)))
                side = -1 if cx < frame.shape[1] / 2 else 1
                if tid in previous_side and previous_side[tid] != side:
                    crossings["left_to_right" if side > 0 else "right_to_left"] += 1
                previous_side[tid] = side

            click = ui.pop("click", None)
            if click:
                x, y = click
                chosen = next((i for i, b in enumerate(boxes) if b[0] <= x <= b[2] and b[1] <= y <= b[3]), None)
                if chosen is not None:
                    target_id, last_seen = ids[chosen], now
                    banner_message, banner_until = f"Selected person #{target_id}", now + 2
            if target_id is None and ids:
                cx, cy = frame.shape[1] / 2, frame.shape[0] / 2
                i = min(range(len(ids)), key=lambda j: (centers[ids[j]][0]-cx)**2 + (centers[ids[j]][1]-cy)**2)
                target_id, last_seen = ids[i], now

            # Mid-frame line is an optional directional counter, not an alarm or identity feature.
            if line_enabled:
                x = frame.shape[1] // 2
                cv2.line(frame, (x, 48), (x, frame.shape[0]-32), (255, 190, 45), 2, cv2.LINE_AA)
                cv2.putText(frame, "COUNT LINE", (x+8, 70), cv2.FONT_HERSHEY_SIMPLEX, .42,
                            (255, 190, 45), 1, cv2.LINE_AA)

            for i, (tid, box) in enumerate(zip(ids, boxes)):
                active = tid == target_id
                color = RED if active else DIM
                thickness = 3 if active else 2
                if boxes_enabled:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                if all_poses or active:
                    points = poses[i] if i < len(poses) else []
                    scores = confs[i] if i < len(confs) else [1.0] * len(points)
                    draw_skeleton(frame, points, scores, args.keypoint_conf, color, thickness)
                cx, cy = map(int, centers[tid])
                label = f"ID {tid}" + ("  FOLLOWING" if active else "")
                cv2.putText(frame, label, (cx-28, max(64, int(box[1])-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .54, color, 2 if active else 1, cv2.LINE_AA)
                if trails_enabled and len(histories[tid]) > 1 and (all_poses or active):
                    trail = list(histories[tid])
                    for j in range(1, len(trail)):
                        alpha = j / len(trail)
                        trail_color = tuple(int(v * alpha) for v in color)
                        cv2.line(frame, trail[j-1], trail[j], trail_color, 2, cv2.LINE_AA)
                if active:
                    prev = previous_centers.get(tid)
                    if prev:
                        dx, dy = cx - prev[0], cy - prev[1]
                        mag = (dx*dx + dy*dy) ** .5
                        if mag > 3:
                            cv2.arrowedLine(frame, prev, (cx, cy), (255, 225, 70), 2, cv2.LINE_AA, tipLength=.35)
                    previous_centers[tid] = (cx, cy)
            # Clean temporal state for tracks that have expired in the tracker.
            for stale in list(histories):
                if stale not in ids and now - last_seen > args.lost_seconds and stale != target_id:
                    histories.pop(stale, None)
                    previous_side.pop(stale, None)
                    previous_centers.pop(stale, None)

            if target_id in ids:
                last_seen = now
                status, color = f"FOLLOWING  |  ID {target_id}", (80, 235, 100)
            elif target_id is not None and now - last_seen <= args.lost_seconds:
                status, color = f"OCCLUDED  |  holding ID {target_id}  |  reacquiring", (0, 190, 255)
            else:
                target_id = None
                status, color = "SEARCHING  |  click a person or wait for center selection", (0, 190, 255)

            dt = max(now - last_tick, 1e-6)
            fps = .85 * fps + .15 / dt if fps else 1 / dt
            last_tick = now
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (24, 30, 35), -1)
            cv2.putText(frame, status, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .68, color, 2, cv2.LINE_AA)
            stats = f"PEOPLE {len(ids)}  |  {fps:.1f} FPS"
            if line_enabled:
                stats += f"  |  L>R {crossings['left_to_right']}  R>L {crossings['right_to_left']}"
            cv2.putText(frame, stats, (max(14, frame.shape[1]-350), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, WHITE, 1, cv2.LINE_AA)
            footer = "CLICK select   N next   M all/target   T trails   B boxes   K count line   V video   P snapshot   Q quit"
            cv2.rectangle(frame, (0, frame.shape[0]-32), (frame.shape[1], frame.shape[0]), (24, 30, 35), -1)
            cv2.putText(frame, footer, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, .43, WHITE, 1, cv2.LINE_AA)
            if recording is not None:
                cv2.circle(frame, (frame.shape[1]-18, 65), 7, RED, -1, cv2.LINE_AA)
                cv2.putText(frame, "REC", (frame.shape[1]-68, 70), cv2.FONT_HERSHEY_SIMPLEX, .5, RED, 2)
                recording.write(frame)
            elif now < banner_until:
                cv2.putText(frame, banner_message, (15, 72), cv2.FONT_HERSHEY_SIMPLEX, .53,
                            (255, 235, 140), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('n') and ids:
                current = ids.index(target_id) if target_id in ids else -1
                target_id = ids[(current + 1) % len(ids)]
                last_seen = now
            elif key == ord('r'):
                target_id = None
            elif key == ord('m'):
                all_poses = not all_poses
            elif key == ord('t'):
                trails_enabled = not trails_enabled
            elif key == ord('b'):
                boxes_enabled = not boxes_enabled
            elif key == ord('k'):
                line_enabled = not line_enabled
                previous_side.clear()
                crossings = {"left_to_right": 0, "right_to_left": 0}
            elif key == ord('p'):
                outdir = Path("captures")
                outdir.mkdir(exist_ok=True)
                shot = outdir / time.strftime("pose-%Y%m%d-%H%M%S.jpg")
                cv2.imwrite(str(shot), frame)
                banner_message, banner_until = f"Saved snapshot: {shot}", now + 3
            elif key == ord('v'):
                if recording is None:
                    outdir = Path("recordings")
                    outdir.mkdir(exist_ok=True)
                    recording_path = outdir / time.strftime("tracking-%Y%m%d-%H%M%S.mp4")
                    rate = cap.get(cv2.CAP_PROP_FPS)
                    if not rate or rate < 1 or rate > 120:
                        rate = 20.0
                    height, width = frame.shape[:2]
                    recording = cv2.VideoWriter(str(recording_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                rate, (width, height))
                    if not recording.isOpened():
                        recording.release()
                        recording = None
                        banner_message, banner_until = "Could not start video recording", now + 3
                    else:
                        banner_message, banner_until = f"Recording locally: {recording_path}", now + 3
                else:
                    recording.release()
                    recording = None
                    banner_message, banner_until = f"Saved video: {recording_path}", now + 4
        if recording is not None:
            recording.release()
    finally:
        if recording is not None:
            recording.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
