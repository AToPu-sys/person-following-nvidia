"""Camera-to-NVIDIA Body Pose segment pipeline.

Captures short CFR clips, tracks people locally to create NVIDIA's required bbox
annotations, sends each clip to the official NVIDIA NIM client, then previews the
actual NVIDIA 77-joint overlay. This is segment-based, not zero-latency live pose.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

WINDOW = "NVIDIA Body Pose | segment pipeline"


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Run NVIDIA Body Pose on short camera segments")
    ap.add_argument("--source", default="0", help="webcam index or authorized RTSP/HTTP URL")
    ap.add_argument("--model", default="yolo11n.pt", help="local person detector for bbox annotations")
    ap.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    ap.add_argument("--duration", type=float, default=8.0, help="seconds per upload; minimum recommended 6s")
    ap.add_argument("--fps", type=float, default=20.0, help="fixed CFR sampling rate for uploaded segments")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--client", default="nim-clients/body-pose/scripts/body_pose.py",
                    help="path to NVIDIA's official body_pose.py client")
    ap.add_argument("--mode", choices=["build", "self-hosted"], default="build",
                    help="NVIDIA Build cloud endpoint or a self-hosted NVIDIA NIM server")
    ap.add_argument("--target", default=None, help="gRPC endpoint; self-hosted default: 127.0.0.1:8001")
    ap.add_argument("--function-id", default=os.getenv("NVIDIA_BODY_POSE_FUNCTION_ID",
                    "254b3621-bb66-4b7b-9c89-621c352b4d91"))
    ap.add_argument("--output-dir", default="nvidia-results", help="local location for clips and pose results")
    args = ap.parse_args()

    api_key = os.getenv("NVIDIA_API_KEY")
    target = args.target or (os.getenv("NVIDIA_BODY_POSE_TARGET", "grpc.nvcf.nvidia.com:443")
                             if args.mode == "build" else "127.0.0.1:8001")
    if args.mode == "build" and (not api_key or api_key == "replace_me"):
        raise SystemExit("NVIDIA_API_KEY is missing. Put it in .env beside this script.")
    client = Path(args.client).expanduser()
    if not client.is_file():
        raise SystemExit(f"Official NVIDIA client not found: {client}\nSee README.md to install nim-clients.")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to convert camera segments to H.264 CFR. Install with: brew install ffmpeg")
    if args.duration * args.fps < 120:
        raise SystemExit("Use a segment with at least 120 frames (duration × fps >= 120) for NVIDIA's temporal window.")

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION) if isinstance(source, int) and hasattr(cv2, "CAP_AVFOUNDATION") else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. On macOS, allow camera access for Terminal/Codex in System Settings > Privacy & Security > Camera.")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    model = YOLO(args.model)
    outdir = Path(args.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    segment_frames = max(120, round(args.duration * args.fps))
    tempdir = Path(tempfile.mkdtemp(prefix="nvidia-body-pose-"))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)
    segment_no = 0
    keep_running = True
    try:
        while keep_running:
            segment_no += 1
            raw_path = tempdir / f"segment-{segment_no:04d}-raw.mp4"
            video_path = outdir / f"segment-{segment_no:04d}.mp4"
            bbox_path = outdir / f"segment-{segment_no:04d}-bbox.txt"
            json_path = outdir / f"segment-{segment_no:04d}-pose.json"
            overlay_path = outdir / f"segment-{segment_no:04d}-nvidia-overlay.mp4"
            writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (args.width, args.height))
            if not writer.isOpened():
                raise SystemExit("Could not create a camera segment. Check available disk space and camera format.")
            annotations: list[dict[int, list[float] | None]] = []
            known_ids: set[int] = set()
            frames = 0
            last_frame = None
            start = time.monotonic()
            while frames < segment_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cv2.waitKey(100)
                    continue
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                result = model.track(frame, persist=True, tracker=args.tracker, classes=[0],
                                     imgsz=args.imgsz, verbose=False)[0]
                row: dict[int, list[float] | None] = {}
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().tolist()
                    ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else []
                    for tid, box in zip(ids, boxes):
                        known_ids.add(int(tid))
                        x1, y1, x2, y2 = box
                        row[int(tid)] = [max(0.0, x1), max(0.0, y1), max(1.0, x2-x1), max(1.0, y2-y1)]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (50, 220, 80), 2)
                        cv2.putText(frame, f"TRACK ID {tid}", (int(x1), max(22, int(y1)-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .48, (50, 220, 80), 2, cv2.LINE_AA)
                writer.write(frame)
                annotations.append(row)
                frames += 1
                last_frame = frame.copy()
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (24, 30, 35), -1)
                cv2.putText(frame, f"NVIDIA BODY POSE  |  BUFFERING {frames}/{segment_frames}  |  Q quit",
                            (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (70, 235, 100), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    keep_running = False
                    break
            writer.release()
            if not keep_running:
                break
            if not known_ids:
                raw_path.unlink(missing_ok=True)
                continue
            if len(known_ids) > 50:
                raise SystemExit(f"NVIDIA accepts at most 50 tracked bodies per frame; this segment found {len(known_ids)} IDs.")

            # NVIDIA's annotation header counts tracked IDs across the segment. Sentinel rows mark gaps.
            ordered_ids = sorted(known_ids)
            with bbox_path.open("w", encoding="utf-8") as f:
                f.write(f"{len(ordered_ids)}\n")
                for frame_id, row in enumerate(annotations):
                    for tid in ordered_ids:
                        box = row.get(tid)
                        if box is None:
                            f.write(f"{frame_id} {tid} -1 -1 -1 -1\n")
                        else:
                            x, y, w, h = box
                            f.write(f"{frame_id} {tid} {x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")

            # The local MP4 writer has a fixed rate; re-encode to H.264/CFR for NIM.
            convert = subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(args.fps),
                "-vsync", "cfr", "-movflags", "+faststart", str(video_path),
            ], capture_output=True, text=True)
            raw_path.unlink(missing_ok=True)
            if convert.returncode:
                raise SystemExit("ffmpeg failed to prepare the segment: " + convert.stderr[-1000:])

            state = last_frame.copy() if last_frame is not None else None
            if state is not None:
                cv2.rectangle(state, (0, 0), (state.shape[1], 42), (24, 30, 35), -1)
                cv2.putText(state, f"NVIDIA INFERENCE  |  segment {segment_no}  |  uploading & processing...",
                            (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .52, (70, 235, 100), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, state)
                cv2.waitKey(1)
            cmd = ["python3", str(client), "--target", target,
                   "--video-input", str(video_path), "--bbox-input", str(bbox_path),
                   "--output", str(json_path), "--overlay-output", str(overlay_path)]
            if args.mode == "build":
                cmd.extend(["--preview-mode", "--function-id", args.function_id, "--api-key", api_key])
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode:
                raise SystemExit("NVIDIA Body Pose request failed:\n" + process.stderr[-2500:] + "\n" + process.stdout[-800:])
            print(process.stdout.strip())
            print(f"NVIDIA pose JSON: {json_path}")

            # Preview NVIDIA's rendered 77-joint result, not the local 17-joint detector output.
            rendered = cv2.VideoCapture(str(overlay_path))
            while rendered.isOpened():
                ok, pose_frame = rendered.read()
                if not ok:
                    break
                cv2.imshow(WINDOW, pose_frame)
                key = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
                if key in (ord('q'), 27):
                    keep_running = False
                    break
            rendered.release()
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                keep_running = False
            if keep_running:
                print(f"Completed segment {segment_no} in {time.monotonic()-start:.1f}s")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        shutil.rmtree(tempdir, ignore_errors=True)

if __name__ == "__main__":
    main()
