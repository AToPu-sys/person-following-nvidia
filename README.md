# Person following — camera prototype

A visible, single-person tracking demo for a webcam or an authorized IP-camera feed. It draws a red full-body skeleton over the camera image, similar in visual style to the supplied NVIDIA example, and can track multiple people at once. It adds temporary IDs, movement trails, a direction arrow, optional line-crossing counts, snapshots, and opt-in local video recording. A clear status bar stays visible while following, searching, or reacquiring after occlusion. It has no face recognition, identity inference, recording, or covert mode.

## NVIDIA Body Pose API findings (checked 2026-10-31)

The official NVIDIA Build page is [NVIDIA 3D Body Pose API](https://build.nvidia.com/nvidia/body-pose/api), with the detailed [NIM documentation](https://docs.nvidia.com/nim/maxine/body-pose/latest/overview.html).

- **Endpoint/protocol:** Build's free endpoint uses gRPC streaming at `grpc.nvcf.nvidia.com:443`, function ID `254b3621-bb66-4b7b-9c89-621c352b4d91`. It is not a REST/JSON per-frame endpoint.
- **Authentication:** `NVIDIA_API_KEY` passed to the official NVIDIA client as `--api-key`; the NGC key needs NGC Catalog access.
- **Input:** One compressed video file (H.264 MP4 preferred; accepted codecs/containers and pixel constraints are documented). In addition, provide tracked bounding boxes for each frame in a text annotation. The service does not detect people. Constant frame rate is required. The cloud endpoint's Build example currently documents a 50 MB video limit.
- **Output:** JSON Lines, one record per frame: 2D keypoints, 3D camera-space joints, confidence, rotations/rest pose/root pose and the supplied tracking ID. The model has a temporal window; results are delayed (about 120 frames before first output).
- **Consequence for this prototype:** NIM itself cannot serve the synchronous webcam/IP-camera loop here. The runnable live demo therefore uses Ultralytics YOLO pose (COCO-17 2D keypoints) plus BoT-SORT by default. For production 3D pose, feed CFR video segments and matching stable-ID bounding-box annotations to NVIDIA NIM using NVIDIA's official client, then consume its JSONL output. The tracker must remain the source of tracking IDs. For true live 3D NIM service, deploy the NIM on an NVIDIA GPU host and build a buffered segment/async result pipeline; don't treat its delayed clip interface as real-time.

## macOS setup

1. Install Python 3.10+ and a recent FFmpeg (Homebrew: `brew install ffmpeg`).
2. In this folder, create and activate a virtual environment:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. macOS: allow the app running Python (Terminal, iTerm, or Codex) to use the camera under **System Settings → Privacy & Security → Camera**. Quit and reopen that app after changing access.
4. Start the webcam:

   ```sh
   python person_follow.py --source 0
   ```

   Or use an authorized IP camera (keep credentials private):

   ```sh
   python person_follow.py --source 'rtsp://user:password@camera-host:554/path'
   ```

   First run downloads `yolo11n-pose.pt` from Ultralytics. Press `q` or Escape to stop. Controls: click to select a person; `n` next person; `r` automatic selection; `m` switch between all skeletons and selected person only; `t` trails on/off; `b` boxes on/off; `k` toggle the center crossing line and reset its counts; `p` save a snapshot to `captures/`; `v` start/stop local video recording in `recordings/`. Recording is off until you press `v`. Use `--imgsz 512` for more speed on a slower computer.

## If macOS says it cannot open the camera

The Python packages and pose model can install correctly while camera access is still denied. Open **System Settings → Privacy & Security → Camera**, turn on access for the application that launched Python, then quit and reopen that application. If it is not listed, launch the script from the standard Terminal app so macOS can show its camera permission prompt. The app keeps its capture window open while reading frames; press `q` to exit.

## Behavior and limits

At startup, the person closest to the center is selected. You can override that with a click or `n`. BoT-SORT maintains temporary IDs and attempts to preserve them through short occlusions; after `--lost-seconds` (default 3) the app returns to search. Low-confidence joints are hidden to reduce skeleton flicker; adjust with `--keypoint-conf`. Line-cross counts are directional image-plane counts, not a reliable people counter if detections are missed or IDs switch. Track IDs are temporary within a run, not identity claims. Long occlusion or people crossing can still cause an ID switch.

The banner remains visible while the application is running, including search and occlusion states. The local preview does not transmit video. If you press `v`, it saves a local recording; `p` saves a still image. The separate NVIDIA mode uploads short video segments for inference and saves the results locally. NVIDIA Build inference is a separate, opt-in video upload workflow; use it only for camera footage you are authorized to process.

## Difference from NVIDIA Body Pose

The red rendering follows the look of the provided screenshot. Its pose model is still 17 two-dimensional COCO landmarks. NVIDIA Body Pose uses the Nova-77 skeleton and can return 77 2D/3D joints, confidence scores, joint rotations and root pose. Those 3D results are not produced by this live macOS demo; the current Build endpoint accepts video plus tracked-box annotations, not live webcam frames.

## NVIDIA NIM follow-up (offline/segment processing)

Install and use NVIDIA's maintained gRPC client from [NVIDIA-Maxine/nim-clients](https://github.com/NVIDIA-Maxine/nim-clients), following its Body Pose instructions. Its documented invocation for the Build endpoint is:

```sh
python body_pose.py --preview-mode \
  --target grpc.nvcf.nvidia.com:443 \
  --function-id 254b3621-bb66-4b7b-9c89-621c352b4d91 \
  --api-key "$NVIDIA_API_KEY" \
  --video-input segment.mp4 \
  --bbox-input segment_bbox.txt \
  --output segment_pose.json \
  --overlay-output segment_overlay.mp4
```

The annotation file format is a first line with the number of tracked bodies (1–50), followed by one row per tracked person per frame: `frame_id tracking_id x y width height` (pixel coordinates, frame IDs start at zero). Keep the video CFR and ensure its frame indices exactly match the annotations. The official client repo provides dependencies and the generated gRPC stubs; this project deliberately does not duplicate NVIDIA's client protocol.

## CCTV / RTSP deployment notes

- Use an authorized, dedicated camera account and restrict RTSP network access; avoid embedding credentials in saved shell history or source.
- For a pilot, run one process per camera and supervise/restart it with a service manager. Add reconnect/backoff, health reporting, and per-camera configuration before unattended operation.
- Keep tracking and inference near the camera/network edge. For NVIDIA NIM, the documented self-hosted container needs a supported NVIDIA GPU; macOS is suitable for this CPU/local demo but not the documented NIM GPU deployment.
- For multi-camera scaling, use one capture worker per stream and a bounded queue so slow inference cannot grow memory without limit. Define retention/access rules before adding recording, which this prototype intentionally omits.

## NVIDIA Body Pose mode (actual NVIDIA model)

`nvidia_follow.py` is the mode to use when you want NVIDIA's own 77-joint output rather than the quick local 17-point preview. It uses a local person detector/tracker to create the tracked-box annotations NVIDIA requires, uploads short camera segments to NVIDIA Build, and previews the official NIM rendered skeleton. It also saves each clip, annotation, JSON result and overlay under `nvidia-results/` for inspection. The Build cloud endpoint receives the camera segments; only use a feed you are authorized to send.

This runs in segments, so the preview pauses during upload/inference. NVIDIA documents that its temporal model can delay the first results by about 120 frames. The default segment is 160 frames at 20 fps, and each uploaded segment should stay under Build's documented 50 MB limit. It is not true zero-latency tracking. The detector/tracker assigns temporary IDs; NVIDIA estimates pose for those supplied boxes.

Install NVIDIA's official client in this project folder and its dependencies:

```sh
cd ~/Documents/person-following-nvidia
git clone https://github.com/NVIDIA-Maxine/nim-clients.git
/opt/homebrew/bin/python3 -m pip install -r nim-clients/body-pose/requirements.txt
```

Make sure `.env` beside `nvidia_follow.py` contains your NVIDIA key:

```env
NVIDIA_API_KEY=your_key_here
```

Then run the NVIDIA pipeline:

```sh
/opt/homebrew/bin/python3 nvidia_follow.py --source 0
```

Press `q` to stop. You can use an authorized RTSP source with `--source 'rtsp://…'`. Install FFmpeg first (`brew install ffmpeg`). If the NVIDIA client is in another folder, set `--client /full/path/to/body_pose.py`.

The local `person_follow.py` remains an immediate 2D preview. On a MacBook Air, NVIDIA's Build cloud service is the path to actual Body Pose output. Self-hosting the NIM keeps inference at the site and avoids sending footage to Build, but the documented video-and-annotation gRPC workflow still has buffering and model delay; it is not zero-latency camera inference.

## Long-term product target: CCTV cameras worldwide

This repository is an early edge pilot. The intended product is a camera analytics system that can be deployed across many authorized CCTV cameras, sites and regions worldwide. The practical download-and-install target is an edge gateway beside the cameras/NVR: cameras expose authorized RTSP/ONVIF streams, and the gateway runs the software. Installing an app directly into every camera is only possible on supported models with a vendor SDK; there is no universal camera app format.

- **Edge-first camera gateway:** connect to authorized RTSP/ONVIF feeds at each site, keep capture resilient to disconnects, and run time-sensitive tracking close to the cameras where practical.
- **Regional inference:** route work to GPU/NIM services near each site, rather than sending every camera stream to one central region. Keep model versions and inference settings explicit and controllable per deployment.
- **Fleet operations:** support camera/site enrollment, health status, safe configuration rollout, bounded queues, back-pressure, retries, upgrades, and clear per-camera diagnostics.
- **Tenant and site isolation:** separate organizations, sites, camera credentials, events and access permissions. Store credentials in a secret manager; never in source code or logs.
- **Privacy by design:** process only feeds the operator is authorized to use; visibly indicate active analytics at the site; avoid face recognition and identity inference. Prefer sending pose metadata/events instead of continuous raw footage, and make any recording opt-in with explicit retention controls.
- **Global deployment readiness:** make region, language, time zone, data location, retention, and operator roles configurable. Before operating in any country, the deployment owner must establish its applicable privacy, workplace/CCTV, notice, retention, and data-transfer requirements.
- **Service boundaries:** keep camera ingestion, person detection/tracking, NVIDIA pose inference, event generation, storage, and operator UI as separate components so one camera or regional service failure does not stop the entire fleet.

The repository now has a single-site web edge console and a Linux installer helper. The helper is currently a source-tree installer for a single site; customers cannot yet download a signed one-click installer from a public release page or centrally manage a worldwide fleet. See [the distribution roadmap](startup/04_GLOBAL_DISTRIBUTION_ROADMAP_RU.md).

## Modular camera-fleet prototype

`camera_fleet/` contains a first multi-camera edge runtime. Each feed has an isolated capture/tracker worker with reconnect backoff. The web dashboard reads the latest frame per camera, so a slow client does not create an unbounded frame queue. SQLite event records contain aggregate counts and tenant/region/site/camera/zone metadata, not person tracker IDs or frames. Each enabled feed must explicitly declare `authorized: true`.

The included `cameras.yaml` defaults to built-in webcam `0`, so no camera variable is needed for the MacBook camera. For an RTSP source, store the URL in `.env.cameras` as `CAMERA_FRONT_DOOR=rtsp://...`; the config uses it instead of the default. Keep camera credentials separate from dashboard and NVIDIA keys.

Install the added dependencies if needed and launch:

```sh
cd ~/Documents/person-following-nvidia
/opt/homebrew/bin/python3 -m pip install -r requirements.txt
/opt/homebrew/bin/python3 run_fleet.py --config cameras.yaml
```

The grid shows each enabled feed, status, people currently visible and processing rate. Press `q` to stop. Metadata events are written to `events/camera-events.jsonl`; frames are not written to disk.

To add another authorized camera, add a unique camera entry in `cameras.yaml` with `site_id`, its environment-variable source and `authorized: true`; then add that variable to `.env.cameras`. For example:

```yaml
  - id: loading-area
    site_id: warehouse-01
    source: "${CAMERA_LOADING_AREA}"
    authorized: true
```

```env
CAMERA_LOADING_AREA=rtsp://operator:encoded-password@camera-host:554/stream
```

Enable camera access in macOS privacy settings for local webcams. RTSP credentials belong only in the ignored `.env` file; URL-encode special characters in passwords. The local prototype loads one pose model per enabled camera, so keep the number of local feeds small on a MacBook Air. For a large deployment, this capture layer should publish bounded frame work to a GPU inference pool instead of loading a model inside each camera worker.

The fleet code is still an edge pilot, not a global production deployment. The web console adds local authentication, TLS at the reverse proxy, and SQLite, but there is no central enrollment service, per-user identity/SSO, alert delivery, signed OTA/rollback, or multi-site policy service yet.

## Browser-based edge console

The project now includes `web_app.py`, a protected browser dashboard for a single edge site. It serves live MJPEG previews, per-camera health/FPS/person counts and a metadata-only event feed. It uses a visible login form with a signed session cookie, hides the FastAPI docs, stores event metadata in SQLite with a configurable 30-day default retention, and does not save camera frames. Local preview uses 2D YOLO pose; the NVIDIA NIM segment path remains separate because Body Pose's gRPC API consumes video plus tracked-box annotations.

### Run on this Mac for development

Install the updated dependencies. On first local launch, if dashboard credentials are missing, the app creates a random operator password in `.env` with owner-only file permissions and prints it once in the Terminal. Save that password. If you already set dashboard credentials in `.env`, the app uses them. `cameras.yaml` defaults to the Mac's camera index 0.

```sh
cd ~/Documents/person-following-nvidia
/opt/homebrew/bin/python3 -m pip install -r requirements.txt
/opt/homebrew/bin/python3 -m uvicorn web_app:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`; a login form asks for the dashboard username and password. macOS must grant camera access to Terminal. For an RTSP source, add its URL to `.env.cameras` as `CAMERA_FRONT_DOOR=...`; the `.env.cameras` file is ignored and separate from NVIDIA and dashboard credentials. If the old browser authentication prompt still appears, stop the old server with Ctrl+C and restart it so it loads this version.

### Deploy the edge console on a Linux host

Use an always-on Linux server at the camera site, not the MacBook, for an unattended edge pilot. If the files do not exist, copy `.env.example` to `.env` and `.env.cameras.example` to `.env.cameras`; preserve any existing secret file and edit it in place. Point a public DNS name at the server and allow inbound TCP 80/443 for Caddy's automatic HTTPS certificate. Set `DASHBOARD_USERNAME`, a strong `DASHBOARD_PASSWORD`, and `CADDY_DOMAIN` in `.env`. Set authorized RTSP URLs in `.env.cameras` and ensure `cameras.yaml` references their variable names. Then run:

```sh
docker compose -f deploy/compose.yaml up --build -d
```

Open `https://<your-CADDY_DOMAIN>`. The application container has no published host port; Caddy is the only public entry point. SQLite events and model cache are kept in a Docker volume. The Compose file passes dashboard secrets only to the camera service, camera URLs only to that service, and the NGC key only to the optional NVIDIA NIM service. Back up the event volume securely and set a retention value appropriate for the site.

See [deploy/README.md](deploy/README.md) for operator setup and NVIDIA NIM commands.

## NVIDIA self-hosted NIM deployment

The [NVIDIA Deploy instructions](https://build.nvidia.com/nvidia/body-pose/deploy) target Linux with Docker and an NVIDIA GPU. They require compute capability 8.0 or newer (the page says T4 is unsupported), the NVIDIA Container Toolkit, and an NGC API key to download the container and models. This is separate from `NVIDIA_API_KEY`, which is used for the Build cloud endpoint.

On a supported Linux NVIDIA host, set `NGC_API_KEY` in `.env`, authenticate to `nvcr.io` with the NGC key, and start the optional profile:

```sh
docker compose -f deploy/compose.yaml --profile nvidia up -d body-pose-nim
```

The gRPC port is bound to host loopback only. From that host, run the segment client against the local NIM:

```sh
python3 nvidia_follow.py --mode self-hosted --target 127.0.0.1:8001 --source 0
```

This uses NVIDIA pose on buffered segments; it does not make the Body Pose API a per-frame camera endpoint. For a remote edge client, first secure the gRPC channel with TLS/mTLS and firewall rules; do not expose port 8001 publicly. The MacBook Air cannot host the documented NVIDIA GPU container.

### Anonymous zone analytics

The web edge console can now draw normalized polygon zones, report current occupancy and write aggregate `zone_entered` / `zone_cleared` events. Configure zones under a camera in `cameras.yaml`; coordinates range from 0 to 1 relative to image width and height. Restart the service after changing YAML. Temporary tracker IDs are used only in memory for continuity; the event database migration removes any stored tracker-ID column and new event records contain only aggregate counts and zone/camera metadata.
