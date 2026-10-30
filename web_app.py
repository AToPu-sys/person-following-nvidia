from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
import os
import base64
from collections import defaultdict, deque
import hashlib
import hmac
import secrets
import tempfile
import threading
import time

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse

from camera_fleet.config import load_config
from camera_fleet.event_store import EventStore
from camera_fleet.manager import CameraManager

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
load_dotenv(APP_DIR / ".env.cameras", override=True)


def bootstrap_local_login():
    """Create a one-time local operator credential; remote/container deploys must configure it."""
    password = os.getenv("DASHBOARD_PASSWORD", "")
    username = os.getenv("DASHBOARD_USERNAME", "")
    if username and len(password) >= 16 and "replace_with" not in password:
        return
    if os.getenv("APP_HOST") == "0.0.0.0":
        return
    env_file = APP_DIR / ".env"
    current = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    username = username if len(username) >= 3 else "operator"
    password = secrets.token_urlsafe(32)
    lines = [line for line in current.splitlines()
             if not line.startswith("DASHBOARD_USERNAME=") and not line.startswith("DASHBOARD_PASSWORD=")]
    lines.extend([f"DASHBOARD_USERNAME={username}", f"DASHBOARD_PASSWORD={password}"])
    content = "\n".join(lines) + "\n"
    fd, temp_path = tempfile.mkstemp(prefix=".env.", dir=APP_DIR, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, env_file)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    load_dotenv(env_file, override=True)
    print(f"First-run dashboard login: username={username} password={password}")


bootstrap_local_login()
CONFIG_PATH = Path(os.getenv("CAMERA_CONFIG", str(APP_DIR / "cameras.yaml")))
EVENT_DB = os.getenv("EVENT_DB", str(APP_DIR / "data" / "camera-events.sqlite3"))
USER = os.getenv("DASHBOARD_USERNAME", "")
PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
SESSION_TTL = 12 * 60 * 60
LOGIN_KEY = b""
_LOGIN_LOCK = threading.Lock()
_LOGIN_ATTEMPTS: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(USER) < 3 or len(PASSWORD) < 16:
        raise RuntimeError("Set DASHBOARD_USERNAME and a DASHBOARD_PASSWORD of at least 16 characters in .env")
    config = load_config(CONFIG_PATH)
    if os.getenv("APP_HOST") == "0.0.0.0" and any(isinstance(camera.source, int) for camera in config.cameras):
        raise RuntimeError("Container deployment needs authorized RTSP sources in .env.cameras; webcam index 0 is for local macOS only")
    retention_days = int(os.getenv("EVENT_RETENTION_DAYS", "30"))
    store = EventStore(EVENT_DB, retention_days=retention_days)
    manager = CameraManager(config, store)
    app.state.manager = manager
    app.state.store = store
    app.state.started_at = time.time()
    manager.start()
    yield
    manager.stop()


app = FastAPI(title="Camera Edge Console", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
LOGIN_KEY = hashlib.sha256((USER + "\0" + PASSWORD).encode()).digest()


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_login_rate(request: Request):
    key = _client_key(request)
    now = time.monotonic()
    with _LOGIN_LOCK:
        attempts = _LOGIN_ATTEMPTS[key]
        while attempts and now - attempts[0] > 300:
            attempts.popleft()
        if len(attempts) >= 8:
            raise HTTPException(status_code=429, detail="Too many login attempts; wait five minutes")
        return key


def _record_login_failure(key: str):
    with _LOGIN_LOCK:
        _LOGIN_ATTEMPTS[key].append(time.monotonic())


def _create_session(username: str) -> str:
    expires = int(time.time()) + SESSION_TTL
    payload = base64.urlsafe_b64encode(f"{username}\n{expires}".encode()).decode().rstrip("=")
    signature = hmac.new(LOGIN_KEY, payload.encode(), hashlib.sha256).hexdigest()
    return payload + "." + signature


def _valid_session(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    payload, signature = token.rsplit(".", 1)
    expected = hmac.new(LOGIN_KEY, payload.encode(), hashlib.sha256).hexdigest()
    if not secrets.compare_digest(signature, expected):
        return None
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode()
        username, expires_text = raw.rsplit("\n", 1)
        if username != USER or int(expires_text) <= int(time.time()):
            return None
        return username
    except (ValueError, UnicodeDecodeError):
        return None


def require_login(request: Request):
    username = _valid_session(request.cookies.get("camera_session"))
    if username is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")
    return username


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(APP_DIR / "web" / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/api/login")
def login(payload: dict, request: Request, response: Response):
    ip_key = _check_login_rate(request)
    username = str(payload.get("username", ""))
    password = str(payload.get("password", ""))
    user_ok = secrets.compare_digest(username.encode(), USER.encode())
    password_ok = secrets.compare_digest(password.encode(), PASSWORD.encode())
    if not (user_ok and password_ok):
        _record_login_failure(ip_key)
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    with _LOGIN_LOCK:
        _LOGIN_ATTEMPTS.pop(ip_key, None)
    response.set_cookie("camera_session", _create_session(username), max_age=SESSION_TTL,
                        httponly=True, secure=request.url.scheme == "https",
                        samesite="strict", path="/")
    return {"status": "ok", "username": username}


@app.post("/api/logout")
def logout(response: Response, _user=Depends(require_login)):
    response.delete_cookie("camera_session", path="/")
    return {"status": "ok"}


@app.get("/api/status")
def get_status(_user=Depends(require_login)):
    manager: CameraManager = app.state.manager
    snapshots = manager.hub.snapshots()
    now = time.monotonic()
    result = []
    for camera_id in manager.camera_ids():
        item = snapshots.get(camera_id)
        result.append({
            "camera_id": camera_id,
            "status": item.status if item else "starting",
            "people_visible": item.people_visible if item else 0,
            "zones": item.zone_counts if item and item.zone_counts else {},
            "fps": round(item.fps, 1) if item else 0,
            "frame_age_seconds": round(now-item.updated_at, 1) if item and item.updated_at else None,
        })
    return {"region": manager.config.region, "tenant_id": manager.config.tenant_id, "cameras": result}


@app.get("/api/events")
def get_events(limit: int = Query(default=100, ge=1, le=500), _user=Depends(require_login)):
    return app.state.store.recent(limit)


@app.get("/api/cameras/{camera_id}/mjpeg")
def stream_camera(camera_id: str, _user=Depends(require_login)):
    manager: CameraManager = app.state.manager
    if camera_id not in manager.camera_ids():
        raise HTTPException(404, "Unknown camera")

    def frames():
        last_update = 0.0
        while True:
            snapshot = manager.hub.wait_for_update(camera_id, last_update, timeout=1.0)
            if snapshot is None:
                continue
            last_update = snapshot.updated_at
            if snapshot.jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + snapshot.jpeg + b"\r\n"
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
