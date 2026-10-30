#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer deploys the edge gateway on Linux. Use the local macOS command in README.md for development."
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker Engine is required. Install Docker and the Compose plugin, then run this installer again."
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin is required. Install it, then run this installer again."
  exit 1
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -e .env || -e .env.cameras ]]; then
  echo "A .env or .env.cameras file already exists. To avoid overwriting secrets, configure it manually; see deploy/README.md."
  exit 1
fi

python3 - <<'PY'
from getpass import getpass
from pathlib import Path
import os
import re
import secrets

print("Camera Edge Gateway setup. Use this only for camera feeds you are authorized to process.")
domain = input("Public DNS name for this site (must point to this server): ").strip()
if not re.fullmatch(r"[A-Za-z0-9.-]+", domain) or "." not in domain:
    raise SystemExit("Enter a valid public DNS name, such as cameras.example.com")
username = input("Dashboard username [operator]: ").strip() or "operator"
if not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", username):
    raise SystemExit("Username must be 3-64 letters, numbers, dot, underscore or hyphen")
password = getpass("Dashboard password (leave blank to generate a random one): ")
if not password:
    password = secrets.token_urlsafe(32)
    print(f"Generated dashboard password (save it now): {password}")
if len(password) < 24 or not re.fullmatch(r"[A-Za-z0-9_-]+", password):
    raise SystemExit("Use at least 24 characters from letters, numbers, underscore and hyphen")
source = getpass("Authorized RTSP URL (input hidden; URL-encode special characters): ").strip()
if not source.startswith(("rtsp://", "rtsps://")):
    raise SystemExit("Enter an authorized RTSP or RTSPS URL")

env = "\n".join([
    "NVIDIA_API_KEY=", "NVIDIA_BODY_POSE_TARGET=grpc.nvcf.nvidia.com:443",
    "NVIDIA_BODY_POSE_FUNCTION_ID=254b3621-bb66-4b7b-9c89-621c352b4d91",
    f"DASHBOARD_USERNAME={username}", f"DASHBOARD_PASSWORD={password}",
    "EVENT_RETENTION_DAYS=30", f"CADDY_DOMAIN={domain}",
    "NGC_API_KEY=", "NIM_CONCURRENCY=1", "",
])
cameras = "# Site-local authorized camera sources. Restrict permissions.\nCAMERA_FRONT_DOOR=" + source + "\n"
Path(".env").write_text(env, encoding="utf-8")
Path(".env.cameras").write_text(cameras, encoding="utf-8")
os.chmod(".env", 0o600)
os.chmod(".env.cameras", 0o600)
print("Created protected site configuration. Camera URLs stay on this gateway.")
PY

printf '\nStarting the edge gateway. First startup downloads the pose model and may take several minutes.\n'
docker compose -f deploy/compose.yaml up --build -d
printf '\nDeployment submitted. Open https://%s after Caddy reports a healthy certificate.\n' "$(sed -n 's/^CADDY_DOMAIN=//p' .env)"
