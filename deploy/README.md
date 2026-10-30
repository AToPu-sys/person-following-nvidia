# Edge deployment operations

This Compose setup is for a **single Linux edge site** with authorized RTSP camera feeds. The dashboard and SQLite event database live on that host; Caddy terminates HTTPS. It is an edge pilot, not a worldwide hosted control plane.

## Required values

1. Copy `.env.example` to `.env` and `.env.cameras.example` to `.env.cameras` if those files do not already exist. Preserve existing secret files and edit them in place. Set `DASHBOARD_USERNAME`, a unique random `DASHBOARD_PASSWORD` (24+ characters recommended; `openssl rand -hex 24` can generate one), and `CADDY_DOMAIN`. Restrict secret file access with `chmod 600 .env .env.cameras`.
2. Set the DNS A/AAAA records for `CADDY_DOMAIN` to the edge host. Allow incoming TCP ports 80 and 443; do not open the application port 8080.
3. Put camera RTSP URLs in `.env.cameras`, with URL-encoded credentials. Protect that file with owner-only permissions. Keep `authorized: true` in `cameras.yaml` only for feeds the operator is authorized to process.
4. Ensure each enabled camera's `source` in `cameras.yaml` refers to the corresponding environment variable.
5. Start from the project root:

   ```sh
   docker compose -f deploy/compose.yaml up --build -d
   ```

Caddy obtains and renews the site's TLS certificate. The dashboard uses a visible login form and a signed session cookie, with one shared account configured through environment variables. Use a private network/VPN or an identity-aware proxy in front of it if multiple operator accounts, MFA, or centralized access control are required.

For the guided single-site installer, run `./deploy/install-edge.sh` from a checked-out copy of this project on the Linux edge host. It prompts for the site domain, dashboard password, and one authorized RTSP feed, then writes owner-only local secret files and starts Compose. This helper is for a pilot; it is not yet a signed public download or centrally managed product release.

## NVIDIA Body Pose NIM (optional)

Only use this profile on a supported Linux host with the NVIDIA Container Toolkit and an eligible NVIDIA GPU. NVIDIA's current deploy page requires compute capability 8.0+ and explicitly excludes T4. The NIM's gRPC port is published on `127.0.0.1:8001`, never on all interfaces.

Set `NGC_API_KEY` in `.env`, authenticate Docker to NVIDIA Container Registry (`nvcr.io`) with that key, then start. The NGC key is only passed to the NIM container; it is not the NVIDIA Build inference key.

```sh
printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

Then start:

```sh
docker compose -f deploy/compose.yaml --profile nvidia up -d body-pose-nim
```

To use it from the segment client on the same host:

```sh
python3 nvidia_follow.py --mode self-hosted --target 127.0.0.1:8001 --source 0
```

The NIM estimates pose for supplied tracked boxes from video segments. For a client on another machine, configure TLS/mTLS, restrict network access, and update NIM's SSL configuration before connecting. Do not publish its gRPC port to the public internet.

## Operational limits

- The local preview loads one YOLO model per camera worker. Size the edge host for the configured streams and resolution.
- Events contain aggregate counts and camera/site/zone metadata; frames and person tracker IDs are not stored in the SQLite event database. Default retention is 30 days and can be changed with `EVENT_RETENTION_DAYS`.
- Keep backups of the Docker `camera_data` volume encrypted and access-controlled.
- Rotate dashboard, camera, NVIDIA Build and NGC credentials independently. Never put camera URLs or keys into YAML, source, issue reports, or screenshots.
- Multi-site enrollment, tenant administration, SSO/MFA, central fleet management, alert delivery, and cross-region event storage are not implemented in this edge pilot.
