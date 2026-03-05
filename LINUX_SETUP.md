# Linux (Ubuntu) Setup Guide

This guide covers pulling the latest changes onto an Ubuntu machine that
already has the project at commit `7dddf51`.

---

## What's new since 7dddf51

| Area | Change |
|---|---|
| **Live Inference** | New `/ws/infer` WebSocket — real-time face + object detection as video plays, no batch job needed |
| **Performance** | Celery models cached per-worker (no reload per task); frame seeking instead of decoding every frame |
| **Accuracy** | Ghost track fix, high-quality embedding filter, vectorised cosine matching |
| **Frontend** | Canvas letterbox fix, non-blocking WebSocket pubsub, ⚡ Live Inference button |
| **Dependency** | `websockets` now required for uvicorn WebSocket support |

---

## 1. Pull the changes

```bash
cd ~/face_recognition      # adjust path if different
git pull origin main
```

---

## 2. Install the new dependency

`websockets` is now listed in `pyproject.toml`. Install it:

```bash
# If you used pip install -e .
pip install 'uvicorn[standard]'     # installs websockets + httptools

# Or install websockets directly
pip install 'websockets>=11.0'
```

> **Why?** Without a WebSocket library, uvicorn silently rejects all `ws://`
> upgrade requests with 404. This broke `/ws/infer` (live inference) and the
> existing `/ws/realtime` (batch live-stream). This is the #1 gotcha on a
> fresh Linux install.

---

## 3. System services (if not already running)

### PostgreSQL

```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create DB and user (match your .env DATABASE_URL)
sudo -u postgres psql -c "CREATE USER ava WITH SUPERUSER;"
sudo -u postgres createdb videointel
```

Enable pgvector extension:

```bash
sudo apt install postgresql-15-pgvector   # adjust version to match your PG
# or build from source: https://github.com/pgvector/pgvector
```

### Redis

```bash
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
redis-cli ping    # should return PONG
```

---

## 4. Environment file

Copy and edit if you haven't already:

```bash
cp .env.example .env    # or create .env manually
```

Key values to check on Linux (paths are relative so should be fine):

```ini
DATABASE_URL=postgresql://ava@localhost:5432/videointel
REDIS_URL=redis://localhost:6379/0
FACE_THRESHOLD=0.45
YOLO_CONFIDENCE=0.5
FRAME_SAMPLE_INTERVAL=2
MAX_BATCH_SIZE=32
```

---

## 5. YOLO model weights

`yolov8m.pt` is **not** committed (it's in `.gitignore`). On first run the
Celery worker and the FastAPI server will both auto-download it from
Ultralytics (~52 MB). If the machine has no internet access, copy the file
manually:

```bash
# On Mac (source):
scp yolov8m.pt user@linux-host:~/face_recognition/

# Or download directly on Linux:
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

---

## 6. Start all three processes

Open three terminals (or use tmux/screen):

```bash
# Terminal 1 — FastAPI server
uvicorn src.video_intel.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Celery worker (batch pipeline)
celery -A src.video_intel.celery_worker.celery_app worker --loglevel=info

# Terminal 3 — (optional) Celery flower monitoring
celery -A src.video_intel.celery_worker.celery_app flower
```

Confirm startup looks like this in the uvicorn log:

```
INFO — InsightFace model prepared (CPU).
INFO — YOLO model loaded for live inference.
INFO — Startup complete — 997 actor(s) registered.
```

---

## 7. Using Live Inference

1. Upload a video and process it (or use an already-processed video)
2. Open `http://<server-ip>:8000/results/<video_id>`
3. In the legend bar below the player, click **⚡ Live Inference**
4. Button turns **🔴 Live ON** — press Play on the video
5. Bounding boxes appear in real time, frame by frame

Click the button again to stop and return to batch-results mode.

---

## Linux vs macOS differences

| | macOS | Ubuntu |
|---|---|---|
| Package manager | brew | apt |
| PostgreSQL service | `brew services start postgresql` | `sudo systemctl start postgresql` |
| Redis service | `brew services start redis` | `sudo systemctl start redis-server` |
| Python binary | `python3` / `python` | `python3` (use `python3 -m pip`) |
| InsightFace model cache | `~/.insightface/` | `~/.insightface/` (same) |
| YOLO cache | `~/.config/Ultralytics/` | `~/.config/Ultralytics/` (same) |

Everything else (pip packages, FastAPI, Celery, Redis config) is identical.
