# AI Video Intelligence Platform

A production-ready Python FastAPI backend that delivers **Netflix X-Ray style** video analysis: face detection, actor matching, object recognition, cross-frame tracking, and a full JSON timeline — all running on CPU with async Celery workers.

---

## Features

| Feature | Technology |
|---|---|
| REST API | FastAPI + Uvicorn |
| Async processing | Celery + Redis |
| Face detection & embedding | InsightFace (VGGFace2, CPU) |
| Actor matching | 512-d cosine similarity via pgvector |
| Face tracking | DeepSORT |
| Object detection | YOLOv8 nano (CPU) |
| Storage | PostgreSQL + pgvector |
| Frame extraction | OpenCV |

---

## Project Structure

```
video-intel/
├── src/
│   └── video_intel/
│       ├── __init__.py        # package metadata
│       ├── database.py        # PostgreSQL / pgvector layer
│       ├── pipeline.py        # frame extraction, detection, tracking
│       ├── celery_worker.py   # Celery task definition
│       └── main.py            # FastAPI app & endpoints
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # unit tests (no GPU required)
├── uploads/                   # raw uploaded videos
├── processed/                 # result JSON files
├── actors/                    # reference actor photos
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🤖 AI Agent Execution Guide

If you are an AI agent tasked with setting up and running this project automatically, execute these exact sequential steps using your command execution tools. This ensures background processes don't block your execution loop.

### 1. Project Initialization
```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Setup environment variables
cp .env.example .env
```

### 2. Infrastructure (Redis & Database)
Ensure Redis and PostgreSQL are running before starting the app.
```bash
# Start Redis in the background (Linux systemd)
sudo systemctl start redis || sudo systemctl start redis-server || redis-server --daemonize yes

# Start PostgreSQL and create the database (Linux systemd)
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE DATABASE videointel;" || psql -U postgres -c "CREATE DATABASE videointel;"
```
*(Note to Agent: The database requires the `pgvector` extension).*

### 3. Start Application Services
Run the services in the background using `nohup` so your execution tool does not hang.
```bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Start Celery worker in background
nohup celery -A src.video_intel.celery_worker worker --loglevel=info > celery.log 2>&1 &

# Start FastAPI server in background
nohup uvicorn src.video_intel.main:app --port 8000 > api.log 2>&1 &
```

### 4. Verify Health
```bash
# Wait for services to boot and verify
sleep 5
curl http://localhost:8000/health
```

---

## 👨‍💻 Quick Start (Manual Setup)

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 14+ with `pgvector` extension
- Redis 7+

### 2. Clone and install

```bash
git clone <repo-url> video-intel
cd video-intel
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if your PostgreSQL credentials differ
```

### 4. Start services

**Terminal 1 — Redis**
```bash
sudo systemctl start redis
redis-cli ping   # → PONG
```

**Terminal 2 — PostgreSQL** (ensure it is running, then create the DB)
```bash
psql -U postgres -c "CREATE DATABASE videointel;"
```

**Terminal 3 — Celery worker**
```bash
cd video-intel
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
celery -A src.video_intel.celery_worker worker --loglevel=info
```

**Terminal 4 — FastAPI server**
```bash
cd video-intel
source venv/bin/activate
uvicorn src.video_intel.main:app --reload --port 8000
```

### 5. Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "redis": "ok",
  "db": "ok",
  "actors_loaded": 0
}
```

---

## API Reference

### `POST /upload`

Upload a video file for processing.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/movie.mp4"
```

Response:
```json
{ "video_id": "a1b2c3d4", "filename": "a1b2c3d4_movie.mp4", "size_mb": 142.7 }
```

---

### `POST /process/{video_id}`

Dispatch async processing for an uploaded video.

```bash
curl -X POST http://localhost:8000/process/a1b2c3d4
```

Response:
```json
{ "task_id": "...", "video_id": "a1b2c3d4", "status": "queued" }
```

---

### `GET /status/{video_id}`

Poll the current processing status.

```bash
curl http://localhost:8000/status/a1b2c3d4
```

Response:
```json
{ "status": "detecting_faces", "progress_pct": 30, "current_step": "Detecting faces and objects..." }
```

Progress stages:

| Stage | `progress_pct` |
|---|---|
| `queued` | 0 |
| `extracting_frames` | 10 |
| `detecting_faces` | 30 |
| `tracking` | 50 |
| `matching` | 70 |
| `saving` | 90 |
| `done` | 100 |
| `error` | 0 |

---

### `GET /metadata/{video_id}`

Retrieve the full timeline JSON after processing completes.

```bash
curl http://localhost:8000/metadata/a1b2c3d4
```

Response schema:
```json
{
  "video_id": "a1b2c3d4",
  "timeline": {
    "5": { "actors": ["Tom Hanks"], "objects": ["car"], "track_ids": [1] }
  },
  "screentime": {
    "Tom Hanks": { "seconds": 48, "scenes": 3, "first_seen": 4 }
  },
  "objects_summary": { "car": 12 },
  "metadata": {
    "duration_seconds": 900.0,
    "fps": 30.0,
    "frames_processed": 450.0,
    "processing_time_seconds": 142.3
  }
}
```

---

### `POST /actors/add`

Register an actor with a reference photo.

```bash
curl -X POST http://localhost:8000/actors/add \
  -F "name=Tom Hanks" \
  -F "photo=@/path/to/tom_hanks.jpg"
```

Response:
```json
{ "success": true, "actor_name": "Tom Hanks", "embedding_id": 1 }
```

---

### `GET /actors`

List all registered actors.

```bash
curl http://localhost:8000/actors
```

Response:
```json
[{ "id": 1, "name": "Tom Hanks" }]
```

---

### `GET /health`

Service health check.

---

## Running Tests

```bash
# From the project root with PYTHONPATH set:
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest tests/ -v --cov=src --cov-report=term-missing
```

No running Redis or PostgreSQL is required — all tests use mocks and temporary files.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:Passw0rd@localhost:5432/videointel` | PostgreSQL DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL (broker + backend) |
| `UPLOAD_DIR` | `uploads` | Dir for raw video uploads |
| `PROCESSED_DIR` | `processed` | Dir for result JSON files |
| `ACTORS_DIR` | `actors` | Dir for reference actor photos |
| `FACE_THRESHOLD` | `0.45` | Min cosine similarity for actor match |
| `YOLO_CONFIDENCE` | `0.5` | Min YOLO detection confidence |
| `FRAME_SAMPLE_INTERVAL` | `2` | Seconds between sampled frames |
| `MAX_BATCH_SIZE` | `32` | Max frames per face-model batch |

---

## Architecture

```
Client
  │
  ▼
FastAPI (main.py)
  │
  ├──► /upload        → save to uploads/
  ├──► /process/{id}  → dispatch Celery task
  ├──► /status/{id}   → read Redis key
  ├──► /metadata/{id} → read JSON from processed/
  └──► /actors/add    → InsightFace → pgvector

Celery Worker (celery_worker.py)
  │
  └──► run_pipeline() (pipeline.py)
          │
          ├── extract_frames()       (OpenCV)
          ├── detect_objects()       (YOLOv8n)
          └── track_and_match_faces()
                  ├── InsightFace → normed_embedding
                  ├── DeepSORT    → track_id
                  └── _cosine_match() → actor name

PostgreSQL + pgvector (database.py)
  └── actors table with vector(512) + IVFFlat index
```

---

## License

MIT
