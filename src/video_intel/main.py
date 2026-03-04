"""FastAPI application for the AI Video Intelligence Platform.

Exposes REST endpoints for video upload, async processing
dispatch, status polling, result retrieval, and actor
management.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import insightface
import numpy as np
import psycopg2
import redis as redis_lib
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.video_intel.celery_worker import (
    process_video_task,
)
from src.video_intel.database import (
    get_all_actors,
    init_db,
    insert_actor,
)
from src.video_intel.pipeline import ProcessingStatus

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# --- environment ---

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Passw0rd@localhost:5432/videointel",
)
REDIS_URL: str = os.getenv(
    "REDIS_URL", "redis://localhost:6379/0"
)
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "processed")
ACTORS_DIR: str = os.getenv("ACTORS_DIR", "actors")
FACE_THRESHOLD: float = float(
    os.getenv("FACE_THRESHOLD", "0.45")
)

# --- global model state ---
# Populated in the startup event; typed as Any to avoid
# import-time side effects in test environments.
face_model: Any = None
redis_client: Any = None
db_conn: Any = None

app = FastAPI(
    title="AI Video Intelligence Platform",
    description=(
        "Netflix X-Ray style video analysis: face detection,"
        " actor matching, object detection, and timeline."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")  # type: ignore[misc]
async def startup_event() -> None:
    """Initialise directories, database, models, and Redis.

    Runs once when the FastAPI process starts.  All
    expensive resources (face model, DB connection, Redis)
    are stored as module-level globals so endpoint handlers
    can reuse them without re-initialising on every request.

    Raises:
        RuntimeError: If any critical resource cannot be
            initialised.
    """
    global face_model, redis_client, db_conn

    for directory in (UPLOAD_DIR, PROCESSED_DIR, ACTORS_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", directory)

    try:
        init_db(DATABASE_URL)
        db_conn = psycopg2.connect(DATABASE_URL)
        logger.info("PostgreSQL connection established.")
    except psycopg2.OperationalError as exc:
        raise RuntimeError(
            "Cannot connect to PostgreSQL on startup."
        ) from exc

    try:
        redis_client = redis_lib.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("Redis connection established.")
    except redis_lib.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Redis on startup."
        ) from exc

    # ctx_id=-1 → CPU mode (never 0, never positive)
    face_model = insightface.app.FaceAnalysis(
        allowed_modules=["detection", "recognition"]
    )
    face_model.prepare(ctx_id=-1)
    logger.info("InsightFace model prepared (CPU).")

    actors = get_all_actors(db_conn)
    logger.info(
        "Startup complete — %d actor(s) registered.",
        len(actors),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Accept and persist an uploaded video file.

    Args:
        file (UploadFile): The video file sent by the client.

    Returns:
        dict[str, Any]: Dict with ``video_id`` (str),
            ``filename`` (str), and ``size_mb`` (float).

    Raises:
        HTTPException: 400 if the file is not a video.
        HTTPException: 500 if saving the file fails.
    """
    content_type: str = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid content type '{content_type}'. "
                "Only video/* files are accepted."
            ),
        )

    video_id: str = uuid.uuid4().hex[:8]
    safe_name: str = (
        f"{video_id}_{file.filename or 'video'}"
    )
    save_path: Path = Path(UPLOAD_DIR) / safe_name

    try:
        content: bytes = await file.read()
        save_path.write_bytes(content)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from exc

    size_mb: float = round(save_path.stat().st_size / 1e6, 2)
    logger.info(
        "Uploaded video_id=%s → %s (%.2f MB).",
        video_id,
        save_path,
        size_mb,
    )
    return {
        "video_id": video_id,
        "filename": safe_name,
        "size_mb": size_mb,
    }


@app.post("/process/{video_id}")
async def process_video(
    video_id: str,
) -> dict[str, Any]:
    """Dispatch async video processing via Celery.

    Locates the uploaded file in ``UPLOAD_DIR``, sends the
    processing task to the Celery queue, and sets an initial
    QUEUED status in Redis.

    Args:
        video_id (str): The 8-character hex ID returned by
            ``/upload``.

    Returns:
        dict[str, Any]: Dict with ``task_id``, ``video_id``,
            and ``status``.

    Raises:
        HTTPException: 404 if no matching file is found.
    """
    matches = list(
        Path(UPLOAD_DIR).glob(f"{video_id}_*")
    )
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded file found for video_id='{video_id}'.",
        )

    video_path: str = str(matches[0].resolve())

    redis_client.set(
        f"status:{video_id}",
        json.dumps(
            {
                "status": ProcessingStatus.QUEUED.value,
                "progress_pct": 0,
                "current_step": "Queued for processing...",
            }
        ),
    )

    task = process_video_task.delay(video_path, video_id)

    logger.info(
        "Dispatched Celery task %s for video_id=%s.",
        task.id,
        video_id,
    )
    return {
        "task_id": task.id,
        "video_id": video_id,
        "status": ProcessingStatus.QUEUED.value,
    }


@app.get("/status/{video_id}")
async def get_status(
    video_id: str,
) -> dict[str, Any]:
    """Retrieve the current processing status from Redis.

    Args:
        video_id (str): Unique video identifier.

    Returns:
        dict[str, Any]: Parsed status payload with
            ``status``, ``progress_pct``, and
            ``current_step``.

    Raises:
        HTTPException: 404 if no status key exists.
        HTTPException: 500 if the stored value is malformed.
    """
    raw: bytes | None = redis_client.get(
        f"status:{video_id}"
    )
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"No status found for video_id='{video_id}'.",
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Status value in Redis is not valid JSON.",
        ) from exc


@app.get("/metadata/{video_id}")
async def get_metadata(
    video_id: str,
) -> dict[str, Any]:
    """Return the full pipeline result for a processed video.

    Args:
        video_id (str): Unique video identifier.

    Returns:
        dict[str, Any]: Full result JSON matching SECTION 10.

    Raises:
        HTTPException: 404 if the result file is missing.
        HTTPException: 500 if the file cannot be parsed.
    """
    result_path = Path(PROCESSED_DIR) / f"{video_id}.json"
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No result file found for video_id='"
                f"{video_id}'. "
                "Has the video finished processing?"
            ),
        )
    try:
        with result_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Result file exists but could not be read.",
        ) from exc


@app.post("/actors/add")
async def add_actor(
    name: str = Form(...),
    photo: UploadFile = File(...),
) -> dict[str, Any]:
    """Register a new actor by uploading a reference photo.

    Saves the photo, runs InsightFace face detection,
    extracts the ``normed_embedding``, and persists it to
    pgvector.

    Args:
        name (str): Display name for the actor.
        photo (UploadFile): JPEG or PNG reference photo
            containing exactly one clearly visible face.

    Returns:
        dict[str, Any]: Dict with ``success`` (bool),
            ``actor_name`` (str), and ``embedding_id`` (int).

    Raises:
        HTTPException: 422 if no face is detected in the
            uploaded photo.
        HTTPException: 500 if saving or DB insert fails.
    """
    photo_path = Path(ACTORS_DIR) / f"{name}.jpg"
    try:
        content: bytes = await photo.read()
        photo_path.write_bytes(content)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to save actor reference photo.",
        ) from exc

    import cv2

    img = cv2.imread(str(photo_path))
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Uploaded photo could not be decoded as an image.",
        )

    faces = face_model.get(img)
    if not faces:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No face detected in the photo for '{name}'. "
                "Please upload a clear, front-facing photo."
            ),
        )

    # Always use normed_embedding, never embedding
    normed: list[float] = faces[0].normed_embedding.tolist()

    try:
        actor_id: int = insert_actor(
            name=name,
            embedding=normed,
            photo_path=str(photo_path),
            conn=db_conn,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database insert failed for actor '{name}'.",
        ) from exc

    logger.info(
        "Registered actor '%s' with id=%d.", name, actor_id
    )
    return {
        "success": True,
        "actor_name": name,
        "embedding_id": actor_id,
    }


@app.get("/actors")
async def list_actors() -> list[dict[str, Any]]:
    """List all registered actors.

    Returns:
        list[dict[str, Any]]: Each item contains ``id`` and
            ``name``; embedding is intentionally omitted to
            keep the response lightweight.
    """
    rows = get_all_actors(db_conn)
    return [{"id": row["id"], "name": row["name"]} for row in rows]


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Report health of all dependent services.

    Pings Redis and runs ``SELECT 1`` against PostgreSQL,
    returning a structured status dict.

    Returns:
        dict[str, Any]: Dict with ``status`` ("ok"),
            ``redis`` ("ok" | "error"),
            ``db`` ("ok" | "error"),
            and ``actors_loaded`` (int).
    """
    redis_status: str = "error"
    try:
        redis_client.ping()
        redis_status = "ok"
    except redis_lib.RedisError:
        logger.warning("Redis health check failed.")

    db_status: str = "error"
    actors_loaded: int = 0
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT 1;")
        actors = get_all_actors(db_conn)
        actors_loaded = len(actors)
        db_status = "ok"
    except psycopg2.DatabaseError:
        logger.warning("PostgreSQL health check failed.")

    return {
        "status": "ok",
        "redis": redis_status,
        "db": db_status,
        "actors_loaded": actors_loaded,
    }


# Static files mount MUST come after all route definitions
app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads",
)
