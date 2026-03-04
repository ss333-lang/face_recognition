"""Celery worker configuration and video processing task.

Defines the Celery application and the async task that is
dispatched when a user requests video processing.
"""

import json
import logging
import os
from typing import Any

import insightface
import psycopg2
import redis as redis_lib
from celery import Celery
from deep_sort_realtime.deepsort_tracker import DeepSort  # noqa: F401
from dotenv import load_dotenv
from ultralytics import YOLO

from src.video_intel.database import (
    get_all_actors,
    init_db,
)
from src.video_intel.pipeline import (
    ProcessingStatus,
    run_pipeline,
)

load_dotenv()

logger = logging.getLogger(__name__)

# --- constants ---

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:Passw0rd@localhost:5432/videointel",
)

celery_app = Celery(
    "video_intel",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(
    bind=True,
    name="video_intel.process_video",
    max_retries=2,
    default_retry_delay=10,
)
def process_video_task(
    self: Any,
    video_path: str,
    video_id: str,
) -> dict[str, Any]:
    """Celery task that runs the complete video pipeline.

    Initialises all heavyweight models inside the task
    worker process, runs the pipeline, and returns the
    result payload.  A fresh ``DeepSort`` tracker is created
    inside ``run_pipeline`` so it is never shared between
    concurrent tasks.

    Args:
        self (Any): Celery task instance (injected by
            ``bind=True``).
        video_path (str): Absolute path to the uploaded
            video file.
        video_id (str): Unique identifier for status
            tracking and output naming.

    Returns:
        dict[str, Any]: Complete pipeline result matching
            the SECTION 10 schema.

    Raises:
        Exception: Any unhandled exception causes the task
            to transition to ERROR state in Redis, then
            re-raises to allow Celery retry logic.
    """
    redis_client = redis_lib.from_url(REDIS_URL)

    redis_client.set(
        f"status:{video_id}",
        json.dumps(
            {
                "status": ProcessingStatus.QUEUED.value,
                "progress_pct": 0,
                "current_step": "Task queued, waiting for worker...",
            }
        ),
    )

    try:
        # Initialise face model — always ctx_id=-1 (CPU)
        face_model = insightface.app.FaceAnalysis(
            allowed_modules=["detection", "recognition"]
        )
        face_model.prepare(ctx_id=-1)

        # Nano YOLO model for fastest CPU inference
        yolo_model = YOLO("yolov8n.pt")

        # Open a dedicated DB connection for this task
        conn = psycopg2.connect(DATABASE_URL)
        init_db(DATABASE_URL)

        actors = get_all_actors(conn)
        actor_db: dict[str, list[float]] = {
            row["name"]: row["embedding"]
            for row in actors
            if row.get("embedding") is not None
        }

        logger.info(
            "Task %s starting pipeline for video_id=%s "
            "with %d actors in DB.",
            self.request.id,
            video_id,
            len(actor_db),
        )

        result = run_pipeline(
            video_path=video_path,
            video_id=video_id,
            face_model=face_model,
            yolo_model=yolo_model,
            actor_db=actor_db,
            redis_client=redis_client,
        )

        conn.close()
        return result

    except Exception as exc:
        # Write error status to Redis so the frontend knows
        redis_client.set(
            f"status:{video_id}",
            json.dumps(
                {
                    "status": ProcessingStatus.ERROR.value,
                    "progress_pct": 0,
                    "current_step": (
                        f"Processing failed: {exc!s}"
                    ),
                }
            ),
        )
        logger.exception(
            "Pipeline error for video_id=%s: %s",
            video_id,
            exc,
        )
        raise
