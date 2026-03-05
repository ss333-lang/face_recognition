"""Core processing pipeline for the AI Video Intelligence Platform.

Handles frame extraction, face detection & matching,
object detection, DeepSORT tracking, and final timeline
assembly.
"""
from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- constants ---

FACE_THRESHOLD: float = float(
    os.getenv("FACE_THRESHOLD", "0.45")
)
FRAME_SAMPLE_INTERVAL: int = int(
    os.getenv("FRAME_SAMPLE_INTERVAL", "1")
)
YOLO_CONFIDENCE: float = float(
    os.getenv("YOLO_CONFIDENCE", "0.5")
)
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "32"))

RELEVANT_OBJECTS: list[str] = [
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "dog",
    "cat",
    "gun",
    "knife",
    "phone",
    "laptop",
    "book",
    "bottle",
    "chair",
    "tv",
    "airplane",
    "boat",
]


class ProcessingStatus(Enum):
    """Status values for the video processing pipeline.

    Each value corresponds to a distinct stage that is
    reported to Redis so the frontend can show live progress.
    """

    QUEUED = "queued"
    EXTRACTING = "extracting_frames"
    DETECTING = "detecting_faces"
    TRACKING = "tracking"
    MATCHING = "matching"
    SAVING = "saving"
    DONE = "done"
    ERROR = "error"


def load_actor_db(conn: Any) -> dict[str, list[float]]:
    """Build an in-memory mapping of actor name → embedding.

    Fetches all actor rows from the database and converts
    them into a plain dict for fast cosine-similarity lookup
    during frame processing.

    Args:
        conn (Any): An open ``psycopg2`` connection.

    Returns:
        dict[str, list[float]]: Keys are actor names, values
            are 512-d normalised embedding vectors.
    """
    from src.video_intel.database import get_all_actors

    rows = get_all_actors(conn)
    actor_db: dict[str, list[float]] = {}
    for row in rows:
        emb = row.get("embedding")
        if emb is not None:
            actor_db[row["name"]] = emb
    logger.info(
        "Loaded %d actors into memory.", len(actor_db)
    )
    return actor_db


def extract_frames(
    video_path: str,
    interval: int,
) -> tuple[dict[int, Any], dict[str, float]]:
    """Read a video file and sample one frame every N seconds.

    Args:
        video_path (str): Absolute or relative path to the
            video file.
        interval (int): Sample one frame every this many
            wall-clock seconds of video.

    Returns:
        tuple[dict[int, Any], dict[str, float]]:
            - frames: mapping of timestamp-in-seconds (int)
              to BGR numpy array.
            - meta: dict with keys ``duration_seconds``,
              ``fps``, and ``frames_processed``.

    Raises:
        ValueError: If the video file cannot be opened.
        IOError: If reading a frame fails unexpectedly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(
            f"Cannot open video file: {video_path}"
        )

    raw_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames: dict[int, Any] = {}
    last_sampled_sec: int = -1
    last_ms: float = 0.0
    frame_idx: int = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Use the video's own position clock — immune to bad
            # FPS metadata (e.g. WebM screencasts reporting 1000fps)
            pos_ms: float = cap.get(cv2.CAP_PROP_POS_MSEC)
            last_ms = pos_ms
            pos_sec: int = int(pos_ms / 1000)

            # Grab one frame per interval seconds of real video time
            if pos_sec >= last_sampled_sec + interval:
                frames[pos_sec] = frame
                last_sampled_sec = pos_sec
            frame_idx += 1
    except Exception as exc:
        raise IOError(
            f"Error reading frame {frame_idx} from "
            f"{video_path}."
        ) from exc
    finally:
        cap.release()

    duration_seconds: float = last_ms / 1000.0

    meta: dict[str, float] = {
        "duration_seconds": duration_seconds,
        "fps": raw_fps,
        "frames_processed": float(len(frames)),
    }
    logger.info(
        "Extracted %d sample frames (%.1f s duration).",
        len(frames),
        duration_seconds,
    )
    return frames, meta


def detect_objects(
    frame: Any,
    model: Any,
    confidence: float,
) -> list[str]:
    """Run YOLOv8 on a frame and return relevant class names.

    Only class names that appear in ``RELEVANT_OBJECTS`` are
    kept to avoid surfacing noisy detections.

    Args:
        frame (Any): A BGR numpy array (H×W×3).
        model (Any): A loaded ``ultralytics.YOLO`` instance.
        confidence (float): Minimum detection confidence
            threshold (0–1).

    Returns:
        list[dict[str, Any]]: Deduplicated list of detected object dicts
            with keys ``label``, ``confidence``, and ``bbox`` ([x, y, w, h]).
    """
    results = model(frame, verbose=False)
    found_objs: list[dict[str, Any]] = []
    # Deduplicate by keeping the highest confidence per class-box combo roughly
    # But for bounding boxes, we actually want ALL distinct instances of an object
    # (e.g. 3 different cars). 
    
    for result in results:
        for box in result.boxes:
            score: float = float(box.conf[0])
            if score < confidence:
                continue
            class_idx: int = int(box.cls[0])
            class_name: str = (
                result.names.get(class_idx, "")
            )
            if class_name in RELEVANT_OBJECTS:
                # box.xyxy[0] is [x1, y1, x2, y2]
                x1, y1, x2, y2 = (
                    float(box.xyxy[0][0]),
                    float(box.xyxy[0][1]),
                    float(box.xyxy[0][2]),
                    float(box.xyxy[0][3]),
                )
                w = x2 - x1
                h = y2 - y1
                found_objs.append({
                    "label": class_name,
                    "confidence": round(score, 2),
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)]
                })
    return found_objs


def _cosine_match(
    embedding: list[float],
    actor_db: dict[str, list[float]],
    threshold: float,
) -> tuple[str | None, float]:
    """Find the best matching actor via cosine similarity.

    Because all embeddings are already L2-normalised the
    cosine similarity is simply the dot product.

    Args:
        embedding (list[float]): Query face embedding
            (512-d, normed).
        actor_db (dict[str, list[float]]): In-memory mapping
            from actor name to normalised embedding.
        threshold (float): Minimum similarity required to
            accept a match.

    Returns:
        tuple[str | None, float]: (actor_name, score) if a
            match is found, or (None, 0.0) otherwise.
    """
    if not actor_db:
        return None, 0.0

    query = np.array(embedding, dtype=np.float32)
    best_name: str | None = None
    best_score: float = 0.0

    for name, ref_emb in actor_db.items():
        ref = np.array(ref_emb, dtype=np.float32)
        # Dot product of two L2-normed vectors equals cosine
        score: float = float(np.dot(query, ref))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, best_score
    return None, 0.0


def track_and_match_faces(
    frame: Any,
    tracker: Any,
    face_model: Any,
    actor_db: dict[str, list[float]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Detect, track, and match faces in a single frame.

    Steps:
      1. Run InsightFace detection on the frame.
      2. Feed bounding boxes into DeepSORT for track IDs.
      3. Match each tracked face embedding to the actor DB.

    Args:
        frame (Any): BGR numpy array (H×W×3).
        tracker (Any): A ``DeepSort`` instance shared across
            frames of the SAME video (never across tasks).
        face_model (Any): An ``insightface.app.FaceAnalysis``
            instance prepared with ``ctx_id=-1``.
        actor_db (dict[str, list[float]]): In-memory actor
            embeddings.
        threshold (float): Cosine similarity threshold for
            accepting an actor match.

    Returns:
        list[dict[str, Any]]: One dict per tracked face with
            keys ``track_id`` (int), ``actor`` (str | None),
            and ``score`` (float).
    """
    faces = face_model.get(frame)
    if not faces:
        return []

    # Build DeepSORT detection tuples:
    # ([x, y, w, h], confidence, "face")
    detections: list[tuple[list[float], float, str]] = []
    embeddings: list[list[float]] = []

    for face in faces:
        bbox = face.bbox  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )
        w = x2 - x1
        h = y2 - y1
        det_conf: float = float(face.det_score)
        detections.append(([x1, y1, w, h], det_conf, "face"))
        # Always use normed_embedding, never embedding
        normed_emb: list[float] = (
            face.normed_embedding.tolist()
        )
        embeddings.append(normed_emb)

    # Build a {detection_index: embedding} lookup BEFORE calling
    # DeepSORT, since DeepSORT may reorder tracks.
    det_emb_map: dict[int, list[float]] = {}
    for det_i, (det_bbox, _, _) in enumerate(detections):
        det_emb_map[det_i] = embeddings[det_i]

    tracks = tracker.update_tracks(
        detections, frame=frame
    )

    results: list[dict[str, Any]] = []
    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id: int = int(track.track_id)
        # Use track index to map embedding (best-effort proxy)
        emb = det_emb_map.get(i)
        if emb is None and embeddings:
            emb = embeddings[min(i, len(embeddings) - 1)]
        if emb is not None:
            actor_name, score = _cosine_match(
                emb, actor_db, threshold
            )
        else:
            actor_name, score = None, 0.0

        # get the bounding box from the deepsort track
        # to_tlwh() returns [top_left_x, top_left_y, width, height]
        track_bbox = track.to_tlwh()
        x, y, w, h = (
            float(track_bbox[0]),
            float(track_bbox[1]),
            float(track_bbox[2]),
            float(track_bbox[3]),
        )

        results.append(
            {
                "track_id": track_id,
                "actor": actor_name,
                "score": round(score, 2),
                "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
            }
        )
    # Deduplicate: keep only the best-scoring match per actor.
    # Without this, stale tracks all match the same face and
    # the same actor appears multiple times in one frame.
    best_per_actor: dict[str, tuple[int, float]] = {}
    for r in results:
        name = r["actor"]
        if name is None:
            continue
        if name not in best_per_actor or r["score"] > best_per_actor[name][1]:
            best_per_actor[name] = (r["track_id"], r["score"])

    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for r in results:
        name = r["actor"]
        if name is None:
            deduped.append(r)
        elif name not in seen_names and best_per_actor.get(name, (None, 0))[0] == r["track_id"]:
            deduped.append(r)
            seen_names.add(name)
    return deduped


def run_pipeline(
    video_path: str,
    video_id: str,
    face_model: Any,
    yolo_model: Any,
    actor_db: dict[str, list[float]],
    redis_client: Any,
) -> dict[str, Any]:
    """Execute the full video intelligence pipeline.

    Orchestrates frame extraction → face tracking & matching
    → object detection → timeline assembly → result
    persistence.  Reports granular progress to Redis so the
    frontend can display a live progress bar.

    Args:
        video_path (str): Path to the uploaded video file.
        video_id (str): Unique identifier used for the
            output filename and Redis keys.
        face_model (Any): Prepared InsightFace
            ``FaceAnalysis`` instance (CPU mode).
        yolo_model (Any): Loaded YOLOv8 ``YOLO`` instance.
        actor_db (dict[str, list[float]]): In-memory actor
            name → normalised embedding mapping.
        redis_client (Any): Connected ``redis.Redis`` client
            for status updates.

    Returns:
        dict[str, Any]: The complete result payload matching
            the SECTION 10 schema.

    Raises:
        ValueError: If video extraction fails.
        IOError: If the result JSON cannot be written.
        RuntimeError: If pipeline processing encounters an
            unrecoverable error.
    """
    from deep_sort_realtime.deepsort_tracker import DeepSort

    processed_dir: str = os.getenv("PROCESSED_DIR", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    def _set_status(
        status: ProcessingStatus,
        progress: int,
        step: str,
    ) -> None:
        """Push a JSON status blob to Redis."""
        redis_client.set(
            f"status:{video_id}",
            json.dumps(
                {
                    "status": status.value,
                    "progress_pct": progress,
                    "current_step": step,
                }
            ),
        )

    start_time: float = time.time()

    # --- Stage: EXTRACTING ---
    _set_status(
        ProcessingStatus.EXTRACTING,
        10,
        "Extracting frames from video...",
    )
    frames, meta = extract_frames(
        video_path, FRAME_SAMPLE_INTERVAL
    )

    # --- Stage: DETECTING ---
    _set_status(
        ProcessingStatus.DETECTING,
        30,
        "Detecting faces and objects...",
    )

    # n_init=1: confirm immediately. max_age=3: kill stale tracks
    # after 3 missed sampled frames (~6 real seconds), not 60.
    tracker = DeepSort(max_age=3, n_init=1)

    timeline: dict[str, dict[str, Any]] = {}
    # actor_name → {seconds, scenes, first_seen}
    screentime: dict[str, dict[str, Any]] = {}
    objects_summary: dict[str, int] = {}

    # --- Stage: TRACKING ---
    _set_status(
        ProcessingStatus.TRACKING,
        50,
        "Tracking faces across frames...",
    )

    sorted_timestamps = sorted(frames.keys())

    for ts in sorted_timestamps:
        frame = frames[ts]

        face_hits = track_and_match_faces(
            frame,
            tracker,
            face_model,
            actor_db,
            FACE_THRESHOLD,
        )

        # --- Stage: MATCHING (reported once at mid-point) ---
        if ts == sorted_timestamps[len(sorted_timestamps) // 2]:
            _set_status(
                ProcessingStatus.MATCHING,
                70,
                "Matching actor identities...",
            )

        obj_hits = detect_objects(
            frame, yolo_model, YOLO_CONFIDENCE
        )

        actors_in_frame: list[str] = []
        track_ids_in_frame: list[int] = []

        for hit in face_hits:
            track_ids_in_frame.append(hit["track_id"])
            actors_in_frame.append(hit)  # Store the full hit dict (actor, score, bbox)
            
            if hit["actor"] is not None:
                actor_name: str = hit["actor"]
                if actor_name not in screentime:
                    screentime[actor_name] = {
                        "seconds": 0,
                        "scenes": 0,
                        "first_seen": ts,
                    }
                screentime[actor_name]["seconds"] += (
                    FRAME_SAMPLE_INTERVAL
                )
                screentime[actor_name]["scenes"] += 1

        for obj in obj_hits:
            label = obj["label"]
            objects_summary[label] = (
                objects_summary.get(label, 0) + 1
            )

        timeline[str(ts)] = {
            "actors": actors_in_frame,
            "objects": obj_hits,
            "track_ids": track_ids_in_frame,
        }

    elapsed: float = round(time.time() - start_time, 2)

    result: dict[str, Any] = {
        "video_id": video_id,
        "timeline": timeline,
        "screentime": screentime,
        "objects_summary": objects_summary,
        "metadata": {
            "duration_seconds": meta["duration_seconds"],
            "fps": meta["fps"],
            "frames_processed": meta["frames_processed"],
            "processing_time_seconds": elapsed,
        },
    }

    # --- Stage: SAVING ---
    _set_status(
        ProcessingStatus.SAVING,
        90,
        "Saving timeline to disk...",
    )

    output_path = os.path.join(
        processed_dir, f"{video_id}.json"
    )
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    except OSError as exc:
        raise IOError(
            f"Cannot write result JSON to {output_path}."
        ) from exc

    # --- Stage: DONE ---
    _set_status(
        ProcessingStatus.DONE,
        100,
        "Processing complete.",
    )

    logger.info(
        "Pipeline finished for video_id=%s in %.2f s.",
        video_id,
        elapsed,
    )
    return result
