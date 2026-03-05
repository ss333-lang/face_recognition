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
# Max embeddings stored per track — prevents O(N) mean on long videos
MAX_TRACK_EMBEDDINGS: int = 30

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
    """Status values for the video processing pipeline."""
    QUEUED = "queued"
    EXTRACTING = "extracting_frames"
    DETECTING = "detecting_faces"
    TRACKING = "tracking"
    MATCHING = "matching"
    SAVING = "saving"
    DONE = "done"
    ERROR = "error"


def _get_overlap_ratio(boxA: list[float], boxB: list[float]) -> float:
    """Calculates the intersection over Area A (is A inside B?)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    return interArea / float(areaA) if areaA > 0 else 0


def load_actor_db(conn: Any) -> dict[str, list[float]]:
    """Build an in-memory mapping of actor name -> embedding."""
    from src.video_intel.database import get_all_actors

    rows = get_all_actors(conn)
    actor_db: dict[str, list[float]] = {}
    for row in rows:
        emb = row.get("embedding")
        if emb is not None:
            actor_db[row["name"]] = emb
    logger.info("Loaded %d actors into memory.", len(actor_db))
    return actor_db


def extract_frames(
    video_path: str,
    interval: int,
) -> tuple[dict[int, Any], dict[str, float]]:
    """Sample one frame every N seconds using direct seeking.

    Uses ``cap.set(cv2.CAP_PROP_POS_MSEC)`` to jump directly to
    target timestamps rather than decoding every frame.  For a 60 s
    video at 30 fps with interval=2, this reads 30 frames instead of
    1 800 — a ~60× speed-up on typical content.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Guard against bogus FPS metadata (the '1000 FPS' bug)
    raw_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if raw_fps > 120.0 or raw_fps < 1.0:
        raw_fps = 25.0

    total_frames: float = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration_ms: float = (total_frames / raw_fps) * 1000.0

    frames: dict[int, Any] = {}
    target_ms: float = 0.0

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
            ret, frame = cap.read()
            if not ret:
                break
            pos_sec: int = int(target_ms / 1000)
            frames[pos_sec] = frame
            target_ms += interval * 1000.0
            # Stop if we've passed the video end
            if duration_ms > 0 and target_ms > duration_ms + 500:
                break
    except Exception as exc:
        raise IOError(
            f"Error seeking to {target_ms:.0f} ms in {video_path}."
        ) from exc
    finally:
        cap.release()

    # Use estimate when duration_ms is available, else last target
    duration_seconds: float = (
        duration_ms / 1000.0 if duration_ms > 0 else target_ms / 1000.0
    )
    meta: dict[str, float] = {
        "duration_seconds": duration_seconds,
        "fps": raw_fps,
        "frames_processed": float(len(frames)),
    }
    return frames, meta


def detect_objects(
    frame: Any,
    model: Any,
    confidence: float,
) -> list[dict[str, Any]]:
    """Run YOLOv8 on a frame and return relevant class names."""
    results = model(frame, verbose=False)
    found_objs: list[dict[str, Any]] = []
    
    for result in results:
        for box in result.boxes:
            score: float = float(box.conf[0])
            if score < confidence:
                continue
            class_idx: int = int(box.cls[0])
            class_name: str = result.names.get(class_idx, "")
            if class_name in RELEVANT_OBJECTS:
                x1, y1, x2, y2 = (
                    float(box.xyxy[0][0]),
                    float(box.xyxy[0][1]),
                    float(box.xyxy[0][2]),
                    float(box.xyxy[0][3]),
                )
                w, h = x2 - x1, y2 - y1
                found_objs.append({
                    "label": class_name,
                    "confidence": round(score, 2),
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)]
                })
    return found_objs


def _cosine_match(
    embedding: list[float],
    actor_names: list[str],
    actor_matrix: "np.ndarray | None",
    threshold: float,
) -> tuple[str | None, float]:
    """Find the best matching actor via vectorized cosine similarity.

    Uses a pre-built (N × 512) matrix so all scores are computed with
    a single matrix-vector multiply instead of a Python loop.  For a
    database of 1 000 actors this is ~1 000× faster than the loop.

    Both the query embedding and all rows in ``actor_matrix`` must be
    L2-normalised so that the dot product equals cosine similarity.
    """
    if not actor_names or actor_matrix is None:
        return None, 0.0

    query = np.array(embedding, dtype=np.float32)
    # Shape: (N,) — cosine similarities for all actors at once
    scores: "np.ndarray" = actor_matrix @ query
    best_idx: int = int(np.argmax(scores))
    best_score: float = float(scores[best_idx])

    if best_score >= threshold:
        return actor_names[best_idx], round(best_score, 2)
    return None, 0.0


def _calc_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculates Intersection over Union for two bounding boxes.

    Args:
        box_a (list[float]): Bounding box [x, y, w, h].
        box_b (list[float]): Bounding box [x, y, w, h].

    Returns:
        float: the intersection-over-union score.
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y_b = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    if inter_area == 0:
        return 0.0

    box_a_area = box_a[2] * box_a[3]
    box_b_area = box_b[2] * box_b[3]
    return inter_area / float(box_a_area + box_b_area - inter_area)


def track_and_match_faces(
    frame: Any,
    tracker: Any,
    face_model: Any,
    actor_names: list[str],
    actor_matrix: "np.ndarray | None",
    threshold: float,
    track_embeddings: dict[int, list[list[float]]],
    track_identities: dict[int, tuple[str | None, float]],
) -> list[dict[str, Any]]:
    """Detect, track, and match faces in a single frame.

    Args:
        frame: The image matrix (BGR, uint8).
        tracker: DeepSORT instance.
        face_model: InsightFace FaceAnalysis model.
        actor_names: Ordered list of actor names matching
            rows in ``actor_matrix``.
        actor_matrix: Pre-built (N × 512) float32 array of
            L2-normalised actor embeddings.  Pass ``None``
            when no actors are registered.
        threshold: Minimum cosine similarity to accept a match.
        track_embeddings: Accumulated embeddings per track_id
            (capped at MAX_TRACK_EMBEDDINGS per track).
        track_identities: Best confirmed identity per track_id.

    Returns:
        List of dicts with ``track_id``, ``actor``, ``score``,
        and ``bbox`` [x, y, w, h] for confirmed tracks only.
    """
    faces = face_model.get(frame)
    if not faces:
        # Still update tracker with empty detections so ages increment correctly
        tracker.update_tracks([], frame=frame)
        return []

    detections: list[tuple[list[float], float, str]] = []
    face_boxes: list[list[float]] = []
    embeddings: list[list[float]] = []
    det_scores: list[float] = []

    for face in faces:
        bbox = face.bbox
        x1, y1 = float(bbox[0]), float(bbox[1])
        x2, y2 = float(bbox[2]), float(bbox[3])
        tlwh = [x1, y1, x2 - x1, y2 - y1]
        score_f = float(face.det_score)
        detections.append((tlwh, score_f, "face"))
        face_boxes.append(tlwh)
        embeddings.append(face.normed_embedding.tolist())
        det_scores.append(score_f)

    tracks = tracker.update_tracks(detections, frame=frame)

    results: list[dict[str, Any]] = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        # GHOST TRACK FIX: skip tracks that were NOT matched to a real detection
        # in this frame.  time_since_update == 0 means InsightFace found the face
        # right now.  Values > 0 mean DeepSORT is predicting position via Kalman
        # filter — the person may not even be on screen.  Including these caused:
        #   • phantom bboxes in frames where actors are absent
        #   • scenes always counting as 1 (no real gaps)
        #   • inflated screentime seconds
        if track.time_since_update > 0:
            continue

        track_id = int(track.track_id)
        bbox = track.to_tlwh()

        best_iou = 0.0
        best_idx = -1
        for i, f_tlwh in enumerate(face_boxes):
            iou = _calc_iou(bbox, f_tlwh)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou > 0.3:
            det_score = det_scores[best_idx]
            # ACCURACY FIX: only accumulate HIGH-QUALITY embeddings.
            # Low-confidence detections (blurry, occluded, side-profile)
            # drag down the running average and reduce match scores.
            # det_score >= 0.5 keeps sharp, frontal faces only.
            if det_score >= 0.5:
                if track_id not in track_embeddings:
                    track_embeddings[track_id] = []
                track_embeddings[track_id].append(embeddings[best_idx])
                # Cap history to prevent O(N) mean on long videos
                if len(track_embeddings[track_id]) > MAX_TRACK_EMBEDDINGS:
                    track_embeddings[track_id] = (
                        track_embeddings[track_id][-MAX_TRACK_EMBEDDINGS:]
                    )

        actor_name: str | None = None
        score: float = 0.0

        if track_id in track_embeddings and track_embeddings[track_id]:
            embs = track_embeddings[track_id]
            avg_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm

            avg_emb_list: list[float] = avg_emb.tolist()
            tmp_name, tmp_score = _cosine_match(
                avg_emb_list, actor_names, actor_matrix, threshold
            )

            if track_id in track_identities and \
                    track_identities[track_id][1] > 0.65:
                actor_name, score = track_identities[track_id]
            else:
                track_identities[track_id] = (tmp_name, tmp_score)
                actor_name, score = tmp_name, tmp_score

        results.append({
            "track_id": track_id,
            "actor": actor_name,
            "score": score,
            "bbox": [
                round(bbox[0], 1),
                round(bbox[1], 1),
                round(bbox[2], 1),
                round(bbox[3], 1),
            ],
        })

    best_per_actor: dict[str, tuple[int, float]] = {}
    for r in results:
        name = r["actor"]
        if name is not None:
            if name not in best_per_actor or \
               r["score"] > best_per_actor[name][1]:
                best_per_actor[name] = (r["track_id"], r["score"])

    deduped: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for r in results:
        name = r["actor"]
        if name is None:
            deduped.append(r)
        elif name not in seen_names and \
             best_per_actor.get(name, (0, 0.0))[0] == r["track_id"]:
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
    """Execute the full video intelligence pipeline."""
    from deep_sort_realtime.deepsort_tracker import DeepSort

    processed_dir = os.getenv("PROCESSED_DIR", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    def _set_status(status: ProcessingStatus, progress: int, step: str) -> None:
        redis_client.set(
            f"status:{video_id}",
            json.dumps(
                {"status": status.value, "progress_pct": progress, "current_step": step}
            ),
        )

    start_time = time.time()
    _set_status(ProcessingStatus.EXTRACTING, 10, "Extracting frames...")
    frames, meta = extract_frames(video_path, FRAME_SAMPLE_INTERVAL)

    _set_status(ProcessingStatus.DETECTING, 30, "Analyzing video content...")

    # Pre-build actor name list + L2-normalised embedding matrix ONCE.
    # _cosine_match uses a single matrix-vector multiply (actor_matrix @ query)
    # instead of a Python loop, giving ~1000× speed-up for large DBs.
    actor_names: list[str] = list(actor_db.keys())
    actor_matrix: np.ndarray | None = None
    if actor_names:
        raw = np.array(
            [actor_db[n] for n in actor_names], dtype=np.float32
        )  # shape (N, 512)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        actor_matrix = raw / norms

    # n_init=3: require 3 consecutive detections before confirming a track
    # (was 1 — caused many false/ghost tracks on noisy detections).
    # max_age=60: keep tracks alive longer to survive brief occlusions.
    # max_cosine_distance=0.4: stricter ReID feature matching.
    tracker = DeepSort(max_age=60, n_init=3, max_cosine_distance=0.4)

    timeline: dict[str, Any] = {}
    screentime: dict[str, Any] = {}
    objects_summary: dict[str, int] = {}
    prev_person_boxes: list[list[float]] = []

    track_embeddings: dict[int, list[list[float]]] = {}
    track_identities: dict[int, tuple[str | None, float]] = {}

    sorted_timestamps = sorted(frames.keys())
    for ts in sorted_timestamps:
        frame = frames[ts]
        face_hits = track_and_match_faces(
            frame,
            tracker,
            face_model,
            actor_names,
            actor_matrix,
            FACE_THRESHOLD,
            track_embeddings,
            track_identities,
        )
        obj_hits = detect_objects(frame, yolo_model, YOLO_CONFIDENCE)

        current_person_boxes: list[list[float]] = []
        actors_in_frame: list[dict[str, Any]] = []
        track_ids_in_frame: list[int] = []

        for hit in face_hits:
            track_ids_in_frame.append(hit["track_id"])
            actors_in_frame.append(hit)

        for obj in obj_hits:
            label = obj["label"]
            if label == "person":
                is_actor = False
                for fhit in face_hits:
                    if _get_overlap_ratio(obj["bbox"], fhit["bbox"]) > 0.7:
                        is_actor = True
                        break
                if is_actor: 
                    continue
                
                is_duplicate = False
                for pbox in prev_person_boxes:
                    if _get_overlap_ratio(obj["bbox"], pbox) > 0.8:
                        is_duplicate = True
                        break
                
                current_person_boxes.append(obj["bbox"])
                if is_duplicate: 
                    continue
            
            objects_summary[label] = objects_summary.get(label, 0) + 1
            
        prev_person_boxes = current_person_boxes
        frame_payload = {
            "actors": actors_in_frame, 
            "objects": obj_hits, 
            "track_ids": track_ids_in_frame
        }
        timeline[str(ts)] = frame_payload
        
        # Publish live frame tracking to Redis for Real-Time clients
        redis_client.publish(
            f"realtime:{video_id}",
            json.dumps({
                "type": "frame_result",
                "timestamp_seconds": float(ts),
                "data": frame_payload
            })
        )

    # Lookahead / Backdating pass
    for ts_str, frame_data in timeline.items():
        for fhit in frame_data.get("actors", []):
            tid = fhit.get("track_id")
            if tid in track_identities:
                final_name, final_score = track_identities[tid]
                fhit["actor"] = final_name
                fhit["score"] = final_score

    # Re-calculate screentime accurately
    prev_actors_list: list[str] = []
    for ts_str in sorted(timeline.keys(), key=float):
        c_actors = set()
        for fhit in timeline[ts_str].get("actors", []):
            name = fhit.get("actor")
            if name is not None:
                c_actors.add(name)
                if name not in screentime:
                    screentime[name] = {
                        "seconds": 0, "scenes": 0, "first_seen": float(ts_str)
                    }
                screentime[name]["seconds"] += FRAME_SAMPLE_INTERVAL
                if name not in prev_actors_list:
                    screentime[name]["scenes"] += 1
        prev_actors_list = list(c_actors)

    elapsed = round(time.time() - start_time, 2)
    
    final_fps = meta["fps"]
    if final_fps > 100.0 or final_fps < 1.0:
        final_fps = 25.0

    result = {
        "video_id": video_id,
        "timeline": timeline,
        "screentime": screentime,
        "objects_summary": objects_summary,
        "metadata": {
            "duration_seconds": meta["duration_seconds"],
            "fps": final_fps,
            "frames_processed": meta["frames_processed"],
            "processing_time_seconds": elapsed,
        },
    }

    output_path = os.path.join(processed_dir, f"{video_id}.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _set_status(ProcessingStatus.DONE, 100, "Processing Complete.")
    redis_client.publish(f"realtime:{video_id}", "DONE")
    return result
