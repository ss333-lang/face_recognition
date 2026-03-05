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
    """Read a video file and sample one frame every N seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # BUG FIX: Handle '1000 FPS' bug
    raw_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if raw_fps > 100.0 or raw_fps < 1.0:
        raw_fps = 25.0

    frames: dict[int, Any] = {}
    last_sampled_sec: int = -1
    last_ms: float = 0.0
    frame_idx: int = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pos_ms: float = cap.get(cv2.CAP_PROP_POS_MSEC)
            last_ms = pos_ms
            pos_sec: int = int(pos_ms / 1000)

            if pos_sec >= last_sampled_sec + interval:
                frames[pos_sec] = frame
                last_sampled_sec = pos_sec
            frame_idx += 1
    except Exception as exc:
        raise IOError(f"Error reading frame {frame_idx} from {video_path}.") from exc
    finally:
        cap.release()

    duration_seconds: float = last_ms / 1000.0
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
    actor_db: dict[str, list[float]],
    threshold: float,
) -> tuple[str | None, float]:
    """Find the best matching actor via cosine similarity."""
    if not actor_db:
        return None, 0.0

    query = np.array(embedding, dtype=np.float32)
    best_name, best_score = None, 0.0

    for name, ref_emb in actor_db.items():
        ref = np.array(ref_emb, dtype=np.float32)
        score: float = float(np.dot(query, ref))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, round(best_score, 2)
    return None, 0.0


def track_and_match_faces(
    frame: Any,
    tracker: Any,
    face_model: Any,
    actor_db: dict[str, list[float]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Detect, track, and match faces in a single frame."""
    faces = face_model.get(frame)
    if not faces:
        return []

    detections, embeddings = [], []
    for face in faces:
        bbox = face.bbox
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        detections.append(([x1, y1, x2 - x1, y2 - y1], float(face.det_score), "face"))
        embeddings.append(face.normed_embedding.tolist())

    det_emb_map = {i: emb for i, emb in enumerate(embeddings)}
    tracks = tracker.update_tracks(detections, frame=frame)

    results = []
    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = int(track.track_id)
        # Standard proxy mapping
        emb = det_emb_map.get(i)
        if emb is None and embeddings:
            emb = embeddings[min(i, len(embeddings) - 1)]
        
        actor_name, score = (None, 0.0)
        if emb is not None:
            actor_name, score = _cosine_match(emb, actor_db, threshold)

        bbox = track.to_tlwh()
        results.append({
            "track_id": track_id,
            "actor": actor_name,
            "score": score,
            "bbox": [round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1)],
        })

    # Global Deduplication (to avoid Ryan Gosling being 2 tracks at once)
    best_per_actor = {}
    for r in results:
        name = r["actor"]
        if name and (name not in best_per_actor or r["score"] > best_per_actor[name][1]):
            best_per_actor[name] = (r["track_id"], r["score"])

    deduped = []
    seen_names = set()
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
    """Execute the full video intelligence pipeline."""
    from deep_sort_realtime.deepsort_tracker import DeepSort

    processed_dir = os.getenv("PROCESSED_DIR", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    def _set_status(status, progress, step):
        redis_client.set(f"status:{video_id}", json.dumps({"status": status.value, "progress_pct": progress, "current_step": step}))

    start_time = time.time()
    _set_status(ProcessingStatus.EXTRACTING, 10, "Extracting frames...")
    frames, meta = extract_frames(video_path, FRAME_SAMPLE_INTERVAL)

    _set_status(ProcessingStatus.DETECTING, 30, "Analyzing video content...")
    tracker = DeepSort(max_age=3, n_init=1)

    timeline, screentime, objects_summary = {}, {}, {}
    prev_actors = set()
    # Spatial memory for unidentified persons
    prev_person_boxes = [] 
    
    sorted_timestamps = sorted(frames.keys())
    for ts in sorted_timestamps:
        frame = frames[ts]
        face_hits = track_and_match_faces(frame, tracker, face_model, actor_db, FACE_THRESHOLD)
        obj_hits = detect_objects(frame, yolo_model, YOLO_CONFIDENCE)

        current_actors, current_person_boxes = set(), []
        actors_in_frame, track_ids_in_frame = [], []

        # 1. Process Faces/Actors
        for hit in face_hits:
            track_ids_in_frame.append(hit["track_id"])
            actors_in_frame.append(hit)
            name = hit["actor"]
            if name:
                current_actors.add(name)
                if name not in screentime:
                    screentime[name] = {"seconds": 0, "scenes": 0, "first_seen": ts}
                screentime[name]["seconds"] += FRAME_SAMPLE_INTERVAL
                # Phenomenal Counting Logic: Appearance Based
                if name not in prev_actors:
                    screentime[name]["scenes"] += 1

        # 2. Process Objects and Deduplicate Persons
        for obj in obj_hits:
            label = obj["label"]
            if label == "person":
                # Phenomenal Logic A: Actor overlapping person?
                is_actor = False
                for fhit in face_hits:
                    if _get_overlap_ratio(obj["bbox"], fhit["bbox"]) > 0.7:
                        is_actor = True; break
                if is_actor: continue
                
                # Phenomenal Logic B: Spatial memory (Same person stationary?)
                is_duplicate = False
                for pbox in prev_person_boxes:
                    if _get_overlap_ratio(obj["bbox"], pbox) > 0.8:
                        is_duplicate = True; break
                
                current_person_boxes.append(obj["bbox"])
                if is_duplicate: continue
            
            objects_summary[label] = objects_summary.get(label, 0) + 1
            
        prev_actors = current_actors
        prev_person_boxes = current_person_boxes
        timeline[str(ts)] = {"actors": actors_in_frame, "objects": obj_hits, "track_ids": track_ids_in_frame}

    elapsed = round(time.time() - start_time, 2)
    
    # Final metadata guard for 1000 FPS bug
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
    return result
