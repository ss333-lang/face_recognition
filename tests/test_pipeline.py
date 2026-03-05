"""Unit tests for the video intelligence pipeline module.

Tests cover frame extraction, cosine matching, object
detection filtering, and the full run_pipeline flow using
synthetic data so no real GPU or video file is required.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.video_intel.pipeline import (
    ProcessingStatus,
    _cosine_match,
    detect_objects,
    extract_frames,
    run_pipeline,
    track_and_match_faces,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_black_frame(
    height: int = 240,
    width: int = 320,
) -> Any:
    """Return a black BGR numpy array of the given size.

    Args:
        height (int): Frame height in pixels. Defaults to 240.
        width (int): Frame width in pixels. Defaults to 320.

    Returns:
        Any: BGR numpy array shaped (height, width, 3).
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_video_file(
    path: str,
    num_frames: int = 60,
    fps: float = 10.0,
) -> None:
    """Write a minimal synthetic video to disk.

    Uses the MJPG codec (widely available on CI systems)
    and fills every frame with a solid colour.

    Args:
        path (str): Destination file path.
        num_frames (int): Number of frames to write.
        fps (float): Frames per second for the output file.
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(num_frames):
        # Alternate frame colours so the video is non-trivial
        colour = (i * 4 % 256, 100, 200)
        frame = np.full((240, 320, 3), colour, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _unit_vector(dim: int = 512) -> list[float]:
    """Return a random L2-normalised vector.

    Args:
        dim (int): Dimensionality. Defaults to 512.

    Returns:
        list[float]: Normalised vector.
    """
    vec = np.random.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()


# ---------------------------------------------------------------------------
# ProcessingStatus
# ---------------------------------------------------------------------------


class TestProcessingStatus:
    """Tests for the ProcessingStatus enum."""

    def test_all_values_are_strings(self) -> None:
        """Every enum member value should be a plain string."""
        for member in ProcessingStatus:
            assert isinstance(member.value, str)

    def test_done_value(self) -> None:
        """DONE sentinel must equal the string 'done'."""
        assert ProcessingStatus.DONE.value == "done"

    def test_error_value(self) -> None:
        """ERROR sentinel must equal the string 'error'."""
        assert ProcessingStatus.ERROR.value == "error"

    def test_unique_values(self) -> None:
        """Each enum member must have a distinct value."""
        values = [m.value for m in ProcessingStatus]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# _cosine_match
# ---------------------------------------------------------------------------


class TestCosineMatch:
    """Tests for the _cosine_match helper function."""

    def test_exact_match_returns_actor(self) -> None:
        """A query identical to a known embedding should match."""
        emb = _unit_vector()
        actor_db = {"Alice": emb}
        name, score = _cosine_match(emb, actor_db, threshold=0.45)
        assert name == "Alice"
        assert score > 0.99

    def test_orthogonal_vectors_no_match(self) -> None:
        """Orthogonal embeddings have zero similarity."""
        dim = 512
        ref = np.zeros(dim, dtype=np.float32)
        ref[0] = 1.0
        query = np.zeros(dim, dtype=np.float32)
        query[1] = 1.0
        actor_db = {"Bob": ref.tolist()}
        name, score = _cosine_match(
            query.tolist(), actor_db, threshold=0.45
        )
        assert name is None
        assert score == 0.0

    def test_empty_actor_db_returns_none(self) -> None:
        """An empty actor database should always return None."""
        emb = _unit_vector()
        name, score = _cosine_match(emb, {}, threshold=0.45)
        assert name is None
        assert score == 0.0

    def test_below_threshold_returns_none(self) -> None:
        """A match below threshold should be suppressed."""
        emb = _unit_vector()
        # Negate to get cosine similarity ≈ -1
        opposite = [-v for v in emb]
        actor_db = {"Carol": opposite}
        name, score = _cosine_match(
            emb, actor_db, threshold=0.45
        )
        assert name is None

    def test_best_of_multiple_actors_is_selected(self) -> None:
        """Highest-similarity actor should be returned."""
        base = np.random.randn(512).astype(np.float32)
        base /= np.linalg.norm(base) + 1e-9
        # Slightly perturbed version — still very similar
        close = base + np.random.randn(512).astype(np.float32) * 0.01
        close /= np.linalg.norm(close) + 1e-9
        # Orthogonal reference
        orth = np.random.randn(512).astype(np.float32)
        orth -= orth.dot(base) * base
        orth /= np.linalg.norm(orth) + 1e-9

        actor_db = {
            "NearActor": close.tolist(),
            "FarActor": orth.tolist(),
        }
        name, score = _cosine_match(
            base.tolist(), actor_db, threshold=0.1
        )
        assert name == "NearActor"
        assert score > 0.9


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------


class TestExtractFrames:
    """Tests for extract_frames()."""

    def test_raises_on_missing_file(self) -> None:
        """Non-existent video path should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot open video"):
            extract_frames("/nonexistent/path.mp4", interval=2)

    def test_returns_frames_and_metadata(self) -> None:
        """Should return a non-empty frames dict and meta."""
        with tempfile.NamedTemporaryFile(
            suffix=".avi", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            # 30 frames at 10 fps = 3 s; interval=1 → ~3 frames
            _make_video_file(tmp_path, num_frames=30, fps=10.0)
            frames, meta = extract_frames(tmp_path, interval=1)
            assert isinstance(frames, dict)
            assert len(frames) > 0
            assert "duration_seconds" in meta
            assert "fps" in meta
            assert "frames_processed" in meta
            assert meta["fps"] == pytest.approx(10.0, abs=1.0)
        finally:
            os.unlink(tmp_path)

    def test_frame_keys_are_integers(self) -> None:
        """Timestamp keys must be plain Python ints."""
        with tempfile.NamedTemporaryFile(
            suffix=".avi", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            _make_video_file(tmp_path, num_frames=20, fps=10.0)
            frames, _ = extract_frames(tmp_path, interval=1)
            for key in frames:
                assert isinstance(key, int)
        finally:
            os.unlink(tmp_path)

    def test_frame_values_are_numpy_arrays(self) -> None:
        """Every sampled frame should be a numpy ndarray."""
        with tempfile.NamedTemporaryFile(
            suffix=".avi", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            _make_video_file(tmp_path, num_frames=20, fps=10.0)
            frames, _ = extract_frames(tmp_path, interval=1)
            for frame in frames.values():
                assert isinstance(frame, np.ndarray)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# detect_objects
# ---------------------------------------------------------------------------


class TestDetectObjects:
    """Tests for detect_objects()."""

    def _make_mock_yolo(
        self,
        detections: list[tuple[int, float]],
    ) -> MagicMock:
        """Build a mock YOLO model returning given detections.

        Args:
            detections (list[tuple[int, float]]): List of
                (class_index, confidence) pairs.

        Returns:
            MagicMock: Callable mock with ultralytics-like
                output structure.
        """
        results = []
        result_mock = MagicMock()
        result_mock.names = {
            0: "person",
            1: "car",
            2: "banana",
            3: "phone",
        }
        boxes = []
        for cls_idx, conf in detections:
            box = MagicMock()
            box.conf = [conf]
            box.cls = [cls_idx]
            boxes.append(box)
        result_mock.boxes = boxes
        results.append(result_mock)

        model = MagicMock()
        model.return_value = results
        return model

    def test_relevant_objects_are_returned(self) -> None:
        """Detections above threshold in RELEVANT_OBJECTS included."""
        model = self._make_mock_yolo(
            [(0, 0.9), (1, 0.7)]  # person, car
        )
        frame = _make_black_frame()
        found = detect_objects(frame, model, confidence=0.5)
        labels = [x["label"] for x in found]
        assert "person" in labels
        assert "car" in labels

    def test_irrelevant_objects_excluded(self) -> None:
        """Class names not in RELEVANT_OBJECTS must be filtered."""
        model = self._make_mock_yolo([(2, 0.9)])  # banana
        frame = _make_black_frame()
        found = detect_objects(frame, model, confidence=0.5)
        labels = [x["label"] for x in found]
        assert "banana" not in labels

    def test_low_confidence_excluded(self) -> None:
        """Detections below the confidence threshold are dropped."""
        model = self._make_mock_yolo([(0, 0.1)])  # person, low conf
        frame = _make_black_frame()
        found = detect_objects(frame, model, confidence=0.5)
        labels = [x["label"] for x in found]
        assert "person" not in labels

    def test_empty_result_on_no_detections(self) -> None:
        """Empty detection list should return empty list."""
        model = self._make_mock_yolo([])
        frame = _make_black_frame()
        found = detect_objects(frame, model, confidence=0.5)
        assert found == []


# ---------------------------------------------------------------------------
# track_and_match_faces
# ---------------------------------------------------------------------------


class TestTrackAndMatchFaces:
    """Tests for track_and_match_faces()."""

    def test_returns_empty_on_no_faces(self) -> None:
        """If InsightFace finds no faces, return empty list."""
        face_model = MagicMock()
        face_model.get.return_value = []
        tracker = MagicMock()
        actor_db: dict[str, list[float]] = {}
        frame = _make_black_frame()

        result = track_and_match_faces(
            frame, tracker, face_model, actor_db, 0.45, {}, {}
        )
        assert result == []

    def test_matched_actor_appears_in_result(self) -> None:
        """A face with a high-similarity embedding should match."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9

        face = MagicMock()
        face.bbox = [10.0, 10.0, 50.0, 50.0]
        face.det_score = 0.95
        face.normed_embedding = emb

        face_model = MagicMock()
        face_model.get.return_value = [face]

        track = MagicMock()
        track.is_confirmed.return_value = True
        track.track_id = 42
        track.to_tlwh.return_value = [10.0, 10.0, 40.0, 40.0]
        tracker = MagicMock()
        tracker.update_tracks.return_value = [track]

        actor_db = {"TestActor": emb.tolist()}
        frame = _make_black_frame()

        results = track_and_match_faces(
            frame, tracker, face_model, actor_db, 0.45, {}, {}
        )
        assert len(results) == 1
        assert results[0]["track_id"] == 42
        assert results[0]["actor"] == "TestActor"

    def test_unconfirmed_tracks_are_skipped(self) -> None:
        """Tracks not yet confirmed by DeepSORT are ignored."""
        emb = _unit_vector()
        face = MagicMock()
        face.bbox = [10.0, 10.0, 50.0, 50.0]
        face.det_score = 0.95
        face.normed_embedding = np.array(emb)

        face_model = MagicMock()
        face_model.get.return_value = [face]

        track = MagicMock()
        track.is_confirmed.return_value = False  # not confirmed
        tracker = MagicMock()
        tracker.update_tracks.return_value = [track]

        frame = _make_black_frame()
        results = track_and_match_faces(
            frame, tracker, face_model, {}, 0.45, {}, {}
        )
        assert results == []


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Integration-style tests for run_pipeline()."""

    def _make_redis(self) -> MagicMock:
        """Build a minimal in-memory Redis mock.

        Returns:
            MagicMock: Mock with ``set`` / ``get`` / ``ping``.
        """
        store: dict[str, bytes] = {}
        client = MagicMock()
        client.set.side_effect = lambda k, v: store.__setitem__(
            k, v.encode() if isinstance(v, str) else v
        )
        client.get.side_effect = lambda k: store.get(k)
        client.ping.return_value = True
        return client

    def test_returns_expected_keys(self) -> None:
        """Output dict must contain all top-level schema keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.avi")
            _make_video_file(video_path, num_frames=20, fps=10.0)

            face_model = MagicMock()
            face_model.get.return_value = []

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.names = {}
            yolo_model = MagicMock()
            yolo_model.return_value = [mock_result]

            redis = self._make_redis()

            with patch.dict(
                os.environ,
                {
                    "PROCESSED_DIR": tmpdir,
                    "FRAME_SAMPLE_INTERVAL": "1",
                },
            ):
                result = run_pipeline(
                    video_path=video_path,
                    video_id="test123",
                    face_model=face_model,
                    yolo_model=yolo_model,
                    actor_db={},
                    redis_client=redis,
                )

            required_keys = {
                "video_id",
                "timeline",
                "screentime",
                "objects_summary",
                "metadata",
            }
            assert required_keys.issubset(result.keys())
            assert result["video_id"] == "test123"

    def test_result_json_written_to_disk(self) -> None:
        """Pipeline must write a valid JSON file to PROCESSED_DIR."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test2.avi")
            _make_video_file(video_path, num_frames=20, fps=10.0)

            face_model = MagicMock()
            face_model.get.return_value = []

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.names = {}
            yolo_model = MagicMock()
            yolo_model.return_value = [mock_result]

            redis = self._make_redis()

            vid_id = "disk_test"
            with patch.dict(
                os.environ,
                {
                    "PROCESSED_DIR": tmpdir,
                    "FRAME_SAMPLE_INTERVAL": "1",
                },
            ):
                run_pipeline(
                    video_path=video_path,
                    video_id=vid_id,
                    face_model=face_model,
                    yolo_model=yolo_model,
                    actor_db={},
                    redis_client=redis,
                )

            out_path = Path(tmpdir) / f"{vid_id}.json"
            assert out_path.exists()
            with out_path.open("r") as fh:
                data = json.load(fh)
            assert data["video_id"] == vid_id

    def test_metadata_has_fps_and_duration(self) -> None:
        """Metadata block must expose fps and duration_seconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test3.avi")
            _make_video_file(video_path, num_frames=20, fps=10.0)

            face_model = MagicMock()
            face_model.get.return_value = []

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.names = {}
            yolo_model = MagicMock()
            yolo_model.return_value = [mock_result]

            redis = self._make_redis()

            with patch.dict(
                os.environ,
                {
                    "PROCESSED_DIR": tmpdir,
                    "FRAME_SAMPLE_INTERVAL": "1",
                },
            ):
                result = run_pipeline(
                    video_path=video_path,
                    video_id="meta_test",
                    face_model=face_model,
                    yolo_model=yolo_model,
                    actor_db={},
                    redis_client=redis,
                )

            meta = result["metadata"]
            assert "fps" in meta
            assert "duration_seconds" in meta
            assert meta["fps"] > 0
            assert meta["duration_seconds"] >= 0

    def test_redis_status_set_to_done(self) -> None:
        """Redis status for video_id must be DONE after completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test4.avi")
            _make_video_file(video_path, num_frames=10, fps=10.0)

            face_model = MagicMock()
            face_model.get.return_value = []

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.names = {}
            yolo_model = MagicMock()
            yolo_model.return_value = [mock_result]

            store: dict[str, bytes] = {}
            redis = MagicMock()
            redis.set.side_effect = lambda k, v: store.__setitem__(
                k, v.encode() if isinstance(v, str) else v
            )
            redis.get.side_effect = lambda k: store.get(k)

            vid_id = "done_test"
            with patch.dict(
                os.environ,
                {
                    "PROCESSED_DIR": tmpdir,
                    "FRAME_SAMPLE_INTERVAL": "1",
                },
            ):
                run_pipeline(
                    video_path=video_path,
                    video_id=vid_id,
                    face_model=face_model,
                    yolo_model=yolo_model,
                    actor_db={},
                    redis_client=redis,
                )

            final_raw = store.get(f"status:{vid_id}")
            assert final_raw is not None
            final = json.loads(final_raw)
            assert final["status"] == ProcessingStatus.DONE.value
            assert final["progress_pct"] == 100
