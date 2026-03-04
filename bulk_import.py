#!/usr/bin/env python3
"""Bulk import celebrities from the HuggingFace dataset
tonyassi/celebrity-1000-embeddings into the pgvector actors table.

Loads ~18k face images of 1,000 celebrities, runs InsightFace on each
to extract a 512-d normed embedding (ignoring the dataset's pre-computed
768-d embeddings which are model-incompatible), and upserts every
identity into PostgreSQL via the project's database layer.

Usage (from project root):
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    python3 bulk_import.py

Environment variables read from .env (or shell):
    DATABASE_URL    PostgreSQL connection string
"""
from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict

import cv2
import insightface
import numpy as np
import psycopg2
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres@localhost:5432/videointel",
)

HF_DATASET: str = "tonyassi/celebrity-1000-embeddings"

# How many photos to try per celebrity before giving up
MAX_PHOTOS_TO_TRY: int = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pil_to_bgr(pil_image: object) -> np.ndarray:
    """Convert a PIL Image to a BGR numpy array for InsightFace.

    Args:
        pil_image: A PIL Image object (any mode).

    Returns:
        np.ndarray: BGR image array shaped (H, W, 3).
    """
    rgb = np.array(pil_image.convert("RGB"))  # type: ignore[union-attr]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_embedding(
    pil_image: object,
    face_model: object,
) -> list[float] | None:
    """Run InsightFace on a PIL image and return the normed embedding.

    Args:
        pil_image: A PIL Image object.
        face_model: A prepared InsightFace FaceAnalysis instance.

    Returns:
        list[float] | None: 512-d normed embedding, or None if no face
            was detected.
    """
    img = pil_to_bgr(pil_image)
    faces = face_model.get(img)  # type: ignore[union-attr]
    if faces:
        return faces[0].normed_embedding.tolist()
    return None


def insert_actor_raw(
    name: str,
    embedding: list[float],
    conn: object,
) -> int:
    """Upsert one actor row into the actors table.

    Args:
        name (str): Celebrity display name.
        embedding (list[float]): 512-d normed embedding.
        conn: Open psycopg2 connection.

    Returns:
        int: Inserted or updated row ID.

    Raises:
        psycopg2.DatabaseError: If the upsert fails.
    """
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    with conn.cursor() as cur:  # type: ignore[union-attr]
        cur.execute(
            """
            INSERT INTO actors (name, embedding, photo_path)
            VALUES (%s, %s::vector, %s)
            ON CONFLICT (name) DO UPDATE
                SET embedding  = EXCLUDED.embedding,
                    photo_path = EXCLUDED.photo_path
            RETURNING id;
            """,
            (name, vec_str, HF_DATASET),
        )
        row = cur.fetchone()
    conn.commit()  # type: ignore[union-attr]
    return int(row[0])  # type: ignore[index]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full bulk import pipeline.

    1. Load dataset from HuggingFace.
    2. Build a label → image index map.
    3. For each celebrity: try up to MAX_PHOTOS_TO_TRY images,
       extract a 512-d InsightFace embedding, upsert into pgvector.
    4. Report final counts.
    """
    logger.info("Loading dataset '%s' from HuggingFace...", HF_DATASET)
    dataset = load_dataset(HF_DATASET, split="train")

    label_names: list[str] = dataset.features["label"].names  # type: ignore[index]
    logger.info(
        "Dataset loaded — %d images, %d celebrities.",
        len(dataset),
        len(label_names),
    )

    # Group dataset indices by label for efficient per-celebrity access
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        label_to_indices[dataset[idx]["label"]].append(idx)

    # Initialise InsightFace — always ctx_id=-1 (CPU)
    face_model = insightface.app.FaceAnalysis(
        allowed_modules=["detection", "recognition"]
    )
    face_model.prepare(ctx_id=-1)
    logger.info("InsightFace model ready (CPU mode).")

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError as exc:
        logger.error("Cannot connect to DB: %s", exc)
        sys.exit(1)

    # Ensure schema exists
    from src.video_intel.database import init_db
    init_db(DATABASE_URL)

    inserted: int = 0
    skipped_no_face: int = 0

    for label_id, name in enumerate(label_names):
        indices = label_to_indices.get(label_id, [])

        if not indices:
            logger.warning("No images found for '%s', skipping.", name)
            skipped_no_face += 1
            continue

        embedding: list[float] | None = None
        for idx in indices[:MAX_PHOTOS_TO_TRY]:
            pil_image = dataset[idx]["image"]
            embedding = extract_embedding(pil_image, face_model)
            if embedding is not None:
                break

        if embedding is None:
            logger.warning(
                "No face detected in any photo for '%s', skipping.", name
            )
            skipped_no_face += 1
            continue

        try:
            actor_id = insert_actor_raw(
                name=name,
                embedding=embedding,
                conn=conn,
            )
            logger.info("✔ Inserted '%s' → id=%d", name, actor_id)
            inserted += 1
        except psycopg2.DatabaseError as exc:
            logger.error("DB error for '%s': %s", name, exc)
            conn.rollback()

    conn.close()

    logger.info(
        "\n=== Bulk import complete ===\n"
        "  Inserted / updated : %d\n"
        "  Skipped (no face)  : %d\n"
        "  Total celebrities  : %d",
        inserted,
        skipped_no_face,
        len(label_names),
    )


if __name__ == "__main__":
    main()
