#!/usr/bin/env python3
"""Backfill actor thumbnails from the HuggingFace dataset.

Grabs exactly ONE 256×256 face image per celebrity and saves it to
``actors/<name>.jpg``.  Does NOT re-run InsightFace — just copies the
first PIL image from the dataset, which is already face-cropped.

Run from the project root AFTER activating the virtualenv:
    python3 backfill_photos.py

The dataset is usually already cached from the original bulk_import run
so this completes in ~2-5 minutes instead of 30-60 minutes.

Environment variables read (via .env):
    ACTORS_DIR  directory to write photos to (default: "actors")
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ACTORS_DIR: str = os.getenv("ACTORS_DIR", "actors")
HF_DATASET: str = "tonyassi/celebrity-1000-embeddings"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Download and save one reference photo per celebrity."""
    # Import here so the script stays importable without heavy deps
    from datasets import load_dataset  # type: ignore[import]

    logger.info("Loading dataset '%s' (may use local cache)…", HF_DATASET)
    dataset = load_dataset(HF_DATASET, split="train")
    label_names: list[str] = dataset.features["label"].names  # type: ignore[index]
    logger.info(
        "Dataset ready — %d images, %d celebrities.",
        len(dataset),
        len(label_names),
    )

    # Group dataset indices by label for O(1) per-celebrity access
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        label_to_indices[dataset[idx]["label"]].append(idx)

    actors_dir = Path(ACTORS_DIR)
    actors_dir.mkdir(parents=True, exist_ok=True)

    saved: int = 0
    skipped: int = 0

    for label_id, name in enumerate(label_names):
        save_path = actors_dir / f"{name}.jpg"

        if save_path.exists():
            skipped += 1
            continue  # Already have a photo for this celebrity

        indices = label_to_indices.get(label_id, [])
        if not indices:
            logger.warning("No images in dataset for '%s', skipping.", name)
            continue

        # Grab just the first image — already 256×256 face-cropped
        pil_image = dataset[indices[0]]["image"]
        try:
            pil_image.save(str(save_path), format="JPEG", quality=85)
            saved += 1
        except Exception as exc:
            logger.warning(
                "Could not save photo for '%s': %s", name, exc
            )

        if saved % 100 == 0 and saved > 0:
            logger.info("  … saved %d photos so far.", saved)

    logger.info(
        "\n=== Backfill complete ===\n"
        "  Photos saved   : %d\n"
        "  Already existed: %d\n"
        "  Output dir     : %s",
        saved,
        skipped,
        actors_dir.resolve(),
    )


if __name__ == "__main__":
    main()
