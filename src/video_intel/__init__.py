"""AI Video Intelligence Platform.

A FastAPI backend that processes videos to detect faces,
match known actors, track objects, and produce a full
timeline JSON — Netflix X-Ray style.

Typical usage:
    uvicorn src.video_intel.main:app --reload --port 8000
"""

__version__ = "1.0.0"
__author__ = "Video Intelligence Team"
