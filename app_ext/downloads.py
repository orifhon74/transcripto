# app_ext/downloads.py

from uuid import uuid4
from datetime import datetime, timedelta
from threading import Lock

from config import Config

DOWNLOADS: dict[str, dict] = {}
DL_LOCK = Lock()


def register_download(data: bytes, mimetype: str, filename: str) -> str:
    token = uuid4().hex
    with DL_LOCK:
        DOWNLOADS[token] = {
            "data": data,
            "mimetype": mimetype,
            "filename": filename,
            "created_at": datetime.utcnow(),
        }
    return token


def cleanup_downloads(now: datetime | None = None):
    """Remove expired downloads from memory."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(minutes=Config.DOWNLOAD_TTL_MIN)
    to_delete: list[str] = []

    with DL_LOCK:
        for t, item in list(DOWNLOADS.items()):
            if item["created_at"] < cutoff:
                to_delete.append(t)
        for t in to_delete:
            DOWNLOADS.pop(t, None)