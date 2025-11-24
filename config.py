# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")

    DOWNLOAD_TTL_MIN = int(os.getenv("DOWNLOAD_TTL_MIN", "30"))
    JOB_WORKERS = int(os.getenv("JOB_WORKERS", "2"))
    JOB_TTL_MIN = int(os.getenv("JOB_TTL_MIN", "60"))
    MAX_JOB_LOG_LINES = int(os.getenv("MAX_JOB_LOG_LINES", "200"))
    CLEANUP_INTERVAL_SEC = int(os.getenv("CLEANUP_INTERVAL_SEC", "120"))