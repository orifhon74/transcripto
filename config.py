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

    # ---- Transcription switching ----
    # local | openai
    TRANSCRIBE_BACKEND = os.getenv("TRANSCRIBE_BACKEND", "local")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
    OPENAI_DIARIZE_MODEL = os.getenv("OPENAI_DIARIZE_MODEL", "gpt-4o-transcribe-diarize")
    OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "120"))

    # ---- Diarization ----
    # cpu | openai
    DIARIZATION_BACKEND = os.getenv("DIARIZATION_BACKEND", "cpu")
    DIARIZATION_MODE = os.getenv("DIARIZATION_MODE", "fast")  # off|fast|accurate|auto

    # ---- PDF Fonts ----
    # Put DejaVuSans.ttf under static/fonts and set this path accordingly
    PDF_FONT_TTF = os.getenv("PDF_FONT_TTF", "static/fonts/DejaVuSans.ttf")