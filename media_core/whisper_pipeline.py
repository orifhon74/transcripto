import os
from typing import List, Dict, Tuple, Optional

# Local
from faster_whisper import WhisperModel

# OpenAI
from openai import OpenAI


def _pick_device() -> str:
    return os.getenv("DEVICE", "cpu")


def _pick_compute_type(device: str) -> str:
    return os.getenv("COMPUTE_TYPE", "int8" if device == "cpu" else "float16")


# -------------------------
# ENV CONFIG
# -------------------------
TRANSCRIBE_BACKEND = (os.getenv("TRANSCRIBE_BACKEND", "local") or "local").lower()
WHISPER_LANGUAGE = (os.getenv("WHISPER_LANGUAGE", "en") or "en").strip().lower()

# Local whisper settings
_WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
_DEVICE = _pick_device()
_COMPUTE = _pick_compute_type(_DEVICE)

# OpenAI settings
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
OPENAI_DIARIZE_MODEL = os.getenv("OPENAI_DIARIZE_MODEL", "gpt-4o-transcribe-diarize")
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "120"))

print(
    f"[transcribe] backend={TRANSCRIBE_BACKEND} "
    f"local_model={_WHISPER_MODEL_NAME} device={_DEVICE} compute={_COMPUTE} "
    f"lang={WHISPER_LANGUAGE} openai_model={OPENAI_TRANSCRIBE_MODEL}"
)

_fw: Optional[WhisperModel] = None
_oa: Optional[OpenAI] = None


def _get_local_model() -> WhisperModel:
    global _fw
    if _fw is None:
        _fw = WhisperModel(_WHISPER_MODEL_NAME, device=_DEVICE, compute_type=_COMPUTE)
    return _fw


def _get_openai() -> OpenAI:
    global _oa
    if _oa is None:
        _oa = OpenAI(timeout=OPENAI_TIMEOUT_SEC)
    return _oa


def _run_local(path: str) -> Tuple[List[Dict], str]:
    results: List[Dict] = []
    model = _get_local_model()
    segments, _info = model.transcribe(
        path,
        language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=1,
        best_of=1,
    )
    for s in segments:
        results.append(
            {
                "start": float(s.start or 0.0),
                "end": float(s.end or 0.0),
                "text": (s.text or "").strip(),
            }
        )
    transcript = " ".join(seg["text"] for seg in results).strip()
    return results, transcript


def _run_openai_transcribe(path: str, *, model_name: str) -> Tuple[List[Dict], str]:
    """
    Uses OpenAI transcription. Returns (segments, transcript).

    IMPORTANT:
    - Some models may return rich segments; some may return just `.text`.
    - We normalize to your existing segment schema.
    """
    client = _get_openai()

    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model_name,
            file=f,
            # Keep language explicit if you want consistent results.
            # If you want auto-detect, set WHISPER_LANGUAGE="" in env.
            language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None,
            response_format="verbose_json",
        )

    # Newer responses: dict-like with .text and maybe .segments
    text = (getattr(resp, "text", None) or "").strip()

    segs_out: List[Dict] = []

    segs = getattr(resp, "segments", None)
    if segs:
        for s in segs:
            segs_out.append(
                {
                    "start": float(getattr(s, "start", 0.0) or 0.0),
                    "end": float(getattr(s, "end", 0.0) or 0.0),
                    "text": (getattr(s, "text", "") or "").strip(),
                }
            )
        # if .text empty, rebuild from segments
        if not text:
            text = " ".join(x["text"] for x in segs_out).strip()
        return segs_out, text

    # Fallback: no timestamps returned (rare) -> single segment
    if not text:
        # some SDK versions might store content differently
        text = str(resp).strip()

    segs_out = [{"start": 0.0, "end": 0.0, "text": text}]
    return segs_out, text


def run_whisper(path: str) -> Tuple[List[Dict], str]:
    """
    Core transcription call. Returns (segments, transcript).
    Backend chosen by TRANSCRIBE_BACKEND.
    """
    if TRANSCRIBE_BACKEND == "openai":
        return _run_openai_transcribe(path, model_name=OPENAI_TRANSCRIBE_MODEL)

    # default local
    return _run_local(path)


def run_whisper_diarize_model(path: str) -> Tuple[List[Dict], str]:
    """
    Optional: use OpenAI diarization model (speaker labels).
    If the model returns speaker labels, we try to map them into `speaker` field.

    If your model/account doesn't return speaker labels, it still returns segments+text.
    """
    if TRANSCRIBE_BACKEND != "openai":
        # if not using OpenAI, just reuse local
        return _run_local(path)

    client = _get_openai()
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=OPENAI_DIARIZE_MODEL,
            file=f,
            language=WHISPER_LANGUAGE if WHISPER_LANGUAGE else None,
            response_format="verbose_json",
        )

    text = (getattr(resp, "text", None) or "").strip()
    segs_out: List[Dict] = []

    segs = getattr(resp, "segments", None)
    if segs:
        for s in segs:
            segs_out.append(
                {
                    "start": float(getattr(s, "start", 0.0) or 0.0),
                    "end": float(getattr(s, "end", 0.0) or 0.0),
                    "text": (getattr(s, "text", "") or "").strip(),
                    # some diarize models may expose speaker-like fields
                    "speaker": getattr(s, "speaker", None) or getattr(s, "speaker_label", None),
                }
            )
        if not text:
            text = " ".join(x["text"] for x in segs_out).strip()
        return segs_out, text

    if not text:
        text = str(resp).strip()
    segs_out = [{"start": 0.0, "end": 0.0, "text": text, "speaker": None}]
    return segs_out, text