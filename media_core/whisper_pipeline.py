# media_core/whisper_pipeline.py

import os
from typing import List, Dict, Tuple

from faster_whisper import WhisperModel


def _pick_device() -> str:
    return os.getenv("DEVICE", "cpu")


def _pick_compute_type(device: str) -> str:
    return os.getenv("COMPUTE_TYPE", "int8" if device == "cpu" else "float16")


_WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
_DEVICE = _pick_device()
_COMPUTE = _pick_compute_type(_DEVICE)
print(f"[whisper] model={_WHISPER_MODEL_NAME} device={_DEVICE} compute_type={_COMPUTE}")

_fw = WhisperModel(_WHISPER_MODEL_NAME, device=_DEVICE, compute_type=_COMPUTE)


def run_whisper(path: str) -> Tuple[List[Dict], str]:
    """Core Whisper call. Return (segments, transcript)."""
    results: List[Dict] = []
    segments, _info = _fw.transcribe(
        path,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=1,
        best_of=1,
    )
    for s in segments:
        results.append({
            "start": float(s.start or 0.0),
            "end": float(s.end or 0.0),
            "text": (s.text or "").strip(),
        })
    transcript = " ".join(seg["text"] for seg in results).strip()
    return results, transcript