import os
import shlex
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional

# ---------------- OpenAI summarizer ----------------
from openai import OpenAI
_client = OpenAI()  # reads OPENAI_API_KEY

def summarize_text(text: str, model: str = "gpt-4o-mini") -> str:
    text = (text or "").strip()
    if not text:
        return ""
    prompt = (
        "Provide a clear, concise summary of the following content. "
        "Keep the original language when possible. If there are lists, preserve bullets or emojis. "
        "Aim for 3–6 short bullet points followed by a brief overall conclusion.\n\n"
        f"{text[:8000]}"
    )
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise technical summarizer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Summary unavailable: {e})"

# ---------------- Device/compute for faster-whisper ----------------
def _pick_device():
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    # On Railway assume CPU
    return "cpu"

def _pick_compute_type(device: str):
    forced = os.getenv("COMPUTE_TYPE")
    if forced:
        return forced
    return "int8" if device == "cpu" else "float16"

# ---------------- Transcription (faster-whisper) ----------------
from faster_whisper import WhisperModel

_WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
_DEVICE = _pick_device()
_COMPUTE = _pick_compute_type(_DEVICE)
print(f"[whisper] model={_WHISPER_MODEL_NAME} device={_DEVICE} compute_type={_COMPUTE}")

_fw = WhisperModel(_WHISPER_MODEL_NAME, device=_DEVICE, compute_type=_COMPUTE)

def _run_whisper(path: str) -> Tuple[List[Dict], str]:
    """Return (segments, transcript) from faster-whisper."""
    segs_out: List[Dict] = []
    segments, _info = _fw.transcribe(
        path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=1,   # greedy decoding (faster)
        best_of=1,
    )
    for s in segments:
        segs_out.append({
            "start": float(s.start or 0.0),
            "end": float(s.end or 0.0),
            "text": (s.text or "").strip(),
        })
    transcript = " ".join(s["text"] for s in segs_out).strip()
    return segs_out, transcript

# ---------------- Utilities: ffmpeg CLI ----------------
def _to_wav_mono_16k(src_path: str) -> str:
    """Extract mono 16 kHz WAV using ffmpeg CLI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(tmp.name)}'
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name

# ---------------- Diarization: pyannote (accurate) ----------------
_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
_DIARIZATION_MODE = (os.getenv("DIARIZATION_MODE") or "fast").lower()  # fast | off (accurate if token+terms)

def _diarize_pyannote(path: str):
    """Try pyannote diarization. Returns (diarization, pretty_map) or (None, {})."""
    if _DIARIZATION_MODE == "off":
        print("[diar] mode=off")
        return None, {}
    if not _HF_TOKEN:
        print("[diar] no HUGGINGFACE_TOKEN -> off")
        return None, {}
    try:
        from pyannote.audio import Pipeline
        # Must accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
        wav_path = _to_wav_mono_16k(path)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=_HF_TOKEN,
        )

        # Optional hints
        kwargs = {}
        if os.getenv("NUM_SPK"):
            kwargs["num_speakers"] = int(os.getenv("NUM_SPK"))
        else:
            if os.getenv("MIN_SPK"):
                kwargs["min_speakers"] = int(os.getenv("MIN_SPK"))
            if os.getenv("MAX_SPK"):
                kwargs["max_speakers"] = int(os.getenv("MAX_SPK"))

        diar = pipeline({"audio": wav_path}, **kwargs)

        # Build pretty label map S1/S2/...
        labels = []
        for _, _, lab in diar.itertracks(yield_label=True):
            if lab not in labels:
                labels.append(lab)
        pretty = {lab: f"S{idx+1}" for idx, lab in enumerate(sorted(labels))}
        print(f"[diar] pyannote enabled, speakers={list(pretty.values())}")
        return diar, pretty
    except Exception as e:
        print("[diar] pyannote disabled:", e)
        return None, {}

def _assign_speakers(segments: List[Dict], diarization, pretty_map: Dict[str, str]) -> List[Dict]:
    """Assign best-overlap speaker to each Whisper segment, then merge consecutive same-speaker lines."""
    if diarization is None:
        for s in segments:
            s["speaker"] = "S1"
        return segments

    turns = []
    for turn, _, lab in diarization.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), lab))

    def overlap(a0, a1, b0, b1):
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        dur = max(1e-6, a1 - a0)
        return inter / dur

    for s in segments:
        best, score = None, 0.0
        for ts, te, lab in turns:
            r = overlap(s["start"], s["end"], ts, te)
            if r > score:
                score = r
                best = lab
        s["speaker"] = pretty_map.get(best, None) if best else None

    merged: List[Dict] = []
    for s in segments:
        if merged and merged[-1].get("speaker") == s.get("speaker"):
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))
    return merged

# ---------------- Public API used by app.py ----------------
def transcribe_video_simple(path: str):
    return _run_whisper(path)

def transcribe_audio_simple(path: str):
    return _run_whisper(path)

def transcribe_video_diarized(path: str):
    segs, transcript = _run_whisper(path)
    diar, pretty = _diarize_pyannote(path)
    segs = _assign_speakers(segs, diar, pretty)
    mode_used = "accurate" if diar is not None else "off"
    return segs, transcript, mode_used

def transcribe_audio_diarized(path: str):
    segs, transcript = _run_whisper(path)
    diar, pretty = _diarize_pyannote(path)
    segs = _assign_speakers(segs, diar, pretty)
    mode_used = "accurate" if diar is not None else "off"
    return segs, transcript, mode_used

# ---------------- Helpers for rendering & exports ----------------
def transcript_with_speakers(segments: List[Dict]) -> str:
    lines = []
    for s in segments:
        spk = s.get("speaker")
        prefix = f"{spk}: " if spk else ""
        lines.append(prefix + s["text"])
    return "\n".join(lines)

def diarization_summary(segments: List[Dict]) -> Dict:
    spks = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    return {"enabled": bool(spks), "speakers": spks, "count": len(spks)}

def build_srt_from_segments(segments: List[Dict]) -> str:
    def fmt_time(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(s['start'])} --> {fmt_time(s['end'])}")
        prefix = f"{s.get('speaker')}: " if s.get('speaker') else ""
        lines.append(prefix + s["text"])
        lines.append("")
    return "\n".join(lines)

def verification_report_from(media_info: Dict, transcript_text: str, segments: List[Dict]) -> Dict:
    if not segments:
        return {
            "media": media_info,
            "duration_sec": 0.0,
            "avg_chars_per_sec": 0.0,
            "suspicious_speed_flag": False,
            "silence_segments_over_3s": [],
            "notes": ["No segments detected."],
        }
    duration = max(0.0, round(segments[-1]["end"] - segments[0]["start"], 2))
    total_chars = len(transcript_text)
    avg_cps = round(total_chars / duration, 2) if duration > 0 else 0.0

    silences = []
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i-1]["end"]
        if gap > 3.0:
            silences.append({
                "start": round(segments[i-1]["end"], 2),
                "end": round(segments[i]["start"], 2),
                "gap": round(gap, 2),
            })

    return {
        "media": media_info,
        "duration_sec": duration,
        "avg_chars_per_sec": avg_cps,
        "suspicious_speed_flag": avg_cps > 25,
        "silence_segments_over_3s": silences,
        "notes": ["Lightweight QA only (not deepfake detection)."],
    }