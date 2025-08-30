import os
import shlex
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional

# ===================== OpenAI summarizer =====================
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

# ===================== Device/compute for faster-whisper =====================
def _pick_device():
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    # Railway: assume CPU
    return "cpu"

def _pick_compute_type(device: str):
    forced = os.getenv("COMPUTE_TYPE")
    if forced:
        return forced
    return "int8" if device == "cpu" else "float16"

# ===================== Transcription (faster-whisper) =====================
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
            "start": float(getattr(s, "start", 0.0) or 0.0),
            "end":   float(getattr(s, "end", 0.0) or 0.0),
            "text":  (getattr(s, "text", "") or "").strip(),
        })
    transcript = " ".join(s["text"] for s in segs_out).strip()
    return segs_out, transcript

# ===================== Utilities: ffmpeg CLI =====================
def _to_wav_mono_16k(src_path: str) -> str:
    """Extract mono 16 kHz WAV using ffmpeg CLI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(tmp.name)}'
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp.name

# ===================== Diarization strategies =====================
# Mode selector
_DIARIZATION_MODE = (os.getenv("DIARIZATION_MODE") or "fast").lower()     # fast | accurate | off | auto
_DIA_STRATEGY     = (os.getenv("DIA_STRATEGY") or "minimal").lower()      # minimal | fast | accurate
_NUM_SPK          = int(os.getenv("NUM_SPK", "0")) or None                # e.g., 2

# ---------- Accurate: pyannote (optional, gated & heavy) ----------
def _diarize_pyannote(path: str):
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("[diar] pyannote disabled: no HF token")
        return None
    try:
        from pyannote.audio import Pipeline  # heavy
        wav = _to_wav_mono_16k(path)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=token)

        kwargs = {}
        if _NUM_SPK:
            kwargs["num_speakers"] = _NUM_SPK
        else:
            if os.getenv("MIN_SPK"): kwargs["min_speakers"] = int(os.getenv("MIN_SPK"))
            if os.getenv("MAX_SPK"): kwargs["max_speakers"] = int(os.getenv("MAX_SPK"))

        diar = pipeline({"audio": wav}, **kwargs)
        print("[diar] pyannote ran OK")
        return diar
    except Exception as e:
        print("[diar] pyannote error:", e)
        return None

def _pyannote_to_turns(diarization) -> List[tuple]:
    if diarization is None:
        return []
    raw = []
    for turn, _, lab in diarization.itertracks(yield_label=True):
        raw.append((float(turn.start), float(turn.end), lab))
    # Map arbitrary labels to S1,S2,...
    uniq = {}
    out = []
    for s, e, lab in raw:
        if lab not in uniq:
            uniq[lab] = f"S{len(uniq)+1}"
        out.append((s, e, uniq[lab]))
    return out

# ---------- Fast: embeddings + clustering (still heavy on tiny servers) ----------
def _fast_diarize_wav(wav_path: str, target_speakers: Optional[int]) -> Optional[List[tuple]]:
    """Try a faster but still ML-ish approach. Returns None if deps are missing."""
    try:
        import webrtcvad
        from pydub import AudioSegment
        from resemblyzer import VoiceEncoder, preprocess_wav
        from spectralcluster import SpectralClusterer
    except Exception as e:
        print("[fast-diar] deps missing -> None:", e)
        return None

    # VAD: cut to speech regions to reduce embedding calls
    vad = webrtcvad.Vad(3)
    audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
    raw = audio.raw_data
    frame_ms = 30
    frame_bytes = int(16000 * (frame_ms/1000.0) * 2)
    frames = [raw[i:i+frame_bytes] for i in range(0, len(raw), frame_bytes)]
    speech = [len(fr)==frame_bytes and vad.is_speech(fr, 16000) for fr in frames]

    regions = []
    i = 0
    while i < len(speech):
        if speech[i]:
            start = i
            while i < len(speech) and speech[i]:
                i += 1
            end = i
            s = start*frame_ms/1000.0; e = end*frame_ms/1000.0
            if e - s >= 0.6:
                regions.append((s, e))
        else:
            i += 1
    if not regions:
        return []

    # Merge tiny gaps
    merged = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    regions = merged

    # Embeddings
    encoder = VoiceEncoder()
    segs = []
    for s, e in regions:
        seg = audio[int(s*1000):int(e*1000)]
        seg_path = wav_path + f".seg_{int(s*1000)}_{int(e*1000)}.wav"
        seg.export(seg_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        segs.append(preprocess_wav(seg_path))
    embeds = encoder.embed_speaker_segments(segs)

    # Clustering
    n = target_speakers
    clusterer = SpectralClusterer(
        min_clusters=n if n else 2,
        max_clusters=n if n else 8,
        p_percentile=0.90,
        gaussian_blur_sigma=1,
    )
    labels = clusterer.predict(embeds)
    return [(float(s), float(e), f"S{int(l)+1}") for (s, e), l in zip(regions, labels)]

# ---------- Minimal: ultra-light heuristic (no ML, tiny RAM) ----------
def _minimal_diarize_wav(wav_path: str, target_speakers: Optional[int]) -> List[tuple]:
    """
    VAD-only segmentation then assign speakers heuristically.
    - If target_speakers == 2: alternate S1/S2 across speech turns (works well for Q&A/interviews).
    - Else: single speaker S1.
    """
    try:
        import webrtcvad
        from pydub import AudioSegment
    except Exception as e:
        print("[minimal-diar] deps missing:", e)
        return []

    vad = webrtcvad.Vad(2)
    audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
    raw = audio.raw_data
    frame_ms = 30
    frame_bytes = int(16000 * (frame_ms/1000.0) * 2)
    frames = [raw[i:i+frame_bytes] for i in range(0, len(raw), frame_bytes)]
    speech = [len(fr)==frame_bytes and vad.is_speech(fr, 16000) for fr in frames]

    # Merge to regions
    regions = []
    i = 0
    while i < len(speech):
        if speech[i]:
            j = i
            while j < len(speech) and speech[j]:
                j += 1
            s = i*frame_ms/1000.0; e = j*frame_ms/1000.0
            if e - s >= 0.5:
                if regions and s - regions[-1][1] < 0.25:
                    regions[-1] = (regions[-1][0], e)
                else:
                    regions.append((s, e))
            i = j
        else:
            i += 1

    if not regions:
        return []

    if target_speakers == 2:
        labels = []
        cur = "S1"
        for _ in regions:
            labels.append(cur)
            cur = "S2" if cur == "S1" else "S1"
        return [(float(s), float(e), lab) for (s, e), lab in zip(regions, labels)]
    else:
        # single speaker fallback
        return [(float(s), float(e), "S1") for s, e in regions]

# ===================== Assign speakers to Whisper segments =====================
def _assign_speakers_from_turns(segments: List[Dict], turns: List[tuple]) -> List[Dict]:
    if not turns:
        for s in segments: s["speaker"] = "S1"
        return segments

    def overlap_ratio(a0, a1, b0, b1):
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        dur = max(1e-6, a1 - a0)
        return inter / dur

    for s in segments:
        best, score = None, 0.0
        for ts, te, lab in turns:
            r = overlap_ratio(s["start"], s["end"], ts, te)
            if r > score:
                score, best = r, lab
        s["speaker"] = best or "S1"

    # merge consecutive identical speakers
    merged: List[Dict] = []
    for s in segments:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))
    return merged

# ===================== Public API =====================
def transcribe_video_simple(path: str):
    return _run_whisper(path)

def transcribe_audio_simple(path: str):
    return _run_whisper(path)

def _diarize_auto(path: str) -> Tuple[str, List[tuple]]:
    """
    Returns (mode_used, turns).
    Order of attempts depends on DIA_STRATEGY:
      - 'accurate' -> pyannote -> fast -> minimal
      - 'fast'     -> fast -> minimal
      - 'minimal'  -> minimal only
    """
    wav = None
    strategy = _DIA_STRATEGY
    if strategy not in {"accurate", "fast", "minimal"}:
        strategy = "minimal"

    def ensure_wav():
        nonlocal wav
        if wav is None:
            wav = _to_wav_mono_16k(path)

    # accurate
    if strategy == "accurate":
        diar = _diarize_pyannote(path)
        if diar is not None:
            return "accurate", _pyannote_to_turns(diar)
        strategy = "fast"

    # fast
    if strategy == "fast":
        ensure_wav()
        turns = _fast_diarize_wav(wav, _NUM_SPK)
        if turns:
            return "fast", turns
        # fall through to minimal

    # minimal
    ensure_wav()
    return "minimal", _minimal_diarize_wav(wav, _NUM_SPK)

def transcribe_video_diarized(path: str):
    segs, transcript = _run_whisper(path)
    mode, turns = _diarize_auto(path)
    segs = _assign_speakers_from_turns(segs, turns)
    return segs, transcript, mode

def transcribe_audio_diarized(path: str):
    segs, transcript = _run_whisper(path)
    mode, turns = _diarize_auto(path)
    segs = _assign_speakers_from_turns(segs, turns)
    return segs, transcript, mode

# ===================== Helpers for rendering & exports =====================
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