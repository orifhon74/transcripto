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
    return "cpu"  # Railway default

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
        beam_size=1,
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
_DIA_STRATEGY = (os.getenv("DIA_STRATEGY") or "minimal").lower()   # minimal | fast | accurate
_NUM_SPK      = int(os.getenv("NUM_SPK", "0")) or None             # e.g., 2

# ----- (Optional) Accurate: pyannote (heavy; not used on hobby tiers) -----
def _diarize_pyannote(path: str):
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("[diar] pyannote disabled: no HF token")
        return None
    try:
        from pyannote.audio import Pipeline
        wav = _to_wav_mono_16k(path)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=token)

        kwargs = {}
        if _NUM_SPK:
            kwargs["num_speakers"] = _NUM_SPK

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
    # map to S1,S2,...
    uniq = {}
    out = []
    for s, e, lab in raw:
        if lab not in uniq:
            uniq[lab] = f"S{len(uniq)+1}"
        out.append((s, e, uniq[lab]))
    return out

# ----- (Optional) Fast: embeddings + clustering (still heavy) -----
def _fast_diarize_wav(wav_path: str, target_speakers: Optional[int]) -> Optional[List[tuple]]:
    try:
        import webrtcvad
        from pydub import AudioSegment
        from resemblyzer import VoiceEncoder, preprocess_wav
        from spectralcluster import SpectralClusterer
    except Exception as e:
        print("[fast-diar] deps missing -> None:", e)
        return None

    # simple VAD to prune non-speech
    vad = webrtcvad.Vad(3)
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
    raw = audio.raw_data
    frame_ms = 30
    frame_bytes = int(16000 * (frame_ms/1000.0) * 2)
    frames = [raw[i:i+frame_bytes] for i in range(0, len(raw), frame_bytes)]
    speech = [len(fr)==frame_bytes and vad.is_speech(fr, 16000) for fr in frames]

    # regions
    regions = []
    i = 0
    while i < len(speech):
        if speech[i]:
            j = i
            while j < len(speech) and speech[j]:
                j += 1
            s = i*frame_ms/1000.0; e = j*frame_ms/1000.0
            if e - s >= 0.6:
                if regions and s - regions[-1][1] < 0.3:
                    regions[-1] = (regions[-1][0], e)
                else:
                    regions.append((s, e))
            i = j
        else:
            i += 1
    if not regions:
        return []

    # embeddings
    encoder = VoiceEncoder()
    segs = []
    for s, e in regions:
        seg = audio[int(s*1000):int(e*1000)]
        seg_path = wav_path + f".seg_{int(s*1000)}_{int(e*1000)}.wav"
        seg.export(seg_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        from resemblyzer import preprocess_wav
        segs.append(preprocess_wav(seg_path))
    embeds = encoder.embed_speaker_segments(segs)

    # clustering
    n = target_speakers
    from spectralcluster import SpectralClusterer
    clusterer = SpectralClusterer(
        min_clusters=n if n else 2,
        max_clusters=n if n else 8,
        p_percentile=0.90,
        gaussian_blur_sigma=1,
    )
    labels = clusterer.predict(embeds)
    return [(float(s), float(e), f"S{int(l)+1}") for (s, e), l in zip(regions, labels)]

# ----- Minimal: zero-ML heuristic (works on tiny servers) -----
def _minimal_diarize_wav(wav_path: str, target_speakers: Optional[int]) -> List[tuple]:
    """
    Try VAD-only turns. If two speakers requested and VAD fails to split,
    we'll fall back to a Whisper-gap heuristic later.
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
        # alternate labels across turns
        lab = "S1"
        out = []
        for s, e in regions:
            out.append((float(s), float(e), lab))
            lab = "S2" if lab == "S1" else "S1"
        return out
    return [(float(s), float(e), "S1") for s, e in regions]

# Heuristic: split by Whisper gaps when VAD/ML produced no usable turns
def _label_two_speakers_from_whisper(segments: List[Dict]) -> List[tuple]:
    """
    Build pseudo-turns from Whisper segments by toggling speakers when:
      - gap between consecutive segments > 0.7s, OR
      - we've had 3 or more consecutive lines by the same speaker, OR
      - the segment looks like a short interjection/question.
    """
    if not segments:
        return []
    TURN_GAP = 0.7
    MAX_RUN  = 3

    def short_interjection(txt: str) -> bool:
        t = txt.strip().lower()
        if not t:
            return False
        return len(t.split()) <= 4 or t.endswith("?")

    turns: List[tuple] = []
    cur_label = "S1"
    run_len = 0
    cur_start = segments[0]["start"]
    prev_end = segments[0]["end"]

    for i, s in enumerate(segments):
        gap = s["start"] - prev_end if i else 0.0
        toggle = (gap > TURN_GAP) or (run_len >= MAX_RUN) or short_interjection(s["text"])

        if toggle and i != 0:
            # close current turn at previous end
            turns.append((float(cur_start), float(prev_end), cur_label))
            # toggle
            cur_label = "S2" if cur_label == "S1" else "S1"
            cur_start = s["start"]
            run_len = 0

        run_len += 1
        prev_end = s["end"]

    # close last
    turns.append((float(cur_start), float(prev_end), cur_label))
    return turns

# ===================== Assign speakers to Whisper segments =====================
def _assign_speakers_from_turns(segments: List[Dict], turns: List[tuple], want_two: bool) -> List[Dict]:
    if not turns:
        if want_two:
            # fallback: build turns from Whisper gaps
            turns = _label_two_speakers_from_whisper(segments)
        else:
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
    """Return (mode_used, turns). Strategy: accurate -> fast -> minimal."""
    wav = None
    strategy = _DIA_STRATEGY if _DIA_STRATEGY in {"accurate", "fast", "minimal"} else "minimal"

    def ensure_wav():
        nonlocal wav
        if wav is None:
            wav = _to_wav_mono_16k(path)

    if strategy == "accurate":
        diar = _diarize_pyannote(path)
        if diar is not None:
            return "accurate", _pyannote_to_turns(diar)
        strategy = "fast"

    if strategy == "fast":
        ensure_wav()
        turns = _fast_diarize_wav(wav, _NUM_SPK)
        if turns:
            return "fast", turns
        # fall through

    ensure_wav()
    return "minimal", _minimal_diarize_wav(wav, _NUM_SPK)

def transcribe_video_diarized(path: str):
    segs, transcript = _run_whisper(path)
    mode, turns = _diarize_auto(path)
    want_two = (_NUM_SPK == 2)
    segs = _assign_speakers_from_turns(segs, turns, want_two)
    return segs, transcript, mode

def transcribe_audio_diarized(path: str):
    segs, transcript = _run_whisper(path)
    mode, turns = _diarize_auto(path)
    want_two = (_NUM_SPK == 2)
    segs = _assign_speakers_from_turns(segs, turns, want_two)
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