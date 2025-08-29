import os
import shlex
import subprocess
import tempfile
from typing import List, Dict, Tuple

# ---------------- OpenAI summarizer ----------------
from openai import OpenAI
_client = OpenAI()

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
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Summary unavailable: {e})"

# ---------------- Device/compute auto-pick for faster-whisper ----------------
def _pick_device():
    forced = os.getenv("DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "metal"  # Apple Silicon via CTranslate2 / Metal
    except Exception:
        pass
    return "cpu"

def _pick_compute_type(device: str):
    forced = os.getenv("COMPUTE_TYPE")
    if forced:
        return forced
    return "float16" if device in ("cuda", "metal") else "int8"

# ---------------- Transcription (faster-whisper) ----------------
from faster_whisper import WhisperModel

_WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_DEVICE = _pick_device()
_COMPUTE = _pick_compute_type(_DEVICE)
print(f"[whisper] model={_WHISPER_MODEL_NAME} device={_DEVICE} compute_type={_COMPUTE}")

_fw = WhisperModel(_WHISPER_MODEL_NAME, device=_DEVICE, compute_type=_COMPUTE)

def _run_whisper(path: str) -> Tuple[List[Dict], str]:
    """Return (segments, transcript) from faster-whisper."""
    segments, _info = _fw.transcribe(
        path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=1,   # greedy decoding (faster)
        best_of=1,
    )
    segs = [{"start": s.start, "end": s.end, "text": (s.text or "").strip()} for s in segments]
    transcript = " ".join(s["text"] for s in segs).strip()
    return segs, transcript

# ---------------- Utilities: ffmpeg CLI (no python wrapper) ----------------
def _to_wav_mono_16k(src_path: str) -> str:
    """Extract mono 16 kHz WAV from any media using the ffmpeg CLI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(tmp.name)}'
    try:
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e}")
    return tmp.name

# ---------------- Diarization: accurate (pyannote) ----------------
_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

def _diarize_pyannote(path: str):
    """Return pyannote diarization object or None (auto #speakers by default)."""
    if not _HF_TOKEN:
        print("[pyannote] disabled: no HUGGINGFACE_TOKEN")
        return None
    try:
        from pyannote.audio import Pipeline
        wav_path = _to_wav_mono_16k(path)

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=_HF_TOKEN,
        )

        env_num = os.getenv("NUM_SPK")
        env_min = os.getenv("MIN_SPK")
        env_max = os.getenv("MAX_SPK")

        kwargs = {}
        if env_num:
            kwargs["num_speakers"] = int(env_num)
        else:
            if env_min: kwargs["min_speakers"] = int(env_min)
            if env_max: kwargs["max_speakers"] = int(env_max)

        diar = pipeline({"audio": wav_path}, **kwargs)
        labels = sorted({lab for _, _, lab in diar.itertracks(yield_label=True)})
        print(f"[pyannote] labels={labels} count={len(labels)} {kwargs or '(auto)'}")
        return diar
    except Exception as e:
        print("[pyannote] error:", e)
        return None

def _pyannote_to_turns(diarization) -> List[tuple]:
    """Convert pyannote result to list[(start,end,label 'S1'..)]."""
    if diarization is None:
        return []
    raw_labels = []
    turns = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        raw_labels.append(label)
        ts, te = float(turn.start), float(turn.end)
        turns.append((ts, te, label))
    # normalize to S1,S2,...
    uniq = {lab: f"S{idx+1}" for idx, lab in enumerate(sorted(set(raw_labels)))}
    return [(ts, te, uniq[lab]) for ts, te, lab in turns]

# ---------------- Diarization: fast (VAD + embeddings + clustering) ----------------
import webrtcvad
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from spectralcluster import SpectralClusterer

def _fast_diarize_wav(wav_path: str, target_speakers: int | None = None):
    """Return list[(start,end,label 'S1','S2',...)] using a fast CPU-friendly pipeline."""
    vad = webrtcvad.Vad(3)  # 0..3 (3 = most aggressive)
    audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
    raw = audio.raw_data

    frame_ms = 30
    frame_bytes = int(16000 * (frame_ms/1000.0) * 2)
    frames = [raw[i:i+frame_bytes] for i in range(0, len(raw), frame_bytes)]
    speech_flags = [len(fr)==frame_bytes and vad.is_speech(fr, 16000) for fr in frames]

    # merge frames into speech regions
    regions = []
    i = 0
    while i < len(speech_flags):
        if speech_flags[i]:
            start = i
            while i < len(speech_flags) and speech_flags[i]:
                i += 1
            end = i
            regions.append((start*frame_ms/1000.0, end*frame_ms/1000.0))
        else:
            i += 1
    if not regions:
        print("[fast-diar] no speech detected")
        return []

    # prune tiny regions and merge tiny gaps to cut embedding calls
    MIN_REGION = 0.6
    regions = [(s, e) for (s, e) in regions if (e - s) >= MIN_REGION]
    merged = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    regions = merged
    if not regions:
        print("[fast-diar] pruned all regions")
        return []

    # embeddings
    encoder = VoiceEncoder()
    seg_wavs = []
    for s, e in regions:
        seg = audio[int(s*1000):int(e*1000)]
        seg_path = wav_path + f".seg_{int(s*1000)}_{int(e*1000)}.wav"
        seg.export(seg_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        seg_wavs.append(preprocess_wav(seg_path))
    embeds = encoder.embed_speaker_segments(seg_wavs)

    # clustering
    n = target_speakers
    clusterer = SpectralClusterer(
        min_clusters=n if n else 2,
        max_clusters=n if n else 8,
        p_percentile=0.90,
        gaussian_blur_sigma=1,
    )
    labels = clusterer.predict(embeds)
    out = [(float(s), float(e), f"S{int(lab)+1}") for (s, e), lab in zip(regions, labels)]
    uniq = sorted({lab for _, _, lab in out})
    print(f"[fast-diar] speakers={uniq} count={len(uniq)}")
    return out

# ---------------- Assign diarization turns to Whisper segments ----------------
def _assign_speakers_from_turns(segments: List[Dict], turns: List[tuple]) -> List[Dict]:
    """turns: list[(start,end,'S1'/'S2'/...)]"""
    if not turns:
        for s in segments: s["speaker"] = "S1"
        print("[mapping] no turns -> all S1")
        return segments

    def overlap_ratio(a0, a1, b0, b1):
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        dur = max(1e-6, a1 - a0)
        return inter / dur

    for s in segments:
        best_label, best_score = None, 0.0
        for ts, te, lab in turns:
            r = overlap_ratio(s["start"], s["end"], ts, te)
            if r > best_score:
                best_score, best_label = r, lab
        s["speaker"] = best_label or "S1"

    # merge consecutive same-speaker segments
    merged: List[Dict] = []
    for s in segments:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))
    final = sorted({seg.get("speaker") for seg in merged})
    print(f"[mapping] final speakers={final} count={len(final)}")
    return merged

# ---------------- Public API used by app.py ----------------
def transcribe_video_simple(path: str):
    return _run_whisper(path)

def transcribe_audio_simple(path: str):
    return _run_whisper(path)

def _diarize_auto(path: str):
    """Choose accurate if available/allowed, else fast. Allow override via DIARIZATION_MODE."""
    mode = (os.getenv("DIARIZATION_MODE") or "auto").lower()
    if mode == "accurate":
        return "accurate", _diarize_pyannote(path)
    if mode == "fast":
        wav = _to_wav_mono_16k(path)
        # Optional fixed speaker count for interviews
        n = int(os.getenv("NUM_SPK")) if os.getenv("NUM_SPK") else None
        return "fast", _fast_diarize_wav(wav, n)

    # auto
    diar = _diarize_pyannote(path)
    if diar is not None:
        return "accurate", diar
    wav = _to_wav_mono_16k(path)
    n = int(os.getenv("NUM_SPK")) if os.getenv("NUM_SPK") else None
    return "fast", _fast_diarize_wav(wav, n)

def transcribe_video_diarized(path: str):
    segs, transcript = _run_whisper(path)
    mode, diar = _diarize_auto(path)
    if mode == "accurate":
        turns = _pyannote_to_turns(diar)
    else:
        turns = diar or []
    segs = _assign_speakers_from_turns(segs, turns)
    return segs, transcript, mode

def transcribe_audio_diarized(path: str):
    segs, transcript = _run_whisper(path)
    mode, diar = _diarize_auto(path)
    if mode == "accurate":
        turns = _pyannote_to_turns(diar)
    else:
        turns = diar or []
    segs = _assign_speakers_from_turns(segs, turns)
    return segs, transcript, mode

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
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
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
    """Basic verification report used by the UI."""
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
                "gap": round(gap, 2)
            })

    return {
        "media": media_info,
        "duration_sec": duration,
        "avg_chars_per_sec": avg_cps,
        "suspicious_speed_flag": avg_cps > 25,
        "silence_segments_over_3s": silences,
        "notes": ["Lightweight QA only (not deepfake detection)."],
    }