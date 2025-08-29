# process_video.py
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
    # Railway default: CPU
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

# =================================================================
#                 DIARIZATION IMPLEMENTATIONS
# =================================================================

# -------- Accurate: pyannote (optional; GPU-friendly, heavy) -----
_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

def _diarize_pyannote(path: str):
    """
    Try pyannote diarization. Returns (diarization, pretty_map) or (None, {}).
    Requires: pyannote.audio installed and model terms accepted on HF.
    """
    try:
        if not _HF_TOKEN:
            print("[diar] pyannote disabled: no HUGGINGFACE_TOKEN")
            return None, {}
        from pyannote.audio import Pipeline
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

        # Pretty label map
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

# -------- Fast: VAD + resemblyzer + spectralcluster (CPU) --------
def _fast_diarize_wav(wav_path: str, target_speakers: Optional[int] = None):
    """
    CPU-friendly diarization:
      1) VAD -> speech regions
      2) Sample 1.5s windows with 0.5s hop inside speech
      3) Resemblyzer embeddings
      4) Spectral clustering -> labels
      5) Merge consecutive labels into turns [(start, end, 'S1'/'S2'...)]
    """
    try:
        import webrtcvad
        from pydub import AudioSegment
        from resemblyzer import VoiceEncoder, preprocess_wav
        from spectralcluster import SpectralClusterer
        import numpy as np
    except Exception as e:
        print("[fast-diar] missing deps:", e)
        return []

    vad = webrtcvad.Vad(3)  # aggressive
    sr = 16000
    audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(sr)
    raw = audio.raw_data

    # --- VAD over 30ms frames
    frame_ms = 30
    bytes_per_sample = 2
    frame_bytes = int(sr * (frame_ms / 1000.0) * bytes_per_sample)
    frames = [raw[i:i + frame_bytes] for i in range(0, len(raw), frame_bytes)]
    speech_flags = [len(fr) == frame_bytes and vad.is_speech(fr, sr) for fr in frames]

    # --- Build regions
    regions = []
    i = 0
    while i < len(speech_flags):
        if speech_flags[i]:
            s = i
            while i < len(speech_flags) and speech_flags[i]:
                i += 1
            e = i
            regions.append((s * frame_ms / 1000.0, e * frame_ms / 1000.0))
        else:
            i += 1

    # prune and merge small gaps
    MIN_REGION = 0.6
    regions = [(s, e) for s, e in regions if (e - s) >= MIN_REGION]
    merged = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < 0.3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    regions = merged
    if not regions:
        print("[fast-diar] no speech regions")
        return []

    # --- Sample windows
    WIN = 1.5
    HOP = 0.5
    encoder = VoiceEncoder()
    embed_times = []
    embeds = []

    for s, e in regions:
        t = s
        while t + WIN <= e:
            seg = audio[int(t * 1000):int((t + WIN) * 1000)]
            seg_path = wav_path + f".seg_{int(t * 1000)}_{int((t + WIN) * 1000)}.wav"
            seg.export(seg_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            wav = preprocess_wav(seg_path)
            if len(wav) == 0:
                t += HOP
                continue
            emb = encoder.embed_utterance(wav)
            embeds.append(emb)
            embed_times.append((t, t + WIN))
            t += HOP

    if len(embeds) < 3:
        print(f"[fast-diar] too few windows: {len(embeds)} -> empty")
        return []

    X = np.stack(embeds)
    n = target_speakers or 2  # for interviews, default to 2
    clusterer = SpectralClusterer(
        min_clusters=n,
        max_clusters=target_speakers or 6,
        p_percentile=0.90,
        gaussian_blur_sigma=1,
    )
    labels = clusterer.predict(X)

    # Merge consecutive window labels → turns
    turns = []
    cur_lab = labels[0]
    cur_start = embed_times[0][0]
    for (ts, te), lab in zip(embed_times, labels):
        if lab != cur_lab:
            turns.append((cur_start, ts, f"S{int(cur_lab) + 1}"))
            cur_lab = lab
            cur_start = ts
    turns.append((cur_start, embed_times[-1][1], f"S{int(cur_lab) + 1}"))

    # Merge tiny gaps with same label
    final = []
    for s, e, lab in turns:
        if final and lab == final[-1][2] and (s - final[-1][1]) < 0.2:
            final[-1] = (final[-1][0], e, lab)
        else:
            final.append((s, e, lab))

    uniq = sorted({lab for _, _, lab in final})
    print(f"[fast-diar] windows={len(embed_times)} speakers={uniq} count={len(uniq)}")
    return final

# -------- Map turns to Whisper segments --------------------------
def _assign_speakers_from_turns(segments: List[Dict], turns: List[tuple]) -> List[Dict]:
    """Assign best-overlap speaker to each Whisper segment, merge consecutive same-speaker segments."""
    if not turns:
        for s in segments:
            s["speaker"] = "S1"
        print("[mapping] no turns -> all S1")
        return segments

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
        s["speaker"] = best or "S1"

    merged: List[Dict] = []
    for s in segments:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))
    final_speakers = sorted({seg.get("speaker") for seg in merged})
    print(f"[mapping] final speakers={final_speakers} count={len(final_speakers)}")
    return merged

# =================================================================
#                    PUBLIC API USED BY app.py
# =================================================================

def transcribe_video_simple(path: str):
    return _run_whisper(path)

def transcribe_audio_simple(path: str):
    return _run_whisper(path)

def _diarize_auto(path: str):
    """Choose fast/accurate/off based on env. Fallbacks applied."""
    mode = (os.getenv("DIARIZATION_MODE") or "fast").lower()  # default fast on Railway
    num = int(os.getenv("NUM_SPK")) if os.getenv("NUM_SPK") else None

    if mode == "off":
        return "off", []

    if mode == "accurate":
        diar, pretty = _diarize_pyannote(path)
        if diar is not None:
            # convert to turns [(start,end,'Sx')]
            turns = []
            for turn, _, lab in diar.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), pretty.get(lab, "S1")))
            return "accurate", turns
        # fallback to fast
        wav = _to_wav_mono_16k(path)
        return "fast", _fast_diarize_wav(wav, num)

    if mode == "fast":
        wav = _to_wav_mono_16k(path)
        return "fast", _fast_diarize_wav(wav, num)

    # auto (try accurate, else fast)
    diar, pretty = _diarize_pyannote(path)
    if diar is not None:
        turns = []
        for turn, _, lab in diar.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), pretty.get(lab, "S1")))
        return "accurate", turns
    wav = _to_wav_mono_16k(path)
    return "fast", _fast_diarize_wav(wav, num)

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