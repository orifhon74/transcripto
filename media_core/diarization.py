import os
import shlex
import subprocess
import tempfile
import shutil
import uuid
from typing import List, Dict, Tuple, Optional

import numpy as np
import webrtcvad
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from spectralcluster import SpectralClusterer

from .whisper_pipeline import run_whisper, run_whisper_diarize_model


# --- ffmpeg helper ---

def _to_wav_mono_16k(src_path: str) -> str:
    """Extract mono 16 kHz WAV using ffmpeg CLI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = f'ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(tmp.name)}'
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp.name


# --- env / config ---

_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
_DIARIZATION_MODE = (os.getenv("DIARIZATION_MODE") or "fast").lower()
if _DIARIZATION_MODE == "auto":
    _DIARIZATION_MODE = "accurate" if _HF_TOKEN else "fast"

# NEW: allow OpenAI diarization model usage
DIARIZATION_BACKEND = (os.getenv("DIARIZATION_BACKEND", "cpu") or "cpu").lower()
# cpu = your existing pipeline; openai = use OpenAI diarize model (if available)


def _get_num_spk() -> Optional[int]:
    v = os.getenv("NUM_SPK")
    try:
        return int(v) if v else None
    except Exception:
        return None


# --- accurate (pyannote) ---

def _diarize_pyannote(path: str):
    """
    Try pyannote/speaker-diarization-3.1 with forced or bounded speaker count.
    Returns (diarization_obj, pretty_map) or (None, {}).
    Ensures temp WAV cleanup.
    """
    if _DIARIZATION_MODE == "off":
        print("[diar] mode=off")
        return None, {}
    if not _HF_TOKEN:
        print("[diar] no HUGGINGFACE_TOKEN -> skipping pyannote")
        return None, {}

    try:
        from pyannote.audio import Pipeline

        wav = _to_wav_mono_16k(path)
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=_HF_TOKEN,
            )

            num_spk = _get_num_spk()
            if num_spk is not None:
                diar = pipeline({"audio": wav}, min_speakers=num_spk, max_speakers=num_spk)
            else:
                min_spk = int(os.getenv("MIN_SPK", "2"))
                max_spk = int(os.getenv("MAX_SPK", "6"))
                diar = pipeline({"audio": wav}, min_speakers=min_spk, max_speakers=max_spk)

            raw_labels = []
            for _, _, lab in diar.itertracks(yield_label=True):
                if lab not in raw_labels:
                    raw_labels.append(lab)
            pretty = {lab: f"S{idx + 1}" for idx, lab in enumerate(sorted(raw_labels))}
            print(f"[diar][accurate] speakers={list(pretty.values())} (mode bounds applied)")
            return diar, pretty
        finally:
            try:
                os.remove(wav)
            except Exception:
                pass
    except Exception as e:
        print("[diar][accurate] disabled:", e)
        return None, {}


# --- fast diarization (VAD + embeddings + spectral clustering) ---

def _fast_diarize_wav(wav_path: str,
                      min_clusters: int = 2,
                      max_clusters: int = 6) -> List[tuple]:
    """
    Returns list of (start_sec, end_sec, 'S#') with auto number of speakers.
    Uses a temp dir for segment files and cleans it up.
    """
    try:
        vad = webrtcvad.Vad(3)
        audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
        raw = audio.raw_data

        frame_ms = 30
        frame_bytes = int(16000 * (frame_ms / 1000.0) * 2)
        frames = [raw[i:i + frame_bytes] for i in range(0, len(raw), frame_bytes)]
        speech_flags = [len(fr) == frame_bytes and vad.is_speech(fr, 16000) for fr in frames]

        regions = []
        i = 0
        while i < len(speech_flags):
            if speech_flags[i]:
                start = i
                while i < len(speech_flags) and speech_flags[i]:
                    i += 1
                end = i
                regions.append((start * frame_ms / 1000.0, end * frame_ms / 1000.0))
            else:
                i += 1

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
            print("[diar][fast] no speech regions")
            return []

        tmp_root = os.path.join(tempfile.gettempdir(), f"diar_{uuid.uuid4().hex}")
        os.makedirs(tmp_root, exist_ok=True)

        try:
            enc = VoiceEncoder()
            seg_wavs = []
            for s, e in regions:
                seg = audio[int(s * 1000):int(e * 1000)]
                seg_path = os.path.join(tmp_root, f"seg_{int(s * 1000)}_{int(e * 1000)}.wav")
                seg.export(seg_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                seg_wavs.append(preprocess_wav(seg_path))

            embeds = np.vstack([enc.embed_utterance(w) for w in seg_wavs])

            clusterer = SpectralClusterer(
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                p_percentile=0.90,
                gaussian_blur_sigma=1,
            )
            labels = clusterer.predict(embeds)

            out = []
            for (s, e), lab in zip(regions, labels):
                out.append((float(s), float(e), f"S{int(lab) + 1}"))

            uniq = sorted({lab for _, _, lab in out})
            print(f"[diar][fast] speakers={uniq} count={len(uniq)} (auto {min_clusters}-{max_clusters})")
            return out
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
    except Exception as e:
        print("[diar][fast] error:", e)
        return []


# --- public diarization helpers ---

def diarize_auto(path: str) -> Tuple[str, List[tuple]]:
    """
    Auto selection:
      - If DIARIZATION_MODE=accurate and HF token present -> pyannote
      - Else -> fast CPU pipeline
    Ensures temp WAV cleanup for the fast path.
    """
    if _DIARIZATION_MODE == "accurate" and _HF_TOKEN:
        diar, pretty = _diarize_pyannote(path)
        if diar is not None:
            turns = []
            for turn, _, lab in diar.itertracks(yield_label=True):
                ts, te = float(turn.start), float(turn.end)
                turns.append((ts, te, pretty.get(lab, "S1")))
            return "accurate", turns

    wav = _to_wav_mono_16k(path)
    try:
        num_spk = _get_num_spk()
        if num_spk is not None:
            turns = _fast_diarize_wav(wav, min_clusters=num_spk, max_clusters=num_spk)
        else:
            min_spk = int(os.getenv("MIN_SPK", "2"))
            max_spk = int(os.getenv("MAX_SPK", "6"))
            turns = _fast_diarize_wav(wav, min_clusters=min_spk, max_clusters=max_spk)
        mode = "fast" if turns else "off"
        return mode, turns
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass


def assign_speakers_from_turns(segments: List[Dict], turns: List[tuple]) -> List[Dict]:
    """
    Map (start,end,label) turns to Whisper segments; merge consecutive same-speaker lines.
    Adds short-blip smoothing to reduce label flicker.
    """
    if not turns:
        for s in segments:
            s["speaker"] = "S1"
        print("[map] no turns -> all S1")
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

    def smooth(labels: List[str], min_dur=0.8):
        out = labels[:]
        for i in range(1, len(labels) - 1):
            prev_l, cur_l, next_l = labels[i - 1], labels[i], labels[i + 1]
            dur = segments[i]["end"] - segments[i]["start"]
            if dur < min_dur and prev_l == next_l != cur_l:
                out[i] = prev_l
        return out

    labels = [s["speaker"] for s in segments]
    labels = smooth(labels)
    for s, lab in zip(segments, labels):
        s["speaker"] = lab

    merged: List[Dict] = []
    for s in segments:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))

    final = sorted({seg.get("speaker") for seg in merged})
    print(f"[map] final speakers={final} count={len(final)}")
    return merged


def diarization_summary(segments: List[Dict]) -> Dict:
    spks = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    return {"enabled": bool(spks), "speakers": spks, "count": len(spks)}


# --- convenience "public" API used by higher layers ---

def transcribe_video_simple(path: str):
    return run_whisper(path)


def transcribe_audio_simple(path: str):
    return run_whisper(path)


def transcribe_video_diarized(path: str):
    # Option A: OpenAI diarization model (if enabled)
    if DIARIZATION_BACKEND == "openai":
        segs, transcript = run_whisper_diarize_model(path)
        # If no speaker labels came back, fall back to CPU speaker assignment
        if not any(s.get("speaker") for s in segs):
            mode, turns = diarize_auto(path)
            segs = assign_speakers_from_turns(segs, turns)
            return segs, transcript, ("minimal" if mode == "off" else mode)
        # Normalize speakers to S1,S2... if needed
        raw = [s.get("speaker") for s in segs if s.get("speaker")]
        uniq = {x for x in raw if x}
        mapping = {lab: f"S{idx+1}" for idx, lab in enumerate(sorted(uniq))}
        for s in segs:
            if s.get("speaker"):
                s["speaker"] = mapping.get(s["speaker"], "S1")
        return segs, transcript, "openai"

    # Option B: CPU diarization (your current approach)
    segs, transcript = run_whisper(path)
    mode, turns = diarize_auto(path)
    segs = assign_speakers_from_turns(segs, turns)
    return segs, transcript, ("minimal" if mode == "off" else mode)


def transcribe_audio_diarized(path: str):
    if DIARIZATION_BACKEND == "openai":
        segs, transcript = run_whisper_diarize_model(path)
        if not any(s.get("speaker") for s in segs):
            mode, turns = diarize_auto(path)
            segs = assign_speakers_from_turns(segs, turns)
            return segs, transcript, ("minimal" if mode == "off" else mode)
        raw = [s.get("speaker") for s in segs if s.get("speaker")]
        uniq = {x for x in raw if x}
        mapping = {lab: f"S{idx+1}" for idx, lab in enumerate(sorted(uniq))}
        for s in segs:
            if s.get("speaker"):
                s["speaker"] = mapping.get(s["speaker"], "S1")
        return segs, transcript, "openai"

    segs, transcript = run_whisper(path)
    mode, turns = diarize_auto(path)
    segs = assign_speakers_from_turns(segs, turns)
    return segs, transcript, ("minimal" if mode == "off" else mode)