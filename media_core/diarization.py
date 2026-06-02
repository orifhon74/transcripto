"""
Speaker diarization for the transcription pipeline.

Two engines are available:

  * accurate  -> pyannote/speaker-diarization-3.1 (requires a HuggingFace token).
                 This is the primary, highest-quality path.
  * fast      -> a fully-local CPU pipeline:
                   WebRTC VAD  ->  sliding-window speaker embeddings (Resemblyzer)
                   ->  spectral clustering with automatic speaker-count estimation
                   ->  majority-vote assignment of speakers onto ASR segments.

The CPU path was rebuilt to embed short *sliding windows* instead of whole VAD
regions. Whole-region embeddings smear two speakers together whenever the VAD
glues their turns into one region; fine windows keep each speaker's voiceprint
clean, which is what makes clustering work.

An optional OpenAI diarization backend is also supported via
``DIARIZATION_BACKEND=openai``.
"""

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import webrtcvad
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav

from .whisper_pipeline import run_whisper, run_whisper_diarize_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# off | fast | accurate | auto
_DIARIZATION_MODE = (os.getenv("DIARIZATION_MODE") or "auto").lower()
if _DIARIZATION_MODE == "auto":
    _DIARIZATION_MODE = "accurate" if _HF_TOKEN else "fast"

# cpu | openai
DIARIZATION_BACKEND = (os.getenv("DIARIZATION_BACKEND", "cpu") or "cpu").lower()

# Sliding-window embedding parameters (seconds).
_WIN_SEC = float(os.getenv("DIAR_WIN_SEC", "1.5"))
_HOP_SEC = float(os.getenv("DIAR_HOP_SEC", "0.75"))

# Lazy singleton so we don't reload the encoder on every request.
_ENCODER: Optional[VoiceEncoder] = None


def _log(msg: str, progress: Optional[Callable[[str], None]] = None) -> None:
    print(msg)
    if progress is not None:
        try:
            progress(msg)
        except Exception:
            pass


def _get_encoder() -> VoiceEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = VoiceEncoder()
    return _ENCODER


def _env_int(name: str) -> Optional[int]:
    v = os.getenv(name)
    try:
        return int(v) if v else None
    except Exception:
        return None


def _speaker_bounds() -> Tuple[int, int, Optional[int]]:
    """Return (min_speakers, max_speakers, forced_count)."""
    forced = _env_int("NUM_SPK")
    if forced is not None:
        return forced, forced, forced
    min_spk = _env_int("MIN_SPK") or 1
    max_spk = _env_int("MAX_SPK") or 8
    return max(1, min_spk), max(min_spk, max_spk), None


# ---------------------------------------------------------------------------
# ffmpeg helper
# ---------------------------------------------------------------------------

def _to_wav_mono_16k(src_path: str) -> str:
    """Extract mono 16 kHz WAV using the ffmpeg CLI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = f"ffmpeg -y -i {shlex.quote(src_path)} -ac 1 -ar 16000 -vn {shlex.quote(tmp.name)}"
    subprocess.run(
        cmd, shell=True, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return tmp.name


# ---------------------------------------------------------------------------
# Voice-activity detection
# ---------------------------------------------------------------------------

def _vad_regions(
    audio: AudioSegment,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_region: float = 0.4,
    merge_gap: float = 0.25,
) -> List[Tuple[float, float]]:
    """Return merged (start, end) speech regions in seconds."""
    vad = webrtcvad.Vad(aggressiveness)
    raw = audio.raw_data
    frame_bytes = int(16000 * (frame_ms / 1000.0) * 2)  # 16-bit mono
    frames = [raw[i:i + frame_bytes] for i in range(0, len(raw), frame_bytes)]
    flags = [len(fr) == frame_bytes and vad.is_speech(fr, 16000) for fr in frames]

    regions: List[Tuple[float, float]] = []
    i = 0
    while i < len(flags):
        if flags[i]:
            start = i
            while i < len(flags) and flags[i]:
                i += 1
            regions.append((start * frame_ms / 1000.0, i * frame_ms / 1000.0))
        else:
            i += 1

    regions = [(s, e) for s, e in regions if (e - s) >= min_region]

    merged: List[Tuple[float, float]] = []
    for s, e in regions:
        if merged and (s - merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Speaker-count estimation
# ---------------------------------------------------------------------------

def _estimate_num_speakers(embeds: np.ndarray, min_k: int, max_k: int) -> int:
    """
    Estimate the number of speakers from the eigengap of the affinity matrix.

    Returns a value in [min_k, max_k]. With a single, near-uniform cluster the
    largest eigengap lands at k=1, which lets single-speaker audio stay single.
    """
    n = embeds.shape[0]
    max_k = min(max_k, n)
    if n <= 1 or max_k <= 1:
        return 1
    if min_k == max_k:
        return min_k

    # Cosine affinity (embeddings are L2-normalised before this call).
    sim = embeds @ embeds.T
    sim = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)  # -> [0, 1]

    # Symmetric normalised Laplacian.
    deg = np.sum(sim, axis=1)
    deg[deg == 0] = 1e-9
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    lap = np.eye(n) - (sim * d_inv_sqrt).T * d_inv_sqrt

    eigvals = np.sort(np.linalg.eigvalsh(lap))
    eigvals = eigvals[: max_k + 1]
    gaps = np.diff(eigvals)
    if gaps.size == 0:
        return min_k

    # Largest gap; +1 because k clusters correspond to k smallest eigenvalues.
    k = int(np.argmax(gaps)) + 1
    return int(min(max(k, min_k), max_k))


def _cluster(embeds: np.ndarray, min_k: int, max_k: int) -> np.ndarray:
    """Cluster window embeddings into speaker labels (0-indexed)."""
    n = embeds.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)

    k = _estimate_num_speakers(embeds, min_k, max_k)
    if k <= 1:
        return np.zeros(n, dtype=int)

    # Prefer spectral clustering; fall back to KMeans if anything goes wrong.
    try:
        from spectralcluster import SpectralClusterer

        clusterer = SpectralClusterer(
            min_clusters=k,
            max_clusters=k,
            p_percentile=0.95,
            gaussian_blur_sigma=1,
        )
        return clusterer.predict(embeds)
    except Exception as e:  # pragma: no cover - depends on optional internals
        print("[diar][fast] spectral clustering failed, using KMeans:", e)
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeds)


# ---------------------------------------------------------------------------
# Fast CPU diarization
# ---------------------------------------------------------------------------

def _fast_diarize_wav(
    wav_path: str,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Tuple[float, float, str]]:
    """
    Local CPU diarization.

    Returns a list of (start_sec, end_sec, 'S#') turns at sliding-window
    granularity. Automatically estimates the number of speakers.
    """
    min_spk, max_spk, _ = _speaker_bounds()

    try:
        audio = AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(16000)
    except Exception as e:
        print("[diar][fast] could not load audio:", e)
        return []

    regions = _vad_regions(audio)
    if not regions:
        _log("[diar][fast] no speech regions detected", progress)
        return []

    # Build sliding windows inside each speech region.
    windows: List[Tuple[float, float]] = []
    for s, e in regions:
        t = s
        while t < e:
            w_end = min(t + _WIN_SEC, e)
            if (w_end - t) >= 0.5:  # ignore slivers too short to embed
                windows.append((t, w_end))
            if w_end >= e:
                break
            t += _HOP_SEC
    if not windows:
        return [(s, e, "S1") for s, e in regions]

    _log(f"[diar][fast] embedding {len(windows)} windows", progress)

    tmp_root = os.path.join(tempfile.gettempdir(), f"diar_{uuid.uuid4().hex}")
    os.makedirs(tmp_root, exist_ok=True)
    try:
        enc = _get_encoder()
        embeds = []
        kept: List[Tuple[float, float]] = []
        for (s, e) in windows:
            seg = audio[int(s * 1000):int(e * 1000)]
            seg_path = os.path.join(tmp_root, f"w_{int(s*1000)}_{int(e*1000)}.wav")
            seg.export(seg_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            try:
                wav = preprocess_wav(seg_path)
                embeds.append(enc.embed_utterance(wav))
                kept.append((s, e))
            except Exception:
                continue

        if not embeds:
            return [(s, e, "S1") for s, e in regions]

        mat = np.vstack(embeds).astype(np.float32)
        # L2-normalise so dot product == cosine similarity.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        mat = mat / norms

        labels = _cluster(mat, min_spk, max_spk)

        turns = [(float(s), float(e), f"S{int(l) + 1}") for (s, e), l in zip(kept, labels)]
        uniq = sorted({t[2] for t in turns})
        _log(f"[diar][fast] speakers={uniq} count={len(uniq)} windows={len(turns)}", progress)
        return turns
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Accurate (pyannote) diarization
# ---------------------------------------------------------------------------

def _diarize_pyannote(
    path: str,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Tuple[float, float, str]]:
    """Run pyannote 3.1. Returns turns or [] on any failure."""
    if not _HF_TOKEN:
        _log("[diar][accurate] no HUGGINGFACE_TOKEN -> skipping pyannote", progress)
        return []

    wav = None
    try:
        from pyannote.audio import Pipeline

        _log("[diar][accurate] loading pyannote/speaker-diarization-3.1", progress)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=_HF_TOKEN,
        )

        # Use GPU if available.
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception:
            pass

        wav = _to_wav_mono_16k(path)
        min_spk, max_spk, forced = _speaker_bounds()
        if forced is not None:
            diar = pipeline(wav, num_speakers=forced)
        else:
            diar = pipeline(wav, min_speakers=min_spk, max_speakers=max_spk)

        # Stable speaker labels ordered by first appearance.
        order: List[str] = []
        for _, _, lab in diar.itertracks(yield_label=True):
            if lab not in order:
                order.append(lab)
        pretty = {lab: f"S{i + 1}" for i, lab in enumerate(order)}

        turns = [
            (float(t.start), float(t.end), pretty.get(lab, "S1"))
            for t, _, lab in diar.itertracks(yield_label=True)
        ]
        _log(f"[diar][accurate] speakers={sorted(set(pretty.values()))}", progress)
        return turns
    except Exception as e:
        _log(f"[diar][accurate] disabled: {e}", progress)
        return []
    finally:
        if wav:
            try:
                os.remove(wav)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------

def diarize_auto(
    path: str,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[Tuple[float, float, str]]]:
    """
    Pick the best available diarization engine.

    Returns (mode_used, turns) where mode_used is one of
    "accurate", "fast", or "off".
    """
    if _DIARIZATION_MODE == "off":
        return "off", []

    # Primary: pyannote when configured.
    if _DIARIZATION_MODE in ("accurate", "auto") and _HF_TOKEN:
        turns = _diarize_pyannote(path, progress)
        if turns:
            return "accurate", turns
        _log("[diar] pyannote unavailable -> falling back to fast CPU pipeline", progress)

    # Fallback: local CPU pipeline.
    wav = _to_wav_mono_16k(path)
    try:
        turns = _fast_diarize_wav(wav, progress)
        return ("fast" if turns else "off"), turns
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mapping turns onto ASR segments
# ---------------------------------------------------------------------------

def assign_speakers_from_turns(
    segments: List[Dict],
    turns: List[Tuple[float, float, str]],
) -> List[Dict]:
    """
    Assign a speaker label to each ASR segment by majority overlap with the
    diarization turns, then smooth single-segment flickers and merge
    consecutive same-speaker segments.
    """
    if not segments:
        return segments

    if not turns:
        for s in segments:
            s["speaker"] = "S1"
        return segments

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    # For each segment, accumulate overlap duration per speaker label and pick
    # the label with the most total overlap (robust to many short turns).
    for s in segments:
        scores: Dict[str, float] = {}
        for ts, te, lab in turns:
            ov = overlap(s["start"], s["end"], ts, te)
            if ov > 0:
                scores[lab] = scores.get(lab, 0.0) + ov
        s["speaker"] = max(scores, key=scores.get) if scores else None

    # Forward/back-fill any segment that had no overlap.
    last = None
    for s in segments:
        if s["speaker"] is None:
            s["speaker"] = last
        else:
            last = s["speaker"]
    nxt = None
    for s in reversed(segments):
        if s["speaker"] is None:
            s["speaker"] = nxt or "S1"
        else:
            nxt = s["speaker"]

    # Smooth: a short segment wedged between two identical neighbours adopts them.
    labels = [s["speaker"] for s in segments]
    for i in range(1, len(labels) - 1):
        dur = segments[i]["end"] - segments[i]["start"]
        if dur < 1.0 and labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]
    for s, lab in zip(segments, labels):
        s["speaker"] = lab

    # Merge consecutive segments from the same speaker.
    merged: List[Dict] = []
    for s in segments:
        if merged and merged[-1]["speaker"] == s["speaker"]:
            merged[-1]["end"] = s["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + s["text"]).strip()
            if s.get("uz"):
                merged[-1]["uz"] = (merged[-1].get("uz", "") + " " + s["uz"]).strip()
        else:
            merged.append(dict(s))

    # Re-number speakers in order of first appearance (S1, S2, ...).
    order: List[str] = []
    for s in merged:
        if s["speaker"] not in order:
            order.append(s["speaker"])
    remap = {old: f"S{i + 1}" for i, old in enumerate(order)}
    for s in merged:
        s["speaker"] = remap[s["speaker"]]

    print(f"[map] final speakers={sorted(set(remap.values()))} count={len(remap)}")
    return merged


def diarization_summary(segments: List[Dict]) -> Dict:
    spks = sorted({s.get("speaker") for s in segments if s.get("speaker")})
    return {"enabled": bool(spks), "speakers": spks, "count": len(spks)}


# ---------------------------------------------------------------------------
# Public API used by the routes / job pipeline
# ---------------------------------------------------------------------------

def _normalise_openai_speakers(segs: List[Dict]) -> None:
    raw = sorted({s.get("speaker") for s in segs if s.get("speaker")})
    mapping = {lab: f"S{i + 1}" for i, lab in enumerate(raw)}
    for s in segs:
        if s.get("speaker"):
            s["speaker"] = mapping.get(s["speaker"], "S1")


def transcribe_video_simple(path: str, progress=None):
    return run_whisper(path)


def transcribe_audio_simple(path: str, progress=None):
    return run_whisper(path)


def _transcribe_diarized(path: str, progress=None):
    """Shared implementation for audio + video diarized transcription."""
    if DIARIZATION_BACKEND == "openai":
        segs, transcript = run_whisper_diarize_model(path)
        if any(s.get("speaker") for s in segs):
            _normalise_openai_speakers(segs)
            return segs, transcript, "openai"
        # No speaker labels returned -> fall back to local assignment.
        mode, turns = diarize_auto(path, progress)
        segs = assign_speakers_from_turns(segs, turns)
        return segs, transcript, ("minimal" if mode == "off" else mode)

    segs, transcript = run_whisper(path)
    mode, turns = diarize_auto(path, progress)
    segs = assign_speakers_from_turns(segs, turns)
    return segs, transcript, ("minimal" if mode == "off" else mode)


def transcribe_video_diarized(path: str, progress=None):
    return _transcribe_diarized(path, progress)


def transcribe_audio_diarized(path: str, progress=None):
    return _transcribe_diarized(path, progress)
