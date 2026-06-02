# app_ext/jobs.py
"""
Unified async job pipeline.

A single worker (`run_media_job`) handles every input kind — uploaded audio /
video / text and YouTube links, with optional speaker diarization and
translation — and stores a complete render payload in the job's ``result``
field. The `/result/<job_id>` route turns that payload into a rendered page.

Jobs report coarse progress (`progress` 0-100 and a human `stage` string) so the
UI can show a live status while long transcriptions run.
"""

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

from config import Config
from app_ext.downloads import register_download
from app_ext.pdf_export import register_pdf_payload
from app_ext.youtube import yt_fetch_captions, yt_download_video, parse_vtt

from media_core.diarization import (
    transcribe_video_simple,
    transcribe_video_diarized,
    transcribe_audio_simple,
    transcribe_audio_diarized,
    diarization_summary,
)
from media_core.summarization import summarize_text
from media_core.translation import translate_texts
from media_core.formatting import build_srt_from_segments, transcript_with_speakers

JOBS: dict[str, dict] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=Config.JOB_WORKERS)

VALID_KINDS = {
    "text",
    "audio_simple", "audio_diarized",
    "video_simple", "video_diarized",
    "youtube_simple", "youtube_diarized",
}


# ---------------------------------------------------------------------------
# Job bookkeeping
# ---------------------------------------------------------------------------

def new_job(kind: str, filename: str) -> dict:
    job = {
        "id": "",
        "status": "queued",
        "progress": 0,
        "stage": "Queued",
        "kind": kind,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "logs": [],
        "error": None,
        "result": None,
    }
    return job


def job_log(job_id: str, msg: str):
    job = JOBS.get(job_id)
    if job is None:
        return
    job["logs"].append(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}")
    if len(job["logs"]) > Config.MAX_JOB_LOG_LINES:
        drop = max(1, Config.MAX_JOB_LOG_LINES // 10)
        del job["logs"][:drop]


def set_progress(job_id: str, pct: int, stage: str):
    job = JOBS.get(job_id)
    if job is None:
        return
    job["progress"] = max(0, min(100, int(pct)))
    job["stage"] = stage
    job_log(job_id, stage)


def cleanup_jobs(now: Optional[datetime] = None):
    """Remove completed/errored jobs older than JOB_TTL_MIN."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(minutes=Config.JOB_TTL_MIN)
    for jid, job in list(JOBS.items()):
        if job["status"] in {"done", "error"}:
            try:
                created = datetime.fromisoformat(job["created_at"].replace("Z", ""))
            except Exception:
                created = now
            if created < cutoff:
                JOBS.pop(jid, None)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_media_job(
    job_id: str,
    kind: str,
    *,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    youtube_url: Optional[str] = None,
    target_lang: str = "",
):
    try:
        JOBS[job_id]["status"] = "running"
        set_progress(job_id, 5, "Preparing")

        if kind == "text":
            _run_text(job_id, file_bytes, filename, target_lang)
        elif kind.startswith("youtube_"):
            _run_youtube(job_id, kind, youtube_url, target_lang)
        else:
            _run_upload(job_id, kind, file_bytes, filename, target_lang)

    except Exception as e:
        job = JOBS.get(job_id)
        if job is not None:
            job["status"] = "error"
            job["error"] = str(e)
        job_log(job_id, f"Error: {e}")


# ---- shared helpers -------------------------------------------------------

def _translate_payload(job_id: str, segments, summary, target_lang):
    """Translate the summary and each segment in place. Returns translated summary."""
    summary_tr = ""
    if not target_lang:
        return summary_tr
    set_progress(job_id, 85, "Translating")
    if summary:
        summary_tr = "\n".join(translate_texts([summary], target_lang))
    if segments:
        tr_lines = translate_texts([s.get("text", "") for s in segments], target_lang)
        for s, tr in zip(segments, tr_lines):
            s["uz"] = tr
    return summary_tr


def _finish_media(
    job_id, *, version, media_type, filename, segments, transcript,
    display_transcript, summary, summary_uz, meta, target_lang,
    file_bytes=None, media_mime="video/mp4",
):
    set_progress(job_id, 95, "Building downloads")
    base = os.path.splitext(filename or "media")[0].replace("/", "_").replace("\\", "_")

    srt = build_srt_from_segments(segments)
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")

    token_media = None
    if file_bytes is not None:
        token_media = register_download(file_bytes, media_mime, filename or f"{base}.mp4")

    token_pdf = register_pdf_payload(
        version=version, media_type=media_type, filename=filename or base,
        summary=summary, summary_uz=summary_uz, meta=meta, segments=segments,
    )

    JOBS[job_id]["result"] = {
        "template": "media",
        "version": version,
        "media_type": media_type,
        "transcript": display_transcript,
        "summary": summary,
        "summary_uz": summary_uz,
        "meta": meta,
        "segments": segments,
        "show_uz": bool(target_lang),
        "tokens": {
            "txt": token_txt,
            "srt": token_srt,
            "media": token_media,
            "pdf": token_pdf,
        },
    }
    JOBS[job_id]["status"] = "done"
    set_progress(job_id, 100, "Done")


# ---- text -----------------------------------------------------------------

def _run_text(job_id, file_bytes, filename, target_lang):
    set_progress(job_id, 30, "Reading text")
    text = (file_bytes or b"").decode("utf-8", errors="ignore")

    set_progress(job_id, 60, "Summarizing")
    summary = summarize_text(text)

    summary_uz = ""
    if target_lang and summary:
        set_progress(job_id, 85, "Translating")
        summary_uz = "\n".join(translate_texts([summary], target_lang))

    base = os.path.splitext(filename or "text")[0]
    token_text = register_download(text.encode("utf-8"), "text/plain", f"{base}_original.txt")
    token_pdf = register_pdf_payload(
        version="Text", media_type="text", filename=filename or "text.txt",
        summary=summary, summary_uz=summary_uz, meta={},
        segments=[{"start": 0, "end": 0, "text": text}],
    )

    JOBS[job_id]["result"] = {
        "template": "text",
        "file_label": filename,
        "summary": summary,
        "summary_uz": summary_uz,
        "show_uz": bool(target_lang),
        "tokens": {"text": token_text, "pdf": token_pdf},
    }
    JOBS[job_id]["status"] = "done"
    set_progress(job_id, 100, "Done")


# ---- uploaded audio / video ----------------------------------------------

def _run_upload(job_id, kind, file_bytes, filename, target_lang):
    is_video = kind.startswith("video")
    diarized = kind.endswith("diarized")
    media_type = "video" if is_video else "audio"
    suffix = ".mp4" if is_video else ".wav"

    def cb(msg):
        job_log(job_id, msg)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        set_progress(job_id, 30, "Transcribing")
        if diarized:
            set_progress(job_id, 45, "Transcribing + separating speakers")
            fn = transcribe_video_diarized if is_video else transcribe_audio_diarized
            segments, transcript, mode_used = fn(tmp.name, progress=cb)
        else:
            fn = transcribe_video_simple if is_video else transcribe_audio_simple
            segments, transcript = fn(tmp.name, progress=cb)
            mode_used = None

    set_progress(job_id, 75, "Summarizing")
    summary = summarize_text(transcript)
    summary_uz = _translate_payload(job_id, segments, summary, target_lang)

    meta = diarization_summary(segments) if diarized else None
    label = "Video" if is_video else "Audio"
    version = (
        f"{label} — Differentiated ({mode_used})" if diarized else f"{label} — Simple"
    )
    display = transcript_with_speakers(segments) if diarized else transcript

    _finish_media(
        job_id, version=version, media_type=media_type, filename=filename,
        segments=segments, transcript=transcript, display_transcript=display,
        summary=summary, summary_uz=summary_uz, meta=meta, target_lang=target_lang,
        file_bytes=file_bytes, media_mime=("video/mp4" if is_video else "audio/wav"),
    )


# ---- youtube --------------------------------------------------------------

def _run_youtube(job_id, kind, url, target_lang):
    diarized = kind.endswith("diarized")

    def cb(msg):
        job_log(job_id, msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        segments, transcript, title = [], "", None

        if not diarized:
            set_progress(job_id, 15, "Fetching captions")
            try:
                vtt_path, title = yt_fetch_captions(url, tmpdir)
                if vtt_path and os.path.exists(vtt_path):
                    with open(vtt_path, "r", encoding="utf-8", errors="ignore") as fh:
                        segments, transcript = parse_vtt(fh.read())
            except Exception as e:
                job_log(job_id, f"Captions fetch failed: {e}")

        set_progress(job_id, 25, "Downloading video")
        video_path, title2 = yt_download_video(url, tmpdir)
        title = title or title2 or "youtube"

        mode_used = None
        if diarized:
            set_progress(job_id, 45, "Transcribing + separating speakers")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpa:
                shutil.copyfile(video_path, tmpa.name)
                segments, transcript, mode_used = transcribe_audio_diarized(tmpa.name, progress=cb)
        elif not segments:
            set_progress(job_id, 40, "Transcribing audio")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpa:
                shutil.copyfile(video_path, tmpa.name)
                segments, transcript = transcribe_audio_simple(tmpa.name, progress=cb)

        with open(video_path, "rb") as fh:
            media_bytes = fh.read()

    set_progress(job_id, 75, "Summarizing")
    summary = summarize_text(transcript)
    summary_uz = _translate_payload(job_id, segments, summary, target_lang)

    meta = diarization_summary(segments) if diarized else None
    version = (
        f"YouTube — Differentiated ({mode_used})" if diarized else "YouTube — Simple"
    )
    display = transcript_with_speakers(segments) if diarized else transcript

    _finish_media(
        job_id, version=version, media_type="video", filename=f"{title}.mp4",
        segments=segments, transcript=transcript, display_transcript=display,
        summary=summary, summary_uz=summary_uz, meta=meta, target_lang=target_lang,
        file_bytes=media_bytes, media_mime="video/mp4",
    )
