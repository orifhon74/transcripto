# app_ext/jobs.py

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from config import Config
from app_ext.transcription import (
    transcribe_video_simple,
    transcribe_video_diarized,
    transcribe_audio_simple,
    transcribe_audio_diarized,
    summarize_text,
    build_srt_from_segments,
    transcript_with_speakers,
    diarization_summary,
)
from app_ext.downloads import register_download

JOBS: dict[str, dict] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=Config.JOB_WORKERS)


def job_log(job_id: str, msg: str):
    job = JOBS.get(job_id)
    if job is not None:
        job["logs"].append(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}")
        if len(job["logs"]) > Config.MAX_JOB_LOG_LINES:
            drop = max(1, Config.MAX_JOB_LOG_LINES // 10)
            del job["logs"][:drop]


def finish_job(job_id: str, *, status: str, artifacts=None, meta=None, error=None):
    job = JOBS.get(job_id)
    if job is None:
        return
    job["status"] = status
    if artifacts is not None:
        job["artifacts"] = artifacts
    if meta is not None:
        job["meta"] = meta
    if error is not None:
        job["error"] = str(error)


def cleanup_jobs(now: datetime | None = None):
    """Remove completed/errored jobs older than JOB_TTL_MIN."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(minutes=Config.JOB_TTL_MIN)
    to_delete: list[str] = []

    for jid, job in list(JOBS.items()):
        if job["status"] in {"done", "error"}:
            try:
                created = datetime.fromisoformat(job["created_at"].replace("Z", ""))
            except Exception:
                created = now
            if created < cutoff:
                to_delete.append(jid)
    for jid in to_delete:
        JOBS.pop(jid, None)


def run_media_job(job_id: str, kind: str, file_bytes: bytes, orig_name: str):
    """
    Worker for async /jobs endpoint. Behavior copied from old _run_media_job.
    """
    try:
        JOBS[job_id]["status"] = "running"
        job_log(job_id, f"Started {kind} for {orig_name}")

        suffix = ".txt" if kind == "text" else (".wav" if kind.startswith("audio_") else ".mp4")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()

            base = os.path.splitext(orig_name)[0]

            if kind == "text":
                text = file_bytes.decode("utf-8", errors="ignore")
                job_log(job_id, "Summarizing text…")
                summary = summarize_text(text)
                token_text_original = register_download(
                    text.encode("utf-8"), "text/plain", f"{base}_original.txt"
                )
                artifacts = {
                    "summary": summary,
                    "text_original": {"token": token_text_original, "filename": f"{base}_original.txt"},
                }
                meta = {"version": "Text", "diarization_mode": "off", "speakers": [], "count": 0}
                finish_job(job_id, status="done", artifacts=artifacts, meta=meta)
                job_log(job_id, "Completed.")
                return

            # MEDIA
            if kind == "video_simple":
                job_log(job_id, "Transcribing (simple)…")
                segments, transcript = transcribe_video_simple(tmp.name)
                mode_used = "off"
            elif kind == "video_diarized":
                job_log(job_id, "Transcribing + diarizing (video)…")
                segments, transcript, mode_used = transcribe_video_diarized(tmp.name)
            elif kind == "audio_simple":
                job_log(job_id, "Transcribing (simple)…")
                segments, transcript = transcribe_audio_simple(tmp.name)
                mode_used = "off"
            elif kind == "audio_diarized":
                job_log(job_id, "Transcribing + diarizing (audio)…")
                segments, transcript, mode_used = transcribe_audio_diarized(tmp.name)
            else:
                raise ValueError(f"Unknown job kind: {kind}")

            job_log(job_id, "Building summary/SRT…")
            display_transcript = (
                transcript_with_speakers(segments) if "diarized" in kind else transcript
            )
            summary = summarize_text(transcript)
            srt = build_srt_from_segments(segments)
            meta = diarization_summary(segments) if "diarized" in kind else None

            token_txt = register_download(
                transcript.encode("utf-8"), "text/plain", f"{base}.txt"
            )
            token_srt = register_download(
                srt.encode("utf-8"), "application/x-subrip", f"{base}.srt"
            )

            artifacts = {
                "transcript_display": display_transcript,
                "summary": summary,
                "downloads": {
                    "txt": {"token": token_txt, "filename": f"{base}.txt"},
                    "srt": {"token": token_srt, "filename": f"{base}.srt"},
                },
            }
            header_version = (
                f"{'Video' if kind.startswith('video') else 'Audio'} — "
                f"{'Differentiated' if 'diarized' in kind else 'Simple'}"
                + (f" ({mode_used})" if "diarized" in kind else "")
            )
            meta_block = {
                "version": header_version,
                "diarization_mode": mode_used,
                "speakers": (meta or {}).get("speakers", []),
                "count": (meta or {}).get("count", 0),
            }
            finish_job(job_id, status="done", artifacts=artifacts, meta=meta_block)
            job_log(job_id, "Completed.")

    except Exception as e:
        finish_job(job_id, status="error", error=e)
        job_log(job_id, f"Error: {e}")