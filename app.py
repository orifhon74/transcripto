import os
import json
import tempfile
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from dotenv import load_dotenv
load_dotenv()  # must run before importing process_video

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify

from process_video import (
    # media
    transcribe_video_simple,
    transcribe_video_diarized,
    transcribe_audio_simple,
    transcribe_audio_diarized,
    # helpers
    summarize_text,
    build_srt_from_segments,
    verification_report_from,
    transcript_with_speakers,
    diarization_summary,
    translate_texts_to_uz,   # Uzbek translator helper
)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev")

# ------------------------------------------------------------
# Optional memory guardrail (reject huge uploads)
# Uncomment to enforce a hard limit (bytes). Example: 200 MB.
# app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
# ------------------------------------------------------------

# =========================
# In-memory downloads registry WITH TTL
# =========================
DOWNLOADS = {}  # token -> {"data": bytes, "mimetype": str, "filename": str, "created_at": datetime}
DL_LOCK = Lock()
DOWNLOAD_TTL_MIN = int(os.getenv("DOWNLOAD_TTL_MIN", "30"))  # minutes

def register_download(data: bytes, mimetype: str, filename: str) -> str:
    token = uuid4().hex
    with DL_LOCK:
        DOWNLOADS[token] = {
            "data": data,
            "mimetype": mimetype,
            "filename": filename,
            "created_at": datetime.utcnow(),
        }
    return token

def cleanup_downloads(now: datetime | None = None):
    """Remove expired downloads from memory."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(minutes=DOWNLOAD_TTL_MIN)
    to_delete = []
    with DL_LOCK:
        for t, item in list(DOWNLOADS.items()):
            if item["created_at"] < cutoff:
                to_delete.append(t)
        for t in to_delete:
            DOWNLOADS.pop(t, None)

# =========================
# JOBS store WITH TTL & bounded logs
# =========================
JOBS = {}  # job_id -> dict(status, kind, filename, created_at, logs[], artifacts{}, meta{}, error?)
EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("JOB_WORKERS", "2")))

JOB_TTL_MIN = int(os.getenv("JOB_TTL_MIN", "60"))  # minutes after completion
MAX_JOB_LOG_LINES = int(os.getenv("MAX_JOB_LOG_LINES", "200"))

def _job_log(job_id: str, msg: str):
    job = JOBS.get(job_id)
    if job is not None:
        job["logs"].append(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}")
        # Bound log growth
        if len(job["logs"]) > MAX_JOB_LOG_LINES:
            # drop oldest ~10% to avoid constant popping
            drop = max(1, MAX_JOB_LOG_LINES // 10)
            del job["logs"][:drop]

def _finish_job(job_id: str, *, status: str, artifacts=None, meta=None, error=None):
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
    """Remove completed/errored jobs older than JOB_TTL_MIN (and free their artifacts)."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(minutes=JOB_TTL_MIN)
    to_delete = []
    for jid, job in list(JOBS.items()):
        if job["status"] in {"done", "error"}:
            # created_at saved as ISO; normalize
            try:
                created = datetime.fromisoformat(job["created_at"].replace("Z", ""))
            except Exception:
                created = now
            if created < cutoff:
                to_delete.append(jid)
    for jid in to_delete:
        JOBS.pop(jid, None)

# Run light cleanup every N seconds (cheap & safe)
CLEANUP_INTERVAL_SEC = int(os.getenv("CLEANUP_INTERVAL_SEC", "120"))
_LAST_CLEAN = datetime.utcnow()

@app.before_request
def periodic_cleanup():
    global _LAST_CLEAN
    now = datetime.utcnow()
    if (now - _LAST_CLEAN).total_seconds() >= CLEANUP_INTERVAL_SEC:
        cleanup_downloads(now)
        cleanup_jobs(now)
        _LAST_CLEAN = now

# =========================
# Worker that does the heavy lifting
# =========================
def _run_media_job(job_id: str, kind: str, file_bytes: bytes, orig_name: str):
    """
    Runs in a worker thread. Produces artifacts:
    - transcript_txt (token)
    - subtitles_srt (token) when media
    - verification_json (token)
    Also returns meta: {version, diarization_mode, speakers?}
    """
    try:
        JOBS[job_id]["status"] = "running"
        _job_log(job_id, f"Started {kind} for {orig_name}")

        # Save to a temp file with appropriate suffix
        suffix = ".txt" if kind == "text" else (".wav" if kind.startswith("audio_") else ".mp4")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()

            base = os.path.splitext(orig_name)[0]

            if kind == "text":
                text = file_bytes.decode("utf-8", errors="ignore")
                _job_log(job_id, "Summarizing text…")
                summary = summarize_text(text)
                token_text_original = register_download(
                    text.encode("utf-8"), "text/plain", f"{base}_original.txt"
                )
                artifacts = {
                    "summary": summary,
                    "text_original": {"token": token_text_original, "filename": f"{base}_original.txt"},
                }
                meta = {"version": "Text", "diarization_mode": "off", "speakers": [], "count": 0}
                _finish_job(job_id, status="done", artifacts=artifacts, meta=meta)
                _job_log(job_id, "Completed.")
                return

            # MEDIA (audio/video)
            if kind == "video_simple":
                _job_log(job_id, "Transcribing (simple)…")
                segments, transcript = transcribe_video_simple(tmp.name)
                mode_used = "off"
            elif kind == "video_diarized":
                _job_log(job_id, "Transcribing + diarizing (video)…")
                segments, transcript, mode_used = transcribe_video_diarized(tmp.name)
            elif kind == "audio_simple":
                _job_log(job_id, "Transcribing (simple)…")
                segments, transcript = transcribe_audio_simple(tmp.name)
                mode_used = "off"
            elif kind == "audio_diarized":
                _job_log(job_id, "Transcribing + diarizing (audio)…")
                segments, transcript, mode_used = transcribe_audio_diarized(tmp.name)
            else:
                raise ValueError(f"Unknown job kind: {kind}")

            # Build artifacts
            _job_log(job_id, "Building summary/SRT/report…")
            display_transcript = (
                transcript_with_speakers(segments) if "diarized" in kind else transcript
            )
            summary = summarize_text(transcript)
            srt = build_srt_from_segments(segments)
            verification = verification_report_from(
                media_info={
                    "type": "video" if kind.startswith("video") else "audio",
                    "name": orig_name,
                    "diarization_mode": mode_used,
                },
                transcript_text=transcript,
                segments=segments,
            )
            meta = diarization_summary(segments) if "diarized" in kind else None

            # Register downloads (small text artifacts: OK in memory; they expire via TTL)
            token_txt = register_download(
                transcript.encode("utf-8"), "text/plain", f"{base}.txt"
            )
            token_srt = register_download(
                srt.encode("utf-8"), "application/x-subrip", f"{base}.srt"
            )
            token_json = register_download(
                json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                "application/json",
                f"{base}_verification.json",
            )

            artifacts = {
                "transcript_display": display_transcript,
                "summary": summary,
                "verification": verification,
                "downloads": {
                    "txt": {"token": token_txt, "filename": f"{base}.txt"},
                    "srt": {"token": token_srt, "filename": f"{base}.srt"},
                    "verification_json": {"token": token_json, "filename": f"{base}_verification.json"},
                },
            }
            header_version = (
                f"{'Video' if kind.startswith('video') else 'Audio'} — "
                f"{'Differentiated' if 'diarized' in kind else 'Simple'}"
                + (f" ({mode_used})" if 'diarized' in kind else "")
            )
            meta_block = {
                "version": header_version,
                "diarization_mode": mode_used,
                "speakers": (meta or {}).get("speakers", []),
                "count": (meta or {}).get("count", 0),
            }
            _finish_job(job_id, status="done", artifacts=artifacts, meta=meta_block)
            _job_log(job_id, "Completed.")

    except Exception as e:
        _finish_job(job_id, status="error", error=e)
        _job_log(job_id, f"Error: {e}")

# =========================
# JOB endpoints
# =========================
@app.post("/jobs")
def create_job():
    """
    Enqueue a job.
    Accepts multipart/form-data:
      - kind: audio_simple | audio_diarized | video_simple | video_diarized | text
      - file: the uploaded file (for text/media)
    Returns: { job_id }
    """
    kind = request.form.get("kind", "").strip()
    f = request.files.get("file")
    if kind not in {"audio_simple", "audio_diarized", "video_simple", "video_diarized", "text"}:
        return jsonify({"error": "Invalid 'kind'."}), 400
    if not f:
        return jsonify({"error": "Missing 'file'."}), 400

    job_id = uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "kind": kind,
        "filename": f.filename,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "logs": [],
        "artifacts": {},
        "meta": {},
    }

    file_bytes = f.read()
    EXECUTOR.submit(_run_media_job, job_id, kind, file_bytes, f.filename)

    return jsonify({"job_id": job_id}), 202

@app.get("/jobs/<job_id>")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "id": job["id"],
        "status": job["status"],
        "kind": job["kind"],
        "filename": job["filename"],
        "created_at": job["created_at"],
        "logs": job["logs"][-50:],  # last 50 lines (response is bounded; we also bound growth on write)
        "meta": job.get("meta", {}),
        "artifacts": job.get("artifacts", {}),
    }), 200

# =========================
# Existing page routes (with Uzbek translation support)
# =========================
@app.get("/")
def index():
    return render_template("index.html")

# ---------- TEXT ----------
@app.post("/upload_text")
def upload_text():
    f = request.files.get("text_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    text = f.read().decode("utf-8", errors="ignore")
    summary = summarize_text(text)

    token_text = register_download(
        text.encode("utf-8"),
        "text/plain",
        f"{os.path.splitext(f.filename)[0]}_original.txt",
    )

    return render_template(
        "result_text.html",
        file_label=f.filename,
        summary=summary,
        token_text=token_text,
    )

# ---------- VIDEO (simple) ----------
@app.post("/upload_video_simple")
def upload_video_simple():
    f = request.files.get("video_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    translate_flag = bool(request.form.get("translate_uz"))

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript = transcribe_video_simple(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""
    if translate_flag and summary:
        summary_uz = "\n".join(translate_texts_to_uz([summary]))

    if translate_flag and segments:
        seg_texts = [s.get("text", "") for s in segments]
        uz_lines = translate_texts_to_uz(seg_texts)
        for s, uz in zip(segments, uz_lines):
            s["uz"] = uz

    srt = build_srt_from_segments(segments)
    verification = verification_report_from(
        media_info={"type": "video", "name": f.filename, "diarization_mode": "off"},
        transcript_text=transcript,
        segments=segments,
    )

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                   "application/json", f"{base}_verification.json")

    token_media = register_download(file_bytes, f.mimetype or "video/mp4", f.filename)

    return render_template(
        "result_media.html",
        version="Video — Simple",
        transcript=transcript,
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=None,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
        media_url=url_for("download", token=token_media),
        media_type="video",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
    )

# ---------- VIDEO (differentiated / diarized) ----------
@app.post("/upload_video_diarized")
def upload_video_diarized():
    f = request.files.get("video_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    translate_flag = bool(request.form.get("translate_uz"))

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript, mode_used = transcribe_video_diarized(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""
    if translate_flag and summary:
        summary_uz = "\n".join(translate_texts_to_uz([summary]))

    if translate_flag and segments:
        seg_texts = [s.get("text", "") for s in segments]
        uz_lines = translate_texts_to_uz(seg_texts)
        for s, uz in zip(segments, uz_lines):
            s["uz"] = uz

    srt = build_srt_from_segments(segments)
    verification = verification_report_from(
       media_info={"type": "video", "name": f.filename, "diarization_mode": mode_used},
       transcript_text=transcript,
       segments=segments,
    )
    meta = diarization_summary(segments)

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                   "application/json", f"{base}_verification.json")

    token_media = register_download(file_bytes, f.mimetype or "video/mp4", f.filename)

    return render_template(
        "result_media.html",
        version=f"Video — Differentiated ({mode_used})",
        transcript=transcript_with_speakers(segments),
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=meta,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
        media_url=url_for("download", token=token_media),
        media_type="video",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
    )

# ---------- AUDIO (simple) ----------
@app.post("/upload_audio_simple")
def upload_audio_simple():
    f = request.files.get("audio_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    translate_flag = bool(request.form.get("translate_uz"))

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript = transcribe_audio_simple(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""
    if translate_flag and summary:
        summary_uz = "\n".join(translate_texts_to_uz([summary]))

    if translate_flag and segments:
        seg_texts = [s.get("text", "") for s in segments]
        uz_lines = translate_texts_to_uz(seg_texts)
        for s, uz in zip(segments, uz_lines):
            s["uz"] = uz

    srt = build_srt_from_segments(segments)
    verification = verification_report_from(
        media_info={"type": "audio", "name": f.filename, "diarization_mode": "off"},
        transcript_text=transcript,
        segments=segments,
    )

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                   "application/json", f"{base}_verification.json")

    token_media = register_download(file_bytes, f.mimetype or "audio/wav", f.filename)

    return render_template(
        "result_media.html",
        version="Audio — Simple",
        transcript=transcript,
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=None,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
        media_url=url_for("download", token=token_media),
        media_type="audio",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
    )

# ---------- AUDIO (differentiated / diarized) ----------
@app.post("/upload_audio_diarized")
def upload_audio_diarized():
    f = request.files.get("audio_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    translate_flag = bool(request.form.get("translate_uz"))

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript, mode_used = transcribe_audio_diarized(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""
    if translate_flag and summary:
        summary_uz = "\n".join(translate_texts_to_uz([summary]))

    if translate_flag and segments:
        seg_texts = [s.get("text", "") for s in segments]
        uz_lines = translate_texts_to_uz(seg_texts)
        for s, uz in zip(segments, uz_lines):
            s["uz"] = uz

    srt = build_srt_from_segments(segments)
    verification = verification_report_from(
        media_info={"type": "audio", "name": f.filename, "diarization_mode": mode_used},
        transcript_text=transcript,
        segments=segments,
    )
    meta = diarization_summary(segments)

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                   "application/json", f"{base}_verification.json")

    token_media = register_download(file_bytes, f.mimetype or "audio/wav", f.filename)

    return render_template(
        "result_media.html",
        version=f"Audio — Differentiated ({mode_used})",
        transcript=transcript_with_speakers(segments),
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=meta,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
        media_url=url_for("download", token=token_media),
        media_type="audio",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
    )

# ---------- downloads ----------
@app.get("/download/<token>")
def download(token: str):
    # one-shot + TTL safety (in case user never downloads)
    item = DOWNLOADS.pop(token, None)
    if not item:
        return "Not found or expired", 404
    return send_file(
        BytesIO(item["data"]),
        mimetype=item["mimetype"],
        as_attachment=True,
        download_name=item["filename"],
    )

@app.get("/healthz")
def healthz():
    # force cleanup when health is probed
    now = datetime.utcnow()
    cleanup_downloads(now)
    cleanup_jobs(now)
    return "ok", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)