import os
import json
import tempfile
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import shutil

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

# ---------- PDF (ReportLab) ----------
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

# ---------- YouTube ----------
import yt_dlp
import shutil as _shutil  # for ffmpeg which()

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
# PDF builder
# =========================
def _build_pdf_bytes(payload: dict) -> bytes:
    """
    payload keys we expect:
      title, version, filename, created_at, media_type,
      summary, summary_uz (optional),
      verification: {duration_sec, avg_chars_per_sec, suspicious_speed_flag, silence_segments_over_3s: [...]},
      meta: {speakers[], count} (optional),
      segments: list of {start, end, text, speaker?, uz?}
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=payload.get("title", "Transcription Report"),
        author="Summarizer",
    )

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=body, fontSize=9, leading=12)
    mono = ParagraphStyle("Mono", parent=body, fontName="Courier", fontSize=9, leading=12)
    uzStyle = ParagraphStyle("Uz", parent=small, textColor=colors.green, italic=True)

    story = []

    # Header
    title = payload.get("title") or "Transcription Report"
    story.append(Paragraph(title, h1))
    story.append(Spacer(1, 6))
    story.append(Paragraph(payload.get("version", ""), small))
    story.append(Paragraph(f"File: {payload.get('filename','')}", small))
    story.append(Paragraph(f"Generated: {payload.get('created_at','')}", small))
    story.append(Spacer(1, 10))

    # Summary
    if payload.get("summary"):
        story.append(Paragraph("Summary", h2))
        for line in (payload["summary"] or "").split("\n"):
            story.append(Paragraph(line.strip(), body))
        if payload.get("summary_uz"):
            story.append(Spacer(1, 6))
            story.append(Paragraph("Uzbek Summary", h3))
            for line in (payload["summary_uz"] or "").split("\n"):
                story.append(Paragraph(line.strip(), uzStyle))
        story.append(Spacer(1, 10))

    # Verification
    ver = payload.get("verification") or {}
    if ver:
        story.append(Paragraph("Verification", h2))
        tbl = Table(
            [
                ["Duration (sec)", str(ver.get("duration_sec", ""))],
                ["Avg chars/sec", str(ver.get("avg_chars_per_sec", ""))],
                ["Suspicious speed", str(ver.get("suspicious_speed_flag", ""))],
                ["Long silences", str(len(ver.get("silence_segments_over_3s", []) or []))],
            ],
            colWidths=[2.3 * inch, 3.9 * inch],
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 10))

    # Transcript
    segs = payload.get("segments") or []
    if segs:
        story.append(Paragraph("Transcript", h2))
        for s in segs:
            ts = int(s.get("start", 0) or 0)
            mm = ts // 60
            ss = ts % 60
            ts_label = f"{mm:02d}:{ss:02d}"
            speaker = s.get("speaker")
            prefix = f"{speaker}: " if speaker else ""
            text = (s.get("text") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(f"<b>[{ts_label}]</b> {prefix}{text}", mono))
            if s.get("uz"):
                uz = (s.get("uz") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(uz, uzStyle))
        story.append(Spacer(1, 8))

    doc.build(story)
    buf.seek(0)
    return buf.read()

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
# YouTube helpers
# =========================
def _yt_dl_opts_base(tmpdir: str) -> dict:
    ua = os.getenv(
        "YTDLP_UA",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    )
    ff = _shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
    if not os.path.exists(ff):
        alt = "/opt/homebrew/bin/ffmpeg"   # Apple Silicon default
        if os.path.exists(alt):
            ff = alt

    ydl_opts = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "geo_bypass": True,
        "http_headers": {
            "User-Agent": ua,
            "Accept-Language": "en-US,en;q=0.9",
        },
        "ffmpeg_location": ff,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web_safari", "tv_embedded"],
            }
        },
    }

    # Use browser cookies locally (no path needed)
    cfb = os.getenv("COOKIES_FROM_BROWSER")
    if cfb:
        ydl_opts["cookiesfrombrowser"] = (cfb, None, None, None)

    # Or a cookies.txt file (for servers like Railway)
    cookiefile = os.getenv("YTDLP_COOKIEFILE")
    if cookiefile and os.path.exists(cookiefile):
        ydl_opts["cookiefile"] = cookiefile

    return ydl_opts

def _yt_dl_opts_audio(tmpdir: str) -> dict:
    o = _yt_dl_opts_base(tmpdir)
    o.update({
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    })
    return o

def _yt_dl_opts_subs(tmpdir: str) -> dict:
    o = _yt_dl_opts_base(tmpdir)
    o.update({
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
    })
    return o

def _yt_fetch_captions(url: str, tmpdir: str):
    """Try to fetch English captions (manual or auto). Returns (vtt_path, title) or (None, None)."""
    opts = _yt_dl_opts_subs(tmpdir)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)  # may raise DownloadError
        title = info.get("title")
        vid = info.get("id")
        vtt_path = None
        for fname in os.listdir(tmpdir):
            if fname.startswith(vid) and fname.endswith(".vtt"):
                vtt_path = os.path.join(tmpdir, fname)
                break
        return vtt_path, title

def _yt_download_audio(url: str, tmpdir: str):
    """Download best audio and convert to m4a. Returns (audio_path, title)."""
    opts = _yt_dl_opts_audio(tmpdir)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)  # may raise DownloadError
        title = info.get("title") or info.get("id")
        vid = info.get("id")
        out = None
        for fname in os.listdir(tmpdir):
            if fname.startswith(vid) and fname.endswith(".m4a"):
                out = os.path.join(tmpdir, fname)
                break
        if not out:
            out = ydl.prepare_filename(info)  # fallback
        return out, title

def _yt_download_video(url: str, tmpdir: str):
    """
    Download best video+audio as mp4 for local playback.
    Returns (video_path, title).
    """
    opts = _yt_dl_opts_base(tmpdir)
    opts.update({
        # bestvideo+audio, prefer mp4 container
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or info.get("id")
        out = ydl.prepare_filename(info)
        # yt-dlp might name it .webm etc; prefer .mp4 if merged
        if not out.endswith(".mp4"):
            base, _ = os.path.splitext(out)
            mp4_candidate = base + ".mp4"
            if os.path.exists(mp4_candidate):
                out = mp4_candidate
        return out, title

def _parse_vtt(vtt_text: str):
    """Minimal WebVTT to segments: returns (segments, full_text)."""
    def _parse_ts(ts: str) -> float:
        ts = ts.strip().replace(',', '.')
        parts = ts.split(':')
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = '0', parts[0], parts[1]
        else:
            return 0.0
        return int(h) * 3600 + int(m) * 60 + float(s)

    lines = [ln.rstrip('\n') for ln in vtt_text.splitlines()]
    segs = []
    i = 0
    buff = []
    start = end = None

    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln:
            continue
        if '-->' in ln:
            if start is not None and buff:
                text = ' '.join(buff).strip()
                if text:
                    segs.append({"start": start, "end": end, "text": text})
            buff = []
            try:
                a, b = ln.split('-->')
                start = _parse_ts(a.strip())
                end = _parse_ts(b.strip().split(' ')[0])
            except Exception:
                start = end = None
        else:
            if start is not None:
                buff.append(ln)

    if start is not None and buff:
        text = ' '.join(buff).strip()
        if text:
            segs.append({"start": start, "end": end, "text": text})

    full = " ".join(s["text"] for s in segs).strip()
    return segs, full

# =========================
# Existing page routes (with Uzbek translation support)
# =========================
@app.get("/")
def index():
    return render_template("index.html")

# Helper to prepare a PDF payload and register it
def _register_pdf_payload(*, version, media_type, filename, summary, summary_uz, verification, meta, segments):
    payload = {
        "title": "Transcription Report",
        "version": version,
        "media_type": media_type,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "summary": summary or "",
        "summary_uz": summary_uz or "",
        "verification": verification or {},
        "meta": meta or {},
        "segments": segments or [],
    }
    tok = register_download(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        "application/json",
        f"{os.path.splitext(filename)[0]}_pdf_payload.json",
    )
    return tok

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

    # Minimal PDF for text-only: include just summary and original text as one segment
    segments = [{"start": 0, "end": 0, "text": text}]
    token_pdfdata = _register_pdf_payload(
        version="Text",
        media_type="text",
        filename=f.filename,
        summary=summary,
        summary_uz="",  # not translating text uploads here
        verification={},
        meta={},
        segments=segments,
    )

    return render_template(
        "result_text.html",
        file_label=f.filename,
        summary=summary,
        token_text=token_text,
        # not rendered there, but kept similar API
        token_pdf=token_pdfdata,
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

    # PDF payload token
    token_pdfdata = _register_pdf_payload(
        version="Video — Simple",
        media_type="video",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=None,
        segments=segments,
    )

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
        token_pdf=token_pdfdata,
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

    token_pdfdata = _register_pdf_payload(
        version=f"Video — Differentiated ({mode_used})",
        media_type="video",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=meta,
        segments=segments,
    )

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
        token_pdf=token_pdfdata,
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

    token_pdfdata = _register_pdf_payload(
        version="Audio — Simple",
        media_type="audio",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=None,
        segments=segments,
    )

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
        token_pdf=token_pdfdata,
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

    token_pdfdata = _register_pdf_payload(
        version=f"Audio — Differentiated ({mode_used})",
        media_type="audio",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        verification=verification,
        meta=meta,
        segments=segments,
    )

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
        token_pdf=token_pdfdata,
    )

# ---------- YOUTUBE (simple) ----------
@app.post("/ingest_youtube_simple")
def ingest_youtube_simple():
    url = (request.form.get("youtube_url") or "").strip()
    if not url:
        flash("No YouTube URL.")
        return redirect(url_for("index"))

    translate_flag = bool(request.form.get("translate_uz"))

    with tempfile.TemporaryDirectory() as tmpdir:
        title = None
        segments = []
        transcript = ""

        # 1) Try captions first (build transcript/segments)
        try:
            vtt_path, title = _yt_fetch_captions(url, tmpdir)
            if vtt_path and os.path.exists(vtt_path):
                with open(vtt_path, "r", encoding="utf-8", errors="ignore") as fh:
                    vtt_text = fh.read()
                segments, transcript = _parse_vtt(vtt_text)
        except Exception as e:
            print("[yt] captions fetch error:", e)

        # 2) Always fetch audio for PLAYER (so highlight works), but
        #    only transcribe it if we didn't get captions.
        media_bytes = None
        # media_mime = "audio/m4a"
        media_mime = "video/mp4"
        media_filename = None

        # audio_path, title2 = _yt_download_audio(url, tmpdir)
        video_path, title2 = _yt_download_video(url, tmpdir)
        title = title or title2 or "youtube_audio"

        if not segments:
            # no captions -> transcribe audio
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpa:
                shutil.copyfile(video_path, tmpa.name)
                segs, tx = transcribe_audio_simple(tmpa.name)
                segments, transcript = segs, tx

        # read bytes so player can play it from memory (works in both cases)
        with open(video_path, "rb") as fh:
            media_bytes = fh.read()
        media_filename = os.path.basename(video_path)

        # Build outputs
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
            media_info={"type": "audio", "name": title or "YouTube", "diarization_mode": "off"},
            transcript_text=transcript,
            segments=segments,
        )

        base = (title or "youtube").replace("/", "_").replace("\\", "_").strip()
        token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
        token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
        token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                       "application/json", f"{base}_verification.json")

        token_media = register_download(media_bytes, media_mime, media_filename or f"{base}.mp4")
        media_url = url_for("download", token=token_media)

        token_pdfdata = _register_pdf_payload(
            version="YouTube — Simple",
            media_type="video",
            filename=title or "YouTube",
            summary=summary,
            summary_uz=summary_uz,
            verification=verification,
            meta=None,
            segments=segments,
        )

        return render_template(
            "result_media.html",
            version=f"YouTube — Simple",
            transcript=transcript,
            summary=summary,
            summary_uz=summary_uz,
            verification=verification,
            meta=None,
            token_txt=token_txt,
            token_srt=token_srt,
            token_json=token_json,
            media_url=media_url,
            media_type="video",
            segments_json=json.dumps(segments, ensure_ascii=False),
            show_uz=translate_flag,
            token_pdf=token_pdfdata,
        )

# ---------- YOUTUBE (differentiated/diarized) ----------
@app.post("/ingest_youtube_diarized")
def ingest_youtube_diarized():
    url = (request.form.get("youtube_url_d") or "").strip()
    if not url:
        flash("No YouTube URL.")
        return redirect(url_for("index"))

    translate_flag = bool(request.form.get("translate_uz"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # download audio (always)
        # audio_path, title = _yt_download_audio(url, tmpdir)
        video_path, title = _yt_download_video(url, tmpdir)

        # Run diarized transcription
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=True) as tmpa:
            shutil.copyfile(video_path, tmpa.name)
            segments, transcript, mode_used = transcribe_audio_diarized(tmpa.name)

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
            media_info={"type": "audio", "name": title or "YouTube", "diarization_mode": mode_used},
            transcript_text=transcript,
            segments=segments,
        )
        meta = diarization_summary(segments)

        base = (title or "youtube").replace("/", "_").replace("\\", "_").strip()
        token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
        token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
        token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                       "application/json", f"{base}_verification.json")

        with open(video_path, "rb") as fh:
            media_bytes = fh.read()
        token_media = register_download(media_bytes, "video/mp4", os.path.basename(video_path))

        token_pdfdata = _register_pdf_payload(
            version=f"YouTube — Differentiated",
            media_type="video",
            filename=title or "YouTube",
            summary=summary,
            summary_uz=summary_uz,
            verification=verification,
            meta=None,
            segments=segments,
        )

        return render_template(
            "result_media.html",
            version=f"YouTube — Differentiated ({mode_used})",
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
            token_pdf=token_pdfdata,
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

# ---------- PDF download ----------
@app.get("/download_pdf/<token>")
def download_pdf(token: str):
    """
    token points to a JSON payload (stored in DOWNLOADS) with all data needed for PDF.
    We POP it to keep memory low; PDF can be regenerated by re-running the job page.
    """
    # Get and pop JSON payload
    item = DOWNLOADS.pop(token, None)
    if not item:
        return "Not found or expired", 404
    try:
        payload = json.loads(item["data"].decode("utf-8"))
    except Exception:
        return "Invalid payload", 400

    # Build PDF bytes
    pdf_bytes = _build_pdf_bytes(payload)

    # Provide a friendly filename
    base = os.path.splitext(payload.get("filename", "export"))[0]
    pdf_name = f"{base}.pdf"

    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=pdf_name,
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