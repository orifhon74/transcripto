# app_ext/routes_media.py

import os
import json
import tempfile
from datetime import datetime
from io import BytesIO

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
)

from app_ext.transcription import (
    transcribe_video_simple,
    transcribe_video_diarized,
    transcribe_audio_simple,
    transcribe_audio_diarized,
    summarize_text,
    build_srt_from_segments,
    transcript_with_speakers,
    diarization_summary,
    translate_texts_to_uz,
)
from app_ext.downloads import DOWNLOADS, register_download, cleanup_downloads
from app_ext.jobs import cleanup_jobs
from app_ext.pdf_export import register_pdf_payload, build_pdf_bytes

bp = Blueprint("media", __name__)


@bp.get("/")
def index():
    return render_template("index.html")


# ---------- TEXT ----------
@bp.post("/upload_text")
def upload_text():
    f = request.files.get("text_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

    text = f.read().decode("utf-8", errors="ignore")
    summary = summarize_text(text)

    token_text = register_download(
        text.encode("utf-8"),
        "text/plain",
        f"{os.path.splitext(f.filename)[0]}_original.txt",
    )

    segments = [{"start": 0, "end": 0, "text": text}]
    token_pdfdata = register_pdf_payload(
        version="Text",
        media_type="text",
        filename=f.filename,
        summary=summary,
        summary_uz="",
        meta={},
        segments=segments,
    )

    return render_template(
        "result_text.html",
        file_label=f.filename,
        summary=summary,
        token_text=token_text,
        token_pdf=token_pdfdata,
    )


# ---------- VIDEO (simple) ----------
@bp.post("/upload_video_simple")
def upload_video_simple():
    f = request.files.get("video_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

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

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_media = register_download(file_bytes, f.mimetype or "video/mp4", f.filename)

    token_pdfdata = register_pdf_payload(
        version="Video — Simple",
        media_type="video",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        meta=None,
        segments=segments,
    )

    return render_template(
        "result_media.html",
        version="Video — Simple",
        transcript=transcript,
        summary=summary,
        summary_uz=summary_uz,
        meta=None,
        token_txt=token_txt,
        token_srt=token_srt,
        media_url=url_for("media.download", token=token_media),
        media_type="video",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
        token_pdf=token_pdfdata,
    )


# ---------- VIDEO (differentiated / diarized) ----------
@bp.post("/upload_video_diarized")
def upload_video_diarized():
    f = request.files.get("video_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

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
    meta = diarization_summary(segments)

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_media = register_download(file_bytes, f.mimetype or "video/mp4", f.filename)

    token_pdfdata = register_pdf_payload(
        version=f"Video — Differentiated ({mode_used})",
        media_type="video",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        meta=meta,
        segments=segments,
    )

    return render_template(
        "result_media.html",
        version=f"Video — Differentiated ({mode_used})",
        transcript=transcript_with_speakers(segments),
        summary=summary,
        summary_uz=summary_uz,
        meta=meta,
        token_txt=token_txt,
        token_srt=token_srt,
        media_url=url_for("media.download", token=token_media),
        media_type="video",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
        token_pdf=token_pdfdata,
    )


# ---------- AUDIO (simple) ----------
@bp.post("/upload_audio_simple")
def upload_audio_simple():
    f = request.files.get("audio_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

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

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_media = register_download(file_bytes, f.mimetype or "audio/wav", f.filename)

    token_pdfdata = register_pdf_payload(
        version="Audio — Simple",
        media_type="audio",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        meta=None,
        segments=segments,
    )

    return render_template(
        "result_media.html",
        version="Audio — Simple",
        transcript=transcript,
        summary=summary,
        summary_uz=summary_uz,
        meta=None,
        token_txt=token_txt,
        token_srt=token_srt,
        media_url=url_for("media.download", token=token_media),
        media_type="audio",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
        token_pdf=token_pdfdata,
    )


# ---------- AUDIO (differentiated / diarized) ----------
@bp.post("/upload_audio_diarized")
def upload_audio_diarized():
    f = request.files.get("audio_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

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
    meta = diarization_summary(segments)

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_media = register_download(file_bytes, f.mimetype or "audio/wav", f.filename)

    token_pdfdata = register_pdf_payload(
        version=f"Audio — Differentiated ({mode_used})",
        media_type="audio",
        filename=f.filename,
        summary=summary,
        summary_uz=summary_uz,
        meta=meta,
        segments=segments,
    )

    return render_template(
        "result_media.html",
        version=f"Audio — Differentiated ({mode_used})",
        transcript=transcript_with_speakers(segments),
        summary=summary,
        summary_uz=summary_uz,
        meta=meta,
        token_txt=token_txt,
        token_srt=token_srt,
        media_url=url_for("media.download", token=token_media),
        media_type="audio",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=translate_flag,
        token_pdf=token_pdfdata,
    )


# ---------- downloads ----------
@bp.get("/download/<token>")
def download(token: str):
    item = DOWNLOADS.get(token)
    if not item:
        return "Not found or expired", 404

    return send_file(
        BytesIO(item["data"]),
        mimetype=item["mimetype"],
        as_attachment=True,
        download_name=item["filename"],
    )


# ---------- PDF download ----------
@bp.get("/download_pdf/<token>")
def download_pdf(token: str):
    item = DOWNLOADS.pop(token, None)
    if not item:
        return "Not found or expired", 404
    try:
        payload = json.loads(item["data"].decode("utf-8"))
    except Exception:
        return "Invalid payload", 400

    pdf_bytes = build_pdf_bytes(payload)
    base = os.path.splitext(payload.get("filename", "export"))[0]
    pdf_name = f"{base}.pdf"

    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=pdf_name,
    )


@bp.get("/healthz")
def healthz():
    now = datetime.utcnow()
    cleanup_downloads(now)
    cleanup_jobs(now)
    return "ok", 200