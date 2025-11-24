# app_ext/routes_youtube.py

import os
import json
import shutil
import tempfile

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)

from app_ext.transcription import (
    transcribe_audio_simple,
    transcribe_audio_diarized,
    summarize_text,
    build_srt_from_segments,
    transcript_with_speakers,
    diarization_summary,
    translate_texts_to_uz,
)
from app_ext.downloads import register_download
from app_ext.pdf_export import register_pdf_payload
from app_ext.youtube import yt_fetch_captions, yt_download_video, parse_vtt

bp = Blueprint("youtube", __name__)


# ---------- YOUTUBE (simple) ----------
@bp.post("/ingest_youtube_simple")
def ingest_youtube_simple():
    url = (request.form.get("youtube_url") or "").strip()
    if not url:
        flash("No YouTube URL.")
        return redirect(url_for("media.index"))

    translate_flag = bool(request.form.get("translate_uz"))

    with tempfile.TemporaryDirectory() as tmpdir:
        title = None
        segments = []
        transcript = ""

        try:
            vtt_path, title = yt_fetch_captions(url, tmpdir)
            if vtt_path and os.path.exists(vtt_path):
                with open(vtt_path, "r", encoding="utf-8", errors="ignore") as fh:
                    vtt_text = fh.read()
                segments, transcript = parse_vtt(vtt_text)
        except Exception as e:
            print("[yt] captions fetch error:", e)

        media_bytes = None
        media_mime = "video/mp4"
        media_filename = None

        video_path, title2 = yt_download_video(url, tmpdir)
        title = title or title2 or "youtube_audio"

        if not segments:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpa:
                shutil.copyfile(video_path, tmpa.name)
                segs, tx = transcribe_audio_simple(tmpa.name)
                segments, transcript = segs, tx

        with open(video_path, "rb") as fh:
            media_bytes = fh.read()
        media_filename = os.path.basename(video_path)

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

        base = (title or "youtube").replace("/", "_").replace("\\", "_").strip()
        token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
        token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")

        token_media = register_download(media_bytes, media_mime, media_filename or f"{base}.mp4")
        media_url = url_for("media.download", token=token_media)

        token_pdfdata = register_pdf_payload(
            version="YouTube — Simple",
            media_type="video",
            filename=title or "YouTube",
            summary=summary,
            summary_uz=summary_uz,
            meta=None,
            segments=segments,
        )

        return render_template(
            "result_media.html",
            version="YouTube — Simple",
            transcript=transcript,
            summary=summary,
            summary_uz=summary_uz,
            meta=None,
            token_txt=token_txt,
            token_srt=token_srt,
            media_url=media_url,
            media_type="video",
            segments_json=json.dumps(segments, ensure_ascii=False),
            show_uz=translate_flag,
            token_pdf=token_pdfdata,
        )


# ---------- YOUTUBE (differentiated/diarized) ----------
@bp.post("/ingest_youtube_diarized")
def ingest_youtube_diarized():
    url = (request.form.get("youtube_url_d") or "").strip()
    if not url:
        flash("No YouTube URL.")
        return redirect(url_for("media.index"))

    translate_flag = bool(request.form.get("translate_uz"))

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path, title = yt_download_video(url, tmpdir)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpa:
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
        meta = diarization_summary(segments)

        base = (title or "youtube").replace("/", "_").replace("\\", "_").strip()
        token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
        token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")

        with open(video_path, "rb") as fh:
            media_bytes = fh.read()
        token_media = register_download(media_bytes, "video/mp4", os.path.basename(video_path))

        token_pdfdata = register_pdf_payload(
            version="YouTube — Differentiated",
            media_type="video",
            filename=title or "YouTube",
            summary=summary,
            summary_uz=summary_uz,
            meta=meta,
            segments=segments,
        )

        return render_template(
            "result_media.html",
            version=f"YouTube — Differentiated ({mode_used})",
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