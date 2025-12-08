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

from app_ext.downloads import register_download
from app_ext.pdf_export import register_pdf_payload
from app_ext.youtube import yt_fetch_captions, yt_download_video, parse_vtt

# Use the new media_core pipeline
from media_core.summarization import summarize_text
from media_core.translation import translate_texts
from media_core.formatting import build_srt_from_segments, transcript_with_speakers
from media_core.diarization import (
    transcribe_audio_simple,
    transcribe_audio_diarized,
    diarization_summary,
)

bp = Blueprint("youtube", __name__)


# -------------------------------------------------------------------
# Helper: figure out requested target language
# (same semantics as in routes_media._get_target_lang)
# -------------------------------------------------------------------
def _get_target_lang(form) -> str:
    """
    Determine which language (if any) the user asked for.

    Priority:
      1) form["target_lang"] if present and non-empty
      2) legacy form["translate_uz"] checkbox -> "uz"
    """
    lang = (form.get("target_lang") or "").strip().lower()
    if not lang and form.get("translate_uz"):
        lang = "uz"
    return lang


# -------------------------------------------------------------------
# YOUTUBE (simple, captions-first)
# -------------------------------------------------------------------
@bp.post("/ingest_youtube_simple")
def ingest_youtube_simple():
    url = (request.form.get("youtube_url") or "").strip()
    if not url:
        flash("No YouTube URL.")
        return redirect(url_for("media.index"))

    target_lang = _get_target_lang(request.form)

    with tempfile.TemporaryDirectory() as tmpdir:
        title = None
        segments = []
        transcript = ""

        # 1) Try English captions first
        try:
            vtt_path, title = yt_fetch_captions(url, tmpdir)
            if vtt_path and os.path.exists(vtt_path):
                with open(vtt_path, "r", encoding="utf-8", errors="ignore") as fh:
                    vtt_text = fh.read()
                segments, transcript = parse_vtt(vtt_text)
        except Exception as e:
            print("[yt] captions fetch error:", e)

        # 2) Download the video (we always want the media bytes for playback)
        media_bytes = None
        media_mime = "video/mp4"
        media_filename = None

        video_path, title2 = yt_download_video(url, tmpdir)
        title = title or title2 or "youtube_audio"

        # 3) If no captions, run Whisper on the audio
        if not segments:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpa:
                shutil.copyfile(video_path, tmpa.name)
                segs, tx = transcribe_audio_simple(tmpa.name)
                segments, transcript = segs, tx

        # 4) Read video bytes for the embedded player
        with open(video_path, "rb") as fh:
            media_bytes = fh.read()
        media_filename = os.path.basename(video_path)

        # 5) Summary + translations
        summary = summarize_text(transcript)
        summary_uz = ""

        if target_lang and summary:
            # always pass target_lang
            summary_uz = "\n".join(translate_texts([summary], target_lang))

        if target_lang and segments:
            seg_texts = [s.get("text", "") for s in segments]
            tr_lines = translate_texts(seg_texts, target_lang)

            for s, tr in zip(segments, tr_lines):
                # still store in "uz" field for now so templates keep working
                s["uz"] = tr

        # 6) Subtitles + download tokens
        srt = build_srt_from_segments(segments)

        base = (title or "youtube").replace("/", "_").replace("\\", "_").strip()
        token_txt = register_download(
            transcript.encode("utf-8"),
            "text/plain",
            f"{base}.txt",
        )
        token_srt = register_download(
            srt.encode("utf-8"),
            "application/x-subrip",
            f"{base}.srt",
        )

        token_media = register_download(
            media_bytes,
            media_mime,
            media_filename or f"{base}.mp4",
        )
        media_url = url_for("downloads.download", token=token_media)

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
            show_uz=bool(target_lang),
            token_pdf=token_pdfdata,
        )


# -------------------------------------------------------------------
# YOUTUBE (differentiated / diarized)
# -------------------------------------------------------------------
@bp.post("/ingest_youtube_diarized")
def ingest_youtube_diarized():
    url = (request.form.get("youtube_url_d") or "").strip()
    if not url:
        flash("No YouTube URL.")
        return redirect(url_for("media.index"))

    target_lang = _get_target_lang(request.form)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) Download the video
        video_path, title = yt_download_video(url, tmpdir)

        # 2) Run diarized transcription on the audio track
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpa:
            shutil.copyfile(video_path, tmpa.name)
            segments, transcript, mode_used = transcribe_audio_diarized(tmpa.name)

        # 3) Summary + translations
        summary = summarize_text(transcript)
        summary_uz = ""

        if target_lang and summary:
            summary_uz = "\n".join(translate_texts([summary], target_lang))

        if target_lang and segments:
            seg_texts = [s.get("text", "") for s in segments]
            tr_lines = translate_texts(seg_texts, target_lang)

            for s, tr in zip(segments, tr_lines):
                s["uz"] = tr

        # 4) SRT + diarization meta
        srt = build_srt_from_segments(segments)
        meta = diarization_summary(segments)

        base = (title or "youtube").replace("/", "_").replace("\\", "_").strip()
        token_txt = register_download(
            transcript.encode("utf-8"),
            "text/plain",
            f"{base}.txt",
        )
        token_srt = register_download(
            srt.encode("utf-8"),
            "application/x-subrip",
            f"{base}.srt",
        )

        # 5) Media download token
        with open(video_path, "rb") as fh:
            media_bytes = fh.read()
        token_media = register_download(
            media_bytes,
            "video/mp4",
            os.path.basename(video_path),
        )

        token_pdfdata = register_pdf_payload(
            version=f"YouTube — Differentiated ({mode_used})",
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
            media_url=url_for("downloads.download", token=token_media),
            media_type="video",
            segments_json=json.dumps(segments, ensure_ascii=False),
            show_uz=bool(target_lang),
            token_pdf=token_pdfdata,
        )