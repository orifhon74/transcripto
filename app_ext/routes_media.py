# app_ext/routes_media.py

import json
import os
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

from media_core.summarization import summarize_text
from media_core.translation import translate_texts, get_supported_languages
from media_core.formatting import (
    build_srt_from_segments,
    transcript_with_speakers,
)
from media_core.diarization import (
    transcribe_video_simple,
    transcribe_video_diarized,
    transcribe_audio_simple,
    transcribe_audio_diarized,
    diarization_summary,
)

media_bp = Blueprint("media", __name__)


# -------------------------------------------------------------------
# Small helper: figure out requested target language
# - new way:  target_lang form field
# - old way:  translate_uz checkbox -> "uz"
# -------------------------------------------------------------------
def _get_target_lang(form) -> str:
    """
    Determine which language (if any) the user asked for.

    Priority:
      1) form["target_lang"] if present and non-empty
      2) legacy form["translate_uz"] -> "uz"
    """
    lang = (form.get("target_lang") or "").strip().lower()
    if not lang and form.get("translate_uz"):
        lang = "uz"
    return lang


# -------------------------------------------------------------------
# Home page
# -------------------------------------------------------------------
@media_bp.get("/")
def index():
    supported_langs = get_supported_languages()
    return render_template("index.html", supported_langs=supported_langs)


# -------------------------------------------------------------------
# TEXT
# -------------------------------------------------------------------
@media_bp.post("/upload_text")
def upload_text():
    f = request.files.get("text_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

    text = f.read().decode("utf-8", errors="ignore")
    summary = summarize_text(text)

    base = os.path.splitext(f.filename)[0]
    token_text = register_download(
        text.encode("utf-8"),
        "text/plain",
        f"{base}_original.txt",
    )

    # Minimal PDF for text-only: include summary + original as one segment
    segments = [{"start": 0, "end": 0, "text": text}]

    token_pdfdata = register_pdf_payload(
        version="Text",
        media_type="text",
        filename=f.filename,
        summary=summary,
        # For now we don't auto-translate text uploads.
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


# -------------------------------------------------------------------
# VIDEO (simple)
# -------------------------------------------------------------------
@media_bp.post("/upload_video_simple")
def upload_video_simple():
    f = request.files.get("video_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

    target_lang = _get_target_lang(request.form)

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript = transcribe_video_simple(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""  # still named *_uz for template/pdf compatibility

    # Summary translation
    if target_lang and summary:
        summary_uz = "\n".join(translate_texts([summary], target_lang))

    # Per-segment translation
    if target_lang and segments:
        seg_texts = [s.get("text", "") for s in segments]
        tr_lines = translate_texts(seg_texts, target_lang)

        for s, tr in zip(segments, tr_lines):
            # Keep using "uz" field name for now so result_media.html and PDF stay simple
            s["uz"] = tr

    srt = build_srt_from_segments(segments)

    base = os.path.splitext(f.filename)[0]
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
        file_bytes,
        f.mimetype or "video/mp4",
        f.filename,
    )

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
        media_url=url_for("downloads.download", token=token_media),
        media_type="video",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=bool(target_lang),  # means "show translated lines"
        token_pdf=token_pdfdata,
    )


# -------------------------------------------------------------------
# VIDEO (differentiated / diarized)
# -------------------------------------------------------------------
@media_bp.post("/upload_video_diarized")
def upload_video_diarized():
    f = request.files.get("video_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

    target_lang = _get_target_lang(request.form)

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript, mode_used = transcribe_video_diarized(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""

    if target_lang and summary:
        summary_uz = "\n".join(translate_texts([summary], target_lang))

    if target_lang and segments:
        seg_texts = [s.get("text", "") for s in segments]
        tr_lines = translate_texts(seg_texts, target_lang)

        for s, tr in zip(segments, tr_lines):
            s["uz"] = tr

    srt = build_srt_from_segments(segments)
    meta = diarization_summary(segments)

    base = os.path.splitext(f.filename)[0]
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
        file_bytes,
        f.mimetype or "video/mp4",
        f.filename,
    )

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
        media_url=url_for("downloads.download", token=token_media),
        media_type="video",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=bool(target_lang),
        token_pdf=token_pdfdata,
    )


# -------------------------------------------------------------------
# AUDIO (simple)
# -------------------------------------------------------------------
@media_bp.post("/upload_audio_simple")
def upload_audio_simple():
    f = request.files.get("audio_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

    target_lang = _get_target_lang(request.form)

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript = transcribe_audio_simple(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""

    if target_lang and summary:
        summary_uz = "\n".join(translate_texts([summary], target_lang))

    if target_lang and segments:
        seg_texts = [s.get("text", "") for s in segments]
        tr_lines = translate_texts(seg_texts, target_lang)

        for s, tr in zip(segments, tr_lines):
            s["uz"] = tr

    srt = build_srt_from_segments(segments)

    base = os.path.splitext(f.filename)[0]
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
        file_bytes,
        f.mimetype or "audio/wav",
        f.filename,
    )

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
        media_url=url_for("downloads.download", token=token_media),
        media_type="audio",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=bool(target_lang),
        token_pdf=token_pdfdata,
    )


# -------------------------------------------------------------------
# AUDIO (differentiated / diarized)
# -------------------------------------------------------------------
@media_bp.post("/upload_audio_diarized")
def upload_audio_diarized():
    f = request.files.get("audio_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("media.index"))

    target_lang = _get_target_lang(request.form)

    file_bytes = f.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, transcript, mode_used = transcribe_audio_diarized(tmp.name)

    summary = summarize_text(transcript)
    summary_uz = ""

    if target_lang and summary:
        summary_uz = "\n".join(translate_texts([summary], target_lang))

    if target_lang and segments:
        seg_texts = [s.get("text", "") for s in segments]
        tr_lines = translate_texts(seg_texts, target_lang)

        for s, tr in zip(segments, tr_lines):
            s["uz"] = tr

    srt = build_srt_from_segments(segments)
    meta = diarization_summary(segments)

    base = os.path.splitext(f.filename)[0]
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
        file_bytes,
        f.mimetype or "audio/wav",
        f.filename,
    )

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
        media_url=url_for("downloads.download", token=token_media),
        media_type="audio",
        segments_json=json.dumps(segments, ensure_ascii=False),
        show_uz=bool(target_lang),
        token_pdf=token_pdfdata,
    )