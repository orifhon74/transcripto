import os
import json
import tempfile
from io import BytesIO
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()  # load env BEFORE importing process_video (it reads env at import)

from flask import Flask, render_template, request, redirect, url_for, flash, send_file

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
)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev")

# ---------------- in-memory downloads registry ----------------
DOWNLOADS = {}  # token -> {"data": bytes, "mimetype": str, "filename": str}

def register_download(data: bytes, mimetype: str, filename: str) -> str:
    token = uuid4().hex
    DOWNLOADS[token] = {"data": data, "mimetype": mimetype, "filename": filename}
    return token

# ---------------- routes ----------------

@app.get("/")
def index():
    # Expect templates/index.html and templates/result_media.html / result_text.html in your project
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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        f.save(tmp.name)
        segments, transcript = transcribe_video_simple(tmp.name)

    summary = summarize_text(transcript)
    srt = build_srt_from_segments(segments)
    verification = verification_report_from(
        media_info={"type": "video", "name": f.filename, "diarization_mode": "none"},
        transcript_text=transcript,
        segments=segments,
    )

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                   "application/json", f"{base}_verification.json")

    return render_template(
        "result_media.html",
        version="Video — Simple",
        transcript=transcript,
        summary=summary,
        verification=verification,
        meta=None,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
    )

# ---------- VIDEO (differentiated / diarized) ----------
@app.post("/upload_video_diarized")
def upload_video_diarized():
    f = request.files.get("video_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        f.save(tmp.name)
        segments, transcript, mode_used = transcribe_video_diarized(tmp.name)

    display_transcript = transcript_with_speakers(segments)
    summary = summarize_text(transcript)
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

    return render_template(
        "result_media.html",
        version=f"Video — Differentiated ({mode_used})",
        transcript=display_transcript,
        summary=summary,
        verification=verification,
        meta=meta,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
    )

# ---------- AUDIO (simple) ----------
@app.post("/upload_audio_simple")
def upload_audio_simple():
    f = request.files.get("audio_file")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        f.save(tmp.name)
        segments, transcript = transcribe_audio_simple(tmp.name)

    summary = summarize_text(transcript)
    srt = build_srt_from_segments(segments)
    verification = verification_report_from(
        media_info={"type": "audio", "name": f.filename, "diarization_mode": "none"},
        transcript_text=transcript,
        segments=segments,
    )

    base = os.path.splitext(f.filename)[0]
    token_txt = register_download(transcript.encode("utf-8"), "text/plain", f"{base}.txt")
    token_srt = register_download(srt.encode("utf-8"), "application/x-subrip", f"{base}.srt")
    token_json = register_download(json.dumps(verification, ensure_ascii=False, indent=2).encode("utf-8"),
                                   "application/json", f"{base}_verification.json")

    return render_template(
        "result_media.html",
        version="Audio — Simple",
        transcript=transcript,
        summary=summary,
        verification=verification,
        meta=None,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
    )

# ---------- AUDIO (differentiated / diarized) ----------
@app.post("/upload_audio_diarized")
def upload_audio_diarized():
    f = request.files.get("audio_file_d")
    if not f:
        flash("No file selected.")
        return redirect(url_for("index"))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        f.save(tmp.name)
        segments, transcript, mode_used = transcribe_audio_diarized(tmp.name)

    display_transcript = transcript_with_speakers(segments)
    summary = summarize_text(transcript)
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

    return render_template(
        "result_media.html",
        version=f"Audio — Differentiated ({mode_used})",
        transcript=display_transcript,
        summary=summary,
        verification=verification,
        meta=meta,
        token_txt=token_txt,
        token_srt=token_srt,
        token_json=token_json,
    )

# ---------- downloads ----------
@app.get("/download/<token>")
def download(token: str):
    item = DOWNLOADS.get(token)
    if not item:
        return "Not found", 404
    return send_file(
        BytesIO(item["data"]),
        mimetype=item["mimetype"],
        as_attachment=True,
        download_name=item["filename"],
    )

@app.get("/healthz")
def healthz():
    return "ok", 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Railway sets $PORT automatically
    app.run(host="0.0.0.0", port=port, debug=False)