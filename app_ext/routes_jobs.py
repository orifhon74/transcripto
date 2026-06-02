# app_ext/routes_jobs.py

import json
from uuid import uuid4

from flask import (
    Blueprint, request, jsonify, render_template, url_for, abort,
)

from app_ext.jobs import JOBS, EXECUTOR, VALID_KINDS, new_job, run_media_job

bp = Blueprint("jobs", __name__)


def _target_lang(form) -> str:
    lang = (form.get("target_lang") or "").strip().lower()
    if not lang and form.get("translate_uz"):
        lang = "uz"
    return lang


@bp.post("/jobs")
def create_job():
    """
    Start an async transcription/summarization job.

    Accepts (multipart):
      kind         -> one of VALID_KINDS
      file         -> required for upload/text kinds
      youtube_url  -> required for youtube_* kinds
      target_lang  -> optional translation target
    Returns: {"job_id": "..."} with 202.
    """
    kind = (request.form.get("kind") or "").strip()
    if kind not in VALID_KINDS:
        return jsonify({"error": "Invalid 'kind'."}), 400

    target_lang = _target_lang(request.form)
    job_id = uuid4().hex

    if kind.startswith("youtube_"):
        url = (request.form.get("youtube_url") or "").strip()
        if not url:
            return jsonify({"error": "Missing 'youtube_url'."}), 400
        job = new_job(kind, url)
        job["id"] = job_id
        JOBS[job_id] = job
        EXECUTOR.submit(run_media_job, job_id, kind,
                        youtube_url=url, target_lang=target_lang)
    else:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "Missing 'file'."}), 400
        file_bytes = f.read()
        job = new_job(kind, f.filename)
        job["id"] = job_id
        JOBS[job_id] = job
        EXECUTOR.submit(run_media_job, job_id, kind,
                        file_bytes=file_bytes, filename=f.filename,
                        target_lang=target_lang)

    return jsonify({"job_id": job_id}), 202


@bp.get("/jobs/<job_id>")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404

    payload = {
        "id": job["id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
        "stage": job.get("stage", ""),
        "kind": job["kind"],
        "filename": job["filename"],
        "logs": job["logs"][-30:],
        "error": job.get("error"),
    }
    if job["status"] == "done":
        payload["result_url"] = url_for("jobs.show_result", job_id=job_id)
    return jsonify(payload), 200


@bp.get("/result/<job_id>")
def show_result(job_id: str):
    """Render the finished job's result page from its stored payload."""
    job = JOBS.get(job_id)
    if not job or job["status"] != "done" or not job.get("result"):
        abort(404)

    r = job["result"]

    if r["template"] == "text":
        return render_template(
            "result_text.html",
            file_label=r.get("file_label"),
            summary=r.get("summary"),
            summary_uz=r.get("summary_uz"),
            show_uz=r.get("show_uz", False),
            token_text=r["tokens"].get("text"),
            token_pdf=r["tokens"].get("pdf"),
        )

    tokens = r["tokens"]
    media_url = (
        url_for("downloads.download", token=tokens["media"])
        if tokens.get("media") else None
    )
    return render_template(
        "result_media.html",
        version=r["version"],
        transcript=r["transcript"],
        summary=r["summary"],
        summary_uz=r["summary_uz"],
        meta=r["meta"],
        token_txt=tokens["txt"],
        token_srt=tokens["srt"],
        token_pdf=tokens.get("pdf"),
        media_url=media_url,
        media_type=r["media_type"],
        segments_json=json.dumps(r["segments"], ensure_ascii=False),
        show_uz=r["show_uz"],
    )
