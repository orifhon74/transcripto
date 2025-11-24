# app_ext/routes_jobs.py

from uuid import uuid4

from flask import Blueprint, request, jsonify

from app_ext.jobs import JOBS, EXECUTOR, run_media_job

bp = Blueprint("jobs", __name__)


@bp.post("/jobs")
def create_job():
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
        "created_at": "",
        "logs": [],
        "artifacts": {},
        "meta": {},
    }
    JOBS[job_id]["created_at"] = JOBS[job_id]["created_at"] or \
        __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z"

    file_bytes = f.read()
    EXECUTOR.submit(run_media_job, job_id, kind, file_bytes, f.filename)

    return jsonify({"job_id": job_id}), 202


@bp.get("/jobs/<job_id>")
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
        "logs": job["logs"][-50:],
        "meta": job.get("meta", {}),
        "artifacts": job.get("artifacts", {}),
    }), 200