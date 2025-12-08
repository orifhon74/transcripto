# app_ext/routes_downloads.py

from io import BytesIO
import json

from flask import Blueprint, abort, send_file

from app_ext.downloads import DOWNLOADS, cleanup_downloads
from app_ext.pdf_export import build_pdf_bytes

bp = Blueprint("downloads", __name__)


def _get_item_or_404(token: str) -> dict:
    """Fetch a download item by token, with a quick cleanup + 404 on miss."""
    item = DOWNLOADS.get(token)
    if item is None:
        # Try cleaning up old entries, then re-check
        cleanup_downloads()
        item = DOWNLOADS.get(token)
    if item is None:
        abort(404)
    return item


@bp.get("/download/<token>")
def download(token: str):
    """
    Serve a stored blob (txt, srt, original media, pdf payload json, etc.)
    """
    item = _get_item_or_404(token)

    return send_file(
        BytesIO(item["data"]),
        mimetype=item["mimetype"],
        as_attachment=True,
        download_name=item["filename"],
    )


@bp.get("/download_pdf/<token>")
def download_pdf(token: str):
    """
    Build a PDF on the fly from the JSON payload saved by register_pdf_payload
    and stream it to the user.
    """
    item = _get_item_or_404(token)

    # The payload is JSON produced by register_pdf_payload
    payload = json.loads(item["data"].decode("utf-8"))
    pdf_bytes = build_pdf_bytes(payload)

    filename_base = (payload.get("filename") or "report").rsplit(".", 1)[0]
    pdf_name = f"{filename_base}.pdf"

    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=pdf_name,
    )