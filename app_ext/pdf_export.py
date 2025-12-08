# app_ext/pdf_export.py

import json
import os
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app_ext.downloads import register_download

# ---------------------------------------------------------
# Font setup: use a Unicode TTF font for all translated text
# ---------------------------------------------------------

# Adjust this path if you put the font somewhere else.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "..", "static", "fonts", "DejaVuSans.ttf")

# Name we’ll register the font under
UNICODE_FONT_NAME = "DejaVuSans"
FALLBACK_FONT_NAME = "Helvetica"  # used if TTF missing

try:
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont(UNICODE_FONT_NAME, FONT_PATH))
        ACTIVE_FONT = UNICODE_FONT_NAME
    else:
        print(f"[pdf_export] WARNING: {FONT_PATH} not found, using fallback font.")
        ACTIVE_FONT = FALLBACK_FONT_NAME
except Exception as e:
    print(f"[pdf_export] WARNING: could not register font {FONT_PATH}: {e}")
    ACTIVE_FONT = FALLBACK_FONT_NAME


def build_pdf_bytes(payload: dict) -> bytes:
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

    # Base styles
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    # Ensure base body uses our Unicode font
    body.fontName = ACTIVE_FONT
    h1.fontName = ACTIVE_FONT
    h2.fontName = ACTIVE_FONT
    h3.fontName = ACTIVE_FONT

    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=9,
        leading=12,
        fontName=ACTIVE_FONT,
    )

    mono = ParagraphStyle(
        "Mono",
        parent=body,
        fontName=ACTIVE_FONT,  # not truly mono, but supports Unicode
        fontSize=9,
        leading=12,
    )

    translated_style = ParagraphStyle(
        "Translated",
        parent=small,
        fontName=ACTIVE_FONT,
        textColor=colors.green,
        italic=True,
    )

    story = []

    title = payload.get("title") or "Transcription Report"
    story.append(Paragraph(title, h1))
    story.append(Spacer(1, 6))
    story.append(Paragraph(payload.get("version", ""), small))
    story.append(Paragraph(f"File: {payload.get('filename', '')}", small))
    story.append(Paragraph(f"Generated: {payload.get('created_at', '')}", small))
    story.append(Spacer(1, 10))

    # -----------------------
    # Summary + translation
    # -----------------------
    if payload.get("summary"):
        story.append(Paragraph("Summary", h2))
        for line in (payload["summary"] or "").split("\n"):
            story.append(Paragraph(line.strip(), body))

        if payload.get("summary_uz"):
            story.append(Spacer(1, 6))
            # This is now generic – works for any language we stuffed into summary_uz
            story.append(Paragraph("Translated Summary", h3))
            for line in (payload["summary_uz"] or "").split("\n"):
                story.append(Paragraph(line.strip(), translated_style))

        story.append(Spacer(1, 10))

    # -----------------------
    # Transcript + per-line translation
    # -----------------------
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

            text = (s.get("text") or "")
            # basic escaping for XML
            text = (
                text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
            )

            story.append(Paragraph(f"<b>[{ts_label}]</b> {prefix}{text}", mono))

            # We still store translated line in "uz" field, regardless of target language
            if s.get("uz"):
                uz = (s.get("uz") or "")
                uz = (
                    uz.replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;")
                )
                story.append(Paragraph(uz, translated_style))

        story.append(Spacer(1, 8))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def register_pdf_payload(*, version, media_type, filename, summary, summary_uz, meta, segments) -> str:
    payload = {
        "title": "Transcription Report",
        "version": version,
        "media_type": media_type,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "summary": summary or "",
        # summary_uz is now “translated summary” (any lang), name kept for backward compatibility
        "summary_uz": summary_uz or "",
        "meta": meta or {},
        "segments": segments or [],
    }
    return register_download(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        "application/json",
        f"{filename.rsplit('.', 1)[0]}_pdf_payload.json",
    )