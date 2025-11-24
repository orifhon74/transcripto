# app_ext/pdf_export.py

import json
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from app_ext.downloads import register_download


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
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=body, fontSize=9, leading=12)
    mono = ParagraphStyle("Mono", parent=body, fontName="Courier", fontSize=9, leading=12)
    uz_style = ParagraphStyle("Uz", parent=small, textColor=colors.green, italic=True)

    story = []

    title = payload.get("title") or "Transcription Report"
    story.append(Paragraph(title, h1))
    story.append(Spacer(1, 6))
    story.append(Paragraph(payload.get("version", ""), small))
    story.append(Paragraph(f"File: {payload.get('filename','')}", small))
    story.append(Paragraph(f"Generated: {payload.get('created_at','')}", small))
    story.append(Spacer(1, 10))

    if payload.get("summary"):
        story.append(Paragraph("Summary", h2))
        for line in (payload["summary"] or "").split("\n"):
            story.append(Paragraph(line.strip(), body))
        if payload.get("summary_uz"):
            story.append(Spacer(1, 6))
            story.append(Paragraph("Uzbek Summary", h3))
            for line in (payload["summary_uz"] or "").split("\n"):
                story.append(Paragraph(line.strip(), uz_style))
        story.append(Spacer(1, 10))

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
                story.append(Paragraph(uz, uz_style))
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
        "summary_uz": summary_uz or "",
        "meta": meta or {},
        "segments": segments or [],
    }
    return register_download(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        "application/json",
        f"{filename.rsplit('.', 1)[0]}_pdf_payload.json",
    )