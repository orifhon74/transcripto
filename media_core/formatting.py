# media_core/formatting.py

from typing import List, Dict


def transcript_with_speakers(segments: List[Dict]) -> str:
    lines = []
    for s in segments:
        spk = s.get("speaker")
        prefix = f"{spk}: " if spk else ""
        lines.append(prefix + s["text"])
    return "\n".join(lines)


def build_srt_from_segments(segments: List[Dict]) -> str:
    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(s['start'])} --> {fmt_time(s['end'])}")
        prefix = f"{s.get('speaker')}: " if s.get('speaker') else ""
        lines.append(prefix + s["text"])
        lines.append("")
    return "\n".join(lines)