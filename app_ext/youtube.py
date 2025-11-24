# app_ext/youtube.py

import os
import yt_dlp
import shutil as _shutil

def yt_dl_opts_base(tmpdir: str) -> dict:
    from config import Config  # safe local import

    ua = os.getenv(
        "YTDLP_UA",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    )
    ff = _shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
    if not os.path.exists(ff):
        alt = "/opt/homebrew/bin/ffmpeg"
        if os.path.exists(alt):
            ff = alt

    ydl_opts = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "geo_bypass": True,
        "http_headers": {
            "User-Agent": ua,
            "Accept-Language": "en-US,en;q=0.9",
        },
        "ffmpeg_location": ff,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web_safari", "tv_embedded"],
            }
        },
    }

    cfb = os.getenv("COOKIES_FROM_BROWSER")
    if cfb:
        ydl_opts["cookiesfrombrowser"] = (cfb, None, None, None)

    cookiefile = os.getenv("YTDLP_COOKIEFILE")
    if cookiefile and os.path.exists(cookiefile):
        ydl_opts["cookiefile"] = cookiefile

    return ydl_opts


def yt_dl_opts_audio(tmpdir: str) -> dict:
    o = yt_dl_opts_base(tmpdir)
    o.update({
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    })
    return o


def yt_dl_opts_subs(tmpdir: str) -> dict:
    o = yt_dl_opts_base(tmpdir)
    o.update({
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
    })
    return o


def yt_fetch_captions(url: str, tmpdir: str):
    opts = yt_dl_opts_subs(tmpdir)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title")
        vid = info.get("id")
        vtt_path = None
        for fname in os.listdir(tmpdir):
            if fname.startswith(vid) and fname.endswith(".vtt"):
                vtt_path = os.path.join(tmpdir, fname)
                break
        return vtt_path, title


def yt_download_video(url: str, tmpdir: str):
    opts = yt_dl_opts_base(tmpdir)
    opts.update({
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or info.get("id")
        out = ydl.prepare_filename(info)
        if not out.endswith(".mp4"):
            base, _ = os.path.splitext(out)
            mp4_candidate = base + ".mp4"
            if os.path.exists(mp4_candidate):
                out = mp4_candidate
        return out, title


def parse_vtt(vtt_text: str):
    def _parse_ts(ts: str) -> float:
        ts = ts.strip().replace(',', '.')
        parts = ts.split(':')
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = '0', parts[0], parts[1]
        else:
            return 0.0
        return int(h) * 3600 + int(m) * 60 + float(s)

    lines = [ln.rstrip('\n') for ln in vtt_text.splitlines()]
    segs = []
    i = 0
    buff = []
    start = end = None

    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln:
            continue
        if '-->' in ln:
            if start is not None and buff:
                text = ' '.join(buff).strip()
                if text:
                    segs.append({"start": start, "end": end, "text": text})
            buff = []
            try:
                a, b = ln.split('-->')
                start = _parse_ts(a.strip())
                end = _parse_ts(b.strip().split(' ')[0])
            except Exception:
                start = end = None
        else:
            if start is not None:
                buff.append(ln)

    if start is not None and buff:
        text = ' '.join(buff).strip()
        if text:
            segs.append({"start": start, "end": end, "text": text})

    full = " ".join(s["text"] for s in segs).strip()
    return segs, full