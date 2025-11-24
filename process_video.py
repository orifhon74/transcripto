# process_video.py
"""
Thin compatibility wrapper around the new media_core package.

This keeps app.py working while we gradually move routes to app_ext/transcription.py.
"""

from media_core.summarization import summarize_text
from media_core.translation import translate_texts_to_uz
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

__all__ = [
    "summarize_text",
    "translate_texts_to_uz",
    "build_srt_from_segments",
    "transcript_with_speakers",
    "transcribe_video_simple",
    "transcribe_video_diarized",
    "transcribe_audio_simple",
    "transcribe_audio_diarized",
    "diarization_summary",
]