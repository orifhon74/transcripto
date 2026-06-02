"""
Unit tests for the pure logic in the transcription pipeline.

The diarization / transcription modules pull in heavy native dependencies
(torch, webrtcvad, resemblyzer, faster-whisper, yt-dlp). Those are irrelevant
to the deterministic logic we want to test, so we install lightweight stub
modules into ``sys.modules`` before importing. This lets us exercise the *real*
functions without GPU/model downloads.

Run with:  python -m unittest discover -s tests
"""

import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------
# Stub out heavy optional dependencies so imports succeed on a bare machine.
# --------------------------------------------------------------------------
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub("webrtcvad", Vad=lambda *a, **k: None)
_stub("pydub", AudioSegment=object)
_stub("resemblyzer", VoiceEncoder=object, preprocess_wav=lambda *a, **k: None)
_stub("faster_whisper", WhisperModel=object)
_stub("openai", OpenAI=lambda *a, **k: None)
_stub("yt_dlp", YoutubeDL=object)


# Now the real modules import cleanly.
from media_core.formatting import build_srt_from_segments, transcript_with_speakers
from media_core.diarization import (
    assign_speakers_from_turns,
    diarization_summary,
    _estimate_num_speakers,
)
from app_ext.youtube import parse_vtt

import numpy as np


class TestFormatting(unittest.TestCase):
    def test_srt_basic(self):
        segs = [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 1.5, "end": 3.25, "text": "World"},
        ]
        srt = build_srt_from_segments(segs)
        self.assertIn("00:00:00,000 --> 00:00:01,500", srt)
        self.assertIn("00:00:01,500 --> 00:00:03,250", srt)
        self.assertIn("1\n", srt)
        self.assertIn("Hello", srt)

    def test_srt_with_speaker_prefix(self):
        segs = [{"start": 0, "end": 1, "text": "Hi", "speaker": "S2"}]
        self.assertIn("S2: Hi", build_srt_from_segments(segs))

    def test_transcript_with_speakers(self):
        segs = [
            {"start": 0, "end": 1, "text": "a", "speaker": "S1"},
            {"start": 1, "end": 2, "text": "b", "speaker": "S2"},
        ]
        self.assertEqual(transcript_with_speakers(segs), "S1: a\nS2: b")


class TestSpeakerAssignment(unittest.TestCase):
    def test_no_turns_defaults_single_speaker(self):
        segs = [{"start": 0, "end": 1, "text": "x"}]
        out = assign_speakers_from_turns(segs, [])
        self.assertEqual(out[0]["speaker"], "S1")

    def test_majority_overlap_wins(self):
        # Segment 0-4s overlaps S1 for 1s and S2 for 3s -> S2.
        segs = [{"start": 0.0, "end": 4.0, "text": "hello"}]
        turns = [(0.0, 1.0, "S1"), (1.0, 4.0, "S2")]
        out = assign_speakers_from_turns(segs, turns)
        # Single speaker present -> renumbered to S1.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker"], "S1")

    def test_two_speakers_merge_and_renumber(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "one"},
            {"start": 2.0, "end": 4.0, "text": "two"},
            {"start": 4.0, "end": 6.0, "text": "three"},
        ]
        turns = [(0.0, 2.0, "A"), (2.0, 4.0, "B"), (4.0, 6.0, "A")]
        out = assign_speakers_from_turns(segs, turns)
        speakers = [s["speaker"] for s in out]
        # First-appearance renumbering: A->S1, B->S2.
        self.assertEqual(speakers, ["S1", "S2", "S1"])

    def test_consecutive_same_speaker_merges(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "hello"},
            {"start": 2.0, "end": 4.0, "text": "world"},
        ]
        turns = [(0.0, 4.0, "A")]
        out = assign_speakers_from_turns(segs, turns)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "hello world")
        self.assertEqual(out[0]["end"], 4.0)

    def test_short_flicker_is_smoothed(self):
        # A short (<1s) middle segment between two identical neighbours
        # should adopt the neighbours' label, then everything merges.
        segs = [
            {"start": 0.0, "end": 3.0, "text": "a"},
            {"start": 3.0, "end": 3.5, "text": "b"},
            {"start": 3.5, "end": 6.0, "text": "c"},
        ]
        turns = [(0.0, 3.0, "A"), (3.0, 3.5, "B"), (3.5, 6.0, "A")]
        out = assign_speakers_from_turns(segs, turns)
        self.assertEqual(len(out), 1)              # all merged into one A turn
        self.assertEqual({s["speaker"] for s in out}, {"S1"})

    def test_diarization_summary(self):
        segs = [{"speaker": "S1"}, {"speaker": "S2"}, {"speaker": "S1"}]
        summ = diarization_summary(segs)
        self.assertTrue(summ["enabled"])
        self.assertEqual(summ["count"], 2)
        self.assertEqual(summ["speakers"], ["S1", "S2"])


class TestSpeakerCountEstimation(unittest.TestCase):
    def _normed(self, mat):
        mat = np.asarray(mat, dtype=np.float32)
        return mat / np.linalg.norm(mat, axis=1, keepdims=True)

    def test_single_speaker_stays_single(self):
        # 8 near-identical embeddings -> 1 speaker.
        rng = np.random.default_rng(0)
        base = rng.normal(size=(1, 16))
        mat = self._normed(np.repeat(base, 8, axis=0) + rng.normal(scale=0.01, size=(8, 16)))
        k = _estimate_num_speakers(mat, min_k=1, max_k=6)
        self.assertEqual(k, 1)

    def test_two_clear_clusters(self):
        rng = np.random.default_rng(1)
        a = rng.normal(loc=+5, size=(8, 16))
        b = rng.normal(loc=-5, size=(8, 16))
        mat = self._normed(np.vstack([a, b]))
        k = _estimate_num_speakers(mat, min_k=1, max_k=6)
        self.assertEqual(k, 2)

    def test_respects_forced_bounds(self):
        rng = np.random.default_rng(2)
        mat = self._normed(rng.normal(size=(10, 16)))
        self.assertEqual(_estimate_num_speakers(mat, min_k=3, max_k=3), 3)


class TestVttParsing(unittest.TestCase):
    def test_parse_vtt(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Hello there\n\n"
            "00:00:02.000 --> 00:00:04.500\n"
            "General Kenobi\n"
        )
        segs, full = parse_vtt(vtt)
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0]["start"], 0.0)
        self.assertAlmostEqual(segs[1]["end"], 4.5)
        self.assertEqual(full, "Hello there General Kenobi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
