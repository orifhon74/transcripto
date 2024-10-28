# from moviepy.editor import VideoFileClip
# import whisper
# import os
#
# def extract_audio(video_path, audio_path="audio.wav"):
#     """Extracts audio from video file"""
#     video = VideoFileClip(video_path)
#     video.audio.write_audiofile(audio_path, codec='pcm_s16le')
#     return audio_path
#
# def transcribe_audio(audio_path):
#     """Transcribe audio using Whisper"""
#     model = whisper.load_model("base")  # You can choose a larger model if needed (e.g., "small", "medium", etc.)
#     result = model.transcribe(audio_path)
#     return result['text']
#
# def save_transcription(transcription, output_path="transcription.txt"):
#     """Save the transcription in a text file"""
#     with open(output_path, "w") as f:
#         # f.write(transcription)
#         # write the transcription and make new line after a period
#         f.write(transcription.replace('. ', '.\n'))
#
# def main(video_path):
#     # Define paths for audio and output text
#     audio_path = "audio.wav"
#     output_path = f"static/uploads/{os.path.basename(video_path).split('.')[0]}_transcription.txt"
#
#     # Step 1: Extract audio from the video
#     extract_audio(video_path, audio_path)
#
#     # Step 2: Transcribe the extracted audio
#     transcription = transcribe_audio(audio_path)
#
#     # Step 3: Save the transcriptions to a text file
#     save_transcription(transcription, output_path)
#
#     # Clean up intermediate audio file
#     if os.path.exists(audio_path):
#         os.remove(audio_path)
#
#     return output_path

from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip
from pyannote.audio import Pipeline
import whisper
import os


def extract_audio(video_path, audio_path="audio.wav"):
    """Extracts audio from video file"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    return audio_path


def diarize_audio(audio_path):
    """Perform speaker diarization on the audio"""
    # Load pretrained diarization pipeline (ensure you have your Hugging Face access token)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

    # Perform diarization
    diarization_result = pipeline({"uri": "audio", "audio": audio_path})

    # Extract speaker segments
    speaker_segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segment = {
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        }
        speaker_segments.append(segment)

    return speaker_segments


def transcribe_audio(audio_path, segments):
    """Transcribe each segment separately"""
    model = whisper.load_model("base")
    transcription = []

    with AudioFileClip(audio_path) as audio:
        for segment in segments:
            start, end = segment["start"], segment["end"]
            speaker = segment["speaker"]

            # Extract the audio segment
            segment_audio = audio.subclip(start, end)
            segment_audio_path = f"temp_segment_{start:.2f}_{end:.2f}.wav"
            segment_audio.write_audiofile(segment_audio_path, codec='pcm_s16le')

            # Transcribe the segment audio
            result = model.transcribe(segment_audio_path)
            transcription.append((speaker, result['text']))

            # Cleanup temporary file
            if os.path.exists(segment_audio_path):
                os.remove(segment_audio_path)

    return transcription


def transcribe_audio2(audio_path):
    """Transcribe audio using Whisper"""
    model = whisper.load_model("base")  # You can choose a larger model if needed (e.g., "small", "medium", etc.)
    result = model.transcribe(audio_path)
    return result['text']


def save_transcription(transcription, output_path="transcription.txt"):
    """Save the speaker-labeled transcription in a text file"""
    with open(output_path, "w") as f:
        for speaker, text in transcription:
            f.write(f"{speaker}: {text}\n")

def save_transcription2(transcription, output_path="transcription.txt"):
    """Save the transcription in a text file"""
    with open(output_path, "w") as f:
        f.write(transcription.replace('. ', '.\n'))


def main_simple(file_path):
    """Simpler version without differentiating speakers"""
    # Determine if the file is audio or video
    is_audio = file_path.lower().endswith(('.wav', '.mp3', '.flac'))

    # Define paths for audio and output text
    audio_path = file_path if is_audio else "audio.wav"
    output_path = f"static/uploads/{os.path.basename(file_path).split('.')[0]}_simple_transcription.txt"

    # Step 1: Extract audio if it's a video file
    if not is_audio:
        extract_audio(file_path, audio_path)

    # Step 2: Transcribe the extracted audio or directly from audio file
    transcription = transcribe_audio2(audio_path)

    # Step 3: Save the transcriptions to a text file
    save_transcription2(transcription, output_path)

    # Clean up intermediate audio file if extracted from a video
    if os.path.exists(audio_path) and not is_audio:
        os.remove(audio_path)

    return output_path


def main(file_path):
    # Determine if the file is audio or video
    is_audio = file_path.lower().endswith(('.wav', '.mp3', '.flac'))

    # Define paths for audio and output text
    audio_path = file_path if is_audio else "audio.wav"
    output_path = f"static/uploads/{os.path.basename(file_path).split('.')[0]}_transcription.txt"

    # Step 1: Extract audio if it's a video file
    if not is_audio:
        extract_audio(file_path, audio_path)

    # Step 2: Perform diarization on the extracted audio to separate speakers
    speaker_segments = diarize_audio(audio_path)

    # Step 3: Transcribe audio based on speaker segments
    transcription = transcribe_audio(audio_path, speaker_segments)

    # Step 4: Save the transcriptions to a text file
    save_transcription(transcription, output_path)

    # Clean up intermediate audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return output_path