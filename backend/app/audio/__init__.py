"""Audio processing module for medical speech recognition."""

from app.audio.transcriber import TranscriptionResult, transcribe_audio

__all__ = ["transcribe_audio", "TranscriptionResult"]
