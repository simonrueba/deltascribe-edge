"""Medical speech recognition transcription module.

Implements transcription with fallback chain:
1. MedASR (Google's HuggingFace medical ASR) - if available
2. Whisper (OpenAI's general ASR) - local fallback

Raises RuntimeError if no ASR engine is available.
"""

import asyncio
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Global model instances (lazy loaded)
_whisper_model = None
_medasr_model = None
_medasr_processor = None
_medasr_load_attempted = False


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    confidence: float
    timestamps: list[dict]
    source: str  # "medasr" or "whisper"


async def transcribe_audio(
    audio_base64: str,
    language: str = "en",
) -> TranscriptionResult:
    """
    Transcribe medical audio using available ASR engine.

    Args:
        audio_base64: Base64-encoded audio data (webm, mp4, or wav)
        language: Language code for transcription

    Returns:
        TranscriptionResult with text, confidence, and timestamps
    """
    logger.info("Starting audio transcription")

    # Decode base64 audio
    audio_bytes = base64.b64decode(audio_base64)

    # Try transcription methods in order of preference

    # 1. Try MedASR (Google's medical ASR)
    result = await _try_medasr(audio_bytes, language)
    if result:
        return result

    # 2. Try Whisper (local fallback)
    result = await _try_whisper(audio_bytes, language)
    if result:
        return result

    # 3. No ASR engine available
    raise RuntimeError(
        "No ASR engine available. Install MedASR or Whisper for transcription."
    )


async def _load_medasr():
    """Load MedASR model from HuggingFace."""
    global _medasr_model, _medasr_processor, _medasr_load_attempted

    if _medasr_load_attempted:
        return _medasr_model is not None

    _medasr_load_attempted = True
    logger.info("Loading MedASR model from HuggingFace")

    try:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        from app.core.config import settings

        model_id = "google/medasr"

        # Detect device
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        def _load():
            processor = AutoProcessor.from_pretrained(
                model_id,
                token=settings.hf_token,
                trust_remote_code=True,
            )
            model = AutoModelForCTC.from_pretrained(
                model_id,
                token=settings.hf_token,
                trust_remote_code=True,
            )
            return processor, model, device

        _medasr_processor, _medasr_model, dev = await asyncio.to_thread(_load)
        _medasr_model = _medasr_model.to(dev)
        _medasr_model.eval()

        logger.info("MedASR model loaded", device=dev)
        return True

    except Exception as e:
        logger.warning("Failed to load MedASR model", error=str(e))
        import traceback
        traceback.print_exc()
        return False


async def _try_medasr(audio_bytes: bytes, language: str) -> TranscriptionResult | None:
    """
    Attempt transcription using MedASR from HuggingFace.

    MedASR is Google's medical speech recognition model (Conformer architecture)
    optimized for clinical terminology and dictation patterns.
    """
    global _medasr_model, _medasr_processor

    try:
        import librosa
        import torch

        # Load model if not already loaded
        if _medasr_model is None:
            loaded = await _load_medasr()
            if not loaded:
                return None

        logger.info("Attempting MedASR transcription")

        # Write audio to temp file for librosa
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        try:
            # Load and resample audio to 16kHz (required by MedASR)
            speech, sample_rate = await asyncio.to_thread(
                librosa.load, tmp_in_path, sr=16000
            )

            # Extract features manually to work around a bug in
            # LasrFeatureExtractor.__call__ (passes extra args to
            # _torch_extract_fbank_features which only accepts waveform+device).
            device = next(_medasr_model.parameters()).device
            waveform = torch.tensor(speech, dtype=torch.float32).unsqueeze(0)
            fe = _medasr_processor.feature_extractor
            fbanks = fe._torch_extract_fbank_features(waveform, device=str(device))
            # fbanks shape: (1, time_steps, n_mels) — model expects float32
            inputs = {"input_features": fbanks.to(device=device, dtype=torch.float32)}

            # CTC forward pass + greedy decode with blank collapsing
            with torch.no_grad():
                logits = _medasr_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # CTC decoding: collapse repeated tokens, then remove blanks
            # The blank token (CTC epsilon) is token id 0
            blank_id = _medasr_processor.tokenizer.pad_token_id or 0
            collapsed = []
            prev = -1
            for token_id in predicted_ids[0].tolist():
                if token_id != prev and token_id != blank_id:
                    collapsed.append(token_id)
                prev = token_id

            decoded_text = _medasr_processor.tokenizer.decode(
                collapsed, skip_special_tokens=True
            )

            # MedASR doesn't provide word-level timestamps by default
            # Generate approximate timestamps based on text length
            words = decoded_text.split()
            audio_duration = len(speech) / sample_rate
            timestamps = []
            if words:
                word_duration = audio_duration / len(words)
                for i, word in enumerate(words):
                    timestamps.append({
                        "start": i * word_duration,
                        "end": (i + 1) * word_duration,
                        "text": word,
                    })

            logger.info("MedASR transcription successful", text_length=len(decoded_text))
            return TranscriptionResult(
                text=decoded_text.strip(),
                confidence=0.95,  # MedASR has high confidence for medical text
                timestamps=timestamps,
                source="medasr",
            )

        finally:
            Path(tmp_in_path).unlink(missing_ok=True)

    except ImportError as e:
        logger.debug("MedASR dependencies not available", error=str(e))
        return None
    except Exception as e:
        logger.warning("MedASR transcription failed", error=str(e))
        import traceback
        traceback.print_exc()
        return None


async def _try_whisper(audio_bytes: bytes, language: str) -> TranscriptionResult | None:
    """
    Attempt transcription using OpenAI Whisper.

    Whisper provides good general-purpose transcription and can handle
    medical terminology reasonably well.
    """
    global _whisper_model

    try:
        import whisper
        from pydub import AudioSegment

        logger.info("Attempting Whisper transcription")

        # Load model (lazy)
        if _whisper_model is None:
            logger.info("Loading Whisper model (base)")
            _whisper_model = whisper.load_model("base")

        # Convert audio bytes to wav format for Whisper
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        try:
            # Convert to wav using pydub
            audio = AudioSegment.from_file(tmp_in_path)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                audio.export(tmp_out.name, format="wav")
                tmp_out_path = tmp_out.name

            # Transcribe with Whisper
            result = _whisper_model.transcribe(
                tmp_out_path,
                language=language if language != "en" else None,
                word_timestamps=True,
            )

            # Extract timestamps from segments
            timestamps = []
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    timestamps.append({
                        "start": word["start"],
                        "end": word["end"],
                        "text": word["word"],
                    })

            # Calculate average confidence from segments
            confidences = [s.get("no_speech_prob", 0) for s in result.get("segments", [])]
            avg_confidence = 1.0 - (sum(confidences) / len(confidences)) if confidences else 0.8

            logger.info("Whisper transcription successful")
            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=avg_confidence,
                timestamps=timestamps,
                source="whisper",
            )

        finally:
            # Cleanup temp files
            Path(tmp_in_path).unlink(missing_ok=True)
            if 'tmp_out_path' in locals():
                Path(tmp_out_path).unlink(missing_ok=True)

    except ImportError as e:
        logger.debug("Whisper not available", error=str(e))
        return None
    except Exception as e:
        logger.warning("Whisper transcription failed", error=str(e))
        return None


async def get_transcription_status() -> dict:
    """Check which transcription engines are available."""
    global _medasr_model

    status = {
        "medasr": False,
        "medasr_loaded": _medasr_model is not None,
        "whisper": False,
    }

    # Check MedASR (HuggingFace transformers)
    try:
        import importlib.util
        status["medasr"] = importlib.util.find_spec("transformers") is not None
    except ImportError:
        pass

    # Check Whisper
    import importlib.util
    status["whisper"] = importlib.util.find_spec("whisper") is not None

    return status
