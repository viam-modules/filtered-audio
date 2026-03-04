"""Vosk model utilities."""

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
import zipfile

from vosk import Model as VoskModel, KaldiRecognizer
from viam.components.audio_in import AudioResponse as AudioChunk

from .download import download_file
from .fuzzy_matcher import FuzzyWakeWordMatcher


BASE_VOSK_URL = "https://alphacephei.com/vosk/models"
DEFAULT_VOSK_MODEL = "vosk-model-small-en-us-0.15"


def _get_bundled_models_dir() -> str:
    """Get the path to bundled models, handling PyInstaller bundles."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        # Running in development with run.sh
        base_path = Path(__file__).resolve().parents[2]
    return os.path.join(base_path, "vosk_models")


def _extract_vosk_model(zip_path, logger) -> str:
    """
    Extract a Vosk model from a zip file to VIAM_MODULE_DATA.

    Args:
        zip_path: Path to the zip file
        logger: logger instance

    Returns:
        Path to the extracted model directory
    """
    data_path = os.getenv("VIAM_MODULE_DATA")
    if not data_path:
        raise RuntimeError("VIAM_MODULE_DATA environment variable not set")

    # Model name is the zip filename without .zip
    model_name = os.path.basename(zip_path).removesuffix(".zip")
    model_dir = os.path.join(data_path, model_name)

    # Skip extraction if already extracted
    if os.path.exists(model_dir):
        logger.info(f"Model already extracted at {model_dir}")
        return model_dir

    logger.info(f"Extracting model from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
    except Exception as e:
        logger.error(f"Failed to extract Vosk model: {e}")
        raise RuntimeError(f"Failed to extract Vosk model: {e}")

    logger.info(f"Model extracted to {model_dir}")
    return model_dir


def _resolve_absolute_path(path, logger) -> str:
    """Resolve an absolute path to a Vosk model."""
    if not os.path.exists(path):
        raise RuntimeError(f"Vosk model path does not exist: {path}")

    # If it's a zip file, extract it
    if path.endswith(".zip"):
        return _extract_vosk_model(path, logger)

    logger.info(f"Using vosk model at path: {path}")
    return path


def _resolve_model_name(model_name, logger) -> str:
    """Resolve a model name by checking bundled, cached, then downloading."""
    # Check for bundled model (shipped with module)
    bundled_dir = _get_bundled_models_dir()
    bundled_path = os.path.join(bundled_dir, model_name)
    if os.path.exists(bundled_path):
        logger.info("Found bundled vosk model")
        return bundled_path

    # Check for cached model in VIAM_MODULE_DATA
    data_path = os.getenv("VIAM_MODULE_DATA")
    if data_path:
        cached_path = os.path.join(data_path, model_name)
        if os.path.exists(cached_path):
            logger.info(f"Found cached model at {cached_path}")
            return cached_path

    # Download the model
    logger.info("Model not found locally, downloading...")
    return _download_vosk_model(model_name, logger)


def get_vosk_model(model_name_or_path, logger) -> str:
    """
    Get Vosk model path.

    If model_name_or_path is an absolute path, use it directly.
    Otherwise, check bundled models, then cached, then download.

    Returns:
        Path to the model directory
    """
    logger.info(f"Loading vosk model '{model_name_or_path}'...")

    # Expand ~ to home directory
    model_name_or_path = os.path.expanduser(model_name_or_path)

    if os.path.isabs(model_name_or_path):
        return _resolve_absolute_path(model_name_or_path, logger)

    return _resolve_model_name(model_name_or_path, logger)


def _download_vosk_model(model_name, logger) -> str:
    """
    Download Vosk model from the internet.

    Args:
        model_name: Name of the Vosk model to download
        logger: logger instance

    Returns:
        Path to the model directory
    """
    data_path = os.getenv("VIAM_MODULE_DATA")
    if not data_path:
        raise RuntimeError("VIAM_MODULE_DATA environment variable not set")

    model_dir = os.path.join(data_path, model_name)

    if os.path.exists(model_dir):
        logger.info(f"Vosk model already exists at {model_dir}")
        return model_dir

    url = f"{BASE_VOSK_URL}/{model_name}.zip"
    zip_path = (Path(tempfile.gettempdir()) / f"{model_name}.zip").resolve()

    try:
        download_file(url, str(zip_path), logger)
    except Exception as e:
        logger.error(f"Failed to download Vosk model: {e}")
        raise RuntimeError(f"Failed to download Vosk model: {e}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
    except Exception as e:
        logger.error(f"Failed to extract Vosk model: {e}")
        raise RuntimeError(f"Failed to extract Vosk model: {e}")
    finally:
        # clean up zip file even if extraction fails
        if zip_path.exists():
            zip_path.unlink()

    logger.info("Vosk model downloaded and extracted successfully")
    return model_dir


AUDIO_SAMPLE_RATE_HZ = 16000
DEFAULT_GRAMMAR_CONFIDENCE = 0.7


def setup_vosk(
    instance: Any,
    vosk_model: str = DEFAULT_VOSK_MODEL,
    use_grammar: bool = True,
    grammar_confidence: float = DEFAULT_GRAMMAR_CONFIDENCE,
    fuzzy_threshold: Optional[int] = None,
) -> None:
    """
    Set up Vosk detection engine on a WakeWordFilter instance.

    Loads the Vosk model, creates a KaldiRecognizer (optionally with
    grammar), and configures fuzzy matching if requested.
    """
    # Load Vosk model (checks bundled, then cached, then downloads)
    model_path = get_vosk_model(vosk_model, instance.logger)
    instance.vosk_model = VoskModel(model_path)
    instance.logger.debug("Vosk model loaded")

    instance.use_grammar = use_grammar
    instance.grammar_confidence = grammar_confidence
    instance.logger.info(f"Vosk grammar mode: {instance.use_grammar}")
    instance.logger.info(
        f"Vosk grammar confidence threshold: {instance.grammar_confidence:.2f}"
    )

    # Create recognizer
    if instance.use_grammar and instance.wake_words:
        grammar = json.dumps(instance.wake_words)
        instance.recognizer = KaldiRecognizer(
            instance.vosk_model, AUDIO_SAMPLE_RATE_HZ, grammar
        )
    else:
        instance.recognizer = KaldiRecognizer(instance.vosk_model, AUDIO_SAMPLE_RATE_HZ)

    instance.recognizer.SetWords(True)  # Enable word-level confidence scores
    instance.logger.debug("Vosk recognizer initialized")

    # Fuzzy matching - enabled if fuzzy_threshold is set
    if fuzzy_threshold is not None:
        instance.fuzzy_matcher = FuzzyWakeWordMatcher(threshold=fuzzy_threshold)
        instance.logger.info(f"Fuzzy matching enabled with threshold={fuzzy_threshold}")
    else:
        instance.fuzzy_matcher = None


def vosk_check_for_wake_word(instance: Any, audio_bytes: bytes) -> bool:
    """Check if any wake word is present in the audio using Vosk."""
    try:
        instance.recognizer.AcceptWaveform(audio_bytes)
        result = json.loads(instance.recognizer.FinalResult())
        text = result.get("text", "").lower()

        instance.logger.debug("Vosk result: %s", result)
        if "result" in result and result["result"]:
            avg_conf = sum(w.get("conf", 1.0) for w in result["result"])
            avg_conf /= len(result["result"])
            instance.logger.debug("Vosk confidence: %.2f", avg_conf)
            if avg_conf < instance.grammar_confidence:
                instance.logger.debug(
                    "Rejecting low confidence: '%s' (conf=%.2f < %.2f)",
                    text,
                    avg_conf,
                    instance.grammar_confidence,
                )
                return False

        if text:
            instance.logger.debug(f"Recognized: '{text}'")
        else:
            instance.logger.debug("Vosk returned empty text, no speech recognized")
            return False

        for wake_word in instance.wake_words:
            if instance.fuzzy_matcher and not instance.use_grammar:
                match_details = instance.fuzzy_matcher.match(text, wake_word)
                if match_details:
                    instance.logger.info(
                        f"Wake word '{wake_word}' detected (fuzzy: "
                        f"'{match_details['matched_text']}', "
                        f"distance={match_details['distance']})"
                    )
                    return True
            else:
                pattern = rf"\b{re.escape(wake_word)}\b"
                match = re.search(pattern, text)
                instance.logger.debug(
                    "Checking wake_word='%s' pattern='%s' against text='%s' -> %s",
                    wake_word,
                    pattern,
                    text,
                    match,
                )
                if match:
                    instance.logger.info("Wake word '%s' detected", wake_word)
                    return True

        instance.logger.debug("No wake word match found")
        return False
    except Exception as e:
        instance.logger.error(f"Vosk error: {e}", exc_info=True)
        return False


async def vosk_process_segment(
    instance: Any, speech_chunk_buffer: list, speech_buffer: bytearray
) -> AsyncGenerator:
    """Run Vosk wake word inference on a complete buffered speech segment."""
    if not speech_chunk_buffer:
        return

    if instance.is_shutting_down:
        instance.logger.debug("Skipping speech processing due to shutdown")
        return

    try:
        wake_word_detected = await asyncio.get_running_loop().run_in_executor(
            instance.executor,
            vosk_check_for_wake_word,
            instance,
            bytes(speech_buffer),
        )

        if wake_word_detected:
            instance.logger.info(
                f"Wake word detected! Yielding {len(speech_chunk_buffer)} chunks ({len(speech_buffer)} bytes)"
            )
            for chunk in speech_chunk_buffer:
                yield chunk
            empty_response = AudioChunk()
            empty_response.audio.audio_data = b""
            yield empty_response
            instance.logger.debug("Sent empty chunk to signal segment end")
    except RuntimeError as e:
        if "shutdown" in str(e).lower():
            instance.logger.debug("Executor shutdown during processing, ignoring")
            return
        raise
