"""Vosk model utilities."""

import os
import sys
import tempfile
from pathlib import Path
import ssl
import urllib.request
import zipfile
import certifi


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
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    logger.debug(f"Downloading Vosk model from {url}...")

    try:
        with urllib.request.urlopen(url, context=ssl_context, timeout=120) as response:
            with open(zip_path, "wb") as out_file:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    out_file.write(chunk)
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
