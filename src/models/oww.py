"""OpenWakeWord setup utilities."""

import logging
import os
import pathlib
import ssl
import sys
import types
import urllib.error
import urllib.request
from typing import Any

import certifi


from .download import download_file


def _ensure_preprocessing_models(
    openwakeword: types.ModuleType, models_dir: str, logger: logging.Logger
) -> None:
    os.makedirs(models_dir, exist_ok=True)
    for m in list(openwakeword.FEATURE_MODELS.values()) + list(
        openwakeword.VAD_MODELS.values()
    ):
        url = m["download_url"].replace(".tflite", ".onnx")
        fname = url.split("/")[-1]
        if not os.path.exists(os.path.join(models_dir, fname)):
            logger.info(f"Downloading OWW preprocessing model: {fname}")
            openwakeword.utils.download_file(url, models_dir)


def _resolve_oww_model_path(url: str, logger: logging.Logger) -> str:
    filename = url.split("/")[-1]
    cache_dir = os.getenv("VIAM_MODULE_DATA")
    if cache_dir is None:
        raise RuntimeError("VIAM_MODULE_DATA environment variable is not set")

    cached_path = os.path.join(cache_dir, filename)

    if os.path.exists(cached_path):
        logger.debug(f"Using cached OWW model: {cached_path}")
        return cached_path

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=10):
            pass
    except urllib.error.HTTPError as e:
        raise ValueError(f"oww_model_path URL returned {e.code}: {url}") from e

    download_file(url, cached_path, logger)
    return cached_path


def setup_oww(instance: Any, oww_model_path: str, oww_threshold: float) -> None:
    """
    Set up OpenWakeWord detection engine on a WakeWordFilter instance.

    Handles lazy importing OWW, downloading preprocessing models,
    resolving the user's wakeword model path (local or URL), and
    creating the OWW Model object.
    """
    instance.logger.info("Importing openwakeword runtimes")
    import openwakeword
    from openwakeword.model import Model as OWWModel

    models_dir = os.path.join(
        pathlib.Path(openwakeword.__file__).parent.resolve(),
        "resources",
        "models",
    )
    instance.logger.debug("Checking for OWW preprocessing models...")
    _ensure_preprocessing_models(openwakeword, models_dir, instance.logger)

    oww_model_path = os.path.expanduser(oww_model_path)

    if oww_model_path.startswith("http://") or oww_model_path.startswith("https://"):
        oww_model_path = _resolve_oww_model_path(oww_model_path, instance.logger)
    elif not os.path.exists(oww_model_path):
        raise ValueError(f"oww_model_path does not exist: {oww_model_path}")

    instance.oww_threshold = oww_threshold

    instance.logger.debug("Loading OWW model...")
    instance.oww_model = OWWModel(
        wakeword_models=[oww_model_path],
        inference_framework="onnx",
        enable_speex_noise_suppression=(
            sys.platform == "linux"
        ),  # not supported on mac
    )
    instance.logger.debug("OWW model loaded")

    instance.oww_model_name = os.path.splitext(os.path.basename(oww_model_path))[0]
    instance.logger.info(
        f"openWakeWord model loaded (threshold={instance.oww_threshold})"
    )
