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
import numpy as np

from .download import download_file

# OWW inference chunks: 16kHz * 0.080s = 1280 samples * 2 bytes = 2560 bytes
OWW_CHUNK_SIZE = 2560


def _ensure_preprocessing_models(
    openwakeword: types.ModuleType, models_dir: str, logger: logging.Logger
) -> None:
    os.makedirs(models_dir, exist_ok=True)
    for m in list(openwakeword.FEATURE_MODELS.values()) + list(
        openwakeword.VAD_MODELS.values()
    ):
        tflite_url = m["download_url"]
        onnx_url = tflite_url.replace(".tflite", ".onnx")
        for url in [tflite_url, onnx_url]:
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

    _, extension = os.path.splitext(oww_model_path)

    if extension not in (".onnx", ".tflite"):
        raise ValueError(
            f"oww_model_path file extension must be .onnx or .tflite, got {extension}"
        )
    if extension == ".tflite" and sys.platform != "linux":
        raise ValueError(
            "tflite models are only supported on Linux. "
            "Please use an .onnx model instead."
        )
    instance.oww_model = OWWModel(
        wakeword_models=[oww_model_path],
        inference_framework=extension.lstrip("."),
        enable_speex_noise_suppression=(
            sys.platform == "linux"
        ),  # not supported on mac
    )
    instance.logger.debug("OWW model loaded")

    instance.oww_model_name = os.path.splitext(os.path.basename(oww_model_path))[0]
    instance.logger.info(
        f"openWakeWord model loaded (threshold={instance.oww_threshold})"
    )


def oww_check_for_wake_word(instance: Any, oww_audio_buffer: bytearray) -> bool:
    """Drain oww_audio_buffer in chunks and run OWW inference on each.

    Inference runs synchronously on the event loop thread. OWW inference on CPU is ~5ms
    per 80ms chunk on Pi, so this is fine. If heavier models are needed,
    offload to a ThreadPoolExecutor.
    """
    while len(oww_audio_buffer) >= OWW_CHUNK_SIZE:
        oww_chunk = bytes(oww_audio_buffer[:OWW_CHUNK_SIZE])
        del oww_audio_buffer[:OWW_CHUNK_SIZE]
        audio_int16 = np.frombuffer(oww_chunk, dtype=np.int16)
        prediction = instance.oww_model.predict(audio_int16)
        score = prediction.get(instance.oww_model_name, 0.0)
        if score >= instance.oww_threshold:
            instance.logger.info(
                "Wake word detected (score=%.3f >= %.3f)",
                score,
                instance.oww_threshold,
            )
            return True
    return False
