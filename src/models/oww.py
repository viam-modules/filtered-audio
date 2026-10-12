"""OpenWakeWord setup utilities."""

import collections
import importlib.metadata
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

# The openwakeword version whose AudioFeatures internals _AudioTail and the
# methods below match
OWW_AUDIO_TAIL_VERSION = "0.6.0"


class _AudioTail:
    """The last `capacity` samples of 16-bit audio, in a numpy array: stands in
    for openwakeword's raw-audio deque(maxlen=capacity), so the newest samples
    are a slice rather than a copy of the whole deque.

    Samples are appended to a buffer twice the capacity; when that fills, the
    samples still kept move to its start, so a tail is always contiguous.
    """

    def __init__(self, capacity: int) -> None:
        self.maxlen = capacity
        self._buf = np.zeros(2 * capacity, dtype=np.int16)
        self._end = 0
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def clear(self) -> None:
        self._end = 0
        self._len = 0

    def extend(self, samples: np.ndarray) -> None:
        samples = samples[-self.maxlen :]
        n = len(samples)
        if self._end + n > len(self._buf):
            keep = min(self._len, self.maxlen - n)
            self._buf[:keep] = self._buf[self._end - keep : self._end]
            self._end = keep
        self._buf[self._end : self._end + n] = samples
        self._end += n
        self._len = min(self._len + n, self.maxlen)

    def tail(self, n: int) -> np.ndarray:
        """The newest n samples, or all of them if there are fewer."""
        n = min(n, self._len)
        return self._buf[self._end - n : self._end]


def _buffer_raw_data(self: Any, x: np.ndarray) -> None:
    """openwakeword 0.6.0's AudioFeatures._buffer_raw_data for an _AudioTail:
    the original converts every chunk to a Python list for its deque. The
    audio arrives as int16 (oww_check_for_wake_word, or speex noise
    suppression's output), so the buffer stores it unchanged."""
    self.raw_data_buffer.extend(x)


def _streaming_melspectrogram(self: Any, n_samples: int) -> None:
    """openwakeword 0.6.0's AudioFeatures._streaming_melspectrogram for an
    _AudioTail.

    The original copies its whole raw-audio deque (10 s, 160,000 samples) into
    a list on every 80 ms chunk, then keeps the last n_samples + 480. This
    passes those samples as a slice; _get_melspectrogram converts a list to
    the same int16 array, so the features and scores are bit-identical.
    """
    if len(self.raw_data_buffer) < 400:
        raise ValueError(
            "The number of input frames must be at least 400 samples @ 16khz (25 ms)!"
        )
    self.melspectrogram_buffer = np.vstack(
        (
            self.melspectrogram_buffer,
            self._get_melspectrogram(self.raw_data_buffer.tail(n_samples + 160 * 3)),
        )
    )
    if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
        self.melspectrogram_buffer = self.melspectrogram_buffer[
            -self.melspectrogram_max_len :, :
        ]


def _use_audio_tail(preprocessor: Any, logger: logging.Logger) -> None:
    """Give an openwakeword 0.6.0 AudioFeatures an _AudioTail for its raw
    audio deque, with the methods that read and write it; other versions keep
    their own buffer."""
    try:
        version = importlib.metadata.version("openwakeword")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != OWW_AUDIO_TAIL_VERSION or not isinstance(
        preprocessor.raw_data_buffer, collections.deque
    ):
        logger.debug(f"openwakeword {version}: keeping its own raw-audio buffer")
        return
    audio = _AudioTail(preprocessor.raw_data_buffer.maxlen)
    audio.extend(np.array(preprocessor.raw_data_buffer, dtype=np.int16))
    preprocessor.raw_data_buffer = audio
    preprocessor._buffer_raw_data = types.MethodType(_buffer_raw_data, preprocessor)
    preprocessor._streaming_melspectrogram = types.MethodType(
        _streaming_melspectrogram, preprocessor
    )


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
    _use_audio_tail(instance.oww_model.preprocessor, instance.logger)
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
