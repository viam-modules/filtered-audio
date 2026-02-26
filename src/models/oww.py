"""OpenWakeWord setup utilities."""

import os
import pathlib
import sys
import tempfile

from .download import download_file


def setup_oww(instance, attrs):
    """
    Set up OpenWakeWord detection engine on a WakeWordFilter instance.

    Handles lazy importing OWW, downloading preprocessing models,
    resolving the user's wakeword model path (local or URL), and
    creating the OWW Model object.
    """
    # Lazy import OWW since it's large and import is slow.
    instance.logger.info("Importing openwakeword runtimes")
    import openwakeword
    from openwakeword.model import Model as OWWModel

    instance.logger.debug("Checking for OWW preprocessing models...")
    # Ensure preprocessing models exist (bundled by PyInstaller, or download as fallback)
    models_dir = os.path.join(
        pathlib.Path(openwakeword.__file__).parent.resolve(),
        "resources",
        "models",
    )
    os.makedirs(models_dir, exist_ok=True)
    for m in list(openwakeword.FEATURE_MODELS.values()) + list(
        openwakeword.VAD_MODELS.values()
    ):
        url = m["download_url"]
        fname = url.split("/")[-1]
        if not os.path.exists(os.path.join(models_dir, fname)):
            instance.logger.info(f"Downloading OWW model: {fname}")
            openwakeword.utils.download_file(url, models_dir)

    oww_model_path = os.path.expanduser(str(attrs.get("oww_model_path", "")))

    # Get user's wakeword model
    if oww_model_path.startswith("http://") or oww_model_path.startswith("https://"):
        filename = oww_model_path.split("/")[-1]
        cache_dir = os.getenv("VIAM_MODULE_DATA", tempfile.gettempdir())
        cached_path = os.path.join(cache_dir, filename)

        if os.path.exists(cached_path):
            instance.logger.info(f"Using cached OWW model: {cached_path}")
        else:
            download_file(oww_model_path, cached_path, instance.logger)

        oww_model_path = cached_path

    if not oww_model_path.startswith(("http://", "https://")) and not os.path.exists(
        oww_model_path
    ):
        raise ValueError(f"oww_model_path does not exist: {oww_model_path}")

    instance.oww_threshold = float(attrs.get("oww_threshold", 0.5))

    instance.logger.debug("Loading OWW model...")
    instance.oww_model = OWWModel(
        wakeword_models=[oww_model_path],
        inference_framework="tflite",
        enable_speex_noise_suppression=(sys.platform == "linux"),
    )
    instance.logger.debug("OWW model loaded")

    # Derive model name the same way openwakeword does internally:
    # basename without extension
    instance.oww_model_name = os.path.splitext(os.path.basename(oww_model_path))[0]

    instance.logger.info(
        f"openWakeWord model loaded (threshold={instance.oww_threshold})"
    )
