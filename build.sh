#!/bin/sh
set -eu
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

# Download default Vosk model if not present
VOSK_MODEL="vosk-model-small-en-us-0.15"
VOSK_MODELS_DIR="vosk_models"
if [ ! -d "$VOSK_MODELS_DIR/$VOSK_MODEL" ]; then
    echo "Downloading default Vosk model..."
    mkdir -p "$VOSK_MODELS_DIR"
    curl -L -o "/tmp/$VOSK_MODEL.zip" "https://alphacephei.com/vosk/models/$VOSK_MODEL.zip"
    unzip -q "/tmp/$VOSK_MODEL.zip" -d "$VOSK_MODELS_DIR"
    rm "/tmp/$VOSK_MODEL.zip"
    echo "Vosk model downloaded to $VOSK_MODELS_DIR/$VOSK_MODEL"
else
    echo "Vosk model already exists at $VOSK_MODELS_DIR/$VOSK_MODEL"
fi

# Download only openwakeword preprocessing models (not bundled wake words like alexa/jarvis)
echo "Downloading openwakeword preprocessing models..."
$PYTHON -c "
import os, openwakeword, openwakeword.utils
models_dir = os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models')
os.makedirs(models_dir, exist_ok=True)
for m in openwakeword.FEATURE_MODELS.values():
    url = m['download_url']
    fname = url.split('/')[-1]
    if not os.path.exists(os.path.join(models_dir, fname)):
        openwakeword.utils.download_file(url, models_dir)
    onnx_url = url.replace('.tflite', '.onnx')
    onnx_fname = fname.replace('.tflite', '.onnx')
    if not os.path.exists(os.path.join(models_dir, onnx_fname)):
        openwakeword.utils.download_file(onnx_url, models_dir)
for m in openwakeword.VAD_MODELS.values():
    fname = m['download_url'].split('/')[-1]
    if not os.path.exists(os.path.join(models_dir, fname)):
        openwakeword.utils.download_file(m['download_url'], models_dir)
"

# Install pyinstaller
echo "Installing PyInstaller..."
if ! $PYTHON -m pip install pyinstaller -Uqq; then
    exit 1
fi

# Clean previous builds
rm -rf build dist

# Run PyInstaller using the spec file
echo "Running PyInstaller..."
$PYTHON -m PyInstaller main.spec

# Create archive with meta.json
echo "Creating archive..."
tar -czf dist/archive.tar.gz dist/main meta.json vosk_models
