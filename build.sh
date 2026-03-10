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

# Download only openwakeword preprocessing .onnx models (not bundled wake words)
$PYTHON -c "
import os, openwakeword, openwakeword.utils
d = os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models')
os.makedirs(d, exist_ok=True)
for m in list(openwakeword.FEATURE_MODELS.values()) + list(openwakeword.VAD_MODELS.values()):
    tflite_url = m['download_url']
    onnx_url = m['download_url'].replace('.tflite', '.onnx')
    urls = [onnx_url, tflite_url]
    for url in urls:
        f = os.path.join(d, url.split('/')[-1])
        if not os.path.exists(f):
            openwakeword.utils.download_file(url, d)
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
