#!/bin/sh
set -e
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
tar -czf dist/archive.tar.gz -C dist main -C .. meta.json
