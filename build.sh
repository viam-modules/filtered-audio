#!/bin/sh
set -e
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

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

