#!/bin/sh

set -eu

cd `dirname $0`

# Install unzip if not present (needed for Vosk model extraction)
if ! command -v unzip >/dev/null; then
    echo "Installing unzip..."
    if command -v apt-get >/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y unzip curl
    fi
fi

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

# Try to create virtualenv
if ! python3 -m venv $VENV_NAME 2>&1; then
    echo "Failed to create virtualenv. Installing python3, python3-pip, and python3-venv..."
    if command -v apt-get >/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3.11 python3.11-venv python3-pip
        if ! python3.11 -m venv $VENV_NAME 2>&1; then
            echo "Failed to create virtualenv even after installing dependencies" >&2
            exit 1
        fi
    else
        echo "This module requires Python >=3.10, pip, and virtualenv to be installed." >&2
        exit 1
    fi
fi

# Source uses match/case (PEP 634), which requires Python >=3.10.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    PY_VER=`python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])'`
    echo "This module requires Python >=3.10, but python3 is $PY_VER." >&2
    echo "On Debian/Ubuntu: sudo apt-get install python3.11 python3.11-venv" >&2
    exit 1
fi

# remove -U if viam-sdk should not be upgraded whenever possible
# -qq suppresses extraneous output from pip
echo "Virtualenv found/created. Installing/upgrading Python packages..."
if ! [ -f .installed ]; then
    # Install openwakeword without deps to skip tflite-runtime (we use ONNX only)
    if ! $PYTHON -m pip install "openwakeword==0.6.0" --no-deps -Uqq; then
        echo "Failed to install openwakeword" >&2
        exit 1
    fi
    if ! $PYTHON -m pip install -r requirements.txt -Uqq; then
        echo "Failed to install Python packages" >&2
        exit 1
    else
        touch .installed
    fi
fi
