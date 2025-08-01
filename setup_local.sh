#!/bin/bash

set -e  # Exit immediately on error

# Step 0: Setup virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    python3.12 -m venv .venv
else
    echo ".venv already exists. Skipping creation."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Load Hugging Face token from .env
echo "Loading .env..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found. Aborting."
    exit 1
fi

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "HUGGINGFACE_TOKEN not found in .env. Aborting."
    exit 1
fi

# Step 1: Install Python dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
pip install llama-cpp-python langchain_core

# Step 2: Upgrade mflux via uv
echo "Upgrading mflux using uv..."
uv tool install --upgrade mflux

# Step 3: Install project in editable mode with nightly PyTorch
echo "Installing project in editable mode with PyTorch nightly..."
uv pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly -e .

# Step 4: Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Step 5: Log in to Hugging Face using token
echo "Logging in to Hugging Face with token..."
huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential


cd ../../  # Return to project root
echo "✅ Setup complete."