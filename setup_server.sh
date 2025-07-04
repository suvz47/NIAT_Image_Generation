#!/bin/bash

# 1. Install Python 3.13 using pyenv if not available
if ! command -v python3.13 &> /dev/null; then
    echo "Python 3.13 not found. Installing via pyenv..."
    if ! command -v pyenv &> /dev/null; then
        curl https://pyenv.run | bash
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    fi
    pyenv install 3.13.0
fi

# 2. Set Python version
export PYTHON_VERSION="3.13"
export PYTHON_BINARY="python$PYTHON_VERSION"

# 3. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python $PYTHON_VERSION..."
    $PYTHON_BINARY -m venv .venv
fi

# 4. Activate virtual environment
source .venv/bin/activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

# 7. Install llama-cpp-python with CUDA/cuBLAS support
echo "Installing llama-cpp-python with CUDA/cuBLAS support..."
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-binary :all:

# 8. Download GGUF model
MODEL_DIR="models/prompt_engineering"
mkdir -p "$MODEL_DIR"
echo "Downloading Qwen2.5-Coder-7B-Instruct GGUF model..."
wget -O "$MODEL_DIR/qwen2.5-coder-7b-instruct-q5_k_m-00001-of-00002.gguf" \
     "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q5_k_m-00001-of-00002.gguf?download=true"
wget -O "$MODEL_DIR/qwen2.5-coder-7b-instruct-q5_k_m-00002-of-00002.gguf" \
     "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q5_k_m-00002-of-00002.gguf?download=true"

# 9. Clean up filenames (remove `?download=true` if it somehow sneaks in)
for file in "$MODEL_DIR"/*.gguf\?download=true; do
    if [ -e "$file" ]; then
        mv "$file" "${file%\?download=true}"
    fi
done

echo "✅ Setup complete. To activate the environment, run: source .venv/bin/activate"