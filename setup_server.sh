#!/bin/bash

# # 1. Install Python 3.12 using pyenv if not available
# if ! command -v python3.12 &> /dev/null; then
#     echo "Python 3.12 not found. Installing via pyenv..."
#     if ! command -v pyenv &> /dev/null; then
#         curl https://pyenv.run | bash
#         export PATH="$HOME/.pyenv/bin:$PATH"
#         eval "$(pyenv init -)"
#         eval "$(pyenv virtualenv-init -)"
#     fi
#     pyenv install 3.12.0
# fi

# # 2. Set Python version
# export PYTHON_VERSION="3.12"
# export PYTHON_BINARY="python$PYTHON_VERSION"

# # 3. Create virtual environment if it doesn't exist
# if [ ! -d ".venv" ]; then
#     echo "Creating virtual environment with Python $PYTHON_VERSION..."
#     $PYTHON_BINARY -m venv .venv
# fi

# # 4. Activate virtual environment
# source .venv/bin/activate

# # 5. Upgrade pip
# pip install --upgrade pip

# # 6. Install dependencies from requirements.txt
# echo "Installing Python dependencies..."
# pip install -r requirements.txt

# 7. Hugging Face Login using token from environment variable
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "❌ HUGGINGFACE_TOKEN environment variable not set. Please export it before running the script."
    exit 1
else
    echo "🔐 Logging into Hugging Face CLI..."
    echo "$HUGGINGFACE_TOKEN" | huggingface-cli login --token
fi


# 9. Run Uvicorn server
echo "🌐 Starting Uvicorn server..."
uvicorn app:app --host 0.0.0.0 --port 8047

echo "Run python src/gradio_ui.py"
echo "✅ Setup complete. Gradio and Uvicorn are now running."