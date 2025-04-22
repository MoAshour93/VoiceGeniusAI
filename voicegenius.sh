#!/bin/bash

# Title
echo "================================================================="
echo "🎙️ VoiceGenius AI - Speech-to-Text App"
echo "================================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Define the Python environment name
ENV_NAME="SpeechToText1"

# Parse command line options
MODE="run"
if [ "$1" == "--fix-dependencies" ] || [ "$1" == "-f" ]; then
    MODE="fix"
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./voicegenius.sh [OPTION]"
    echo
    echo "Options:"
    echo "  -f, --fix-dependencies   Fix OpenAI integration dependencies"
    echo "  -h, --help               Display this help message"
    echo "  (no option)              Run the application normally"
    echo
    exit 0
fi

# Check if the environment exists
if [ ! -d "$SCRIPT_DIR/$ENV_NAME" ]; then
    echo "Creating Python environment '$ENV_NAME'..."
    python3 -m venv "$ENV_NAME"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create Python environment."
        echo "Make sure Python 3 is installed and try again."
        exit 1
    fi
    
    echo "Environment created successfully."
else
    echo "Found existing Python environment."
fi

# Activate the environment
source "$SCRIPT_DIR/$ENV_NAME/bin/activate"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Python environment."
    exit 1
fi

# If fix mode is selected, fix the dependencies
if [ "$MODE" == "fix" ]; then
    echo "================================================================="
    echo "Fixing dependencies for VoiceGenius AI OpenAI integration"
    echo "================================================================="
    
    echo "Uninstalling problematic packages..."
    pip uninstall -y langchain-openai langchain-anthropic pydantic langchain openai anthropic
    
    echo "Installing fresh versions of core packages..."
    pip install pydantic==2.0.3
    pip install langchain==0.0.267
    
    echo "Installing direct API packages instead of LangChain integrations..."
    pip install openai==1.1.1
    pip install anthropic==0.5.0
    
    echo "================================================================="
    echo "Dependencies fixed! Run this script again without options to start the app."
    echo "================================================================="
    exit 0
fi

# Normal run mode
echo "Installing/updating required packages..."
pip install --upgrade pip wheel setuptools

# Clear pip cache to avoid using cached problematic versions
pip cache purge

# Try to install with binary preference first
echo "Attempting to install packages with binary preference..."
pip install --prefer-binary -r "$SCRIPT_DIR/requirements.txt" || {
    echo "Some packages failed to install. Trying an alternative approach..."
    
    # Try installing packages one by one
    echo "Installing base packages..."
    pip install --prefer-binary streamlit==1.27.2 moviepy==1.0.3 python-dotenv==1.0.0 pydub==0.25.1 requests uuid
    
    echo "Installing PyTorch (this may take a while)..."
    pip install --prefer-binary torch>=2.6.0 torchaudio>=2.6.0
    
    echo "Installing ML libraries..."
    pip install --prefer-binary transformers>=4.37.2 accelerate>=0.26.1 numpy
    
    echo "Installing remaining packages..."
    pip install --prefer-binary langchain langchain-anthropic langchain-community faiss-cpu sentence-transformers
    pip install --prefer-binary imageio decorator tqdm audiofile
    pip install --prefer-binary opencv-python Pillow scipy
    
    echo "Installing YouTube/social media support..."
    pip install --prefer-binary "yt-dlp>=2023.7.6"
    
    echo "Installing Whisper..."
    pip install --prefer-binary git+https://github.com/openai/whisper.git
}

if [ $? -ne 0 ]; then
    echo "Error: Failed to install required packages."
    echo "Check the requirements.txt file and try again."
    echo "If you're trying to use OpenAI models, run with --fix-dependencies option."
    exit 1
fi

echo "Starting the application..."
streamlit run "$SCRIPT_DIR/Speech-To-Text-App_v0.6_Explained.py"