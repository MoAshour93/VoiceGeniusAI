#############################################################################
# SECTION 1: IMPORTS AND DEPENDENCIES
#############################################################################
# Core libraries
import streamlit as st              # Web interface framework
import moviepy.editor as mp         # For video processing and audio extraction
import os                           # File and directory operations
import whisper                      # OpenAI's Whisper for transcription
import time                         # Time tracking and delays
import uuid                         # Unique ID generation
import requests                     # HTTP requests
from datetime import datetime       # Date and time utilities
import tiktoken                     # Token counting for LLMs
import re                           # Regular expressions
import urllib.parse                 # URL parsing
import subprocess                   # For running external commands (e.g., yt-dlp)
import tempfile                     # Temporary file handling
from pathlib import Path            # Path manipulation

# LangChain components for LLM integrations
from langchain_community.llms import Ollama           # Local LLM (Ollama)
from langchain_anthropic import ChatAnthropic         # Anthropic Claude models
try:
    from langchain_openai import ChatOpenAI           # Modern OpenAI client
except ImportError as e:
    # Fallback to older API style if needed
    from langchain.chat_models import ChatOpenAI      # Legacy OpenAI client
from langchain.chains import ConversationalRetrievalChain    # For chat with documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking
from langchain_community.vectorstores import FAISS              # Vector database
from langchain_community.embeddings import HuggingFaceEmbeddings  # Embeddings

# Machine learning components
import torch                        # PyTorch for ML operations
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor  # HuggingFace transformers

#############################################################################
# SECTION 2: COMPATIBILITY PATCHES AND APP CONFIGURATION
#############################################################################
# Patch for Pydantic compatibility issues
try:
    import pydantic
    pydantic_version = getattr(pydantic, "__version__", "unknown")
    
    # For Pydantic v2, fix compatibility issues with SecretStr
    if pydantic_version.startswith("2."):
        from pydantic import SecretStr
        if hasattr(SecretStr, "__modify_schema__") and not hasattr(SecretStr, "__get_pydantic_json_schema__"):
            # Add the missing method to avoid errors
            def get_pydantic_json_schema(self, *args, **kwargs):
                schema = {"type": "string", "format": "password"}
                return schema
            
            SecretStr.__get_pydantic_json_schema__ = get_pydantic_json_schema
except Exception as e:
    print(f"Pydantic patch failed: {e}")

# App title and configuration
st.set_page_config(
    page_title="VoiceGenius AI", 
    layout="wide",
    page_icon="🎙️"
)
st.title("🎙️ VoiceGenius AI: Transcribe & Chat with Audio/Video")

#############################################################################
# SECTION 3: UTILITY FUNCTIONS
#############################################################################

# Token counting utility
def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens using tiktoken for different models
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for token counting
        
    Returns:
        int: The estimated number of tokens
    """
    try:
        # Try to use tiktoken first (fastest and most accurate)
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except (KeyError, ImportError, ModuleNotFoundError):
        # Fallback method 1: Approximate using regex (for words)
        words = re.findall(r'\b\w+\b', text)
        # Estimate: 1 token ~= 0.75 words (rough approximation)
        return int(len(words) / 0.75)

def format_time(seconds):
    """Format time in seconds to a human-readable string"""
    if seconds < 0.001:
        return f"{seconds*1000:.2f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"

def calculate_tokens_per_second(token_count, time_taken, is_simulated=False):
    """Calculate tokens per second generation rate
    
    Args:
        token_count: Number of tokens generated
        time_taken: Time in seconds
        is_simulated: Whether the response was displayed with simulated typing
                     (if True, we adjust the calculation to remove display delay)
    """
    if time_taken <= 0:
        return 0
        
    if is_simulated:
        # Adjust for added display delays
        # Calculate the approximate time added by sleep calls during display
        # For streaming with 0.01s delay per chunk, estimate 1 chunk per 4 tokens (average)
        # For char-by-char with 0.005s delay, subtract total delay based on character count
        
        # Estimate true processing time by subtracting display delay
        # For character-by-character display, we slept 0.005s per character
        # 1 token ~= 4 characters on average
        estimated_char_count = token_count * 4  # rough estimate
        simulated_delay = estimated_char_count * 0.005  # 5ms per character
        
        # Ensure we don't end up with negative time
        adjusted_time = max(0.001, time_taken - simulated_delay)
        return token_count / adjusted_time
    else:
        return token_count / time_taken

#############################################################################
# SECTION 4: EXPORT FUNCTIONALITY
#############################################################################

def export_session_data(transcription, chat_history, model_metrics, filename, source_url=None):
    """Export complete session data including transcription, chat history, and metrics
    
    Creates a Markdown file with all session information including:
    - Transcription text
    - Q&A history with all model responses
    - Performance metrics for each model
    - Overall session statistics
    
    Args:
        transcription (str): The transcript text
        chat_history (list): List of (question, {model: answer}) tuples
        model_metrics (dict): Dictionary of model metrics
        filename (str): Base filename to use
        source_url (str, optional): URL source of the media
    
    Returns:
        str: Path to the exported file
    """
    # Create Exports directory if it doesn't exist
    if not os.path.exists("Exports"):
        os.makedirs("Exports")
    
    # Create export filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"{os.path.splitext(filename)[0]}_export_{timestamp}.md"
    export_path = os.path.join("Exports", export_filename)
    
    with open(export_path, "w") as f:
        # Write header
        f.write(f"# Session Export: {os.path.splitext(filename)[0]}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add source URL if available
        if source_url:
            f.write(f"Source URL: {source_url}\n\n")
        
        # Write transcription section
        f.write("## Transcription\n\n")
        f.write("```\n")
        f.write(transcription)
        f.write("\n```\n\n")
        
        # Write chat history section with model responses
        f.write("## Q&A Session\n\n")
        
        for i, (question, answers) in enumerate(chat_history):
            f.write(f"### Question {i+1}\n\n")
            f.write(f"**User**: {question}\n\n")
            
            for model_name, answer in answers.items():
                f.write(f"**{model_name}**: {answer}\n\n")
                
                # Add metrics for this model if available
                if model_name in model_metrics:
                    metrics = model_metrics[model_name]
                    f.write("*Performance Metrics:*\n\n")
                    f.write(f"- Input Tokens: {metrics.get('input_tokens', 'N/A')}\n")
                    f.write(f"- Output Tokens: {metrics.get('output_tokens', 'N/A')}\n")
                    f.write(f"- Time to First Token: {format_time(metrics.get('time_to_first_token', 0))}\n")
                    f.write(f"- Tokens/Second: {metrics.get('tokens_per_second', 0):.2f}\n")
                    f.write(f"- Total Time: {format_time(metrics.get('total_time', 0))}\n\n")
            
            f.write("---\n\n")
        
        # Write summary section with overall metrics
        f.write("## Session Summary\n\n")
        
        # Calculate total questions and responses
        total_questions = len(chat_history)
        total_responses = sum(len(answers) for _, answers in chat_history)
        
        f.write(f"- Total Questions: {total_questions}\n")
        f.write(f"- Total Responses: {total_responses}\n")
        f.write(f"- Models Used: {', '.join(model_metrics.keys())}\n\n")
        
        # Write per-model summary
        f.write("### Model Performance Summary\n\n")
        
        for model_name, metrics in model_metrics.items():
            # Calculate averages if metrics exist
            avg_tokens_per_second = metrics.get('tokens_per_second', 0)
            total_input_tokens = metrics.get('input_tokens', 0)
            total_output_tokens = metrics.get('output_tokens', 0)
            
            f.write(f"**{model_name}**:\n")
            f.write(f"- Total Input Tokens: {total_input_tokens}\n")
            f.write(f"- Total Output Tokens: {total_output_tokens}\n")
            f.write(f"- Average Tokens/Second: {avg_tokens_per_second:.2f}\n\n")
        
        # Add export timestamp
        f.write(f"\n\n*This report was generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    return export_path

#############################################################################
# SECTION 5: SESSION STATE INITIALIZATION
#############################################################################
# Initialize session state variables for persistence between app interactions
# These variables store the app's state across reruns

# Core functionality states
if "transcription" not in st.session_state:
    st.session_state.transcription = ""               # Stores the current transcription text
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []                # List of Q&A interactions
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None                  # LangChain QA chain (if used)
if "active_models" not in st.session_state:
    st.session_state.active_models = []               # Currently active LLM models
if "model_metrics" not in st.session_state:
    st.session_state.model_metrics = {}               # Performance metrics for each model
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())   # Unique ID for this session

# Model-related states
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []               # Available Ollama models
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = "base"           # Selected Whisper model
if "whisper_model_type" not in st.session_state:
    st.session_state.whisper_model_type = "openai"    # can be 'openai' or 'huggingface'

# File/URL caching states
if "last_url" not in st.session_state:
    st.session_state.last_url = None                  # Last processed URL
if "downloaded_file_info" not in st.session_state:
    st.session_state.downloaded_file_info = {"file_path": None, "source_filename": None}
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None        # Last uploaded file name
if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = {"file_path": None, "source_filename": None, "is_video": False}

# API keys and model lists
if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = None         # Anthropic API key
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None            # OpenAI API key
if "anthropic_models" not in st.session_state:
    st.session_state.anthropic_models = [
        "claude-3-opus-20240229",                     # Most powerful Claude model
        "claude-3-sonnet-20240229",                   # Balanced Claude model
        "claude-3-haiku-20240307",                    # Fast Claude model
        "claude-3.5-sonnet-20240620",                 # New version with dot notation
        "claude-3-5-sonnet-20240620",                 # New version with hyphen notation
        "claude-3.7-sonnet-20240307"                  # Latest Claude model
    ]
if "openai_models" not in st.session_state:
    st.session_state.openai_models = [
        "gpt-3.5-turbo",                              # Fast, economical model
        "gpt-3.5-turbo-0125",                         # Version-specific variant
        "gpt-4-turbo",                                # Upgraded GPT-4
        "gpt-4-turbo-preview",                        # Preview of next GPT-4
        "gpt-4o",                                     # Omni model (multimodal)
        "gpt-4o-2024-05-13",                          # Date-specific version
        "gpt-4"                                       # Original GPT-4
    ]

# API call status tracking
if "anthropic_models_fetched" not in st.session_state:
    st.session_state.anthropic_models_fetched = False # Whether we've fetched Anthropic models
if "openai_models_fetched" not in st.session_state:
    st.session_state.openai_models_fetched = False    # Whether we've fetched OpenAI models

#############################################################################
# SECTION 6: MODEL DISCOVERY FUNCTIONS
#############################################################################
# Functions to discover and fetch available models from different providers

# Function to fetch available Ollama models
def get_ollama_models():
    """Fetch available models from local Ollama instance
    
    Returns:
        list: List of available model names from Ollama server
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            # Extract just the model names
            model_names = [model["name"] for model in models_data]
            return model_names
        else:
            st.warning(f"Failed to fetch Ollama models: HTTP {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Error connecting to Ollama: {str(e)}")
        return ["llama3", "mistral", "gemma", "vicuna"]  # Fallback default models

# Function to fetch available Anthropic models
def get_anthropic_models(api_key):
    """Fetch available models from Anthropic API or use fallback list"""
    if not api_key:
        return st.session_state.anthropic_models  # Use default list
    
    try:
        # Try to import the Anthropic module
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            try:
                # Try to list models (newer API)
                models = client.models.list()
                if hasattr(models, 'data') and len(models.data) > 0:
                    model_names = [model.id for model in models.data]
                    st.session_state.anthropic_models_fetched = True
                    return model_names
            except (AttributeError, Exception) as e:
                # Fallback for older API versions or API access issues
                st.warning(f"Cannot fetch models from Anthropic API: {str(e)}")
        except ImportError:
            st.warning("Anthropic package not installed or incompatible version")
    except Exception as e:
        st.warning(f"Error connecting to Anthropic API: {str(e)}")
    
    # Use manually maintained list as fallback
    return st.session_state.anthropic_models

# Function to fetch available OpenAI models
def get_openai_models(api_key):
    """Fetch available models from OpenAI API or use fallback list"""
    if not api_key:
        return st.session_state.openai_models  # Use default list
    
    try:
        # Try to import the OpenAI module
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            try:
                # Try to list models
                models = client.models.list()
                
                # Filter for chat models only
                chat_models = []
                for model in models.data:
                    model_id = model.id
                    # Only include GPT models suitable for chat
                    if any(prefix in model_id.lower() for prefix in ["gpt-", "ft:"]):
                        # Skip certain models that aren't generally useful
                        if not any(skip in model_id.lower() for skip in ["-instruct", "vision", "embedding", "silent"]):
                            chat_models.append(model_id)
                
                # Sort models by name to group similar models together
                chat_models.sort()
                
                # Update session state
                st.session_state.openai_models_fetched = True
                return chat_models
                
            except Exception as e:
                # Fallback for API access issues
                st.warning(f"Cannot fetch models from OpenAI API: {str(e)}")
        except ImportError:
            st.warning("OpenAI package not installed or incompatible version")
    except Exception as e:
        st.warning(f"Error connecting to OpenAI API: {str(e)}")
    
    # Use an expanded list as fallback
    latest_models = [
        "gpt-3.5-turbo", 
        "gpt-3.5-turbo-0125",
        "gpt-4-turbo", 
        "gpt-4-turbo-preview",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4o", 
        "gpt-4o-2024-05-13",
        "gpt-4"
    ]
    return latest_models

#############################################################################
# SECTION 7: MEDIA PROCESSING FUNCTIONS
#############################################################################
# Functions for handling file uploads, URL downloads and audio extraction

def extract_audio_from_video(video_file, output_path):
    """Extract audio from video file
    
    Args:
        video_file (str): Path to the video file
        output_path (str): Path where the extracted audio will be saved
        
    Returns:
        str: Path to the extracted audio file
    """
    with st.spinner("Extracting audio from video..."):
        video = mp.VideoFileClip(video_file)
        video.audio.write_audiofile(output_path, verbose=False)
    return output_path

def download_from_url(url):
    """Download media from URL using yt-dlp and return the file path.
    
    Supports YouTube, Vimeo, and other platforms via yt-dlp.
    Creates a sanitized filename based on the URL components.
    Shows progress indicators during download.
    Falls back to direct download for direct media links.
    
    Args:
        url (str): The URL to download media from
        
    Returns:
        tuple: (file_path, source_filename) or (None, None) if download fails
    """
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Generate a unique filename based on URL components
    url_parts = urllib.parse.urlparse(url)
    hostname = url_parts.netloc.replace("www.", "")
    path_sanitized = url_parts.path.strip("/").replace("/", "-")
    query_sanitized = url_parts.query.replace("=", "-").replace("&", "_") if url_parts.query else ""
    
    # Combine parts to make a reasonable filename
    if query_sanitized:
        base_filename = f"{hostname}-{path_sanitized}-{query_sanitized}"
    else:
        base_filename = f"{hostname}-{path_sanitized}"
        
    # Generate temp path for downloading
    temp_dir = os.path.join("temp", base_filename)
    
    # Status updates
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        with st.spinner(f"Downloading media from {hostname}..."):
            status_text.text("Starting download... This might take a while based on the size.")
            progress_bar.progress(10)
            
            # Use yt-dlp to download the best audio
            output_template = os.path.join("temp", f"{base_filename}.%(ext)s")
            
            # For YouTube and other streaming sites, get the audio
            cmd = [
                "yt-dlp", 
                "-f", "bestaudio/best", 
                "--extract-audio",
                "--audio-format", "wav",
                "-o", output_template,
                url
            ]
            
            # Execute the command
            status_text.text("Downloading media...")
            progress_bar.progress(30)
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if successful
            if result.returncode != 0:
                status_text.text("Error downloading. Trying direct download...")
                # If yt-dlp failed, try direct download for direct media links
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    direct_file_path = os.path.join("temp", f"{base_filename}.mp4")
                    with open(direct_file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    status_text.text("Direct download successful")
                    progress_bar.progress(80)
                    return direct_file_path, Path(direct_file_path).name
                else:
                    raise Exception(f"Failed to download: {result.stderr}")
            
            # Find the downloaded file - it will have the base filename but with an extension
            status_text.text("Processing downloaded file...")
            progress_bar.progress(90)
            
            # Look for any file with the base filename in the temp directory
            downloaded_files = list(Path("temp").glob(f"{base_filename}.*"))
            if not downloaded_files:
                raise Exception(f"Download completed but couldn't locate the file")
            
            downloaded_file = str(downloaded_files[0])
            original_filename = os.path.basename(downloaded_file)
            status_text.text(f"Download completed: {original_filename}")
            progress_bar.progress(100)
            
            return downloaded_file, original_filename
            
    except Exception as e:
        st.error(f"Error downloading: {str(e)}")
        return None, None

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and extract audio if needed.
    
    Handles both audio and video files:
    - Saves uploaded file to the temp directory
    - For video files: extracts audio using moviepy
    - Uses caching to avoid reprocessing the same file multiple times
    - Stores file metadata in session state for future reference
    
    Args:
        uploaded_file (UploadedFile): Streamlit uploaded file object
        
    Returns:
        tuple: (file_path, is_video) containing the path to the processed file
               and a boolean indicating if it was a video file
    """
    
    # Check if we're processing the same file as before
    if (st.session_state.last_uploaded_file is not None and 
        uploaded_file.name == st.session_state.last_uploaded_file and 
        st.session_state.uploaded_file_info["file_path"]):
        # Use cached file information
        file_path = st.session_state.uploaded_file_info["file_path"]
        is_video = st.session_state.uploaded_file_info["is_video"]
        st.info(f"🔄 Using cached processed file: {uploaded_file.name}")
        return file_path, is_video
    
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Save the uploaded file
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Check if it's a video file
    is_video = uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video:
        st.info(f"🔊 Extracting audio from video: {uploaded_file.name}")
        audio_path = os.path.join("temp", f"{os.path.splitext(uploaded_file.name)[0]}.wav")
        result_path = extract_audio_from_video(file_path, audio_path)
        
        # Store file info in session state
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.uploaded_file_info = {
            "file_path": result_path,
            "source_filename": uploaded_file.name,
            "is_video": True
        }
        return result_path, True
    else:
        # Store file info in session state
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.uploaded_file_info = {
            "file_path": file_path,
            "source_filename": uploaded_file.name,
            "is_video": False
        }
        return file_path, False

#############################################################################
# SECTION 8: TRANSCRIPTION FUNCTIONALITY
#############################################################################
# Functions for speech-to-text transcription using various models

def transcribe_audio(audio_file, model_type="huggingface", model_name="openai/whisper-large-v3", language=None):
    """
    Transcribe audio file using HuggingFace Whisper models.
    
    Uses transformer models to convert speech to text with progress indicators
    and GPU acceleration when available.
    
    Args:
        audio_file (str): Path to the audio file
        model_type (str): Always "huggingface" in this version
        model_name (str): Model ID for HuggingFace (e.g., "openai/whisper-large-v3")
        language (str, optional): Language to use for transcription. If None, will auto-detect.
        
    Returns:
        str: The transcribed text
    """
    with st.spinner(f"Transcribing audio with {model_name.split('/')[-1]} model..."):
        # Use HuggingFace implementation
        try:
            # Show progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Check device
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            status_text.text(f"Step 1/4: Using device: {device} with {torch_dtype}")
            progress_bar.progress(10)
            
            # Step 2: Load model - this can take time
            status_text.text(f"Step 2/4: Loading model {model_name}... (this may take several minutes on first run)")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(device)
            progress_bar.progress(40)
            
            # Step 3: Load processor
            status_text.text(f"Step 3/4: Loading processor...")
            processor = AutoProcessor.from_pretrained(model_name)
            progress_bar.progress(60)
            
            # Step 4: Set up pipeline and run transcription
            status_text.text(f"Step 4/4: Transcribing audio...")
            
            # Determine if this is a turbo model
            is_turbo = "turbo" in model_name.lower()
            
            # Configure pipeline parameters based on model type and device
            if device == "cpu":
                # CPU configuration - use smaller chunks and batch size
                batch_size = 4
                chunk_length_s = 15
            else:
                # GPU configuration
                batch_size = 16 if is_turbo else 8
                chunk_length_s = 30
            
            transcriber = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=256,  # Increased for better completeness
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
                # Additional parameters for better quality
                generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"}
            )
            
            # Perform transcription
            result = transcriber(audio_file)
            progress_bar.progress(100)
            status_text.text("Transcription complete!")
            
            # Check if transcription was successful
            if not result or not result.get("text"):
                raise ValueError("Transcription returned empty result")
                
            return result["text"]
            
        except Exception as e:
            # More detailed error handling
            st.error(f"Error using Whisper model: {str(e)}")
            
            if "CUDA out of memory" in str(e):
                st.warning("""
                💡 GPU memory error detected! Try one of these solutions:
                - Restart your computer to free up GPU memory
                - Try the Turbo model which uses less memory
                - Use a shorter audio file
                """)
            elif "not find" in str(e) or "404" in str(e):
                st.warning("""
                💡 Model not found! Make sure:
                - You entered a valid HuggingFace model ID
                - You have internet connectivity
                - Try one of the built-in options (Large or Turbo)
                """)
            else:
                st.warning("""
                💡 General error. Try:
                - Restarting the app
                - Using a different model
                - Checking your internet connection
                """)
                
            # Ask if user wants to try with the Turbo model instead
            if model_name != "openai/whisper-large-v3-turbo" and st.button("Try with Turbo model instead?"):
                st.session_state.whisper_model = "openai/whisper-large-v3-turbo"
                st.experimental_rerun()
                
            return "Error during transcription. Please try again with a different model or shorter audio."

#############################################################################
# SECTION 9: QA SYSTEM SETUP
#############################################################################
# Functions to create question-answering systems with different LLM providers

def setup_qa_system(transcription, model_choice="ollama", model_name="llama3", model_label=None):
    """Set up the QA system with Ollama, Anthropic, or OpenAI.
    
    Creates a question-answering system that can respond to queries about the transcription.
    Supports three types of backends:
    1. Local Ollama models
    2. Anthropic Claude models (via API)
    3. OpenAI GPT models (via API)
    
    Args:
        transcription (str): The transcript text to answer questions about
        model_choice (str): The provider to use ('ollama', 'anthropic', or 'openai')
        model_name (str): The specific model to use
        model_label (str, optional): A user-friendly label for the model
        
    Returns:
        object: A QA system object with a __call__ method that accepts questions
    """
    with st.spinner(f"Setting up the QA system for {model_name}..."):
        try:
            # Split the transcription into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(transcription)
            
            # Simplified embeddings handling to avoid dependency issues
            # Skip sentence-transformers and use simple QA mode
            use_simple_qa = True
            
            # Set up the LLM based on user choice
            if model_choice == "ollama":
                llm = Ollama(model=model_name)
                
                # Create a simple QA system without retrieval
                class SimpleQA:
                    def __init__(self, llm, transcript, model_name, model_label=None):
                        self.llm = llm
                        self.transcript = transcript
                        self.model_name = model_name
                        self.model_label = model_label or model_name
                        
                    def __call__(self, inputs):
                        question = inputs["question"]
                        chat_history = inputs.get("chat_history", [])
                        model_id = inputs.get("model_id", None)
                        
                        # Skip if this query isn't meant for this model
                        if model_id is not None and model_id != id(self):
                            return {"answer": None}
                        
                        # Format chat history if available
                        chat_context = ""
                        if chat_history:
                            chat_context = "Previous conversation:\n"
                            for q, a in chat_history[-3:]:  # Only include last 3 for context
                                chat_context += f"Q: {q}\nA: {a}\n\n"
                        
                        prompt = f"""You are a helpful assistant answering questions about a transcript.
                        
                        TRANSCRIPT:
                        {self.transcript}
                        
                        {chat_context}
                        QUESTION: {question}
                        
                        Please answer the question based only on the information in the transcript.
                        If the answer is not in the transcript, politely say so.
                        """
                        return {"answer": self.llm.invoke(prompt), "model": self.model_label}
                
                return SimpleQA(llm, transcription, model_name, model_label)
            
            elif model_choice == "anthropic":
                # Use API key from session state, environment variables or Streamlit secrets
                api_key = st.session_state.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    st.error("Anthropic API key not found. Please add it in the sidebar configuration.")
                    return None
                
                # Use direct Anthropic API (most reliable method)
                try:
                    # Import anthropic and set up the client
                    import anthropic
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                    
                    # Create direct client for API calls
                    anthropic_client = anthropic.Anthropic(api_key=api_key)
                    
                    # Use model name as provided
                    model_id = model_name
                    
                    # Clean up model name if needed (handle different format versions)
                    if "claude-3-5" in model_name:
                        model_id = model_name.replace("claude-3-5", "claude-3.5")
                    elif "claude-3-7" in model_name:
                        model_id = model_name.replace("claude-3-7", "claude-3.7")
                    
                    # Test if model exists by making a minimal request
                    try:
                        test_response = anthropic_client.messages.create(
                            model=model_id,
                            max_tokens=10,
                            messages=[{"role": "user", "content": "test"}]
                        )
                        st.success(f"✅ Connected to Anthropic API with model: {model_id}")
                    except Exception as model_error:
                        # If specific model fails, try removing the version date suffix
                        st.warning(f"Model {model_id} not found: {str(model_error)}. Trying alternative model formats...")
                        
                        # Try removing the version date if present
                        if "-2024" in model_id:
                            base_model = model_id.split("-2024")[0]
                            try:
                                test_response = anthropic_client.messages.create(
                                    model=base_model,
                                    max_tokens=10,
                                    messages=[{"role": "user", "content": "test"}]
                                )
                                model_id = base_model
                                st.success(f"✅ Connected to Anthropic API with model: {model_id}")
                            except Exception as e2:
                                # Try just using the latest model in the series
                                if "sonnet" in model_id:
                                    try:
                                        latest_model = "claude-3-sonnet"
                                        test_response = anthropic_client.messages.create(
                                            model=latest_model,
                                            max_tokens=10,
                                            messages=[{"role": "user", "content": "test"}]
                                        )
                                        model_id = latest_model
                                        st.success(f"✅ Connected to Anthropic API with model: {model_id}")
                                    except Exception as e3:
                                        st.error(f"Cannot find a compatible Anthropic model: {str(e3)}")
                                        return None
                                else:
                                    st.error(f"Cannot find a compatible Anthropic model: {str(e2)}")
                                    return None
                    
                    # Create a direct QA system for Anthropic
                    class DirectAnthropicQA:
                        def __init__(self, client, transcript, model_id, model_label=None):
                            self.client = client
                            self.transcript = transcript
                            self.model_name = model_id
                            self.model_label = model_label or model_id
                            
                        def __call__(self, inputs):
                            question = inputs["question"]
                            chat_history = inputs.get("chat_history", [])
                            model_id = inputs.get("model_id", None)
                            
                            # Skip if this query isn't meant for this model
                            if model_id is not None and model_id != id(self):
                                return {"answer": None}
                            
                            # Format chat history if available
                            chat_context = ""
                            if chat_history:
                                chat_context = "Previous conversation:\n"
                                for q, a in chat_history[-3:]:  # Only include last 3 for context
                                    chat_context += f"Q: {q}\nA: {a}\n\n"
                            
                            prompt = f"""You are a helpful assistant answering questions about a transcript.
                            
                            TRANSCRIPT:
                            {self.transcript}
                            
                            {chat_context}
                            QUESTION: {question}
                            
                            Please answer the question based only on the information in the transcript.
                            If the answer is not in the transcript, politely say so.
                            """
                            
                            try:
                                # Use direct API call
                                message = self.client.messages.create(
                                    model=self.model_name,
                                    max_tokens=1000,
                                    messages=[{"role": "user", "content": prompt}]
                                )
                                
                                # Extract the response text
                                if hasattr(message, 'content') and len(message.content) > 0:
                                    # New API format
                                    if hasattr(message.content[0], 'text'):
                                        return {"answer": message.content[0].text, "model": self.model_label}
                                    else:
                                        return {"answer": str(message.content[0]), "model": self.model_label}
                                else:
                                    # Fallback to string representation
                                    return {"answer": str(message), "model": self.model_label}
                            except Exception as e:
                                return {"answer": f"Error: {str(e)}", "model": self.model_label}
                    
                    # Return the direct API implementation
                    return DirectAnthropicQA(anthropic_client, transcription, model_id, model_label)
                    
                except ImportError:
                    st.error("Anthropic package not installed. Run the voicegenius.sh/bat script with --fix-dependencies option.")
                    return None
                except Exception as e:
                    st.error(f"Error setting up Anthropic: {str(e)}")
                    return None
            
            elif model_choice == "openai":
                # Use API key from session state, environment variables or Streamlit secrets
                api_key = st.session_state.openai_api_key or os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
                if not api_key:
                    st.error("OpenAI API key not found. Please add it in the sidebar configuration.")
                    return None
                
                # Set the environment variable for the API key (most reliable method)
                os.environ["OPENAI_API_KEY"] = api_key
                
                # Create the chat model with fallbacks
                try:
                    # First try with direct OpenAI client
                    import openai  # Direct import
                    
                    # Test if model exists by making a minimal request
                    try:
                        client = openai.OpenAI(api_key=api_key)
                        # Create a minimal test call to verify model access
                        test_response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=10
                        )
                        st.success(f"✅ Connected to OpenAI API with model: {model_name}")
                    except Exception as model_error:
                        # For custom/fine-tuned models that might have different format
                        if "ft:" in model_name or ":" in model_name:
                            # Skip validation for custom models
                            pass
                        else:
                            # Try finding a similar model if the exact one isn't available
                            st.warning(f"Model {model_name} not found: {str(model_error)}. Trying alternative model formats...")
                            
                            # Get base model without version
                            if "-" in model_name and not model_name.endswith("-"):
                                parts = model_name.split("-")
                                if parts[-1].startswith("20") or parts[-1].isdigit():  # Looks like a date/version
                                    base_model = "-".join(parts[:-1])
                                    try:
                                        test_response = client.chat.completions.create(
                                            model=base_model,
                                            messages=[{"role": "user", "content": "test"}],
                                            max_tokens=10
                                        )
                                        model_name = base_model
                                        st.success(f"✅ Using base model instead: {model_name}")
                                    except Exception:
                                        # Try the latest known version of each model family
                                        if "gpt-4" in model_name:
                                            fallback_options = ["gpt-4o", "gpt-4-turbo", "gpt-4"]
                                        else:
                                            fallback_options = ["gpt-3.5-turbo"]
                                            
                                        for fallback in fallback_options:
                                            try:
                                                test_response = client.chat.completions.create(
                                                    model=fallback,
                                                    messages=[{"role": "user", "content": "test"}],
                                                    max_tokens=10
                                                )
                                                model_name = fallback
                                                st.success(f"✅ Using fallback model: {model_name}")
                                                break
                                            except Exception:
                                                continue
                    
                    class DirectOpenAIQA:
                        def __init__(self, client, transcript, model_name, model_label=None):
                            self.client = client
                            self.transcript = transcript
                            self.model_name = model_name
                            self.model_label = model_label or model_name
                        
                        def __call__(self, inputs):
                            question = inputs["question"]
                            chat_history = inputs.get("chat_history", [])
                            model_id = inputs.get("model_id", None)
                            
                            # Skip if this query isn't meant for this model
                            if model_id is not None and model_id != id(self):
                                return {"answer": None}
                            
                            # Format chat history if available
                            chat_context = ""
                            if chat_history:
                                chat_context = "Previous conversation:\n"
                                for q, a in chat_history[-3:]:  # Only include last 3 for context
                                    chat_context += f"Q: {q}\nA: {a}\n\n"
                            
                            prompt = f"""You are a helpful assistant answering questions about a transcript.
                            
                            TRANSCRIPT:
                            {self.transcript}
                            
                            {chat_context}
                            QUESTION: {question}
                            
                            Please answer the question based only on the information in the transcript.
                            If the answer is not in the transcript, politely say so.
                            """
                            
                            try:
                                # Use the direct OpenAI client
                                response = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.2,
                                    max_tokens=800
                                )
                                
                                if hasattr(response.choices[0].message, 'content'):
                                    return {"answer": response.choices[0].message.content, "model": self.model_label}
                                else:
                                    return {"answer": str(response.choices[0].message), "model": self.model_label}
                            except Exception as e:
                                # Provide helpful context based on the error message
                                if "rate limit" in str(e).lower():
                                    error_msg = "⚠️ OpenAI rate limit reached. Please try again in a moment."
                                elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                                    error_msg = "⚠️ Authentication error. Please check your API key."
                                elif "model" in str(e).lower() and "not found" in str(e).lower():
                                    error_msg = f"⚠️ Model {self.model_name} not found. Please try a different model."
                                elif "context length" in str(e).lower() or "token" in str(e).lower():
                                    error_msg = "⚠️ The transcript is too long for this model. Please try chunking it or using a model with more context."
                                else:
                                    error_msg = f"Error: {str(e)}"
                                    
                                return {"answer": error_msg, "model": self.model_label}
                    
                    # Use the direct implementation that bypasses langchain
                    openai_client = openai.OpenAI(api_key=api_key)
                    return DirectOpenAIQA(openai_client, transcription, model_name, model_label)
                    
                except (ImportError, Exception) as e:
                    # Only attempt LangChain fallback if it's likely to work
                    st.warning(f"Direct OpenAI client failed: {str(e)}")
                    st.info("Attempting direct API call method instead...")
                    
                    try:
                        # Create a simplified direct API caller that doesn't depend on OpenAI package structure
                        class SimpleDirectOpenAI:
                            def __init__(self, api_key, transcript, model_name, model_label=None):
                                self.api_key = api_key
                                self.transcript = transcript
                                self.model_name = model_name
                                self.model_label = model_label or model_name
                                self.api_url = "https://api.openai.com/v1/chat/completions"
                                
                            def __call__(self, inputs):
                                question = inputs["question"]
                                chat_history = inputs.get("chat_history", [])
                                model_id = inputs.get("model_id", None)
                                
                                # Skip if this query isn't meant for this model
                                if model_id is not None and model_id != id(self):
                                    return {"answer": None}
                                
                                # Format chat history if available
                                chat_context = ""
                                if chat_history:
                                    chat_context = "Previous conversation:\n"
                                    for q, a in chat_history[-3:]:  # Only include last 3 for context
                                        chat_context += f"Q: {q}\nA: {a}\n\n"
                                
                                prompt = f"""You are a helpful assistant answering questions about a transcript.
                                
                                TRANSCRIPT:
                                {self.transcript}
                                
                                {chat_context}
                                QUESTION: {question}
                                
                                Please answer the question based only on the information in the transcript.
                                If the answer is not in the transcript, politely say so.
                                """
                                
                                headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {self.api_key}"
                                }
                                
                                payload = {
                                    "model": self.model_name,
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0.2,
                                    "max_tokens": 800
                                }
                                
                                try:
                                    # Use requests to make a direct API call
                                    response = requests.post(
                                        self.api_url,
                                        headers=headers,
                                        json=payload,
                                        timeout=60
                                    )
                                    
                                    if response.status_code == 200:
                                        try:
                                            result = response.json()
                                            if "choices" in result and len(result["choices"]) > 0:
                                                return {"answer": result["choices"][0]["message"]["content"], 
                                                        "model": self.model_label}
                                            else:
                                                return {"answer": "No response content returned.", "model": self.model_label}
                                        except Exception as e:
                                            return {"answer": f"Error parsing response: {str(e)}", "model": self.model_label}
                                    else:
                                        return {"answer": f"API Error: {response.status_code} - {response.text}", 
                                                "model": self.model_label}
                                except Exception as e:
                                    return {"answer": f"Connection error: {str(e)}", "model": self.model_label}
                        
                        return SimpleDirectOpenAI(api_key, transcription, model_name, model_label)
                    
                    except Exception as e2:
                        st.error(f"Failed to initialize OpenAI model: {str(e2)}")
                        return None
            
        except Exception as e:
            st.error(f"Error setting up QA system: {str(e)}")
            return None

#############################################################################
# SECTION 10: TRANSCRIPTION SAVING
#############################################################################
# Functions to save transcriptions to disk

def save_transcription(transcription, filename, force_new=False):
    """Save transcription to a text file.
    
    Saves transcription with metadata including source and timestamp.
    Can reuse the existing path or create a new file with a fresh timestamp.
    
    Args:
        transcription (str): The text content to save
        filename (str): Original filename of the source media
        force_new (bool): If True, always create a new file with current timestamp
                          If False, use cached path if exists in session state
                          
    Returns:
        str: Path to the saved transcription file
    """
    # Create Transcripts directory if it doesn't exist
    if not os.path.exists("Transcripts"):
        os.makedirs("Transcripts")
    
    # Check if we already have a saved path for this transcription
    if not force_new and "saved_transcript_path" in st.session_state:
        # Just return the existing path to avoid creating duplicate files
        return st.session_state.saved_transcript_path
    
    # Generate new timestamp and path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.txt"
    output_path = os.path.join("Transcripts", output_filename)
    
    # Write the file
    with open(output_path, "w") as f:
        f.write("# TRANSCRIPTION\n\n")
        f.write(f"Source: {filename}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(transcription)
    
    # Store the path in session state to reuse later
    st.session_state.saved_transcript_path = output_path
    
    return output_path

#############################################################################
# SECTION 11: SIDEBAR UI - CONFIGURATION AND INPUTS
#############################################################################
# Sidebar UI for file upload, URL input, and model selection

# Sidebar for file upload and model selection
with st.sidebar:
    st.header("🚀 Upload and Configuration")
    
    # App description
    with st.expander("ℹ️ About VoiceGenius AI", expanded=False):
        st.markdown("""
        **VoiceGenius AI** is a powerful tool that lets you:
        
        - 🎤 **Transcribe** audio/video files or online content
        - 🔄 **Process** media from YouTube or direct URLs
        - 🤖 **Chat** with the content using various LLM models
        - 📊 **Compare** different LLM responses side-by-side
        - 📝 **Export** transcriptions and conversation history
        
        Perfect for researchers, content creators, journalists, students, 
        and anyone who needs to extract and interact with spoken content.
        """)
    
    # Author information
    with st.expander("👨‍💼 About the Author", expanded=False):
        st.markdown("""
        **Mohamed Ashour** is a managing data analyst and AI enthusiast focused on 
        construction data analytics and artificial intelligence applications.
        
        - 🔗 **LinkedIn**: [Mohamed Ashour](https://www.linkedin.com/in/mohamed-ashour-0727/)
        - 📧 **Email**: mo_ashour1@outlook.com
        - 💻 **GitHub**: [MoAshour93](https://github.com/MoAshour93)
        - 📺 **YouTube**: [APCMasteryPath](https://www.youtube.com/channel/APCMasteryPath)
        - 🌐 **Website**: [www.apcmasterypath.co.uk](https://www.apcmasterypath.co.uk)
        
        Mohamed shares insights on RICS APC, construction data analytics, 
        and AI-related projects through his various platforms.
        """)
    
    # License information
    with st.expander("📜 License Information", expanded=False):
        st.markdown("""
        **VoiceGenius AI** is licensed under the MIT License.
        
        Copyright (c) 2024 Mohamed Ashour
        
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files, to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED.
        
        **Note**: This application uses various open-source libraries and APIs that 
        may have their own licensing terms.
        """)
    
    st.markdown("---")
    
    # Input source selection
    source_option = st.radio("📂 Select Source Type", ["Upload File", "Enter URL"])
    
    if source_option == "Upload File":
        upload_col1, upload_col2 = st.columns([3, 1])
        
        with upload_col1:
            uploaded_file = st.file_uploader("Upload Audio or Video File", type=["mp3", "wav", "mp4", "avi", "mov", "mkv"])
        
        with upload_col2:
            # Only show Clear Cache button if there's a cached file
            if st.session_state.last_uploaded_file:
                if st.button("🧹 Clear Cache", help="Clear the cached file processing"):
                    st.session_state.last_uploaded_file = None
                    st.session_state.uploaded_file_info = {"file_path": None, "source_filename": None, "is_video": False}
                    st.success("✅ File cache cleared.")
        
        url_input = None
    else:
        url_col1, url_col2 = st.columns([3, 1])
        
        with url_col1:
            url_input = st.text_input(
                "Enter URL", 
                placeholder="https://www.youtube.com/watch?v=...",
                help="YouTube, Vimeo, and direct media URLs are supported"
            )
        
        with url_col2:
            # Only show Clear Cache button if there's a cached URL
            if st.session_state.last_url:
                if st.button("🧹 Clear Cache", help="Clear the cached download for this URL"):
                    st.session_state.last_url = None
                    st.session_state.downloaded_file_info = {"file_path": None, "source_filename": None}
                    st.success("✅ URL cache cleared. Next use will download fresh media.")
        
        uploaded_file = None
    
    # Transcription model configuration - focusing only on HuggingFace models
    st.subheader("🔊 Transcription Model")
    
    # Always use HuggingFace models
    st.session_state.whisper_model_type = "huggingface"
    
    # Model selection options
    whisper_model_option = st.radio(
        "Choose Whisper Model", 
        ["Large ✨", "Turbo ⚡", "Custom 🛠️"],
        index=0,
        help="Large: Best quality, slower. Turbo: Faster, slightly less accurate. Custom: Enter your own model ID."
    )
    
    if whisper_model_option == "Large ✨":
        st.session_state.whisper_model = "openai/whisper-large-v3"
        st.info("✨ Large model selected: Highest accuracy, optimized for many languages")
    elif whisper_model_option == "Turbo ⚡":
        st.session_state.whisper_model = "openai/whisper-large-v3-turbo"
        st.info("⚡ Turbo model selected: Faster transcription, still high quality")
    else:  # Custom
        custom_model = st.text_input(
            "Enter HuggingFace Model ID", 
            value="openai/whisper-large-v3",
            help="Enter a valid HuggingFace model ID for speech recognition (e.g., 'distil-whisper/distil-large-v3')"
        )
        st.session_state.whisper_model = custom_model
        st.info("🛠️ Custom model selected. Make sure it's a valid Whisper model on HuggingFace")
    
    # Language selection
    st.subheader("🌐 Language Settings")
    common_languages = [
        "None (auto-detect) 🔍", 
        "english 🇬🇧", "french 🇫🇷", "spanish 🇪🇸", "german 🇩🇪", 
        "italian 🇮🇹", "portuguese 🇵🇹", "arabic 🇸🇦", "russian 🇷🇺", 
        "japanese 🇯🇵", "chinese 🇨🇳", "korean 🇰🇷", "hindi 🇮🇳"
    ]
    language = st.selectbox("Audio Language", common_languages, index=0)
    
    # Convert language selection to actual language code
    language_clean = language.split()[0] if " " in language else language
    language_clean = None if language_clean == "None" else language_clean
    
    # Store language in session state but convert "None" to None (auto-detect)
    st.session_state.language = language_clean
    
    # Show language auto-detection note
    if language_clean is None:
        st.info("🔍 Language will be automatically detected")
    else:
        st.info(f"🌐 Audio will be transcribed as {language_clean}")
    
    # Display info about model download
    st.warning("⚠️ HuggingFace models will be downloaded when first used. This may require a good internet connection and might take several minutes on first run.")
    
    # LLM Configuration for QA
    st.subheader("🤖 Chat Model Configuration")
    
    # Display active models
    if st.session_state.active_models:
        st.write("🔄 Active Models:")
        for i, model in enumerate(st.session_state.active_models):
            model_id = id(model)
            # Handle class type checking more safely
            model_type = "Ollama 🦙"
            if hasattr(model, "model_name"):
                if "claude" in model.model_name.lower():
                    model_type = "Claude 🧠"
                elif "gpt" in model.model_name.lower():
                    model_type = "OpenAI 🔮"
            
            model_label = getattr(model, "model_label", model.model_name)
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.info(f"{i+1}. {model_label} ({model_type})")
            with col2:
                if st.button(f"🗑️ Remove", key=f"remove_{model_id}"):
                    st.session_state.active_models.remove(model)
                    st.experimental_rerun()
    else:
        st.info("🤖 No active models. Add a model below to ask questions about your transcription.")
    
    # API Key Management
    with st.expander("🔑 API Key Management", expanded=False):
        # Anthropic API Key
        anthropic_key = st.text_input(
            "🧠 Anthropic API Key",
            type="password",
            value=st.session_state.anthropic_api_key or "",
            help="Your Anthropic API key for Claude models"
        )
        if anthropic_key:
            st.session_state.anthropic_api_key = anthropic_key
            st.success("✅ Anthropic API key saved!")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "🔮 OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key or "",
            help="Your OpenAI API key for GPT models"
        )
        if openai_key:
            st.session_state.openai_api_key = openai_key
            st.success("✅ OpenAI API key saved!")
        
        st.info("Note: API keys are stored only for this session and are not saved permanently.")
    
    # Add new model section
    st.write("➕ Add a new model:")
    
    # Provider selection
    qa_model_choice = st.radio("Choose LLM Provider", ["ollama 🦙", "anthropic 🧠", "openai 🔮"], 
                              key="qa_model_provider")
                              
    # Clean up the option
    qa_model_choice = qa_model_choice.split()[0]  # Remove the emoji
    
    # Custom model label
    custom_label = st.text_input("✏️ Custom Label (optional)", 
                                placeholder="Enter a friendly name for this model",
                                help="This label will be displayed in the chat interface")
    
    # Model selection
    if qa_model_choice == "ollama":
        # Fetch or refresh the available Ollama models
        if st.button("🔄 Refresh Ollama Models"):
            st.session_state.ollama_models = get_ollama_models()
            
        # If no models have been fetched yet or list is empty, fetch them
        if not st.session_state.ollama_models:
            st.session_state.ollama_models = get_ollama_models()
            
        # Display all available models or fallback to defaults
        if st.session_state.ollama_models:
            qa_model_name = st.selectbox(
                "Select Ollama Model", 
                st.session_state.ollama_models, 
                index=0,
                key="qa_ollama_model"
            )
            st.info(f"🦙 Found {len(st.session_state.ollama_models)} Ollama models")
        else:
            default_models = ["llama3", "mistral", "gemma", "vicuna"]
            qa_model_name = st.selectbox(
                "Select Ollama Model", 
                default_models, 
                index=0,
                key="qa_ollama_default"
            )
            st.warning("⚠️ Using default models. Ollama server might not be running.")
    
    elif qa_model_choice == "anthropic":
        # Check if API key is available
        if not st.session_state.anthropic_api_key:
            st.warning("⚠️ Please enter your Anthropic API key in the API Key Management section first.")
            
            # Show default models even without API key
            anthropic_models_display = [
                "claude-3-opus-20240229 (Powerful)",
                "claude-3-sonnet-20240229 (Balanced)",
                "claude-3-haiku-20240307 (Fast)",
                "claude-3.5-sonnet-20240620 (Improved)",
                "claude-3.7-sonnet-20240307 (Latest)"
            ]
        else:
            # Try to fetch available models if API key is provided
            available_models = get_anthropic_models(st.session_state.anthropic_api_key)
            
            # Format models for display with descriptions
            anthropic_models_display = []
            for model in available_models:
                if "opus" in model:
                    display = f"{model} (Powerful)"
                elif "haiku" in model:
                    display = f"{model} (Fast)"
                elif "3.5" in model or "3-5" in model:
                    display = f"{model} (Improved)"
                elif "3.7" in model or "3-7" in model:
                    display = f"{model} (Latest)"
                else:
                    display = f"{model} (Balanced)"
                anthropic_models_display.append(display)
            
            # Add refresh button to fetch models again
            refresh_col1, refresh_col2 = st.columns([3, 1])
            with refresh_col2:
                if st.button("🔄 Refresh", key="refresh_anthropic"):
                    st.session_state.anthropic_models_fetched = False
                    st.experimental_rerun()
            
        # Model selector with dynamically fetched options
        qa_model_name = st.selectbox(
            "Select Anthropic Model", 
            anthropic_models_display, 
            index=0,
            key="qa_anthropic_model"
        )
        
        # Extract the model name without the description
        qa_model_name = qa_model_name.split()[0]
        
        # Show custom model input option
        use_custom_model = st.checkbox("Use custom model ID", key="use_custom_anthropic")
        if use_custom_model:
            custom_model_id = st.text_input(
                "Enter Anthropic model ID",
                placeholder="e.g., claude-3.7-sonnet-20240307",
                key="custom_anthropic_model"
            )
            if custom_model_id:
                qa_model_name = custom_model_id
    
    elif qa_model_choice == "openai":
        # Check if API key is available
        if not st.session_state.openai_api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the API Key Management section first.")
            
            # Show default models even without API key
            openai_models_display = [
                "gpt-3.5-turbo (Fast)",
                "gpt-4o (Balanced)",
                "gpt-4-turbo (Powerful)",
                "gpt-4 (Legacy)"
            ]
        else:
            # Try to fetch available models if API key is provided
            available_models = get_openai_models(st.session_state.openai_api_key)
            
            # Format models for display with descriptions
            openai_models_display = []
            for model in available_models:
                # Add descriptions based on model type
                if "gpt-3.5" in model:
                    display = f"{model} (Fast)"
                elif "gpt-4o" in model:
                    display = f"{model} (Balanced)"
                elif "gpt-4-turbo" in model or "gpt-4-0" in model or "gpt-4-1" in model:
                    display = f"{model} (Powerful)"
                elif model == "gpt-4":
                    display = f"{model} (Legacy)"
                else:
                    display = f"{model}"
                openai_models_display.append(display)
            
            # Add refresh button to fetch models again
            refresh_col1, refresh_col2 = st.columns([3, 1])
            with refresh_col2:
                if st.button("🔄 Refresh", key="refresh_openai"):
                    st.session_state.openai_models_fetched = False
                    st.experimental_rerun()
        
        # Model selector with dynamically fetched options
        qa_model_name = st.selectbox(
            "Select OpenAI Model", 
            openai_models_display, 
            index=0,
            key="qa_openai_model"
        )
        
        # Extract the model name without the description
        qa_model_name = qa_model_name.split()[0]
        
        # Show custom model input option
        use_custom_model = st.checkbox("Use custom model ID", key="use_custom_openai")
        if use_custom_model:
            custom_model_id = st.text_input(
                "Enter OpenAI model ID",
                placeholder="e.g., ft:gpt-4-1106-preview:your-org:custom-model:abc123",
                key="custom_openai_model"
            )
            if custom_model_id:
                qa_model_name = custom_model_id
    
    # Add model button
    if st.button("➕ Add Model", use_container_width=True):
        if st.session_state.transcription:
            with st.spinner(f"⚙️ Setting up {qa_model_name}..."):
                model_label = custom_label if custom_label else qa_model_name
                new_qa_system = setup_qa_system(
                    st.session_state.transcription, 
                    qa_model_choice, 
                    qa_model_name,
                    model_label
                )
                
                if new_qa_system:
                    st.session_state.active_models.append(new_qa_system)
                    st.success(f"✅ Added {model_label} to active models!")
                    st.experimental_rerun()
        else:
            st.warning("⚠️ Please transcribe content first before adding models.")
    
    # General QA buttons
    qa_buttons_col1, qa_buttons_col2 = st.columns(2)
    
    with qa_buttons_col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.experimental_rerun()
            
    with qa_buttons_col2:
        if st.button("🗑️ Reset Models", use_container_width=True):
            st.session_state.active_models = []
            st.session_state.qa_chain = None
            st.success("✅ All models have been removed.")
            time.sleep(1)
            st.experimental_rerun()

#############################################################################
# SECTION 12: MAIN CONTENT AREA - MEDIA PROCESSING AND TRANSCRIPTION
#############################################################################
# Main app interface for media processing, transcription and Q&A

# Check if we have an input source (uploaded file or URL)
input_source_available = (uploaded_file is not None) or (url_input and url_input.strip())

if input_source_available:
    # Process the input source (file or URL)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Media Processing")
        
        # Handle file upload or URL input
        if uploaded_file:
            # Clear URL-based cache if switching to file upload
            if st.session_state.last_url:
                st.session_state.last_url = None
                st.session_state.downloaded_file_info = {"file_path": None, "source_filename": None}
                
            # Process uploaded file (with caching)
            file_path, is_video = process_uploaded_file(uploaded_file)
            source_filename = uploaded_file.name
            
            # Only show the success message if we're not using cached file info
            if not (st.session_state.last_uploaded_file == uploaded_file.name):
                if is_video:
                    st.success(f"🎬 Audio extracted from {source_filename}")
                else:
                    st.success(f"🎵 Audio file {source_filename} processed successfully")
        else:
            # Process URL input
            url = url_input.strip()
            if url:
                # Check if we already downloaded this URL
                if url == st.session_state.last_url and st.session_state.downloaded_file_info["file_path"]:
                    # Create a row with cached info and redownload button
                    cache_col1, cache_col2 = st.columns([3, 1])
                    
                    with cache_col1:
                        # Use cached file information
                        file_path = st.session_state.downloaded_file_info["file_path"]
                        source_filename = st.session_state.downloaded_file_info["source_filename"]
                        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                        st.success(f"Using previously downloaded media: {source_filename}")
                    
                    with cache_col2:
                        # Add a button to force re-download if needed
                        if st.button("Re-download", help="Force a new download of this URL"):
                            # Clear cache and redownload
                            with st.spinner("Re-downloading media..."):
                                file_path, source_filename = download_from_url(url)
                                
                                if file_path:
                                    # Check if it's a video file that needs audio extraction
                                    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                                        audio_path = os.path.join("temp", f"{os.path.splitext(os.path.basename(file_path))[0]}.wav")
                                        file_path = extract_audio_from_video(file_path, audio_path)
                                        is_video = True
                                        st.success(f"Audio extracted from {source_filename}")
                                    else:
                                        is_video = False
                                        st.success(f"Media re-downloaded successfully: {source_filename}")
                                    
                                    # Update cache with new file info
                                    st.session_state.downloaded_file_info = {
                                        "file_path": file_path,
                                        "source_filename": source_filename
                                    }
                                else:
                                    st.error("Failed to re-download from the provided URL")
                                    # Keep using the previously cached version
                                    file_path = st.session_state.downloaded_file_info["file_path"]
                                    source_filename = st.session_state.downloaded_file_info["source_filename"]
                else:
                    # New URL or not yet downloaded
                    file_path, source_filename = download_from_url(url)
                    
                    if file_path:
                        # Check if it's a video file that needs audio extraction
                        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            audio_path = os.path.join("temp", f"{os.path.splitext(os.path.basename(file_path))[0]}.wav")
                            file_path = extract_audio_from_video(file_path, audio_path)
                            is_video = True
                            st.success(f"Audio extracted from {source_filename}")
                        else:
                            is_video = False
                            st.success(f"Media downloaded successfully: {source_filename}")
                            
                        # Cache the file information for future use
                        st.session_state.last_url = url
                        st.session_state.downloaded_file_info = {
                            "file_path": file_path,
                            "source_filename": source_filename
                        }
                    else:
                        st.error("Failed to download from the provided URL")
                        file_path = None
                        source_filename = None
        
        # Only show transcription options if we have a valid file
        if file_path:
            # Transcribe the audio
            transcribe_col1, transcribe_col2 = st.columns([3, 1])
            
            with transcribe_col1:
                if st.button("🎤 Transcribe Audio", use_container_width=True):
                    # Get the model name from session state
                    model_name = st.session_state.whisper_model
                    
                    # Show which model is being used
                    st.info(f"🔊 Using Whisper model: {model_name}")
                    
                    # Add a message about the first-time download
                    if "first_time" not in st.session_state:
                        st.session_state.first_time = True
                        st.info("ℹ️ First time using this model? It will be downloaded now. This may take a few minutes.")
                    
                    # Get selected language from session state
                    language = st.session_state.language if "language" in st.session_state else None
                    
                    # Show language being used
                    if language:
                        st.info(f"🌐 Transcribing in {language} language")
                    else:
                        st.info("🔍 Using automatic language detection")
                    
                    # Call the updated transcribe_audio function with model parameters and language
                    transcription = transcribe_audio(file_path, "huggingface", model_name, language)
                    
                    # Store the transcription in session state
                    st.session_state.transcription = transcription
                    
                    if "Error during transcription" not in transcription:
                        st.success("✅ Transcription completed successfully!")
                        
                        # Save to file automatically
                        save_path = save_transcription(transcription, source_filename)
                        st.info(f"💾 Transcription saved to {os.path.basename(save_path)}")
            
            with transcribe_col2:
                # Add a button to cancel/stop the transcription
                if st.button("🧹 Clear", use_container_width=True):
                    if "transcription" in st.session_state:
                        st.session_state.transcription = ""
                        # Also clear saved transcript path to allow for new file creation
                        if "saved_transcript_path" in st.session_state:
                            del st.session_state.saved_transcript_path
                        st.experimental_rerun()
    
    with col2:
        st.subheader("📝 Transcription")
        
        if st.session_state.transcription:
            st.text_area("📃 Transcription Result", st.session_state.transcription, height=300)
            
            # Ensure we have a valid source filename
            if 'source_filename' not in locals() or not source_filename:
                source_filename = "transcription"
                
            # Download button for transcription - use existing file if available
            save_path = save_transcription(st.session_state.transcription, source_filename)
            
            # Create columns for download options
            download_col1, download_col2 = st.columns([3, 2])
            
            # Regular download button
            with download_col1:
                with open(save_path, "r") as f:
                    st.download_button(
                        label="💾 Download Transcription",
                        data=f,
                        file_name=os.path.basename(save_path),
                        mime="text/plain"
                    )
            
            # Force new download (with new timestamp)
            with download_col2:
                if st.button("📝 Save New Copy", help="Save a new copy with current timestamp"):
                    new_save_path = save_transcription(st.session_state.transcription, source_filename, force_new=True)
                    st.success(f"New copy saved as {os.path.basename(new_save_path)}")
                    st.experimental_rerun()

    #############################################################################
    # SECTION 13: Q&A INTERFACE - CHAT WITH TRANSCRIPTION
    #############################################################################
    # Interactive chat interface for asking questions about the transcription
    
    # Display chat interface if transcription and active models are available
    if st.session_state.transcription and st.session_state.active_models:
        # Create a row with chat title and export button if there's history
        chat_header_col1, chat_header_col2 = st.columns([6, 1])
        
        with chat_header_col1:
            st.subheader("💬 Ask Questions About the Transcription")
        
        with chat_header_col2:
            # Only show export button if there's chat history
            if st.session_state.chat_history:
                if st.button("📊 Export Session", help="Export the complete session including transcription, Q&A, and metrics"):
                    # Determine source name and URL
                    source_name = source_filename if 'source_filename' in locals() and source_filename else "transcription"
                    source_url = url_input if source_option == "Enter URL" and url_input else None
                    
                    export_path = export_session_data(
                        st.session_state.transcription,
                        st.session_state.chat_history,
                        st.session_state.model_metrics,
                        source_name,
                        source_url
                    )
                    
                    # Create a download link for the exported file
                    with open(export_path, "r") as f:
                        export_content = f.read()
                        
                    st.download_button(
                        label="Download Export",
                        data=export_content,
                        file_name=os.path.basename(export_path),
                        mime="text/markdown"
                    )
                    
                    st.success(f"Session exported to {os.path.basename(export_path)}")
        
        # Display chat history
        for i, (question, answers) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            
            # Display each model's answer in an expandable section
            for model_name, answer in answers.items():
                with st.chat_message("assistant"):
                    st.markdown(f"**🤖 {model_name}:** {answer}")
        
        # Model selector for question targeting
        model_options = ["All Models"] + [model.model_label for model in st.session_state.active_models]
        selected_model = st.selectbox("Ask:", model_options)
        
        # User input for questions
        user_question = st.chat_input(f"Ask a question about the content...")
        
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            
            # Store all responses
            responses = {}
            
            # Process for all models or selected model only
            if selected_model == "All Models":
                # Set up message placeholders for all models
                message_placeholders = {}
                for model in st.session_state.active_models:
                    model_name = model.model_label
                    with st.chat_message("assistant"):
                        message_placeholders[model_name] = st.empty()
                        # Start with just the model name
                        message_placeholders[model_name].markdown(f"**🤖 {model_name}:** _Thinking..._")
                
                # Process each model
                for i, model in enumerate(st.session_state.active_models):
                    model_name = model.model_label
                    
                    # If using Anthropic, we can stream the response
                    if "anthropic" in str(model.__class__).lower() or "claude" in model.model_name.lower():
                        streaming_response = ""
                        prompt = f"""You are a helpful assistant answering questions about a transcript.
                        
                        TRANSCRIPT:
                        {model.transcript}
                        
                        QUESTION: {user_question}
                        
                        Please answer the question based only on the information in the transcript.
                        If the answer is not in the transcript, politely say so.
                        """
                        
                        # Initialize metrics
                        input_tokens = count_tokens(prompt, "claude-3-sonnet-20240229")
                        start_time = time.time()
                        first_token_time = None
                        
                        # Call the LLM directly for streaming if possible
                        try:
                            for i, chunk in enumerate(model.llm.stream(prompt)):
                                # Record time to first token
                                if i == 0:
                                    first_token_time = time.time() - start_time
                                
                                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                                streaming_response += chunk_text
                                message_placeholders[model_name].markdown(f"**🤖 {model_name}:** {streaming_response}")
                                time.sleep(0.01)  # Small delay to make streaming visible
                            
                            # Calculate final metrics
                            end_time = time.time()
                            total_time = end_time - start_time
                            output_tokens = count_tokens(streaming_response, "claude-3-sonnet-20240229")
                            tokens_per_second = calculate_tokens_per_second(output_tokens, total_time, is_simulated=True)
                            
                            # Store metrics
                            st.session_state.model_metrics[model_name] = {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "time_to_first_token": first_token_time or 0,
                                "total_time": total_time,
                                "tokens_per_second": tokens_per_second
                            }
                            
                            # Save the full response
                            responses[model_name] = streaming_response
                            
                        except (AttributeError, TypeError):
                            # Fallback if streaming not supported
                            start_time = time.time()
                            
                            response = model({
                                "question": user_question, 
                                "chat_history": [(q, list(a.values())[0] if len(a)==1 else "") 
                                                 for q, a in st.session_state.chat_history]
                            })
                            
                            # First token time is approximated
                            first_token_time = time.time() - start_time
                            answer = response["answer"]
                            
                            # Simulate typing effect for non-streaming models
                            full_answer = answer
                            display_answer = ""
                            for char in full_answer:
                                display_answer += char
                                message_placeholders[model_name].markdown(f"**🤖 {model_name}:** {display_answer}")
                                time.sleep(0.005)  # Faster typing for simulation
                            
                            # Calculate metrics
                            end_time = time.time()
                            total_time = end_time - start_time
                            output_tokens = count_tokens(answer, "gpt-3.5-turbo")
                            input_tokens = count_tokens(prompt, "gpt-3.5-turbo")
                            tokens_per_second = calculate_tokens_per_second(output_tokens, total_time, is_simulated=True)
                            
                            # Store metrics
                            st.session_state.model_metrics[model_name] = {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "time_to_first_token": first_token_time,
                                "total_time": total_time,
                                "tokens_per_second": tokens_per_second
                            }
                            
                            responses[model_name] = full_answer
                    else:
                        # For models without streaming support, simulate typing
                        start_time = time.time()
                        
                        # Build prompt for token counting
                        prompt = f"""You are a helpful assistant answering questions about a transcript.
                        
                        TRANSCRIPT:
                        {model.transcript}
                        
                        QUESTION: {user_question}
                        
                        Please answer the question based only on the information in the transcript.
                        If the answer is not in the transcript, politely say so.
                        """
                        
                        # Get the response
                        response = model({
                            "question": user_question, 
                            "chat_history": [(q, list(a.values())[0] if len(a)==1 else "") 
                                             for q, a in st.session_state.chat_history]
                        })
                        
                        # First token time is approximated
                        first_token_time = time.time() - start_time
                        answer = response["answer"]
                        
                        # Simulate typing effect
                        full_answer = answer
                        display_answer = ""
                        for char in full_answer:
                            display_answer += char
                            message_placeholders[model_name].markdown(f"**🤖 {model_name}:** {display_answer}")
                            time.sleep(0.005)  # Adjust speed as needed
                        
                        # Calculate metrics
                        end_time = time.time()
                        total_time = end_time - start_time
                        output_tokens = count_tokens(answer, "gpt-3.5-turbo")
                        input_tokens = count_tokens(prompt, "gpt-3.5-turbo") 
                        tokens_per_second = calculate_tokens_per_second(output_tokens, total_time, is_simulated=True)
                        
                        # Store metrics
                        st.session_state.model_metrics[model_name] = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "time_to_first_token": first_token_time,
                            "total_time": total_time,
                            "tokens_per_second": tokens_per_second
                        }
                        
                        responses[model_name] = full_answer
                        
                    # Display metrics after response
                    metrics = st.session_state.model_metrics.get(model_name, {})
                    if metrics:
                        with st.expander(f"📊 {model_name} Performance Metrics", expanded=False):
                            metrics_cols = st.columns(4)
                            with metrics_cols[0]:
                                st.metric("Input Tokens", metrics.get("input_tokens", 0))
                            with metrics_cols[1]:
                                st.metric("Output Tokens", metrics.get("output_tokens", 0))
                            with metrics_cols[2]:
                                st.metric("Time to First Token", format_time(metrics.get("time_to_first_token", 0)))
                            with metrics_cols[3]:
                                st.metric("Tokens/Second", f"{metrics.get('tokens_per_second', 0):.2f}")
            else:
                # Find the selected model and query only that one
                selected_model_obj = next(
                    (model for model in st.session_state.active_models 
                     if model.model_label == selected_model), 
                    None
                )
                
                if selected_model_obj:
                    # Create placeholder for the response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        message_placeholder.markdown(f"**🤖 {selected_model}:** _Thinking..._")
                    
                    # If using Anthropic, we can stream the response
                    if "anthropic" in str(selected_model_obj.__class__).lower() or "claude" in selected_model_obj.model_name.lower():
                        streaming_response = ""
                        prompt = f"""You are a helpful assistant answering questions about a transcript.
                        
                        TRANSCRIPT:
                        {selected_model_obj.transcript}
                        
                        QUESTION: {user_question}
                        
                        Please answer the question based only on the information in the transcript.
                        If the answer is not in the transcript, politely say so.
                        """
                        
                        # Initialize metrics
                        input_tokens = count_tokens(prompt, "claude-3-sonnet-20240229")
                        start_time = time.time()
                        first_token_time = None
                        
                        # Try streaming approach first
                        try:
                            for i, chunk in enumerate(selected_model_obj.llm.stream(prompt)):
                                # Record time to first token
                                if i == 0:
                                    first_token_time = time.time() - start_time
                                
                                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                                streaming_response += chunk_text
                                message_placeholder.markdown(f"**🤖 {selected_model}:** {streaming_response}")
                                time.sleep(0.01)  # Small delay to make streaming visible
                            
                            # Calculate final metrics
                            end_time = time.time()
                            total_time = end_time - start_time
                            output_tokens = count_tokens(streaming_response, "claude-3-sonnet-20240229")
                            tokens_per_second = calculate_tokens_per_second(output_tokens, total_time, is_simulated=True)
                            
                            # Store metrics
                            st.session_state.model_metrics[selected_model] = {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "time_to_first_token": first_token_time or 0,
                                "total_time": total_time,
                                "tokens_per_second": tokens_per_second
                            }
                            
                            # Save the full response
                            responses[selected_model] = streaming_response
                            
                        except (AttributeError, TypeError):
                            # Fallback if streaming not supported
                            start_time = time.time()
                            
                            response = selected_model_obj({
                                "question": user_question, 
                                "chat_history": [(q, list(a.values())[0] if len(a)==1 else "") 
                                                 for q, a in st.session_state.chat_history]
                            })
                            
                            # First token time is approximated
                            first_token_time = time.time() - start_time
                            answer = response["answer"]
                            
                            # Simulate typing effect
                            full_answer = answer
                            display_answer = ""
                            for char in full_answer:
                                display_answer += char
                                message_placeholder.markdown(f"**🤖 {selected_model}:** {display_answer}")
                                time.sleep(0.005)  # Faster typing for simulation
                            
                            # Calculate metrics
                            end_time = time.time()
                            total_time = end_time - start_time
                            output_tokens = count_tokens(answer, "gpt-3.5-turbo")
                            input_tokens = count_tokens(prompt, "gpt-3.5-turbo")
                            tokens_per_second = calculate_tokens_per_second(output_tokens, total_time, is_simulated=True)
                            
                            # Store metrics
                            st.session_state.model_metrics[selected_model] = {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "time_to_first_token": first_token_time,
                                "total_time": total_time,
                                "tokens_per_second": tokens_per_second
                            }
                            
                            responses[selected_model] = full_answer
                    else:
                        # For models without streaming support
                        start_time = time.time()
                        
                        # Build prompt for token counting
                        prompt = f"""You are a helpful assistant answering questions about a transcript.
                        
                        TRANSCRIPT:
                        {selected_model_obj.transcript}
                        
                        QUESTION: {user_question}
                        
                        Please answer the question based only on the information in the transcript.
                        If the answer is not in the transcript, politely say so.
                        """
                        
                        # Get the response
                        response = selected_model_obj({
                            "question": user_question, 
                            "chat_history": [(q, list(a.values())[0] if len(a)==1 else "") 
                                             for q, a in st.session_state.chat_history]
                        })
                        
                        # First token time is approximated
                        first_token_time = time.time() - start_time
                        answer = response["answer"]
                        
                        # Simulate typing effect
                        full_answer = answer
                        display_answer = ""
                        for char in full_answer:
                            display_answer += char
                            message_placeholder.markdown(f"**🤖 {selected_model}:** {display_answer}")
                            time.sleep(0.005)  # Adjust speed as needed
                        
                        # Calculate metrics
                        end_time = time.time()
                        total_time = end_time - start_time
                        output_tokens = count_tokens(answer, "gpt-3.5-turbo")
                        input_tokens = count_tokens(prompt, "gpt-3.5-turbo") 
                        tokens_per_second = calculate_tokens_per_second(output_tokens, total_time, is_simulated=True)
                        
                        # Store metrics
                        st.session_state.model_metrics[selected_model] = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "time_to_first_token": first_token_time,
                            "total_time": total_time,
                            "tokens_per_second": tokens_per_second
                        }
                        
                        responses[selected_model] = full_answer
                    
                    # Display metrics after response
                    metrics = st.session_state.model_metrics.get(selected_model, {})
                    if metrics:
                        with st.expander(f"📊 {selected_model} Performance Metrics", expanded=False):
                            metrics_cols = st.columns(4)
                            with metrics_cols[0]:
                                st.metric("Input Tokens", metrics.get("input_tokens", 0))
                            with metrics_cols[1]:
                                st.metric("Output Tokens", metrics.get("output_tokens", 0))
                            with metrics_cols[2]:
                                st.metric("Time to First Token", format_time(metrics.get("time_to_first_token", 0)))
                            with metrics_cols[3]:
                                st.metric("Tokens/Second", f"{metrics.get('tokens_per_second', 0):.2f}")
            
            # Update chat history with all responses
            if responses:
                st.session_state.chat_history.append((user_question, responses))
                
                # If chat history is getting long, suggest exporting
                if len(st.session_state.chat_history) >= 5 and len(st.session_state.chat_history) % 5 == 0:
                    st.info("💡 Your Q&A session is getting substantial. Consider using the 'Export Session' button to save your analysis.")
else:
    #############################################################################
    # SECTION 14: WELCOME SCREEN
    #############################################################################
    # Initial welcome screen shown when no media is loaded
    
    # Welcome message with app logo/icon
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🎙️ Welcome to VoiceGenius AI!</h1>
        <p style="font-size: 1.2em;">Transform speech into text and have conversations with your audio content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Getting started columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Getting Started")
        st.info("Please upload an audio/video file or enter a URL in the sidebar to begin.")
        
        st.markdown("""
        **Quick Steps:**
        1. 📂 Select your source type (file or URL)
        2. 🎤 Upload a file or enter a media URL
        3. 🔊 Choose your transcription model
        4. 🗣️ Select language (or auto-detect)
        5. 💬 Chat with your transcribed content
        """)
    
    with col2:
        # Show URL support information
        st.markdown("### 🔗 Supported Media Sources")
        st.markdown("""
        **File Types:**
        - 🎵 Audio: MP3, WAV
        - 🎬 Video: MP4, AVI, MOV, MKV
        
        **URL Sources:**
        - 📺 YouTube videos 
        - 📹 Vimeo videos
        - 🌐 Direct links to media files
        - 🎞️ Other platforms supported by yt-dlp
        
        **Tips:**
        - YouTube videos work best for URL sources
        - Some sites may have download restrictions
        - Large files may take several minutes to process
        """)
    
    # Add a fancy divider and feature showcase
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("### ✨ Key Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    with feature_col1:
        st.markdown("""
        **🎤 Advanced Transcription**
        - High-quality speech recognition
        - Multiple language support
        - Fast & accurate models
        """)
    
    with feature_col2:
        st.markdown("""
        **🤖 Multi-Model Chat**
        - Compare different LLM responses
        - Local models via Ollama
        - Claude AI integration
        """)
        
    with feature_col3:
        st.markdown("""
        **📊 Export & Analytics**
        - Save transcriptions as text
        - Full chat history exports
        - Performance metrics tracking
        """)

#############################################################################
# SECTION 15: APP FOOTER
#############################################################################
# Footer with app information and author credits

# Footer with app info
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8em; color: #888;">
    <div>🎙️ VoiceGenius AI © 2025 Mohamed Ashour</div>
    <div>Made with ❤️ and Streamlit | <a href="https://github.com/MoAshour93" target="_blank">GitHub</a> | <a href="https://www.linkedin.com/in/mohamed-ashour-0727/" target="_blank">LinkedIn</a></div>
</div>
""", unsafe_allow_html=True)