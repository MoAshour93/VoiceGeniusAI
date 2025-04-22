# 🎙️ VoiceGenius AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0+-red.svg)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/Whisper-Large--v3-orange.svg)](https://github.com/openai/whisper)
[![LangChain](https://img.shields.io/badge/LangChain-Integrated-green.svg)](https://github.com/langchain-ai/langchain)

**VoiceGenius AI** is a comprehensive application that transforms speech to text and enables sophisticated conversations with audio/video content through multiple AI models. This powerful tool processes media from various sources, transcribes spoken content with high accuracy, and provides an interactive question-answering system using local and cloud-based language models. Extract insights, analyze content, and compare responses from different LLMs - all through an intuitive Streamlit interface.

![VoiceGenius AI Screenshot](https://your-screenshot-url.png) <!-- Replace with actual screenshot URL when available -->

## 📑 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
- [Detailed Usage Guide](#-detailed-usage-guide)
- [Media Processing Pipeline](#-media-processing-pipeline)
- [Transcription Capabilities](#-transcription-capabilities)
- [LLM Integration](#-llm-integration)
- [Performance Metrics & Analytics](#-performance-metrics--analytics)
- [Advanced Configuration](#-advanced-configuration)
- [Technical Specifications](#-technical-specifications)
- [Use Cases & Examples](#-use-cases--examples)
- [Troubleshooting](#-troubleshooting)
- [Development Roadmap](#-development-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)

## ✨ Features

### 🎤 Advanced Transcription
- **High-accuracy speech recognition** using OpenAI's Whisper models (base, large, turbo variants)
- **HuggingFace integration** for custom ASR model support
- **Multi-language support** with automatic detection for 100+ languages
- **GPU acceleration** for faster processing (with CPU fallback)
- **Customizable quality settings** balancing accuracy and speed

### 📹 Versatile Media Handling
- **Process local files** in multiple formats (MP3, WAV, MP4, AVI, MOV, MKV)
- **URL-based processing** supporting:
  - YouTube videos with automatic audio extraction
  - Vimeo content with metadata preservation
  - Direct media URLs with format detection
  - Any media source supported by yt-dlp
- **Intelligent caching** to avoid re-downloading or re-processing
- **Audio extraction** from video sources with quality preservation

### 🤖 Multi-Model Q&A System
- **Simultaneous model comparison** with side-by-side responses
- **Support for multiple LLM providers**:
  - **Local models** via Ollama (Llama3, Mistral, Gemma, Vicuna, etc.)
  - **Anthropic Claude** models (Claude 3 Opus, Sonnet, Haiku, 3.5 Sonnet, 3.7 Sonnet)
  - **OpenAI GPT** models (GPT-3.5-Turbo, GPT-4, GPT-4o, GPT-4-Turbo)
- **Dynamic model discovery** with auto-configuration
- **Interactive chat interface** with conversation history
- **Model streaming support** for real-time responses
- **Context-aware follow-up questions** with chat memory

### 📊 Advanced Analytics & Exports
- **Comprehensive metrics tracking**:
  - Response time analysis (total time, time to first token)
  - Token usage measurement (input and output tokens)
  - Generation speed metrics (tokens per second)
  - Model comparison statistics
- **Rich export formats**:
  - Plain text transcriptions with metadata
  - Markdown session exports with complete Q&A history
  - Performance analytics for model evaluation
- **Session persistence** with automatic saving

## 🏗 System Architecture

VoiceGenius AI employs a modular architecture with several key components:

1. **Frontend Layer** (Streamlit):
   - Responsive web UI with dynamic components
   - State management for session persistence
   - User interaction handling and event processing

2. **Media Processing Layer**:
   - Audio extraction via MoviePy
   - URL processing via yt-dlp
   - File format handling and normalization
   - Caching and optimization logic

3. **Transcription Engine**:
   - Whisper model integration (OpenAI)
   - HuggingFace transformers pipeline
   - Language detection and processing
   - GPU acceleration when available

4. **LLM Integration Layer**:
   - Multi-provider API connections
   - Model discovery and configuration
   - Context processing and prompt engineering
   - Response handling and streaming

5. **Analytics & Storage Layer**:
   - Metrics calculation and tracking
   - Session history management
   - Export formatting and generation
   - File system operations

This architecture ensures high extensibility, allowing for the addition of new models, features, and capabilities with minimal code changes.

## 🚀 Getting Started

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB, 16GB+ recommended for larger models
- **Storage**: 5GB+ for application and models
- **GPU**: Optional but recommended for faster transcription (CUDA compatible)
- **Internet Connection**: Required for model downloads and cloud APIs

### Prerequisites
- **Python 3.8+**: Core runtime environment
- **FFmpeg**: Required for audio processing (automatically installed by scripts)
- **yt-dlp**: For URL media processing (automatically installed by scripts)
- **CUDA Toolkit** (optional): For GPU acceleration if compatible hardware is available

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/MoAshour93/VoiceGenius-AI.git
cd VoiceGenius-AI
```

2. **Launch the application**:

**Windows:**
```bash
voicegenius.bat
```

**macOS/Linux:**
```bash
chmod +x voicegenius.sh
./voicegenius.sh
```

The startup scripts automatically:
- Create a Python virtual environment
- Install required dependencies
- Launch the Streamlit server
- Open your browser to the application

### Dependency Troubleshooting

If you encounter any issues with dependencies, use the fix-dependencies flag:

```bash
# Windows
voicegenius.bat --fix-dependencies

# macOS/Linux
./voicegenius.sh --fix-dependencies
```

This will:
- Update pip to the latest version
- Install/update core dependencies
- Install FFmpeg if missing
- Set up proper CUDA configurations if applicable
- Verify model accessibility

### Optional External Services

For cloud-based models, configure API keys:

1. **Anthropic API Key** (for Claude models):
   - Create an account at [Anthropic Console](https://console.anthropic.com/)
   - Generate an API key
   - Enter it in the app sidebar under "API Key Management"

2. **OpenAI API Key** (for GPT models):
   - Create an account at [OpenAI Platform](https://platform.openai.com/)
   - Generate an API key
   - Enter it in the app sidebar under "API Key Management"

3. **Local LLM Setup** (for Ollama models):
   - Install [Ollama](https://ollama.ai/) on your system
   - Pull desired models (e.g., `ollama pull llama3`)
   - Ensure Ollama is running when using VoiceGenius AI

## 📋 Detailed Usage Guide

### Media Input Options

#### Local File Upload
1. Select "Upload File" in the sidebar
2. Click the upload button to select your file
3. Supported formats:
   - **Audio**: MP3, WAV
   - **Video**: MP4, AVI, MOV, MKV
4. Size limit: Up to 500MB (for larger files, consider processing in segments)

#### URL Processing
1. Select "Enter URL" in the sidebar
2. Paste a valid media URL in the input field
3. Supported platforms:
   - **YouTube**: Standard videos, shorts, and playlists (first item)
   - **Vimeo**: Standard videos with public access
   - **Direct media links**: URLs ending with supported format extensions
   - **Other platforms**: Most sites supported by yt-dlp

### Transcription Configuration

#### Model Selection
- **Large ✨**: Highest accuracy, suitable for most languages and complex audio
- **Turbo ⚡**: Faster processing with slightly reduced accuracy
- **Custom 🛠️**: Enter a specific HuggingFace model ID for specialized needs

#### Language Settings
- **Auto-detect 🔍**: Automatically identifies the spoken language
- **Specific language**: Select from common options for improved accuracy
- **Custom language code**: Enter ISO language code for less common languages

### Model Selection and Management

#### Adding Models
1. Navigate to "Chat Model Configuration" in the sidebar
2. Select a provider (Ollama, Anthropic, or OpenAI)
3. Choose a specific model from the dropdown
4. Optionally add a custom label for easier identification
5. Click "Add Model" to initialize the model

#### Comparing Models
- Add multiple models of different types
- Ask questions to receive side-by-side responses
- Use the model selector to target specific models
- Compare performance metrics for benchmarking
- Remove models that aren't needed

### Advanced Chat Capabilities

1. **Asking Questions**:
   - Type your question in the chat input field
   - Select "All Models" or a specific model
   - View responses in real-time with typing simulation
   - Expand performance metrics for detailed analysis

2. **Follow-up Questions**:
   - Chat history is maintained for context
   - Models receive relevant previous exchanges
   - Ask for clarification or deeper insights

3. **Exporting Sessions**:
   - Click "Export Session" to save complete interaction history
   - Download as markdown with all Q&A and metrics
   - Use for documentation or sharing insights

## 🔄 Media Processing Pipeline

VoiceGenius AI implements a sophisticated pipeline for handling various media sources:

### Local File Processing
1. **Upload Handling**: Secure transfer to server with integrity verification
2. **Format Detection**: Automatic identification of file type and encoding
3. **Video Processing**: For video files, the audio track is extracted using MoviePy
4. **Caching**: Processed files are cached to avoid redundant operations

### URL-based Media Extraction
1. **URL Validation**: Verification of URL format and accessibility
2. **Service Detection**: Identification of source platform (YouTube, Vimeo, etc.)
3. **Metadata Preservation**: Extraction of title, uploader, and other relevant details
4. **Download Optimization**: Selection of appropriate quality and format
5. **Progressive Processing**: Download progress tracking with status updates

### Audio Normalization
- **Channel Standardization**: Conversion to consistent audio channel format
- **Sample Rate Adjustment**: Optimization for transcription models
- **Volume Normalization**: Balancing audio levels for improved recognition
- **Noise Reduction**: Basic cleaning for better transcription results

## 🎯 Transcription Capabilities

### Whisper Model Variants

VoiceGenius AI supports multiple Whisper model variants with different characteristics:

| Model | Size | Accuracy | Speed | Languages | Memory Usage |
|-------|------|----------|-------|-----------|--------------|
| Large | 1.5GB | Highest | Slower | 100+ | High |
| Turbo | 1.5GB | High | Faster | 100+ | High |
| Custom | Varies | Varies | Varies | Varies | Varies |

### Technical Implementation Details

1. **Model Loading**:
   - Dynamic GPU detection for hardware acceleration
   - Automatic fallback to CPU when GPU unavailable
   - Memory optimization for efficient processing

2. **Processing Pipeline**:
   - Audio chunking for large files
   - Parallel processing when possible
   - Progress tracking with detailed status updates

3. **Output Processing**:
   - Text cleanup and formatting
   - Punctuation and capitalization correction
   - Speaker diarization preparation (when available)

4. **Special Capabilities**:
   - Timestamps for long-form content
   - Specialized models for academic/scientific content
   - Background noise resilience

## 🧠 LLM Integration

### Supported Model Details

#### Ollama (Local Models)
- **Connection**: Local API at http://localhost:11434
- **Models**: Any model available in Ollama
- **Features**: No API key required, fully private processing
- **Limitations**: Depends on local hardware capabilities

#### Anthropic Claude
- **Models**:
  - **Claude 3 Opus**: Highest capability, best for complex reasoning
  - **Claude 3 Sonnet**: Balanced performance and speed
  - **Claude 3 Haiku**: Fastest model, good for simple queries
  - **Claude 3.5 Sonnet**: Enhanced version with improved capabilities
  - **Claude 3.7 Sonnet**: Latest version with advanced features
- **Special Features**: Handles longer context, excellent comprehension

#### OpenAI GPT
- **Models**:
  - **GPT-3.5-Turbo**: Fast, economical option
  - **GPT-4**: High capability, reasoning focused
  - **GPT-4o**: Latest version with optimized performance
  - **GPT-4-Turbo**: Enhanced speed with GPT-4 capabilities
- **Special Features**: More widely available, familiar response patterns

### LLM Architecture Integration

VoiceGenius AI implements several integration methods for maximum compatibility:

1. **Direct API Integration**:
   - Native API calls to model providers
   - Streaming support for real-time responses
   - Error handling with graceful fallbacks

2. **LangChain Framework**:
   - Fallback integration when direct API is unavailable
   - Standardized interface across model types
   - Enhanced context management

3. **Custom Implementation**:
   - Direct HTTP requests when SDKs aren't available
   - Lightweight wrappers for consistent interface
   - Performance optimization for specific models

## 📊 Performance Metrics & Analytics

VoiceGenius AI provides comprehensive performance tracking for model evaluation:

### Tracked Metrics

1. **Input Tokens**: Number of tokens in the prompt (transcript + question)
2. **Output Tokens**: Number of tokens in the model's response
3. **Time to First Token**: Latency measure from request to initial response
4. **Total Response Time**: Complete processing duration
5. **Tokens per Second**: Generation speed metric
6. **Memory Usage**: Resource utilization (where available)

### Analytical Capabilities

- **Model Comparison**: Side-by-side performance analysis
- **Cost Estimation**: Token-based usage calculation
- **Efficiency Scoring**: Automated rating based on speed and quality
- **Session Analytics**: Aggregate statistics for the entire interaction

### Export Formats

1. **Markdown Reports**:
   - Complete session documentation
   - Formatted Q&A history
   - Performance metrics tables
   - Source metadata

2. **Text Transcription**:
   - Clean, formatted transcript
   - Source information
   - Timestamp data
   - Processing metadata

## ⚙️ Advanced Configuration

### Environment Variables

VoiceGenius AI supports configuration via environment variables:

- `ANTHROPIC_API_KEY`: API key for Anthropic Claude models
- `OPENAI_API_KEY`: API key for OpenAI GPT models
- `OLLAMA_HOST`: Custom host for Ollama (default: http://localhost:11434)
- `CUDA_VISIBLE_DEVICES`: Control which GPU devices are used
- `WHISPER_MODELS_PATH`: Custom path for storing Whisper models

### Streamlit Secrets

For persistent configuration without environment variables:

1. Create a `.streamlit/secrets.toml` file in the application directory
2. Add configuration:
   ```toml
   ANTHROPIC_API_KEY = "your-key-here"
   OPENAI_API_KEY = "your-key-here"
   OLLAMA_HOST = "http://localhost:11434"
   ```

### Custom Model Configuration

#### Whisper Models
Specify custom HuggingFace model IDs in the interface, such as:
- `openai/whisper-small`
- `distil-whisper/distil-large-v3`
- `facebook/wav2vec2-large-960h`

#### LLM Provider Options
- **Custom Anthropic Models**: Support for specialized Claude models
- **OpenAI Fine-tuned Models**: Compatible with models like `ft:gpt-4:org:custom-model:id`
- **Ollama Custom Models**: Any model pulled to your local Ollama instance

## 🔧 Technical Specifications

### Core Dependencies

- **Streamlit**: Web application framework for UI rendering
- **MoviePy**: Video processing and audio extraction
- **Whisper**: OpenAI's speech recognition models
- **Transformers**: HuggingFace's NLP model toolkit
- **LangChain**: Framework for LLM application development
- **Torch**: Neural network and GPU computation
- **yt-dlp**: Media download utility with broad platform support

### Performance Optimization Techniques

1. **Caching Strategy**:
   - Session-based file caching
   - Transcription result preservation
   - Model response memoization
   - URL download deduplication

2. **Memory Management**:
   - Dynamic resource allocation
   - Model unloading when inactive
   - Cleanup of temporary files
   - Stream processing for large files

3. **Parallel Processing**:
   - Concurrent model queries
   - Multi-threaded downloads
   - Asynchronous UI updates

### Security Considerations

- **API Key Handling**: Keys stored only in session memory
- **File Processing**: Secure handling with proper validation
- **URL Security**: Validation and sanitization of external sources
- **Data Privacy**: Local processing option for sensitive content

## 🔍 Use Cases & Examples

### Content Creation Workflow

**Scenario**: A podcast producer needs to transcribe episodes and extract key topics.

**Process with VoiceGenius**:
1. Upload episode MP3 or provide YouTube link
2. Use Large model for high-quality transcription
3. Add Ollama (local) and Claude (cloud) models
4. Ask questions to identify key segments:
   - "What are the main topics discussed?"
   - "What are the key arguments made about [topic]?"
   - "Extract all statistics mentioned in the conversation"
5. Export the complete analysis for show notes

### Academic Research

**Scenario**: A researcher has recorded interviews and needs to analyze responses.

**Process with VoiceGenius**:
1. Upload interview recordings
2. Configure language setting to match interview language
3. Transcribe with high accuracy setting
4. Add GPT-4 and Claude Opus models for deep analysis
5. Ask interpretive questions:
   - "Summarize the key findings from this interview"
   - "What are the recurring themes across responses?"
   - "Compare this perspective with standard theory in the field"
6. Export session for research documentation

### Language Learning Assistant

**Scenario**: A language student wants to practice with authentic content.

**Process with VoiceGenius**:
1. Provide URL to a video in the target language
2. Select specific language in transcription settings
3. Generate accurate transcript
4. Add models with strengths in language education
5. Ask learning-focused questions:
   - "Explain the grammar pattern at 2:30"
   - "What idiomatic expressions were used?"
   - "Provide alternative ways to express the idea at 4:15"
6. Save session as study materials

### Media Analysis

**Scenario**: A journalist needs to quickly analyze an interview for quotes.

**Process with VoiceGenius**:
1. Upload interview recording or provide link
2. Use Turbo model for quick transcription
3. Add fast models like Claude Haiku or GPT-3.5-Turbo
4. Ask targeted questions:
   - "What are the most quotable statements?"
   - "Find all mentions of [specific topic]"
   - "Identify contradictions in the speaker's statements"
5. Export specific segments for inclusion in article

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### Installation Problems

| Issue | Solution |
|-------|----------|
| "Module not found" errors | Run the fix-dependencies script with `--fix-dependencies` flag |
| CUDA/GPU detection failure | Install appropriate CUDA toolkit for your GPU |
| FFmpeg missing | Run `voicegenius.bat --fix-dependencies` to install automatically |

#### Model Loading Issues

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Use a smaller model variant or free up GPU memory |
| "Model not found" | Check internet connection or use a different model |
| Slow model downloads | Ensure stable internet; downloads happen only once |

#### Audio Processing Errors

| Issue | Solution |
|-------|----------|
| "Failed to extract audio" | Ensure FFmpeg is installed correctly |
| "URL download failed" | Check if the URL is accessible or try a different format |
| "Unsupported file format" | Convert to a supported format (MP3, WAV, MP4, etc.) |

#### LLM Connection Issues

| Issue | Solution |
|-------|----------|
| "Ollama models not found" | Ensure Ollama is running on port 11434 |
| "API key error" | Verify key is entered correctly with no extra spaces |
| "Rate limit exceeded" | Wait and retry, or reduce request frequency |

### Advanced Troubleshooting

For persistent issues:

1. **Check logs**: Look for detailed error messages in the terminal
2. **Clear cache**: Delete the `temp` directory to remove cached files
3. **Update dependencies**: Run the fix script with the `--update` flag
4. **GPU issues**: Try forcing CPU mode with `CUDA_VISIBLE_DEVICES=-1`

## 🔮 Development Roadmap

Upcoming features planned for VoiceGenius AI:

### Near-term Enhancements
- **Speaker Diarization**: Identify and label different speakers
- **Translation Integration**: Directly translate transcribed content
- **Enhanced Video Support**: Process and navigate video visually
- **Batch Processing**: Handle multiple files in a queue

### Medium-term Goals
- **Custom Fine-tuning**: Train models on specific domains
- **Advanced Analytics**: Deeper insight into content patterns
- **PDF Report Generation**: Professional report exports
- **Collaborative Features**: Share sessions and insights

### Long-term Vision
- **Real-time Transcription**: Live audio processing
- **Multimodal Integration**: Process video content semantically
- **Embedded Database**: Persistent storage of insights
- **API Access**: Headless operation for integration

## 🤝 Contributing

Contributions to VoiceGenius AI are welcome! Here's how to contribute effectively:

### Contribution Guidelines

1. **Fork the repository** to your GitHub account
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-amazing-feature
   ```
3. **Implement your changes** with clear, documented code
4. **Add appropriate tests** for new functionality
5. **Update documentation** to reflect your changes
6. **Submit a pull request** with a clear description of the improvement

### Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/VoiceGenius-AI.git
   ```

2. Set up development environment:
   ```bash
   cd VoiceGenius-AI
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -e ".[dev]"  # Install with development dependencies
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

### Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all functions and classes
- Maintain test coverage for new features
- Keep commits focused and with clear messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License grants permission, free of charge, to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to including the copyright notice in all copies.

## 👨‍💼 Author

**Mohamed Ashour**

Mohamed is a managing data analyst and AI enthusiast focused on construction data analytics and artificial intelligence applications.

- **LinkedIn**: [Mohamed Ashour](https://www.linkedin.com/in/mohamed-ashour-0727/)
- **Email**: mo_ashour1@outlook.com
- **GitHub**: [MoAshour93](https://github.com/MoAshour93)
- **YouTube**: [APCMasteryPath](https://www.youtube.com/channel/APCMasteryPath)
- **Website**: [www.apcmasterypath.co.uk](https://www.apcmasterypath.co.uk)

Mohamed shares insights on RICS APC, construction data analytics, and AI-related projects through his various platforms.

## 🙏 Acknowledgments

VoiceGenius AI stands on the shoulders of these amazing projects:

- [OpenAI Whisper](https://github.com/openai/whisper) - Revolutionary speech recognition system
- [Streamlit](https://streamlit.io/) - Powerful framework for data applications
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM application development
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - State-of-the-art NLP
- [Ollama](https://ollama.ai/) - Local LLM running infrastructure
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Media downloading utility
- [MoviePy](https://zulko.github.io/moviepy/) - Video editing with Python
- [PyTorch](https://pytorch.org/) - Machine learning framework
- [Anthropic Claude](https://www.anthropic.com/claude) - Advanced AI assistant API
- [OpenAI GPT](https://openai.com/) - Leading language model technology

Special thanks to all the open-source contributors who make these tools possible.

---

<p align="center">
  <img src="https://your-logo-url.png" alt="VoiceGenius AI Logo" width="120"/><br>
  Made with ❤️ and Python<br>
  Copyright © 2023-2025 Mohamed Ashour
</p>
