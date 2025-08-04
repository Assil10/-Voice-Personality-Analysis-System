# 🎭 Voice Personality Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

A comprehensive **multi-modal voice personality analysis system** that combines voice characteristics, sentiment analysis, and topic clustering to provide deep insights into personality traits through audio analysis.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Configuration](#-api-configuration)
- [Project Structure](#-project-structure)
- [Visualizations](#-visualizations)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🎯 Core Capabilities
- **🎵 Voice Feature Extraction**: Pitch, tempo, energy, spectral characteristics
- **📝 Sentiment Analysis**: Multi-language support (Arabic, English, French)
- **🧠 Topic Clustering**: Automatic message grouping by themes
- **🤖 AI-Powered Analysis**: Advanced psychological insights via Google Gemini API
- **📊 Interactive Visualizations**: Real-time GUI dashboard
- **🔄 Multi-modal Integration**: Combines voice, text, and behavioral analysis

### 🛠️ Technical Features
- **Real-time Processing**: Live audio analysis capabilities
- **Batch Processing**: Handle multiple audio files efficiently
- **Export Options**: PNG visualizations, JSON data, text reports
- **Modular Architecture**: Clean separation of concerns
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher**
- **FFmpeg** (for audio processing)
- **Google Gemini API Key** (for advanced analysis)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/voice-personality-ai.git
cd voice-personality-ai
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install optional topic organization dependencies
pip install -r requirements_topic_org.txt
```

### Step 4: Setup API Configuration

```bash
# Run the setup script
python setup_gemini.py
```

Or manually create a `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🎯 Quick Start

### 1. Prepare Audio Files
Place your audio files in the `audio_files/` directory:
```bash
# Supported formats: WAV, MP3, MP4
cp your_audio_files/* audio_files/
```

### 2. Run Complete Analysis
```bash
python multimodal_analysis.py
```

### 3. Generate Visualizations
```bash
python generate_visualizations.py
```

### 4. View Results
```bash
python gui_visualizations.py
```

## 📖 Usage

### Basic Workflow

```python
# The system automatically:
# 1. Transcribes audio files
# 2. Extracts voice characteristics
# 3. Performs sentiment analysis
# 4. Clusters topics
# 5. Generates personality insights
# 6. Creates visualizations
```

### Advanced Usage

#### Custom Analysis Parameters
```python
# Modify analysis settings in multimodal_analysis.py
analysis_config = {
    'voice_features': ['pitch', 'tempo', 'energy', 'spectral'],
    'sentiment_language': 'arabic',  # or 'english', 'french'
    'clustering_method': 'kmeans',
    'visualization_style': 'modern'
}
```

#### Batch Processing
```bash
# Process multiple audio files
python multimodal_analysis.py --batch --input-dir audio_files/ --output-dir results/
```

## 🔧 API Configuration

### Google Gemini API Setup

1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Configure Environment**:
   ```bash
   python setup_gemini.py
   ```
3. **Verify Setup**:
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', '✓' if os.getenv('GEMINI_API_KEY') else '✗')"
   ```

### Environment Variables

Create a `.env` file in the project root:
```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
ANALYSIS_LANGUAGE=arabic
OUTPUT_FORMAT=json
DEBUG_MODE=false
```

## 📁 Project Structure

```
voice-personality-ai/
├── 📄 Core Scripts
│   ├── multimodal_analysis.py          # Main analysis engine
│   ├── generate_visualizations.py      # Static visualization generator
│   ├── gui_visualizations.py          # Interactive GUI dashboard
│   ├── setup_gemini.py                # API configuration utility
│   └── convert_audio.py               # Audio format converter
│
├── 🛠️ Optional Tools
│   ├── enhanced_topic_organizer.py     # Advanced topic clustering
│   ├── run_enhanced_organization.py    # Topic organization runner
│   ├── transcribe_*.py                 # Various transcription scripts
│   └── organize_by_topic.py            # Basic topic organization
│
├── 📊 Output Directories
│   ├── analysis_results/               # Analysis output (timestamped)
│   ├── visualizations/                 # Generated charts and graphs
│   ├── transcripts/                    # Audio transcriptions
│   └── organized_topics_enhanced/      # Topic clustering results
│
├── 🎵 Input Directories
│   ├── audio_files/                    # Input audio files
│   └── mp4_raw/                        # Raw video files
│
├── 📋 Configuration
│   ├── .env                            # Environment variables
│   ├── requirements.txt                # Core dependencies
│   └── requirements_topic_org.txt      # Topic organization dependencies
│
└── 📚 Documentation
    └── README.md                       # This file
```

## 📊 Visualizations

The system generates comprehensive visualizations:

### 🎨 Available Charts
- **📊 Topic Clusters**: Message clustering by themes with size distribution
- **😊 Sentiment Timeline**: Sentiment distribution (positive/negative/neutral)
- **👤 Personality Radar**: Personality traits radar chart
- **🎵 Voice Heatmap**: Voice characteristics correlation matrix
- **💬 Communication Styles**: Communication pattern analysis

### 📈 Output Formats
- **PNG Files**: High-resolution static visualizations
- **JSON Data**: Structured analysis results
- **Text Reports**: Detailed personality insights
- **Interactive GUI**: Real-time visualization dashboard

### 🎯 Visualization Features
- **Responsive Design**: Adapts to different screen sizes
- **Export Capabilities**: Save charts in multiple formats
- **Interactive Elements**: Clickable charts with detailed information
- **Professional Styling**: Modern, publication-ready graphics

## 🔍 Analysis Capabilities

### Voice Characteristics
- **Pitch Analysis**: Fundamental frequency patterns
- **Tempo Detection**: Speech rate and rhythm
- **Energy Levels**: Volume and intensity variations
- **Spectral Features**: Frequency domain characteristics
- **Duration Patterns**: Speaking time and pauses

### Sentiment Analysis
- **Multi-language Support**: Arabic, English, French
- **Statistical Analysis**: Positive/negative/neutral classification
- **Emotion Detection**: Advanced emotional state analysis
- **Context Awareness**: Contextual sentiment interpretation

### Topic Clustering
- **Automatic Grouping**: Intelligent message categorization
- **Keyword Extraction**: Important term identification
- **Theme Detection**: Topic pattern recognition
- **Hierarchical Organization**: Multi-level clustering

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### 1. Fork the Repository
```bash
git clone https://github.com/yourusername/voice-personality-ai.git
cd voice-personality-ai
```

### 2. Create Feature Branch
```bash
git checkout -b feature/amazing-feature
```

### 3. Make Changes
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation

### 4. Commit Changes
```bash
git add .
git commit -m "Add amazing feature"
```

### 5. Push and Pull Request
```bash
git push origin feature/amazing-feature
```

### Development Guidelines
- **Code Style**: Follow PEP 8 standards
- **Documentation**: Update README and docstrings
- **Testing**: Add unit tests for new features
- **Commits**: Use descriptive commit messages

## 🐛 Troubleshooting

### Common Issues

#### 1. Audio Processing Errors
```bash
# Ensure FFmpeg is installed
ffmpeg -version

# Check audio file formats
python convert_audio.py --check-formats
```

#### 2. API Key Issues
```bash
# Verify API key configuration
python setup_gemini.py --verify

# Test API connection
python -c "from multimodal_analysis import test_gemini_connection; test_gemini_connection()"
```

#### 3. Memory Issues
```bash
# For large audio files, use batch processing
python multimodal_analysis.py --batch --chunk-size 1000
```

#### 4. GUI Display Issues
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Try different backend
export MPLBACKEND=TkAgg
```

### Performance Optimization
- **Batch Processing**: Use for large datasets
- **Memory Management**: Process files in chunks
- **Parallel Processing**: Enable multi-threading for large files
- **Caching**: Enable result caching for repeated analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini API** for advanced AI analysis
- **OpenAI Whisper** for audio transcription
- **Librosa** for audio processing
- **Matplotlib/Seaborn** for visualizations
- **Scikit-learn** for machine learning algorithms

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Assil10/voice-personality-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Assil10/voice-personality-ai/discussions)
- **Email**: Khaldi.assil40@gmail.com

## 🔄 Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Added GUI dashboard and enhanced visualizations
- **v1.2.0** - Multi-language support and improved clustering
- **v1.3.0** - Performance optimizations and bug fixes

---

⭐ **Star this repository if you find it useful!** 