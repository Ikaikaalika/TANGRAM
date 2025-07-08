# TANGRAM - AI-Powered Robotic Scene Understanding

A comprehensive pipeline that transforms video input into actionable robot commands using computer vision, 3D reconstruction, and local AI reasoning.

## 🎥 Demo Videos

### Core Pipeline Demo
https://github.com/Ikaikaalika/TANGRAM/blob/main/data/raw_videos/TANGRAM_Demo_Video.mp4

### Enhanced Features Demo  
https://github.com/Ikaikaalika/TANGRAM/blob/main/data/raw_videos/TANGRAM_Enhanced_Demo.mp4

## Quick Start

```bash
# Launch safe GUI (recommended - prevents system freezing)
python tangram.py gui

# Run lightweight demo (safe and fast)
python tangram.py demo

# Process a video file
python tangram.py process video.mp4

# Advanced options (use with caution)
python scripts/demos/advanced_demo.py --task "Organize all objects"
python safe_gui.py  # Direct safe GUI launch
```

## 🛡️ Safe Mode

TANGRAM includes safe mode options to prevent system freezing:

- **Safe GUI**: `python tangram.py gui` - Progressive component loading
- **Lightweight Demo**: `python tangram.py demo` - Minimal resource usage
- **Component Testing**: Built-in dependency checking and fallbacks

## 🌟 Advanced Demo Features

The TANGRAM Advanced Demo showcases a complete AI-powered robotic pipeline:

- **🔍 State-of-the-Art Computer Vision**: YOLO v8 + SAM + depth estimation
- **🏗️ Advanced 3D Reconstruction**: Neural-inspired scene understanding  
- **🧠 Intelligent Task Planning**: DeepSeek R1 7B local LLM reasoning
- **🦾 Realistic Robot Simulation**: 6-DOF physics-based manipulation
- **📊 Comprehensive Analysis**: Detailed performance metrics and reporting

See [Advanced Demo Guide](docs/ADVANCED_DEMO_GUIDE.md) for detailed documentation.

## Project Structure

```
TANGRAM/
├── tangram.py              # Main entry point
├── main.py                 # Core pipeline
├── config.py               # Configuration
│
├── src/tangram/
│   ├── core/               # Core components
│   │   ├── llm/           # Local LLM client
│   │   ├── tracker/       # Object tracking
│   │   ├── segmenter/     # Image segmentation
│   │   ├── reconstruction/ # 3D reconstruction
│   │   └── robotics/      # Robot simulation
│   └── gui/               # User interfaces
│       └── interactive_gui.py
│
├── scripts/
│   ├── demos/             # Demo scripts
│   └── launchers/         # Launch utilities
│
├── data/
│   ├── raw_videos/        # Input videos
│   ├── tracking/          # Detection results
│   └── models/            # AI models
│
├── docs/                  # Documentation
└── tests/                 # Test scripts
```

## Features

- **Video Analysis**: Real-time object detection and tracking
- **3D Reconstruction**: Scene understanding from 2D video
- **Local AI**: DeepSeek R1 7B running locally (zero external API calls)
- **Robot Simulation**: Virtual robot arm control and task execution
- **Interactive GUI**: User-friendly interface for video upload and control

## Installation

1. **Create conda environment:**
   ```bash
   conda create -n tangram python=3.11
   conda activate tangram
   ```

2. **Install dependencies:**
   ```bash
   pip install -r docs/requirements.txt
   ```

3. **Install PyTorch with MPS support (M1 Mac):**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install and run local DeepSeek:**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull DeepSeek R1 7B model
   ollama pull deepseek-r1:7b
   ```

## Usage

### Interactive GUI
The GUI provides three main tabs:
- **Video Analysis**: Upload videos, see object detection
- **3D Environment**: View reconstructed 3D scene
- **Robot Control**: Chat with AI to control virtual robot

### Command Line
```bash
# Process video through complete pipeline
python main.py

# Create demo video
python scripts/demos/create_enhanced_demo_video.py
```

## Configuration

Edit `config.py` to modify:
- LLM model settings
- Video processing parameters
- Robot simulation settings
- File paths and directories

## System Requirements

- **OS**: macOS, Linux, Windows
- **Python**: 3.11+
- **Memory**: 8GB+ RAM recommended
- **Storage**: 5GB for models and data
- **GPU**: Optional (MPS support for M1 Mac)

## Privacy & Security

- **Local Processing**: All AI inference runs locally
- **Zero External APIs**: No data sent to external services
- **Offline Capable**: Works without internet connection

## License

MIT License - see LICENSE file for details