# TANGRAM Technology Stack & Architecture

## 📋 Complete Technology Stack

### 🤖 Computer Vision & AI
- **YOLOv8** - Real-time object detection and tracking
- **SAM (Segment Anything Model)** - Precise object segmentation
- **ByteTrack** - Multi-object tracking algorithm
- **COLMAP** - Structure-from-Motion 3D reconstruction
- **Open3D** - 3D data processing and visualization
- **OpenCV** - Computer vision operations
- **PyTorch/Torchvision** - Deep learning framework

### 🧠 Large Language Models
- **DeepSeek R1 7B** - Local LLM for task planning and reasoning
- **Ollama** - Local LLM server for DeepSeek hosting
- **OpenAI API** - Cloud LLM fallback option
- **Google Gemini** - Alternative cloud LLM
- **Transformers** - Hugging Face model library

### 🚀 Robotics & Simulation
- **PyBullet** - Physics simulation and robot control
- **NumPy** - Numerical computing for robotics calculations
- **SciPy** - Scientific computing for motion planning

### 🌐 Graph Processing & Visualization
- **NetworkX** - Scene graph creation and analysis
- **Matplotlib** - 2D/3D plotting and visualization
- **Pillow (PIL)** - Image processing and manipulation

### ☁️ Cloud Computing
- **Thunder Compute** - GPU cloud instances for heavy processing
- **SSH/Remote Computing** - Distributed processing capabilities

### 🖥️ User Interface
- **Tkinter** - Native GUI framework
- **PIL/ImageTk** - Image display in GUI
- **Matplotlib Backend** - Embedded plotting in GUI
- **Jupyter** - Interactive notebook interface (optional)

### 📊 Data Processing
- **Pandas** - Data manipulation and analysis
- **JSON** - Configuration and data serialization
- **YAML** - Configuration files
- **Scikit-learn** - Machine learning utilities

### 🛠️ Development & Infrastructure
- **Python 3.9+** - Core programming language
- **Conda** - Environment management
- **Git** - Version control
- **Pytest** - Testing framework
- **Black** - Code formatting
- **Flake8** - Code linting
- **TQDM** - Progress bars
- **PSUtil** - System resource monitoring

### 🎥 Media Processing
- **FFmpeg** (via OpenCV) - Video processing
- **Multi-format support** - MP4, AVI, MOV, JPG, PNG, etc.

### 🍎 Apple Silicon Optimization
- **MPS (Metal Performance Shaders)** - GPU acceleration on M-series Macs
- **MLX** - Apple's ML framework integration
- **Native ARM64** - Optimized for Apple Silicon

## 🔄 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TANGRAM PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

📥 INPUT LAYER
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Video     │    │   Images    │    │  Live Feed  │
│   (.mp4)    │    │ (.jpg/.png) │    │  (Camera)   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌─────────────┐
                    │Frame Extract│
                    │  (OpenCV)   │
                    └─────────────┘
                           │
🔍 PERCEPTION LAYER
                    ┌─────────────┐
                    │   YOLOv8    │
                    │ Detection   │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │ ByteTrack   │
                    │ Tracking    │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │    SAM      │
                    │Segmentation │
                    └─────────────┘
                           │
🏗️ RECONSTRUCTION LAYER
                    ┌─────────────┐
                    │   COLMAP    │◄──── Thunder Compute
                    │3D Recon.    │      (Cloud GPU)
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │  Open3D     │
                    │3D Processing│
                    └─────────────┘
                           │
🕸️ UNDERSTANDING LAYER
                    ┌─────────────┐
                    │ NetworkX    │
                    │Scene Graph  │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │  DeepSeek   │◄──── Ollama Server
                    │   R1 LLM    │      (Local AI)
                    └─────────────┘
                           │
🤖 ACTION LAYER
                    ┌─────────────┐
                    │  PyBullet   │
                    │Robot Sim.   │
                    └─────────────┘
                           │
                    ┌─────────────┐
                    │Task Execute │
                    │Motion Plan  │
                    └─────────────┘
                           │
📤 OUTPUT LAYER
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   3D Viz    │    │Export Data  │    │ Robot Ctrl  │
│ (Matplotlib)│    │   (JSON)    │    │ Commands    │
└─────────────┘    └─────────────┘    └─────────────┘

🖥️ INTERFACE LAYER (Spans All)
┌─────────────────────────────────────────────────────────────────┐
│     Interactive GUI (Tkinter) + Real-time Visualization        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Media   │ │ 3D Env  │ │ Env     │ │ Scene   │ │ Robot   │   │
│  │Analysis │ │ View    │ │Creation │ │ Graph   │ │Control  │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Detailed Data Flow

### 1. Input Processing
```
Video/Image → Frame Extraction → Individual Frames
                    ↓
            Frame Preprocessing (OpenCV)
                    ↓
            Color/Size Normalization
```

### 2. Object Detection Pipeline
```
Frames → YOLOv8 → Bounding Boxes + Classes + Confidence
           ↓
    ByteTrack → Tracked Objects + IDs + Trajectories
           ↓
    SAM → Precise Segmentation Masks
```

### 3. 3D Reconstruction Pipeline
```
Tracked Objects → COLMAP (Structure-from-Motion)
       ↓                    ↓
Multi-view Frames → Camera Poses + 3D Points
       ↓
Object 3D Positions (x,y,z coordinates)
       ↓
Open3D Processing → Refined 3D Scene
```

### 4. Scene Understanding
```
3D Objects + Trajectories → NetworkX Scene Graph
                                    ↓
        Spatial Relationships (near, on, touching)
                                    ↓
        Temporal Relationships (before, during, after)
                                    ↓
                            Rich Scene Representation
```

### 5. AI Planning & Control
```
Scene Graph + User Command → DeepSeek R1 LLM
                                    ↓
                    Natural Language Processing
                                    ↓
                        Task Sequence Planning
                                    ↓
                    Robot Action Commands
                                    ↓
            PyBullet Simulation → Motion Execution
```

### 6. Real-time Visualization
```
All Pipeline Data → GUI Components
                           ↓
    ┌─ Media Display (Tkinter Canvas)
    ├─ 3D Visualization (Matplotlib)
    ├─ Environment Creation Progress
    ├─ Scene Graph Timeline (NetworkX + Matplotlib)
    └─ Robot Control Interface
```

## 🔀 Integration Patterns

### Local vs Cloud Processing
```
Local Processing:          Cloud Processing (Thunder):
- YOLOv8 (MPS GPU)        - COLMAP 3D Reconstruction
- SAM Segmentation        - Heavy ML Training
- DeepSeek R1 LLM         - Large Model Inference
- PyBullet Simulation     - Distributed Processing
```

### Multi-Modal Data Fusion
```
Vision Data + Language Commands + Spatial Knowledge
                    ↓
        Unified Scene Understanding
                    ↓
        Intelligent Robot Actions
```

### Real-time Processing Pipeline
```
Live Video → Real-time Detection → Live 3D Updates → 
Dynamic Scene Graph → Continuous LLM Planning → 
Live Robot Control
```

This architecture enables TANGRAM to process visual input, understand 3D scenes, and control robots through natural language interaction, all with real-time visualization and both local and cloud processing capabilities.