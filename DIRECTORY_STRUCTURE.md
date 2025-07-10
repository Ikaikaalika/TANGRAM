# TANGRAM Enterprise Directory Structure

## 📁 Complete File Organization

```
TANGRAM/                                    # Root directory
├── 📄 README.md                           # Main project documentation
├── 📄 config.py                           # Core configuration
├── 📄 main.py                            # Primary entry point
├── 📄 DIRECTORY_STRUCTURE.md             # This file
│
├── 📚 documentation/                      # 📚 DOCUMENTATION HUB
│   ├── 📄 README.md                      # Main documentation index
│   ├── architecture/                     # 🏗️ System Architecture
│   │   └── 📄 DATAFLOW_ARCHITECTURE.md   # Complete dataflow diagrams
│   ├── api/                              # 🔌 API Documentation
│   │   └── 📄 GEMINI_INTEGRATION.md      # Gemini API integration
│   ├── setup/                            # ⚙️ Setup & Installation
│   │   ├── 📄 LOCAL_DEEPSEEK_SETUP.md    # Local LLM setup guide
│   │   ├── 📄 THUNDER_SETUP.md           # Cloud compute setup
│   │   └── 📄 requirements.txt           # Python dependencies
│   ├── technical/                        # 🔧 Technical Specifications
│   │   ├── 📄 TECHNOLOGY_STACK.md        # Complete tech stack
│   │   ├── 📄 FULL_VS_SIMPLIFIED_COMPARISON.md  # Version comparison
│   │   └── 📄 SIMPLIFICATION_SUMMARY.md  # Codebase optimization
│   └── user_guides/                      # 👥 User Documentation
│       ├── 📄 GUI_GUIDE.md               # GUI usage guide
│       ├── 📄 ADVANCED_DEMO_GUIDE.md     # Advanced features
│       └── 📄 README_SIMPLIFIED.md       # Lightweight version guide
│
├── 🧪 testing/                           # 🧪 COMPREHENSIVE TESTING
│   ├── unit/                             # Unit Tests
│   │   ├── 📄 test_components.py         # Component unit tests
│   │   └── 📄 test_local_llm.py         # LLM unit tests
│   ├── integration/                      # Integration Tests
│   │   └── 📄 test_no_external_apis.py  # API integration tests
│   ├── performance/                      # Performance Benchmarks
│   └── system/                          # End-to-End Tests
│       └── 📄 test_reorganization.py    # System architecture tests
│
├── 💡 examples/                          # 💡 USAGE EXAMPLES
│   ├── basic/                           # Basic Examples
│   │   ├── 📄 demo.py                   # Main demonstration
│   │   └── 📄 lightweight_demo.py       # Minimal example
│   ├── advanced/                        # Advanced Examples
│   │   └── 📄 advanced_demo.py          # Full feature demonstration
│   ├── enterprise/                      # Enterprise Examples
│   └── tutorials/                       # Step-by-Step Tutorials
│
├── 🛠️ tools/                            # 🛠️ DEPLOYMENT & UTILITIES
│   ├── deployment/                      # Deployment Scripts
│   ├── monitoring/                      # System Monitoring
│   ├── utilities/                       # Utility Scripts
│   │   ├── 📄 launch_gui.py             # GUI launcher utility
│   │   └── 📄 safe_gui.py               # Safe mode GUI
│   └── simplified_version/              # Lightweight Archive
│       ├── 📄 tangram_core.py           # Consolidated core module
│       ├── 📄 main_simplified.py        # Simplified entry point
│       ├── 📄 config_simplified.py      # Minimal configuration
│       └── 📄 requirements_simplified.txt # Reduced dependencies
│
├── 🚀 src/tangram/                       # 🚀 CORE ENTERPRISE CODEBASE
│   ├── 📄 __init__.py                   # Main package init
│   │
│   ├── 🔄 pipeline/                     # 🔄 PROCESSING PIPELINE
│   │   ├── 📄 __init__.py               # Pipeline package init
│   │   │
│   │   ├── 👁️ perception/               # 👁️ COMPUTER VISION
│   │   │   ├── 📄 __init__.py           # Perception init
│   │   │   ├── detection/               # Object Detection
│   │   │   │   ├── 📄 model_downloader.py      # Model management
│   │   │   │   └── 📄 yolo_sam_detector.py     # YOLO+SAM integration
│   │   │   ├── tracker/                 # Object Tracking
│   │   │   │   └── 📄 track_objects.py  # ByteTrack implementation
│   │   │   ├── segmenter/               # Object Segmentation
│   │   │   │   └── 📄 run_sam.py        # SAM segmentation
│   │   │   └── computer_vision/         # Advanced CV
│   │   │       ├── 📄 __init__.py       # CV module init
│   │   │       └── 📄 advanced_reconstruction.py # Advanced algorithms
│   │   │   
│   │   ├── 🧠 understanding/            # 🧠 SCENE UNDERSTANDING
│   │   │   ├── 📄 __init__.py           # Understanding init
│   │   │   ├── reconstruction/          # 3D Reconstruction
│   │   │   │   ├── 📄 extract_frames.py        # Frame extraction
│   │   │   │   ├── 📄 object_3d_mapper.py      # 3D object mapping
│   │   │   │   ├── 📄 reconstruction_pipeline.py # Complete pipeline
│   │   │   │   ├── 📄 run_colmap.sh            # COLMAP integration
│   │   │   │   └── 📄 triangulate.py           # Point triangulation
│   │   │   └── scene_graph/             # Scene Graph Construction
│   │   │       └── 📄 build_graph.py    # NetworkX graph building
│   │   │
│   │   ├── 🤖 planning/                 # 🤖 AI TASK PLANNING
│   │   │   ├── 📄 __init__.py           # Planning init
│   │   │   └── llm/                     # Language Model Integration
│   │   │       ├── 📄 interpret_scene.py       # Scene interpretation
│   │   │       └── 📄 local_llm_client.py      # Local LLM interface
│   │   │
│   │   └── ⚡ execution/                # ⚡ ROBOT EXECUTION
│   │       ├── 📄 __init__.py           # Execution init
│   │       └── robotics/                # Robotics Systems
│   │           ├── 📄 llm_robot_controller.py  # LLM-robot interface
│   │           ├── 📄 motion_planner.py        # Motion planning
│   │           ├── 📄 simulation_env.py        # PyBullet simulation
│   │           └── 📄 task_executor.py         # Task execution
│   │
│   ├── 🌐 platforms/                    # 🌐 PLATFORM INTEGRATIONS
│   │   ├── 📄 __init__.py               # Platforms init
│   │   ├── cloud/                       # ☁️ Cloud Platforms
│   │   │   ├── 📄 __init__.py           # Cloud init
│   │   │   └── thunder/                 # Thunder Compute
│   │   │       ├── 📄 thunder_integration.py  # Thunder integration
│   │   │       └── 📄 tnr_client.py           # TNR client
│   │   ├── local/                       # 💻 Local Processing
│   │   └── edge/                        # 📱 Edge Deployment
│   │
│   ├── 🖥️ interfaces/                   # 🖥️ USER INTERFACES
│   │   ├── 📄 __init__.py               # Interfaces init
│   │   ├── gui/                         # Graphical Interfaces
│   │   │   ├── 📄 __init__.py           # GUI init
│   │   │   ├── 📄 interactive_gui.py    # Main interactive GUI
│   │   │   ├── 📄 tangram_manager.py    # Enterprise manager
│   │   │   └── 📄 render_graph.py       # Graph visualization
│   │   ├── api/                         # Programmatic APIs
│   │   └── cli/                         # Command Line Interfaces
│   │
│   └── 🔧 utils/                        # 🔧 CORE UTILITIES
│       ├── 📄 __init__.py               # Utils init
│       ├── 📄 file_utils.py             # File operations
│       ├── 📄 geometry_utils.py         # Geometric calculations
│       ├── 📄 logging_utils.py          # Logging system
│       ├── 📄 mock_data.py              # Mock data generation
│       ├── 📄 video_utils.py            # Video processing
│       └── export/                      # Data Export
│           └── 📄 results_exporter.py   # Results export utility
│
├── 📊 data/                             # 📊 DATA STORAGE
│   ├── 3d_points/                      # 3D point clouds
│   ├── 3d_reconstruction_live/         # Live reconstruction data
│   ├── 3d_reconstruction_test/         # Test reconstruction data
│   ├── frames/                         # Extracted video frames
│   ├── frames_live/                    # Live processing frames
│   ├── frames_test/                    # Test frames
│   ├── graphs/                         # Scene graphs
│   ├── masks/                          # Segmentation masks
│   ├── raw_videos/                     # Input video files
│   ├── sample_videos/                  # Sample data
│   ├── simulation/                     # Simulation data
│   └── tracking/                       # Object tracking data
│
├── 🤖 models/                          # 🤖 AI MODELS
│   ├── sam/                            # SAM model weights
│   │   ├── 📄 sam_vit_b.pth            # ViT-Base model
│   │   ├── 📄 sam_vit_b_01ec64.pth     # Alternative ViT-Base
│   │   └── 📄 sam_vit_h_4b8939.pth     # ViT-Huge model
│   └── yolo/                           # YOLO model weights
│       └── 📄 yolov8n.pt               # YOLOv8 nano model
│
├── 📈 results/                         # 📈 OUTPUT RESULTS
│   ├── advanced_demo/                  # Advanced demo outputs
│   ├── exports/                        # Exported data
│   ├── logs/                           # System logs
│   └── videos/                         # Generated videos
│
└── 🐍 venv/                           # 🐍 VIRTUAL ENVIRONMENT
    └── [Python virtual environment files]
```

## 📋 File Categories

### **📚 Documentation (12 files)**
Complete documentation covering architecture, APIs, setup, and user guides.

### **🧪 Testing (4 files)** 
Comprehensive testing suite with unit, integration, performance, and system tests.

### **💡 Examples (3 files)**
Usage examples from basic to enterprise-level implementations.

### **🛠️ Tools (4 utilities + simplified archive)**
Deployment tools, monitoring utilities, and archived simplified version.

### **🚀 Core Codebase (25+ modules)**
Enterprise-grade modular architecture organized by functional pipeline:

- **🔄 Pipeline**: Processing stages (perception → understanding → planning → execution)
- **🌐 Platforms**: Deployment targets (cloud, local, edge)
- **🖥️ Interfaces**: User interaction methods (GUI, API, CLI)
- **🔧 Utils**: Supporting utilities and helpers

### **📊 Data & Models**
Organized storage for datasets, AI models, and processing results.

## 🎯 Enterprise Benefits

### **Logical Organization**
- **Pipeline-based**: Mirrors actual processing flow
- **Platform-agnostic**: Supports multiple deployment environments
- **Interface-separated**: Clean separation of user interactions
- **Utility-centralized**: Shared components in dedicated location

### **Scalability**
- **Modular architecture**: Easy to add new components
- **Clear dependencies**: Well-defined module relationships
- **Testing integrated**: Comprehensive test coverage
- **Documentation complete**: Full technical documentation

### **Maintainability**
- **Predictable structure**: Logical file placement
- **Import clarity**: Clear module hierarchies
- **Separation of concerns**: Each module has single responsibility
- **Version management**: Simplified and full versions organized

This enterprise directory structure provides a professional, scalable foundation for robotic scene understanding at production scale.