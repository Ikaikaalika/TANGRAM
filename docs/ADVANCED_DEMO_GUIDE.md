# TANGRAM Advanced Demo Guide

The TANGRAM Advanced Demo showcases a complete AI-powered robotic scene understanding pipeline with state-of-the-art computer vision, intelligent task planning, and realistic robot simulation.

## 🌟 What the Demo Does

The demo creates a complete workflow from video input to robot task execution:

1. **🔍 Advanced Computer Vision** - Processes video with YOLO v8, SAM segmentation, and depth estimation
2. **🏗️ 3D Scene Reconstruction** - Creates detailed virtual environment from video analysis  
3. **🧠 Intelligent Task Planning** - Uses DeepSeek R1 7B LLM for complex task decomposition
4. **🦾 Robot Simulation** - Executes tasks with physics-based 6-DOF robot arm simulation
5. **📊 Comprehensive Reporting** - Generates detailed analysis and performance metrics

## 🚀 Running the Demo

### Quick Start

```bash
# 1. Activate environment
conda activate tangram

# 2. Run advanced demo
python tangram.py demo

# Or run directly with options
python scripts/demos/advanced_demo.py --task "Stack all objects in a pile"
```

### Command Line Options

```bash
python scripts/demos/advanced_demo.py [OPTIONS]

Options:
  --video PATH       Input video file (uses generated demo scene if not provided)
  --task TEXT        Task description (default: "Organize all objects into a neat pile")
  --no-gui          Run without 3D visualization
  --no-save         Don't save results to disk
  --help            Show help message
```

### Examples

```bash
# Basic demo with default task
python scripts/demos/advanced_demo.py

# Custom task with your video
python scripts/demos/advanced_demo.py --video my_video.mp4 --task "Pick up all red objects"

# Headless mode for servers
python scripts/demos/advanced_demo.py --no-gui --task "Organize objects by color"
```

## 🎯 Demo Pipeline Overview

### Step 1: Advanced Computer Vision Processing
- **YOLO v8** object detection with confidence scoring
- **SAM 2.0** precise object segmentation  
- **Depth estimation** using state-of-the-art neural networks
- **3D object tracking** across video frames
- **Spatial relationship analysis** between objects

### Step 2: 3D Scene Reconstruction
- Dense **point cloud generation** from video frames
- **Physics-aware object positioning** in 3D space
- **Workspace boundary detection** for robot planning
- **Scene understanding** with object relationships

### Step 3: Intelligent Task Planning
- **DeepSeek R1 7B** local LLM for task analysis
- **Multi-step task decomposition** into executable actions
- **Constraint satisfaction** with safety validation
- **Success probability estimation** for each plan
- **Alternative plan generation** for robustness

### Step 4: Advanced Robot Simulation
- **6-DOF robot arm** with realistic kinematics
- **Physics-based simulation** using PyBullet
- **Collision detection** and avoidance
- **Advanced gripper mechanics** for object manipulation
- **Force/torque sensing** for realistic interaction

### Step 5: Results and Visualization
- **Real-time 3D visualization** of robot execution
- **Performance metrics** tracking and analysis
- **Comprehensive reporting** in JSON and Markdown
- **Error analysis** and recovery suggestions

## 🔧 Technical Features

### Computer Vision Pipeline
- **Multi-class object detection** with YOLO v8
- **Instance segmentation** using Segment Anything Model
- **Monocular depth estimation** for 3D positioning
- **Temporal tracking** for consistent object identification
- **Fallback detection** using color-based segmentation

### Robot Simulation
- **UR5-style 6-DOF arm** with realistic joint limits
- **Advanced physics** with mass, friction, and restitution
- **Collision detection** between robot and environment
- **Path planning** with inverse kinematics
- **Gripper control** with force sensing

### LLM Integration
- **Local DeepSeek R1 7B** model (zero external API calls)
- **Advanced prompting** for scene understanding
- **Task decomposition** with safety constraints
- **Execution monitoring** with error recovery
- **Natural language** command interface

### Performance Optimization
- **GPU acceleration** support (CUDA/MPS)
- **Efficient processing** with frame sampling
- **Memory management** for large point clouds
- **Real-time visualization** with 60fps rendering

## 📊 Output and Results

### Generated Files

The demo creates a timestamped output directory with:

```
results/advanced_demo/demo_[timestamp]/
├── demo_report.md              # Human-readable report
├── demo_report.json            # Structured data report  
├── scene_reconstruction.json   # 3D scene data
├── task_plan.json             # LLM-generated plan
├── execution_results.json     # Robot execution metrics
└── demo_scene.mp4             # Generated demo video (if used)
```

### Report Contents

- **Computer Vision Metrics**: Objects detected, confidence scores, 3D positions
- **Scene Analysis**: Spatial relationships, workspace bounds, object properties
- **Task Planning**: Step-by-step robot actions with timing estimates
- **Execution Results**: Success rates, actual vs estimated timing
- **Performance Analysis**: Total pipeline time, bottleneck identification

### Visualization Features

- **Interactive 3D scene** with object highlighting
- **Robot arm visualization** with joint angle display
- **Real-time execution** progress tracking
- **Camera controls** for optimal viewing angles
- **Object property** inspection on hover

## 🎮 Interactive Features

### GUI Controls

When running with GUI enabled:

- **Mouse controls**: Rotate camera view
- **Scroll wheel**: Zoom in/out
- **Space bar**: Pause/resume simulation
- **R key**: Reset camera view
- **ESC**: Exit demo

### Real-time Monitoring

- **Task progress** display with step-by-step updates
- **Robot status** showing joint angles and gripper state
- **Object tracking** with ID and position information
- **Performance metrics** updated in real-time

## 🔍 Understanding the Output

### Success Metrics

- **Detection Accuracy**: Percentage of objects correctly identified
- **Planning Quality**: Task complexity and safety scores (0-10)
- **Execution Success**: Percentage of steps completed successfully
- **Overall Performance**: End-to-end pipeline efficiency

### Error Analysis

Common issues and solutions:

- **Low detection confidence**: Improve lighting or object contrast
- **Planning failures**: Simplify task or adjust workspace constraints  
- **Execution errors**: Check robot reachability and collision clearance
- **Performance issues**: Reduce video resolution or disable GUI

## 🚀 Advanced Usage

### Custom Video Input

```python
# Process your own video
python scripts/demos/advanced_demo.py --video path/to/your/video.mp4
```

Requirements for input videos:
- **Format**: MP4, AVI, MOV supported
- **Resolution**: 640x480 minimum, 1920x1080 maximum  
- **Content**: Tabletop scene with clearly visible objects
- **Duration**: 3-30 seconds optimal

### Custom Task Commands

The LLM understands natural language commands:

```bash
# Object manipulation
--task "Pick up the red apple and place it in the corner"

# Organization tasks  
--task "Arrange all objects in a straight line"

# Sorting operations
--task "Group objects by color"

# Stacking challenges
--task "Build a tower with the largest objects at the bottom"
```

### Integration with Your Code

```python
from scripts.demos.advanced_demo import TangramAdvancedDemo

# Create demo instance
demo = TangramAdvancedDemo(gui=True, save_results=True)

# Initialize systems
demo.initialize_systems()

# Run custom workflow
success = demo.run_complete_demo(
    video_path="my_video.mp4",
    task_description="Custom task description"
)

# Access results
scene_data = demo.scene_reconstruction
task_plan = demo.task_plan
execution_results = demo.execution_results

# Cleanup
demo.cleanup()
```

## 🎯 Demo Scenarios

### Scenario 1: Object Organization
- **Input**: Video of scattered objects on table
- **Task**: "Organize all objects into a neat arrangement"
- **Output**: Robot systematically arranges objects in grid pattern

### Scenario 2: Selective Manipulation  
- **Input**: Multi-object scene with various colors
- **Task**: "Pick up only the red objects"
- **Output**: Robot identifies and manipulates specific objects

### Scenario 3: Complex Stacking
- **Input**: Objects of different sizes and shapes
- **Task**: "Create a stable tower with largest objects at bottom"
- **Output**: Robot plans and executes size-based stacking

### Scenario 4: Workspace Cleaning
- **Input**: Cluttered workspace
- **Task**: "Clear the workspace by moving all objects to one side"
- **Output**: Robot systematically organizes workspace

## 🏆 Performance Benchmarks

### Typical Performance

On an M1 Mac with conda environment:

- **Computer Vision**: 2-5 seconds for 30-frame video
- **Scene Reconstruction**: 1-2 seconds for point cloud generation
- **Task Planning**: 3-8 seconds for LLM reasoning
- **Robot Execution**: 5-15 seconds per manipulation task
- **Total Pipeline**: 15-45 seconds end-to-end

### Optimization Tips

- **Use GPU acceleration** when available (CUDA/MPS)
- **Reduce video resolution** for faster processing
- **Limit frame sampling** to every 10th frame
- **Disable GUI** for headless operation
- **Use SSD storage** for faster I/O

## 🐛 Troubleshooting

### Common Issues

**"No module named 'cv2'"**
```bash
# Install OpenCV
conda install opencv

# Or via pip
pip install opencv-python
```

**"YOLO model not found"**
```bash
# Ensure models directory exists
mkdir -p models/yolo
# Model will be downloaded automatically on first run
```

**"PyBullet GUI not displaying"**
```bash
# Check display configuration
export DISPLAY=:0  # On Linux
# Or run with --no-gui flag
```

**"DeepSeek model not available"**
```bash
# Install and start Ollama
ollama pull deepseek-r1:7b
ollama serve
```

### Performance Issues

**Slow computer vision processing**:
- Reduce video resolution
- Use frame sampling (every 10th frame)
- Disable advanced features temporarily

**LLM timeout errors**:
- Check Ollama service is running
- Increase timeout in config
- Use simpler task descriptions

**Robot simulation lag**:
- Disable real-time simulation
- Reduce physics time step
- Use headless mode

### Getting Help

- **Check logs**: All output is logged with timestamps
- **Review results**: JSON files contain detailed error information
- **Test components**: Use `test_demo_structure.py` to verify setup
- **Simple tasks**: Start with basic commands before complex scenarios

## 🔮 Future Enhancements

Planned improvements for the advanced demo:

- **Multiple robot arms** coordination
- **Dynamic obstacle avoidance** during execution  
- **Learn from demonstration** capabilities
- **Voice command** interface
- **Augmented reality** visualization
- **Remote operation** over network
- **Multi-camera** scene reconstruction
- **Advanced materials** recognition

---

This advanced demo represents the cutting edge of AI-powered robotics, combining state-of-the-art computer vision, intelligent reasoning, and realistic physics simulation in a unified pipeline.