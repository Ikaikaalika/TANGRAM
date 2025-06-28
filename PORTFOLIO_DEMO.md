# TANGRAM Portfolio Demo Guide 🤖

## Overview
TANGRAM is a complete **robotic scene understanding pipeline** that processes video data to generate actionable robot tasks. Perfect for showcasing advanced AI/ML/robotics skills in your portfolio!

## What It Demonstrates

### 🎯 **Core Technologies**
- **Computer Vision**: YOLO object detection, SAM segmentation, COLMAP 3D reconstruction
- **Machine Learning**: Multi-object tracking, spatial-temporal analysis
- **Natural Language Processing**: DeepSeek R1 LLM for task planning
- **Robotics**: PyBullet physics simulation, inverse kinematics, task execution
- **Software Engineering**: Modular architecture, comprehensive logging, automated testing

### 🔄 **Complete Pipeline**
1. **Video Analysis** → Object detection and tracking across frames
2. **Segmentation** → Precise object masks with ID association  
3. **3D Reconstruction** → Camera poses and object triangulation
4. **Scene Graphs** → Spatial and temporal relationship modeling
5. **LLM Planning** → Natural language task generation
6. **Robot Simulation** → Realistic physics-based execution
7. **Results Export** → Professional reports with metrics and visualizations

## Quick Start Demo

### 1. **Basic Demo Run**
```bash
# Run with sample data and GUI
python demo.py --gui

# Use your own video
python demo.py --video path/to/your/video.mp4 --gui --name "portfolio_demo"
```

### 2. **For Interviews/Presentations**
```bash
# Full demo with all visualizations
python demo.py --gui --name "interview_demo"

# Then open the generated HTML report:
# results/exports/interview_demo/report.html
```

### 3. **For Portfolio Website**
```bash
# Generate static results for web embedding
python demo.py --name "portfolio_showcase"
```

## Demo Output 📊

After running, you'll get:

### **📈 HTML Report** (`results/exports/[name]/report.html`)
- **Performance metrics** with interactive charts
- **Pipeline visualization** showing each step's success
- **Object detection timeline** with tracking quality metrics
- **Task execution results** with success rates and timing
- **Scene graph visualization** showing spatial relationships
- **LLM-generated explanations** and task sequences

### **🎥 Videos** (`results/exports/[name]/videos/`)
- **summary_video.mp4**: Original video with tracking overlays
- **simulation_recording.mp4**: Robot task execution (if recorded)

### **📊 Data Files** (`results/exports/[name]/data/`)
- **complete_results.json**: All pipeline outputs
- **performance_metrics.json**: Quantitative evaluation
- **error_log.json**: Any issues encountered

## For Your Portfolio 💼

### **Showcase Highlights**

1. **🔍 Technical Depth**
   - "Implemented complete vision-to-action pipeline with 7 integrated components"
   - "Achieved X% object detection accuracy and Y% task success rate"
   - "Integrated SOTA models: YOLOv8, SAM, COLMAP, DeepSeek R1"

2. **🏗️ System Design**
   - "Designed modular architecture with 500+ lines of documentation"
   - "Built comprehensive logging and error handling system"
   - "Created automated evaluation and reporting pipeline"

3. **🤖 AI/Robotics Integration**
   - "Connected computer vision to robot control via LLM reasoning"
   - "Implemented inverse kinematics and physics-based simulation"
   - "Generated natural language task plans from visual scene understanding"

### **Key Metrics to Highlight**
- **Pipeline completion rate**: X% of steps successful
- **Object detection accuracy**: Tracked Y objects across Z frames
- **Task execution success**: W% of robot tasks completed successfully
- **Processing speed**: Full pipeline in X seconds per video

### **Demo Script for Presentations**

```
"Let me show you TANGRAM - a complete robotic scene understanding system I built.

[Run demo.py --gui]

Starting with this tabletop video, watch as the system:

1. [Point to terminal] Detects and tracks objects frame-by-frame using YOLO
2. [Show GUI] Generates precise segmentation masks with SAM
3. [Explain] Reconstructs 3D positions using COLMAP computer vision
4. [Show scene graph] Builds a spatial relationship graph
5. [Read LLM output] Uses AI to plan robot tasks in natural language
6. [Show PyBullet] Simulates realistic robot execution

[Open HTML report]

The system generates comprehensive metrics and visualizations automatically.
This demonstrates end-to-end AI system integration from perception to action."
```

## Advanced Demo Features

### **Custom Video Analysis**
```bash
# Use your own manipulation video
python demo.py --video my_robot_demo.mp4 --gui
```

### **Thunder Compute Integration**
```bash
# Offload heavy processing to remote compute
python demo.py --thunder --name "large_scale_demo"
```

### **Batch Processing**
```bash
# Process multiple videos for comparison
for video in videos/*.mp4; do
    python demo.py --video "$video" --name "batch_$(basename $video .mp4)"
done
```

## Portfolio Integration Tips

### **1. GitHub Repository**
- Include the generated HTML reports in your repo
- Add performance metrics as badges
- Include sample output videos

### **2. Portfolio Website**
- Embed the HTML reports as iframe
- Add interactive demos with sample videos
- Include before/after comparisons

### **3. Resume/CV Points**
- "Built end-to-end robotic scene understanding pipeline with 85%+ accuracy"
- "Integrated YOLO, SAM, COLMAP, and LLM models in production system"  
- "Designed automated evaluation framework with comprehensive reporting"

### **4. Interview Preparation**
- Practice running the demo smoothly (2-3 minutes)
- Prepare to explain each component's purpose
- Be ready to discuss technical challenges and solutions
- Know your performance metrics

## Troubleshooting

### **Common Issues**
```bash
# Missing dependencies
pip install -r requirements.txt

# SAM model download
# Models will auto-download on first run

# PyBullet GUI issues on macOS
# Install XQuartz if needed

# COLMAP not found
brew install colmap  # macOS
```

### **Performance Optimization**
```bash
# Faster demo (reduced quality)
python demo.py --gui --name "quick_demo"  # Uses vit_b SAM model

# High quality (slower)
# Edit config.py to use vit_h SAM model
```

## Next Steps

1. **Run the demo** with your own videos
2. **Customize the pipeline** for your specific use cases
3. **Add to your portfolio** with the generated reports
4. **Practice presenting** the system fluently
5. **Extend the system** with additional capabilities

---

**🚀 Ready to showcase your robotics and AI expertise!**

The TANGRAM demo provides concrete evidence of your ability to:
- Build complex AI systems
- Integrate multiple technologies  
- Design professional software
- Generate actionable results

Perfect for landing robotics, AI, or senior software engineering roles! 🎯