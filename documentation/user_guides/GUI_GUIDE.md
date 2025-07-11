# TANGRAM GUI - Complete User Guide

## 🎯 Overview

The TANGRAM GUI provides a comprehensive visual interface for the robotic scene understanding pipeline, featuring:

- **Interactive video processing** with real-time overlays
- **Integrated PyBullet simulation** with robot visualization  
- **Pipeline monitoring** with progress tracking and metrics
- **Results visualization** and professional export capabilities
- **Thunder Compute integration** for scalable processing

## 🚀 Quick Start

### Launch the GUI

```bash
# Simple launch
python launch_gui.py

# Or directly
python gui/tangram_gui.py
```

### Basic Workflow

1. **Open Video**: File → Open Video... or click 📁 Open Video
2. **Run Pipeline**: Click ▶️ Run Pipeline or Pipeline → Run Complete Pipeline
3. **Monitor Progress**: Watch the progress bar and step status
4. **View Simulation**: Switch to Simulation tab to see robot in action
5. **Export Results**: Results tab → Export HTML Report

## 🖥️ Interface Overview

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Menu Bar: File | Pipeline | View | Help                        │
├─────────────────────────────────────────────────────────────────┤
│ Toolbar: [📁 Open] [▶️ Run] [⏸️ Pause] [⏹️ Stop] [⚡Thunder] │
├─────────────────┬───────────────────────────────────────────────┤
│                 │ Pipeline Status Tab                           │
│ Video Display   │ ┌─ Steps: 🎯 Tracking ✅                     │
│ Area            │ │  🎨 Segmentation 🔄                        │
│                 │ │  📐 3D Reconstruction ⏳                   │
│ [Video Controls]│ └─ Real-time Metrics                          │
│                 ├───────────────────────────────────────────────┤
│                 │ Simulation Tab                                │
│                 │ ┌─ PyBullet Controls                          │
│                 │ │  [▶️ Start] [📷 Views] [🎯 Follow]         │
│                 │ └─ Simulation Status & Log                    │
├─────────────────┼───────────────────────────────────────────────┤
│ Status Bar      │ Configuration | Results | Logs              │
└─────────────────┴───────────────────────────────────────────────┘
```

## 📊 Features Guide

### 1. Video Processing

**Supported Formats**: MP4, AVI, MOV, MKV, FLV

**Features**:
- Live video preview with processing overlays
- Frame-by-frame navigation
- Real-time object detection visualization
- Segmentation mask overlays

**Controls**:
- Video position slider for seeking
- Overlay toggles in Configuration tab
- Real-time processing option

### 2. Pipeline Monitoring

**Real-time Status**:
- ✅ Completed steps
- 🔄 Currently processing  
- ⏳ Pending steps
- ❌ Failed steps

**Live Metrics**:
- Frame processing rate
- Object detection counts
- Memory usage monitoring
- GPU availability status

**Progress Tracking**:
- Overall pipeline progress bar
- Individual step completion status
- Processing time estimates

### 3. PyBullet Simulation

**Robot Support**:
- Franka Panda (preferred)
- KUKA iiwa (fallback)
- R2D2 (simple fallback)

**Simulation Features**:
- Real-time physics simulation
- Robot task execution visualization
- Object manipulation display
- Camera view controls

**Controls**:
- ▶️ Start/Stop simulation
- 📷 Camera views (Top, Side, Follow Robot)
- 🔄 Reset simulation
- Task execution monitoring

### 4. Configuration Options

**Model Selection**:
- YOLO models: n/s/m/l variants
- SAM models: ViT-B/H/L options
- Thunder Compute toggle

**Processing Options**:
- Show tracking overlays
- Show segmentation masks  
- Real-time processing mode
- Log level selection

### 5. Results & Export

**Visualization**:
- Pipeline results summary
- Scene graph JSON viewer
- Detailed metrics display
- Simulation success rates

**Export Options**:
- 📊 HTML Report: Complete interactive report
- 📹 Video Export: Annotated video (coming soon)
- 📋 JSON Export: Raw pipeline data
- 💾 Log Export: Debug information

## 🔧 Advanced Features

### Thunder Compute Integration

**Enable Thunder Compute**:
1. Check ⚡ Thunder Compute in toolbar
2. Configure connection in `config.py`
3. GUI automatically uses remote processing for heavy tasks

**Benefits**:
- Remote PyBullet simulation (solves ARM Mac issues)
- GPU-accelerated SAM segmentation
- Scalable processing for large videos

### Scene Graph Visualization

**Access**: View → Show Scene Graph

**Features**:
- JSON structure display
- Node and edge relationships
- Object properties and connections
- Exportable format

### Simulation Viewer

**Advanced Controls**:
- Camera follow modes
- Real-time physics monitoring
- Task execution visualization
- Multi-robot support (future)

**Status Information**:
- Simulation FPS
- Robot joint status
- Object positions
- Task success rates

## 🎯 Use Cases

### 1. Portfolio Demonstration

**Workflow**:
1. Load demo video (tabletop manipulation)
2. Run complete pipeline
3. Show real-time processing
4. Demonstrate robot simulation
5. Export professional HTML report

**Key Highlights**:
- Real-time computer vision processing
- Seamless cloud integration (Thunder)
- Professional robotics simulation
- Comprehensive results export

### 2. Research & Development

**Features**:
- Individual pipeline step execution
- Detailed logging and debugging
- Parameter tuning interface
- Results comparison tools

### 3. Educational Use

**Benefits**:
- Visual learning of robotics pipeline
- Interactive exploration of concepts
- Real-time feedback
- Professional tooling experience

## ⚠️ Troubleshooting

### Common Issues

**GUI Won't Start**:
```bash
# Check dependencies
python launch_gui.py

# Install missing packages
pip install -r requirements.txt
```

**PyBullet Simulation Issues**:
- If local PyBullet unavailable: Enable Thunder Compute
- Simulation window not showing: Check View → Show Simulation
- Performance issues: Use Thunder Compute for better GPU resources

**Video Loading Problems**:
- Supported formats: MP4, AVI, MOV, MKV, FLV
- Check file permissions
- Try sample videos in `data/inputs/samples/`

**Pipeline Errors**:
- Check Logs tab for detailed error information
- Verify video file integrity
- Ensure sufficient disk space for processing

### Performance Optimization

**For Better Performance**:
1. Enable Thunder Compute for heavy processing
2. Use smaller video files for testing
3. Reduce video resolution if needed
4. Close other GPU-intensive applications

**Memory Management**:
- Monitor memory usage in real-time metrics
- Clear logs periodically
- Reset pipeline between runs for clean state

## 🚀 Future Enhancements

**Planned Features**:
- Real-time video streaming input
- Multiple robot simulation
- Custom task definition interface
- Advanced visualization options
- Collaborative features
- Mobile app companion

**Integration Roadmap**:
- ROS integration
- Isaac Sim support
- Custom model training interface
- Cloud deployment options

## 📝 Tips & Best Practices

### Optimal Workflow

1. **Start Small**: Use short videos for initial testing
2. **Monitor Resources**: Watch memory/GPU usage
3. **Save Frequently**: Export results after successful runs
4. **Use Thunder**: Enable for production-quality processing
5. **Document Results**: Use HTML export for professional reports

### Performance Tips

- **Video Preparation**: Use 30-second clips for demos
- **Thunder Usage**: Enable for complex scenes or ARM Mac users
- **Resource Monitoring**: Check real-time metrics regularly
- **Clean Workspace**: Reset pipeline between different videos

The TANGRAM GUI provides a complete, professional interface for robotic scene understanding - perfect for demos, research, and educational use!