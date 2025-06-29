# TANGRAM Pipeline Manager GUI

A comprehensive graphical interface for managing TANGRAM training, simulation, and Thunder Compute resources.

## Features

### 🌩️ Enhanced Thunder Compute Management
- **Visual Instance Creation**: Interactive dialog with sliders, buttons, and real-time feedback
- **Smart Presets**: One-click Budget, Balanced, and High-End configurations
- **Cost Estimation**: Real-time pricing updates as you modify settings
- **GPU Selection Cards**: Visual cards showing T4, A100, and A100XL options with VRAM specs
- **vCPU Slider**: Smooth slider control with automatic RAM calculation (8GB per vCPU)
- **Mode Toggle**: Easy switching between Prototyping and Production modes
- **Template Support**: Pre-configured software stacks (Ollama, ComfyUI, WebUI Forge)
- **Instance Control**: Start, stop, delete instances with confirmation dialogs

### 🚀 Pipeline Execution
- **Visual Pipeline Control**: Run complete TANGRAM pipeline or individual components
- **Real-time Monitoring**: Live output log and progress tracking
- **Thunder Integration**: Automatic offloading of heavy tasks to Thunder Compute
- **Flexible Configuration**: Customize goals, enable GUI simulation, select processing modes

### 📊 Training & Simulation
- **Interactive Monitoring**: Real-time pipeline execution status
- **Result Management**: Easy access to pipeline outputs and exports
- **Simulation Control**: Enable/disable PyBullet GUI for robot simulation
- **Goal Customization**: Define custom task planning goals

## Quick Start

### 1. Launch the GUI
```bash
python launch_gui.py
```

### 2. Set up Thunder Compute (Enhanced Interface)
1. Go to the "Thunder Compute" tab
2. Click "Create Instance" to open the enhanced creation dialog
3. **Choose a Quick Preset**:
   - **Budget**: T4 GPU, 4 vCPUs, Prototyping (~$0.50/hour)
   - **Balanced**: A100 GPU, 8 vCPUs, Prototyping (~$2.10/hour)  
   - **High-End**: A100XL GPU, 16 vCPUs, Production (~$4.60/hour)
4. **Or Customize Manually**:
   - Select GPU type using visual cards with VRAM specifications
   - Adjust vCPUs with smooth slider (4-32, auto-calculates RAM)
   - Toggle between Prototyping/Production modes
   - Choose optional software templates
5. **Monitor Real-time Cost** estimate as you adjust settings
6. Click "Create Instance" and wait for green status indicator

### 3. Run a Pipeline
1. Switch to the "Pipeline Execution" tab
2. Click "Browse" to select your input video
3. Choose pipeline mode:
   - **full**: Complete pipeline (tracking → segmentation → 3D → scene graph → LLM → simulation)
   - **track**: Object tracking only
   - **segment**: Tracking + segmentation
   - **reconstruct**: Up to 3D reconstruction
   - **graph**: Up to scene graph generation
   - **llm**: Up to LLM interpretation
   - **simulate**: Complete pipeline with simulation
4. Set your goal (e.g., "Clear the table", "Organize objects")
5. Enable Thunder Compute for heavy processing tasks
6. Click "Start Pipeline"

## Interface Overview

### Thunder Compute Tab
- **Status Indicator**: Shows connection status to Thunder Compute
- **Instance Table**: Lists all instances with their current status
- **Control Buttons**: 
  - Refresh: Update instance status
  - Create Instance: Launch new compute instance
  - Start/Stop: Control instance power state
  - Delete: Permanently remove instance

### Pipeline Execution Tab
- **Input Selection**: Browse for video files
- **Mode Selection**: Choose pipeline components to run
- **Options**:
  - Use Thunder Compute: Enable cloud processing
  - Show Simulation GUI: Display PyBullet simulation window
- **Goal Input**: Define the task for LLM planning
- **Progress Bar**: Shows pipeline execution progress
- **Output Log**: Real-time pipeline output and status messages

## Thunder Compute Integration

The GUI automatically manages Thunder Compute for optimal performance:

### Automatic Offloading
- **SAM Segmentation**: For videos >100MB, >200 frames, or >500 detections
- **COLMAP Reconstruction**: For frame sets >50 images
- **PyBullet Simulation**: All simulations for consistent performance

### Cost Management
- Instances are created on-demand when heavy processing is needed
- Use "Stop" button to pause instances and save costs
- Delete unused instances to avoid storage charges

### Performance Benefits
- GPU acceleration for AI models (SAM, YOLO)
- Faster COLMAP reconstruction on high-end CPUs
- Consistent simulation results across different local hardware

## Usage Tips

### 📹 Video Input
- Supported formats: MP4, AVI, MOV, MKV
- Place videos in `data/raw_videos/` for easy access
- Larger videos benefit more from Thunder Compute acceleration

### ⚡ Performance Optimization
- Enable Thunder Compute for videos larger than 100MB
- Use local processing for quick tests and small videos
- Monitor the output log for performance insights

### 💰 Cost Control
- Create instances only when needed
- Stop instances when not actively processing
- Use T4 GPUs for most workloads (A100 for very large models)

### 🔧 Troubleshooting
- Check Thunder Compute status in the status bar
- View logs via the Tools menu
- Ensure input videos exist and are readable
- Verify Thunder CLI is authenticated (`tnr status`)

## Menu Options

### File Menu
- **Open Video**: Quick video selection for pipeline
- **Exit**: Close the application

### Tools Menu
- **Configuration**: Edit TANGRAM settings (coming soon)
- **View Logs**: Open logs directory
- **Thunder CLI**: Open terminal with Thunder CLI

### Help Menu
- **About**: Application information and version

## Keyboard Shortcuts

- `Cmd+O` (macOS) / `Ctrl+O` (Windows/Linux): Open video file
- `Cmd+Q` (macOS) / `Ctrl+Q` (Windows/Linux): Quit application

## System Requirements

- Python 3.8+ with tkinter support
- Thunder Compute CLI (`tnr`) installed and authenticated
- TANGRAM pipeline dependencies (see main README)

## Troubleshooting

### GUI Won't Start
```bash
# Check tkinter availability
python -c "import tkinter; print('tkinter available')"

# On macOS: Should work with system Python
# On Ubuntu/Debian:
sudo apt-get install python3-tk

# On CentOS/RHEL:
sudo yum install tkinter
```

### Thunder Compute Issues
```bash
# Check CLI installation
tnr status

# Re-authenticate if needed
tnr login
```

### Pipeline Failures
- Check the output log in the GUI for detailed error messages
- Ensure input video file exists and is accessible
- Verify all dependencies are installed (see main README)
- Check available disk space for output files

## Advanced Usage

### Batch Processing
- Use the GUI to set up and test pipeline parameters
- Then use the command line for batch processing multiple videos

### Custom Goals
- Experiment with different task planning goals:
  - "Organize objects by color"
  - "Stack all books on the shelf"
  - "Clear the workspace completely"
  - "Group similar objects together"

### Development Mode
- Enable simulation GUI to watch robot behavior
- Monitor output logs for debugging pipeline components
- Use individual pipeline modes to test specific components

---

For more information, see the main project README and documentation.