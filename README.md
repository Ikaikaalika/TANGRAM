# TANGRAM: Robotic Scene Understanding Pipeline

A comprehensive system for video-based robotic scene understanding, combining object tracking, segmentation, 3D reconstruction, and task planning.

## Project Structure

```
project/
├── data/                 # All data storage
│   ├── raw_videos/       # Input video files
│   ├── frames/          # Extracted video frames
│   ├── tracking/        # Object tracking results
│   ├── masks/           # Segmentation masks
│   ├── 3d_points/       # 3D reconstruction data
│   ├── graphs/          # Scene graph representations
│   └── simulation/      # Simulation assets
├── tracker/             # Object tracking module
├── segmenter/           # SAM segmentation
├── reconstruction/      # COLMAP 3D reconstruction
├── scene_graph/         # Graph building and analysis
├── robotics/            # Simulation and planning
├── llm/                 # LLM scene interpretation
├── visualization/       # Rendering and visualization
└── main.py             # Main pipeline entry point
```

## Setup

1. **Environment Setup**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Data Acquisition**
- Place video files in `data/raw_videos/`
- Supported datasets: Ego4D, EPIC-KITCHENS, custom recordings

3. **Usage**
```bash
# Basic tracking
python main.py --input data/raw_videos/scene.mp4 --mode track

# Full pipeline
python main.py --input data/raw_videos/scene.mp4 --output results/

# Simulation mode
python main.py --mode simulate --gui
```

## Hardware Requirements

- **M1 Mac**: PyTorch with MPS support for GPU acceleration
- **Memory**: 16GB+ RAM recommended for large datasets
- **Storage**: 50GB+ for datasets and intermediate results

## External Dependencies

- **COLMAP**: Install via `brew install colmap` (macOS) or build from source
- **Thunder Compute**: SSH access for heavy inference workloads

## Modules

- **Tracker**: Multi-object tracking using CSRT
- **Segmenter**: SAM-based instance segmentation
- **Reconstruction**: COLMAP-based 3D scene reconstruction
- **Scene Graph**: Spatial relationship modeling
- **Robotics**: PyBullet simulation environment
- **LLM**: OpenAI integration for task planning
- **Visualization**: NetworkX and Matplotlib rendering