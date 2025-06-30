#!/bin/bash
"""
Thunder Compute Setup Script

This script prepares a Thunder Compute instance for TANGRAM processing.
Run this on your Thunder Compute instance to install required dependencies.

Usage:
    # On Thunder Compute instance:
    curl -sSL https://raw.githubusercontent.com/your-repo/TANGRAM/main/scripts/setup_thunder.sh | bash
    
    # Or manually:
    chmod +x setup_thunder.sh
    ./setup_thunder.sh
"""

set -e  # Exit on any error

echo "🌩️  Setting up Thunder Compute for TANGRAM..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0

# Install COLMAP
echo "📐 Installing COLMAP..."
sudo apt-get install -y colmap

# Install Python packages
echo "🐍 Installing Python packages..."
pip3 install --upgrade pip

# Core ML packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Computer vision packages
pip3 install opencv-python opencv-contrib-python

# SAM dependencies  
pip3 install git+https://github.com/facebookresearch/segment-anything.git

# Other ML packages
pip3 install \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    pillow \
    tqdm

# Verify GPU access
echo "🔍 Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Download SAM models
echo "📥 Downloading SAM models..."
mkdir -p ~/models/sam
cd ~/models/sam

# Download SAM checkpoints
wget -O sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget -O sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "✅ SAM models downloaded"

# Create working directories
echo "📁 Creating working directories..."
mkdir -p ~/tangram_workspace/{data,logs,results}

# Set up environment
echo "🌍 Setting up environment..."
echo 'export TANGRAM_MODELS_DIR=~/models' >> ~/.bashrc
echo 'export TANGRAM_WORKSPACE=~/tangram_workspace' >> ~/.bashrc

# Test COLMAP
echo "🧪 Testing COLMAP installation..."
colmap --help > /dev/null && echo "✅ COLMAP installed successfully" || echo "❌ COLMAP installation failed"

# Test PyTorch GPU
echo "🧪 Testing PyTorch GPU access..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ No GPU detected')
"

# Create system info script
echo "📊 Creating system info script..."
cat > ~/system_info.py << 'EOF'
#!/usr/bin/env python3
import torch
import cv2
import subprocess
import psutil

print("=== Thunder Compute System Info ===")
print(f"CPU cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'None'}")
print(f"OpenCV: {cv2.__version__}")

try:
    result = subprocess.run(['colmap', '--help'], capture_output=True, text=True)
    print("COLMAP: ✅ Installed")
except:
    print("COLMAP: ❌ Not found")

print("=== Ready for TANGRAM processing! ===")
EOF

chmod +x ~/system_info.py

echo ""
echo "🎉 Thunder Compute setup completed!"
echo ""
echo "📋 Summary:"
echo "  ✅ System packages updated"
echo "  ✅ COLMAP installed for 3D reconstruction"
echo "  ✅ PyTorch with CUDA support"
echo "  ✅ SAM models downloaded"
echo "  ✅ OpenCV for video processing"
echo "  ✅ Working directories created"
echo ""
echo "🔍 Run system info:"
echo "  python3 ~/system_info.py"
echo ""
echo "🚀 Your Thunder Compute instance is ready for TANGRAM!"
echo ""
echo "💡 Next steps:"
echo "  1. Update your local config.py with this instance details"
echo "  2. Test connection: python thunder/thunder_client.py"  
echo "  3. Run demo: python demo_thunder.py --thunder"