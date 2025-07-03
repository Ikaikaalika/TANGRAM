#!/bin/bash
# TANGRAM Quick Launch Script

set -e

echo "🚀 TANGRAM - AI-Powered Robotic Scene Understanding"
echo "=================================================="

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "tangram" ]]; then
    echo "⚠️  Please activate the tangram conda environment first:"
    echo "   conda activate tangram"
    exit 1
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🤖 Starting Ollama service..."
    ollama serve &
    sleep 3
fi

# Check if DeepSeek model is available
if ! ollama list | grep -q "deepseek-r1:7b"; then
    echo "📥 Downloading DeepSeek R1 7B model (this may take a while)..."
    ollama pull deepseek-r1:7b
fi

echo "✅ All services ready"
echo ""
echo "Choose an option:"
echo "1) Launch Interactive GUI"
echo "2) Process a video file"
echo "3) Run demonstration"
echo "4) Run pipeline with existing video"
echo ""

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🖥️  Launching GUI..."
        python tangram.py gui
        ;;
    2)
        read -p "Enter video file path: " video_path
        echo "🎥 Processing video..."
        python tangram.py process "$video_path"
        ;;
    3)
        echo "🎬 Creating demonstration..."
        python tangram.py demo
        ;;
    4)
        echo "⚙️  Running full pipeline..."
        python main.py
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo "✅ TANGRAM operation complete"