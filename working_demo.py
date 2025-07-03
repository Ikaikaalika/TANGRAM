#!/usr/bin/env python3
"""
TANGRAM Working Demo - M1 Mac Ready

Demonstrates all working components without complex dependencies.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_pytorch_mps():
    """Test PyTorch with M1 GPU acceleration"""
    print("🔥 Testing PyTorch with M1 GPU (MPS)...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   ✅ MPS Available: {torch.backends.mps.is_available()}")
        
        # Test tensor operations on MPS
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print(f"   ✅ M1 GPU computation successful: {z.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_computer_vision():
    """Test computer vision stack"""
    print("\n👁️  Testing Computer Vision Stack...")
    
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        print(f"   ✅ OpenCV {cv2.__version__}")
        print(f"   ✅ NumPy {np.__version__}")
        
        # Create a simple test image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 320), 100, (0, 255, 0), -1)
        
        # Test YOLO (lightweight check)
        model = YOLO('yolov8n.pt')
        print("   ✅ YOLO model loaded")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_physics():
    """Test PyBullet physics"""
    print("\n🎮 Testing PyBullet Physics...")
    
    try:
        import pybullet as p
        
        # Simple direct mode test
        physics_client = p.connect(p.DIRECT)
        print("   ✅ PyBullet connected")
        
        # Test basic physics
        p.setGravity(0, 0, -9.81)
        print("   ✅ Gravity set")
        
        p.disconnect()
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_llm():
    """Test local LLM connection"""
    print("\n🧠 Testing Local DeepSeek LLM...")
    
    try:
        import requests
        
        # Test if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            deepseek_models = [m for m in models if 'deepseek' in m.get('name', '').lower()]
            if deepseek_models:
                print(f"   ✅ DeepSeek models available: {len(deepseek_models)}")
                for model in deepseek_models:
                    print(f"      - {model['name']}")
                return True
            else:
                print("   ⚠️  Ollama running but no DeepSeek models found")
                return False
        else:
            print("   ⚠️  Ollama not responding")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_data_processing():
    """Test data processing components"""
    print("\n📊 Testing Data Processing...")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import networkx as nx
        
        print(f"   ✅ Pandas {pd.__version__}")
        print(f"   ✅ Matplotlib available")
        print(f"   ✅ NetworkX {nx.__version__}")
        
        # Test creating a simple graph
        G = nx.Graph()
        G.add_edge("robot", "cup")
        G.add_edge("cup", "table")
        print(f"   ✅ Scene graph created: {len(G.nodes)} nodes")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run the working demo"""
    print("🚀 TANGRAM Working Demo - M1 Mac")
    print("=" * 60)
    print("Testing all major components...")
    print()
    
    tests = [
        ("PyTorch + M1 GPU", test_pytorch_mps),
        ("Computer Vision", test_computer_vision),
        ("Physics Simulation", test_physics),
        ("Data Processing", test_data_processing),
        ("Local LLM", test_llm),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
            results.append((name, False))
    
    print("\n🎯 FINAL RESULTS")
    print("=" * 60)
    
    working = 0
    for name, result in results:
        status = "✅ WORKING" if result else "❌ FAILED"
        print(f"{name:20} | {status}")
        if result:
            working += 1
    
    print(f"\n📈 Score: {working}/{len(results)} components working")
    
    if working >= 4:
        print("\n🎉 TANGRAM IS READY!")
        print("📋 What you can do:")
        print("   • Run object detection on videos")
        print("   • Create 3D scene graphs")
        print("   • Simulate robot tasks")
        print("   • Generate LLM interpretations")
        print("   • Export beautiful visualizations")
        print()
        print("🚀 Try: python launch_gui.py")
    else:
        print("\n⚠️  Some components need attention")
        print("   Check the failed components above")

if __name__ == "__main__":
    main()