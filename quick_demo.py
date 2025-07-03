#!/usr/bin/env python3
"""
TANGRAM Quick Demo - Working Demo for M1 Mac

This creates a simple working demo that shows all components working together.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import LLM_CONFIG
from src.tangram.core.llm.local_llm_client import LocalLLMClient

def test_local_llm():
    """Test local LLM is working"""
    print("🧠 Testing Local DeepSeek LLM...")
    
    try:
        llm = LocalLLMClient()
        prompt = "Analyze this robotic scene: A red cup is on a table next to a blue block. Generate one simple pick and place task."
        
        print(f"   📝 Prompt: {prompt}")
        print("   🤔 Thinking...")
        
        response = llm.generate_response(prompt)
        print(f"   ✅ Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_physics_sim():
    """Test PyBullet physics simulation"""
    print("\n🎮 Testing PyBullet Physics Simulation...")
    
    try:
        import pybullet as p
        
        # Start physics simulation
        physics_client = p.connect(p.DIRECT)  # No GUI for demo
        p.setGravity(0, 0, -9.81)
        
        # Create ground plane
        plane_id = p.loadURDF("plane.urdf")
        
        # Create a simple cube
        cube_start_pos = [0, 0, 1]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orientation)
        
        # Run simulation for a few steps
        for i in range(100):
            p.stepSimulation()
        
        # Get final position
        pos, orn = p.getBasePositionAndOrientation(cube_id)
        print(f"   ✅ Cube dropped from {cube_start_pos} to {[round(x, 2) for x in pos]}")
        
        p.disconnect()
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_computer_vision():
    """Test computer vision components"""
    print("\n👁️  Testing Computer Vision Components...")
    
    try:
        # Test YOLO
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("   ✅ YOLO model loaded")
        
        # Test SAM (just import, don't load heavy model)
        import segment_anything
        print("   ✅ SAM available")
        
        # Test OpenCV
        import cv2
        print("   ✅ OpenCV available")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run the quick demo"""
    print("🚀 TANGRAM Quick Demo - M1 Mac")
    print("=" * 50)
    
    results = []
    
    # Test all components
    results.append(test_computer_vision())
    results.append(test_physics_sim())
    results.append(test_local_llm())
    
    print("\n🎯 Demo Results:")
    print("=" * 50)
    
    if all(results):
        print("✅ ALL SYSTEMS WORKING!")
        print("🎉 TANGRAM is ready for full demos!")
        print("\nNext steps:")
        print("- Run: python launch_gui.py (for GUI)")
        print("- Run: python main.py --input video.mp4 --mode full")
    else:
        print("⚠️  Some components need attention")
        failed = sum(1 for r in results if not r)
        print(f"   {len(results) - failed}/{len(results)} components working")

if __name__ == "__main__":
    main()