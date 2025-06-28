#!/usr/bin/env python3
"""
TANGRAM Thunder Compute PyBullet Demo

This script demonstrates running PyBullet robot simulation on Thunder Compute
while keeping other pipeline components local for optimal performance.

Usage:
    python demo_thunder_pybullet.py [--video path/to/video.mp4] [--thunder-host HOST]

Author: TANGRAM Team
License: MIT
"""

import argparse
import sys
import time
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from utils.logging_utils import setup_logger
from thunder.thunder_integration import ThunderIntegratedDemo

logger = setup_logger(__name__, "thunder_pybullet_demo.log")

def main():
    """Main Thunder PyBullet demo entry point."""
    parser = argparse.ArgumentParser(description="TANGRAM Thunder PyBullet Demo")
    parser.add_argument("--video", "-v", help="Path to input video file")
    parser.add_argument("--thunder-host", help="Thunder Compute host override")
    parser.add_argument("--name", "-n", help="Experiment name for this demo")
    
    args = parser.parse_args()
    
    print("""
    🤖⚡ TANGRAM Thunder Compute PyBullet Demo
    
    This demo showcases:
    • Local video processing and scene understanding
    • Remote PyBullet simulation on Thunder Compute
    • Seamless integration and fallback handling
    • Professional robotics simulation results
    
    Perfect for demonstrating scalable robotics pipelines! 🎯
    """)
    
    # Override Thunder host if specified
    if args.thunder_host:
        HARDWARE_CONFIG["thunder_compute"]["ssh_host"] = args.thunder_host
        print(f"Using Thunder host: {args.thunder_host}")
    
    # Create Thunder-integrated demo
    experiment_name = args.name or f"thunder_pybullet_{int(time.time())}"
    demo = ThunderIntegratedDemo(experiment_name)
    
    # Test Thunder connection first
    if demo.simulator.use_thunder:
        print("🔗 Testing Thunder Compute connection...")
        if demo.simulator.thunder_client.connect():
            print("✅ Thunder Compute connection successful!")
            demo.simulator.thunder_client.disconnect()
        else:
            print("⚠️  Thunder Compute connection failed - will use local simulation")
    else:
        print("ℹ️  Thunder Compute disabled - using local simulation")
    
    # Run demo pipeline
    video_path = args.video or str(SAMPLE_VIDEOS_DIR / "tabletop_manipulation.mp4")
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        print("Please provide a valid video file with --video option")
        return 1
    
    print(f"\n🎬 Processing video: {video_path}")
    
    start_time = time.time()
    
    try:
        # Run Thunder-optimized pipeline
        results = demo.run_thunder_optimized_pipeline(video_path)
        
        # Additional demonstration: Direct PyBullet simulation
        print("\n🤖 Running dedicated PyBullet simulation demonstration...")
        
        # Create mock scene graph and LLM interpretation for demo
        scene_graph_data = create_demo_scene_graph()
        llm_interpretation_data = create_demo_llm_interpretation()
        
        # Save demo data
        demo_dir = Path("demo_data")
        demo_dir.mkdir(exist_ok=True)
        
        scene_graph_file = demo_dir / "demo_scene_graph.json"
        llm_file = demo_dir / "demo_llm_interpretation.json"
        
        import json
        with open(scene_graph_file, 'w') as f:
            json.dump(scene_graph_data, f, indent=2)
        with open(llm_file, 'w') as f:
            json.dump(llm_interpretation_data, f, indent=2)
        
        # Run PyBullet simulation with Thunder
        simulation_results = demo.simulator.execute_task_sequence_with_thunder(
            str(scene_graph_file),
            str(llm_file),
            "data/thunder_simulation"
        )
        
        # Display results
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("THUNDER PYBULLET DEMO RESULTS")
        print("="*60)
        print(f"Experiment: {experiment_name}")
        print(f"Total Time: {total_time:.1f} seconds")
        
        if simulation_results:
            print(f"Robot: {simulation_results.get('robot_info', {}).get('name', 'Unknown')}")
            print(f"Tasks Executed: {simulation_results.get('total_tasks', 0)}")
            print(f"Success Rate: {simulation_results.get('success_rate', 0):.1%}")
            
            mode = simulation_results.get('simulation_mode', 'unknown')
            if mode == 'mock':
                print("🔧 Simulation Mode: Mock (Thunder/PyBullet unavailable)")
            else:
                print("⚡ Simulation Mode: Thunder Compute PyBullet")
        
        print("\nKey Demonstrations:")
        print("✅ Automatic Thunder Compute detection")
        print("✅ Remote PyBullet simulation execution")
        print("✅ Seamless fallback to local simulation")
        print("✅ Professional results export")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("Check the logs for detailed error information")
        return 1

def create_demo_scene_graph():
    """Create demo scene graph data."""
    return {
        "nodes": [
            {
                "id": "obj_1",
                "type": "object",
                "properties": {
                    "class_name": "cup",
                    "x": 0.2,
                    "y": 0.1,
                    "z": 0.75,
                    "confidence": 0.95
                }
            },
            {
                "id": "obj_2", 
                "type": "object",
                "properties": {
                    "class_name": "book",
                    "x": -0.1,
                    "y": 0.2,
                    "z": 0.72,
                    "confidence": 0.88
                }
            },
            {
                "id": "obj_3",
                "type": "object", 
                "properties": {
                    "class_name": "bottle",
                    "x": 0.0,
                    "y": -0.15,
                    "z": 0.78,
                    "confidence": 0.92
                }
            },
            {
                "id": "context_1",
                "type": "scene_context",
                "properties": {
                    "location": "table_surface",
                    "description": "Objects on table ready for manipulation"
                }
            }
        ],
        "edges": [
            {
                "source": "obj_1",
                "target": "context_1",
                "relation": "on"
            },
            {
                "source": "obj_2", 
                "target": "context_1",
                "relation": "on"
            },
            {
                "source": "obj_3",
                "target": "context_1", 
                "relation": "on"
            }
        ],
        "metadata": {
            "num_objects": 3,
            "scene_type": "tabletop_manipulation",
            "generated_at": time.time()
        }
    }

def create_demo_llm_interpretation():
    """Create demo LLM interpretation data."""
    return {
        "scene_explanation": "The scene contains three objects on a table: a cup, a book, and a bottle. These objects are suitable for robotic manipulation tasks such as grasping, moving, and organizing.",
        "goal": "Organize the objects by moving them to designated locations",
        "task_sequence": [
            {
                "id": "task_1",
                "type": "grasp",
                "description": "Pick up the cup from the table",
                "target_object": "obj_1",
                "estimated_duration": 3.0,
                "parameters": {
                    "approach_angle": "top_down",
                    "grip_force": "light"
                }
            },
            {
                "id": "task_2", 
                "type": "place",
                "description": "Place the cup in the designated area",
                "target_object": "obj_1",
                "estimated_duration": 2.5,
                "parameters": {
                    "target_position": [0.3, 0.3, 0.75],
                    "placement_type": "gentle"
                }
            },
            {
                "id": "task_3",
                "type": "push",
                "description": "Push the book to align it properly",
                "target_object": "obj_2", 
                "estimated_duration": 2.0,
                "parameters": {
                    "push_direction": [0.1, 0, 0],
                    "push_force": "medium"
                }
            },
            {
                "id": "task_4",
                "type": "grasp",
                "description": "Pick up the bottle",
                "target_object": "obj_3",
                "estimated_duration": 3.5,
                "parameters": {
                    "approach_angle": "side_grasp",
                    "grip_force": "medium"
                }
            },
            {
                "id": "task_5",
                "type": "place",
                "description": "Place the bottle in storage area", 
                "target_object": "obj_3",
                "estimated_duration": 3.0,
                "parameters": {
                    "target_position": [-0.3, 0.2, 0.75],
                    "placement_type": "careful"
                }
            }
        ],
        "confidence": 0.87,
        "reasoning": "Tasks are ordered to minimize conflicts and maximize efficiency. Lighter objects are moved first to avoid collisions.",
        "generated_at": time.time()
    }

if __name__ == "__main__":
    sys.exit(main())