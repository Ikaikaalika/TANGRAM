#!/usr/bin/env python3

import argparse
import sys
import os

# Add project modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracker.track_objects import ObjectTracker
from segmenter.run_sam import SAMSegmenter
from scene_graph.build_graph import SceneGraphBuilder
from robotics.simulation_env import RoboticsSimulation
from llm.interpret_scene import SceneInterpreter
from visualization.render_graph import GraphVisualizer

def main():
    parser = argparse.ArgumentParser(description="Robotic Scene Understanding Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--mode", choices=["track", "segment", "reconstruct", "graph", "simulate"], 
                       default="track", help="Processing mode")
    parser.add_argument("--gui", action="store_true", help="Show GUI for simulation")
    
    args = parser.parse_args()
    
    print(f"Processing {args.input} in {args.mode} mode")
    print(f"Output directory: {args.output}")
    
    if args.mode == "track":
        tracker = ObjectTracker()
        print("Running object tracking...")
    
    elif args.mode == "segment":
        segmenter = SAMSegmenter()
        print("Running segmentation...")
    
    elif args.mode == "graph":
        graph_builder = SceneGraphBuilder()
        print("Building scene graph...")
    
    elif args.mode == "simulate":
        sim = RoboticsSimulation(gui=args.gui)
        print("Starting simulation...")
    
    else:
        print("Running full pipeline...")
        # Integrate all modules
        pass

if __name__ == "__main__":
    main()