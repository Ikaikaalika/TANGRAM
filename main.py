#!/usr/bin/env python3
"""
TANGRAM Main Pipeline Controller

This is the main entry point for the TANGRAM robotic scene understanding pipeline.
It orchestrates all components and provides flexible execution modes.

For portfolio demos, use demo.py instead of this file.

Usage:
    python main.py --input video.mp4 --mode full
    python main.py --input video.mp4 --mode track --output results/
    
Author: TANGRAM Team
License: MIT
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add project modules to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (
    FRAME_EXTRACTION_CONFIG, SAM_CONFIG, YOLO_CONFIG, 
    COLMAP_CONFIG, PYBULLET_CONFIG, HARDWARE_CONFIG,
    SCENE_GRAPH_CONFIG, LLM_CONFIG, DATA_DIR, RESULTS_DIR,
    TRACKING_DIR, FRAMES_DIR, GRAPHS_DIR, SIMULATION_DIR, MASKS_DIR
)
from src.tangram.utils.logging_utils import setup_logger, log_pipeline_step
from src.tangram.utils.mock_data import create_mock_3d_positions
from src.tangram.pipeline.perception.tracker.track_objects import YOLOByteTracker
from src.tangram.pipeline.perception.segmenter.run_sam import SAMSegmenter
from src.tangram.pipeline.understanding.reconstruction.triangulate import PointTriangulator
from src.tangram.pipeline.understanding.scene_graph.build_graph import SceneGraphBuilder
from src.tangram.pipeline.execution.robotics.simulation_env import RoboticsSimulation
from src.tangram.pipeline.planning.llm.interpret_scene import create_scene_interpreter
from src.tangram.interfaces.gui.render_graph import GraphVisualizer
from src.tangram.utils.export.results_exporter import ResultsExporter

logger = setup_logger(__name__)

class TANGRAMPipeline:
    """
    Main TANGRAM pipeline controller.
    
    Coordinates execution of all pipeline components with flexible
    mode selection and comprehensive logging.
    """
    
    def __init__(self, input_path: str, output_dir: str = "output", 
                 experiment_name: str = None):
        """
        Initialize pipeline controller.
        
        Args:
            input_path: Path to input video file
            output_dir: Output directory for results
            experiment_name: Name for this pipeline run
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"pipeline_{int(time.time())}"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tracker = YOLOByteTracker()
        self.segmenter = SAMSegmenter()
        self.triangulator = PointTriangulator()
        self.scene_builder = SceneGraphBuilder()
        self.llm_interpreter = create_scene_interpreter()
        self.visualizer = GraphVisualizer()
        self.exporter = ResultsExporter(self.experiment_name)
        
        logger.info(f"Initialized pipeline for: {self.input_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    @log_pipeline_step("Object Detection and Tracking")
    def run_tracking(self):
        """Run object detection and tracking for images or videos."""
        logger.info("Starting object detection/tracking...")
        
        # Use the new auto-detection method that handles both images and videos
        tracking_results = self.tracker.process_media(
            str(self.input_path),
            output_dir=str(TRACKING_DIR)
        )
        
        if not tracking_results:
            logger.error("No detection results produced")
            return None
        
        file_type = "image" if len(tracking_results) == 1 else "video"
        logger.info(f"Detection completed: {len(tracking_results)} frame(s) processed ({file_type})")
        return tracking_results
    
    @log_pipeline_step("Object Segmentation")
    def run_segmentation(self):
        """Run SAM segmentation on tracked objects."""
        logger.info("Starting object segmentation...")
        
        # Load tracking results
        tracking_file = TRACKING_DIR / "tracking_results.json"
        if not tracking_file.exists():
            logger.error("Tracking results not found. Run tracking first.")
            return None
        
        import json
        with open(tracking_file, 'r') as f:
            tracking_results = json.load(f)
        
        segmentation_results = self.segmenter.process_video_with_tracking(
            str(self.input_path),
            tracking_results,
            output_dir=str(MASKS_DIR)
        )
        
        logger.info(f"Segmentation completed: {len(segmentation_results)} frames processed")
        return segmentation_results
    
    @log_pipeline_step("3D Reconstruction")
    def run_reconstruction(self):
        """Run 3D reconstruction pipeline."""
        logger.info("Starting 3D reconstruction...")
        
        # Check if input is image or video
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        is_image = self.input_path.suffix.lower() in image_extensions
        
        if is_image:
            # For single images, copy the image to frames directory
            import shutil
            FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.input_path, FRAMES_DIR / f"frame_0001{self.input_path.suffix}")
            logger.info("Single image copied to frames directory")
            success = True
        else:
            # Extract frames from video
            try:
                from reconstruction.extract_frames import extract_frames
                success = extract_frames(
                    str(self.input_path),
                    str(FRAMES_DIR),
                    frame_interval=5,
                    max_frames=100
                )
            except ImportError:
                logger.warning("Frame extraction module not found, using mock extraction")
                success = True
        
        if not success:
            logger.error("Frame extraction failed")
            return None
        
        # Run COLMAP (would need to be implemented)
        if is_image:
            logger.info("Single image detected - generating simplified 3D positions")
        else:
            logger.info("Note: COLMAP reconstruction should be run separately for videos")
        
        logger.info("Using mock 3D positions for pipeline demonstration")
        
        # For now, create mock positions
        mock_positions = create_mock_3d_positions()
        
        # Save to expected location
        import json
        RECONSTRUCTION_DIR.mkdir(parents=True, exist_ok=True)
        with open(RECONSTRUCTION_DIR / "object_3d_positions.json", 'w') as f:
            json.dump(mock_positions, f, indent=2)
        
        return mock_positions
    
    
    @log_pipeline_step("Scene Graph Construction")
    def build_scene_graph(self):
        """Build scene graph from tracking and 3D data."""
        logger.info("Building scene graph...")
        
        tracking_file = TRACKING_DIR / "tracking_results.json"
        positions_file = RECONSTRUCTION_DIR / "object_3d_positions.json"
        
        if not (tracking_file.exists() and positions_file.exists()):
            logger.error("Required input files not found for scene graph")
            return None
        
        scene_graph = self.scene_builder.build_complete_graph(
            str(tracking_file),
            str(positions_file),
            output_dir=str(GRAPHS_DIR)
        )
        
        # Generate visualization
        self.visualizer.set_graph(scene_graph)
        viz_output = GRAPHS_DIR / "scene_graph_visualization.png"
        self.visualizer.render_2d_graph(str(viz_output))
        
        logger.info("Scene graph construction completed")
        return scene_graph
    
    @log_pipeline_step("LLM Scene Interpretation")
    def run_llm_interpretation(self, goal: str = "Clear the table"):
        """Run LLM interpretation and task planning."""
        logger.info(f"Running LLM interpretation with goal: {goal}")
        
        graph_file = GRAPHS_DIR / "scene_graph.json"
        if not graph_file.exists():
            logger.error("Scene graph not found for LLM interpretation")
            return None
        
        interpretation = self.llm_interpreter.process_scene_graph_file(
            str(graph_file),
            goal=goal,
            output_dir=str(GRAPHS_DIR)
        )
        
        logger.info("LLM interpretation completed")
        return interpretation
    
    @log_pipeline_step("Robot Simulation")
    def run_simulation(self, gui: bool = False):
        """Run robot simulation."""
        logger.info("Starting robot simulation...")
        
        # Load required data
        positions_file = RECONSTRUCTION_DIR / "object_3d_positions.json"
        interpretation_file = GRAPHS_DIR / "llm_interpretation.json"
        
        if not (positions_file.exists() and interpretation_file.exists()):
            logger.error("Required files not found for simulation")
            return None
        
        # Initialize simulation
        sim = RoboticsSimulation(gui=gui)
        
        if not sim.initialize_simulation():
            logger.error("Failed to initialize simulation")
            return None
        
        if not sim.load_robot():
            logger.error("Failed to load robot")
            return None
        
        # Load data
        import json
        with open(positions_file, 'r') as f:
            object_positions = json.load(f)
        
        with open(interpretation_file, 'r') as f:
            interpretation = json.load(f)
        
        # Add objects and run tasks
        object_mapping = sim.add_scene_objects(object_positions)
        task_sequence = interpretation.get("task_sequence", [])
        
        task_results = sim.execute_task_sequence(task_sequence, object_mapping)
        
        # Save results
        output_file = SIMULATION_DIR / "simulation_results.json"
        sim.save_simulation_results(str(output_file))
        
        sim.cleanup()
        
        logger.info("Robot simulation completed")
        return task_results
    
    def run_full_pipeline(self, goal: str = "Clear the table", gui: bool = False):
        """Run the complete TANGRAM pipeline."""
        logger.info("Starting full TANGRAM pipeline...")
        
        try:
            # Step 1: Tracking
            tracking_results = self.run_tracking()
            if not tracking_results:
                return False
            
            # Step 2: Segmentation
            segmentation_results = self.run_segmentation()
            if not segmentation_results:
                return False
            
            # Step 3: 3D Reconstruction
            reconstruction_results = self.run_reconstruction()
            if not reconstruction_results:
                return False
            
            # Step 4: Scene Graph
            scene_graph = self.build_scene_graph()
            if not scene_graph:
                return False
            
            # Step 5: LLM Interpretation
            interpretation = self.run_llm_interpretation(goal)
            if not interpretation:
                return False
            
            # Step 6: Robot Simulation
            simulation_results = self.run_simulation(gui)
            if not simulation_results:
                return False
            
            # Step 7: Export Results
            export_summary = self.exporter.export_complete_report(str(self.input_path))
            
            logger.info("Full pipeline completed successfully!")
            logger.info(f"Results exported to: {export_summary['export_directory']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main entry point for TANGRAM pipeline."""
    parser = argparse.ArgumentParser(description="TANGRAM Robotic Scene Understanding Pipeline")
    parser.add_argument("--input", "-i", help="Input video file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--mode", choices=["track", "segment", "reconstruct", "graph", "llm", "simulate", "full", "gui"], 
                       default="gui", help="Processing mode")
    parser.add_argument("--gui", action="store_true", help="Show GUI for simulation")
    parser.add_argument("--goal", default="Clear the table", help="Goal for LLM task planning")
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--no-gui", action="store_true", help="Force command line mode")
    
    args = parser.parse_args()
    
    # Launch interactive GUI by default unless --no-gui is specified
    if not args.no_gui and (args.mode == "gui" or not args.input):
        print("🚀 Launching TANGRAM Interactive GUI...")
        try:
            from src.tangram.interfaces.gui.interactive_gui import TangramGUI
            app = TangramGUI()
            app.run()
            return 0
        except Exception as e:
            print(f"❌ GUI launch failed: {e}")
            print("💡 Use --no-gui to run in command line mode")
            return 1
    
    # Command line mode - require input file
    if not args.input:
        print("Error: --input is required for command line mode")
        print("💡 Run without --no-gui to use the interactive GUI")
        return 1
    
    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"TANGRAM Pipeline")
    print(f"Input: {args.input}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    
    # Initialize pipeline
    pipeline = TANGRAMPipeline(args.input, args.output, args.name)
    
    # Run selected mode
    success = True
    
    if args.mode == "track":
        success = pipeline.run_tracking() is not None
    elif args.mode == "segment":
        success = pipeline.run_segmentation() is not None
    elif args.mode == "reconstruct":
        success = pipeline.run_reconstruction() is not None
    elif args.mode == "graph":
        success = pipeline.build_scene_graph() is not None
    elif args.mode == "llm":
        success = pipeline.run_llm_interpretation(args.goal) is not None
    elif args.mode == "simulate":
        success = pipeline.run_simulation(args.gui) is not None
    elif args.mode == "full":
        success = pipeline.run_full_pipeline(args.goal, args.gui)
    
    if success:
        print("✅ Pipeline completed successfully!")
        return 0
    else:
        print("❌ Pipeline failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())