#!/usr/bin/env python3
"""
TANGRAM Portfolio Demo Script

This script demonstrates the complete robotic scene understanding pipeline
for portfolio presentation. It runs through all steps with sample data
and generates a comprehensive results report.

Usage:
    python demo.py [--gui] [--video path/to/video.mp4] [--thunder]

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

from config import (
    FRAME_EXTRACTION_CONFIG, SAM_CONFIG, YOLO_CONFIG, 
    COLMAP_CONFIG, PYBULLET_CONFIG, HARDWARE_CONFIG,
    SCENE_GRAPH_CONFIG, LLM_CONFIG, DATA_DIR, RESULTS_DIR
)
from utils.logging_utils import setup_logger, log_pipeline_step
from utils.video_utils import validate_video_file, extract_video_info
from tracker.track_objects import YOLOByteTracker
from segmenter.run_sam import SAMSegmenter  
from reconstruction.triangulate import PointTriangulator
from scene_graph.build_graph import SceneGraphBuilder
from llm.interpret_scene import DeepSeekSceneInterpreter
from robotics.simulation_env import RoboticsSimulation
from export.results_exporter import ResultsExporter

logger = setup_logger(__name__, "demo.log")

class TANGRAMDemo:
    """
    Complete TANGRAM pipeline demonstration for portfolio showcase.
    
    Runs the entire robotic scene understanding pipeline from video input
    to robot simulation and generates comprehensive results for presentation.
    """
    
    def __init__(self, video_path: str = None, use_gui: bool = True, 
                 use_thunder: bool = False, experiment_name: str = None):
        """
        Initialize demo environment.
        
        Args:
            video_path: Path to input video (None = use sample)
            use_gui: Show GUI visualizations
            use_thunder: Use Thunder Compute for heavy processing
            experiment_name: Name for this demo run
        """
        self.video_path = video_path
        self.use_gui = use_gui
        self.use_thunder = use_thunder
        self.experiment_name = experiment_name or f"portfolio_demo_{int(time.time())}"
        
        # Initialize components
        self.tracker = None
        self.segmenter = None
        self.triangulator = None
        self.scene_builder = None
        self.llm_interpreter = None
        self.simulator = None
        self.exporter = None
        
        # Results storage
        self.results = {}
        
        logger.info(f"Initialized TANGRAM demo: {self.experiment_name}")
        logger.info(f"Configuration: GUI={use_gui}, Thunder={use_thunder}")
    
    def setup_demo_environment(self):
        """Set up demo environment and validate inputs."""
        logger.info("Setting up demo environment...")
        
        # Ensure all directories exist
        ensure_directories()
        
        # Validate video input
        if self.video_path:
            if not validate_video_file(self.video_path):
                logger.error(f"Invalid video file: {self.video_path}")
                return False
            logger.info(f"Using video: {self.video_path}")
        else:
            # Use sample video
            sample_video = SAMPLE_VIDEOS_DIR / "tabletop_manipulation.mp4"
            if sample_video.exists():
                self.video_path = str(sample_video)
                logger.info(f"Using sample video: {sample_video}")
            else:
                logger.error("No video file available. Please provide a video or ensure sample data exists.")
                return False
        
        # Initialize components
        self.tracker = YOLOByteTracker()
        self.segmenter = SAMSegmenter(model_type="vit_b")  # Use faster model for demo
        self.triangulator = PointTriangulator()
        self.scene_builder = SceneGraphBuilder()
        self.llm_interpreter = DeepSeekSceneInterpreter()
        self.simulator = RoboticsSimulation(gui=self.use_gui)
        self.exporter = ResultsExporter(self.experiment_name)
        
        logger.info("Demo environment setup complete")
        return True
    
    @log_pipeline_step("STEP 1: Object Detection and Tracking")
    def run_object_tracking(self):
        """Run object detection and tracking on input video."""
        logger.info(f"Processing video: {self.video_path}")
        
        # Extract video info for reporting
        video_info = extract_video_info(self.video_path)
        self.results["video_info"] = video_info
        
        # Run tracking
        tracking_results = self.tracker.process_video(
            self.video_path, 
            output_dir=str(TRACKING_DIR)
        )
        
        self.results["tracking_results"] = tracking_results
        logger.info(f"Tracked objects across {len(tracking_results)} frames")
        
        return len(tracking_results) > 0
    
    @log_pipeline_step("STEP 2: Object Segmentation")  
    def run_segmentation(self):
        """Generate segmentation masks for tracked objects."""
        if not self.results.get("tracking_results"):
            logger.error("No tracking results available for segmentation")
            return False
        
        # Run SAM segmentation
        segmentation_results = self.segmenter.process_video_with_tracking(
            self.video_path,
            self.results["tracking_results"],
            output_dir=str(MASKS_DIR)
        )
        
        self.results["segmentation_results"] = segmentation_results
        logger.info(f"Generated masks for {len(segmentation_results)} frames")
        
        return len(segmentation_results) > 0
    
    @log_pipeline_step("STEP 3: 3D Reconstruction")
    def run_3d_reconstruction(self):
        """Perform 3D reconstruction and object triangulation."""
        # Extract frames for COLMAP
        logger.info("Extracting frames for 3D reconstruction...")
        
        from reconstruction.extract_frames import extract_frames
        success = extract_frames(
            self.video_path,
            str(FRAMES_DIR),
            frame_interval=10,  # Every 10th frame for faster processing
            max_frames=50       # Limit frames for demo
        )
        
        if not success:
            logger.error("Failed to extract frames")
            return False
        
        # Note: In a real demo, you would run COLMAP here
        # For this demo, we'll simulate 3D reconstruction results
        logger.info("Simulating 3D reconstruction (COLMAP would run here)...")
        
        # Create mock 3D positions for demo
        mock_3d_positions = self._create_mock_3d_positions()
        self.results["3d_positions"] = mock_3d_positions
        
        # Save mock results
        import json
        with open(RECONSTRUCTION_DIR / "object_3d_positions.json", 'w') as f:
            json.dump(mock_3d_positions, f, indent=2)
        
        logger.info(f"Generated 3D positions for {len(mock_3d_positions)} objects")
        return True
    
    def _create_mock_3d_positions(self):
        """Create mock 3D positions for demo purposes."""
        mock_positions = {}
        
        # Extract unique objects from tracking
        if self.results.get("tracking_results"):
            unique_objects = set()
            for frame in self.results["tracking_results"]:
                for detection in frame.get("detections", []):
                    track_id = detection["track_id"]
                    class_name = detection["class_name"]
                    unique_objects.add((track_id, class_name))
            
            # Generate realistic 3D positions
            import random
            random.seed(42)  # Consistent results
            
            for i, (track_id, class_name) in enumerate(unique_objects):
                # Place objects on table surface with some variation
                x = random.uniform(-0.3, 0.3)
                y = random.uniform(-0.3, 0.3)
                z = 0.7 + random.uniform(0, 0.1)  # Table height + object height
                
                mock_positions[str(track_id)] = {
                    "position": [x, y, z],
                    "class_name": class_name,
                    "num_observations": random.randint(5, 20)
                }
        
        return mock_positions
    
    @log_pipeline_step("STEP 4: Scene Graph Construction")
    def build_scene_graph(self):
        """Build scene graph from tracking and 3D data."""
        tracking_file = TRACKING_DIR / "tracking_results.json"
        positions_file = RECONSTRUCTION_DIR / "object_3d_positions.json"
        
        if not (tracking_file.exists() and positions_file.exists()):
            logger.error("Required files not found for scene graph construction")
            return False
        
        # Build scene graph
        scene_graph = self.scene_builder.build_complete_graph(
            str(tracking_file),
            str(positions_file),
            output_dir=str(GRAPHS_DIR)
        )
        
        self.results["scene_graph"] = scene_graph
        
        # Get summary
        summary = self.scene_builder.get_graph_summary()
        logger.info(f"Built scene graph: {summary['num_nodes']} nodes, {summary['num_edges']} edges")
        
        return True
    
    @log_pipeline_step("STEP 5: LLM Scene Interpretation")
    def run_llm_interpretation(self):
        """Generate LLM interpretation and task planning."""
        graph_file = GRAPHS_DIR / "scene_graph.json"
        
        if not graph_file.exists():
            logger.error("Scene graph file not found")
            return False
        
        # Test multiple goals for demo
        demo_goals = [
            "Clear the table by moving all objects to storage area",
            "Organize objects by grouping similar items together",
            "Stack objects from largest to smallest"
        ]
        
        interpretations = []
        
        for goal in demo_goals:
            logger.info(f"Generating tasks for goal: {goal}")
            
            interpretation = self.llm_interpreter.process_scene_graph_file(
                str(graph_file),
                goal=goal,
                output_dir=str(GRAPHS_DIR)
            )
            
            if interpretation:
                interpretations.append(interpretation)
        
        self.results["llm_interpretations"] = interpretations
        
        # Use first interpretation for simulation
        if interpretations:
            self.results["selected_interpretation"] = interpretations[0]
            return True
        
        return False
    
    @log_pipeline_step("STEP 6: Robot Simulation")
    def run_robot_simulation(self):
        """Execute robot tasks in PyBullet simulation."""
        if not self.results.get("selected_interpretation"):
            logger.error("No LLM interpretation available for simulation")
            return False
        
        # Initialize simulation
        if not self.simulator.initialize_simulation():
            logger.error("Failed to initialize robot simulation")
            return False
        
        # Load robot
        if not self.simulator.load_robot():
            logger.error("Failed to load robot")
            return False
        
        # Add scene objects
        object_positions = self.results.get("3d_positions", {})
        object_mapping = self.simulator.add_scene_objects(object_positions)
        
        # Execute task sequence
        interpretation = self.results["selected_interpretation"]
        task_sequence = interpretation.get("task_sequence", [])
        
        logger.info(f"Executing {len(task_sequence)} robot tasks...")
        
        task_results = self.simulator.execute_task_sequence(task_sequence, object_mapping)
        
        # Save simulation results
        simulation_results = {
            "task_results": task_results,
            "object_mapping": object_mapping,
            "final_state": self.simulator.get_simulation_state()
        }
        
        output_file = SIMULATION_DIR / "demo_simulation_results.json"
        self.simulator.save_simulation_results(str(output_file))
        
        self.results["simulation_results"] = simulation_results
        
        # Calculate success rate
        successful_tasks = len([r for r in task_results if r.get("success", False)])
        success_rate = successful_tasks / len(task_results) if task_results else 0
        
        logger.info(f"Simulation complete: {successful_tasks}/{len(task_results)} tasks successful ({success_rate:.1%})")
        
        if self.use_gui:
            input("Press Enter to continue (simulation will remain open)...")
        
        return True
    
    @log_pipeline_step("STEP 7: Results Export")
    def export_results(self):
        """Export comprehensive results package."""
        logger.info("Exporting demo results...")
        
        # Export complete results package
        export_summary = self.exporter.export_complete_report(
            input_video_path=self.video_path
        )
        
        self.results["export_summary"] = export_summary
        
        # Print summary for user
        print("\n" + "="*60)
        print("TANGRAM DEMO RESULTS SUMMARY")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Export Directory: {export_summary['export_directory']}")
        print(f"Performance Score: {export_summary['performance_score']:.1%}")
        print(f"HTML Report: {export_summary['html_report']}")
        
        if export_summary['summary_video']:
            print(f"Summary Video: {export_summary['summary_video']}")
        
        print("\nFor your portfolio:")
        print(f"1. Open the HTML report in a browser: {export_summary['html_report']}")
        print(f"2. Show the generated visualizations and metrics")
        print(f"3. Demonstrate the robot simulation results")
        print("="*60)
        
        return True
    
    def run_complete_demo(self):
        """Run the complete TANGRAM pipeline demo."""
        logger.info("Starting TANGRAM complete pipeline demo")
        
        start_time = time.time()
        
        try:
            # Setup
            if not self.setup_demo_environment():
                logger.error("Failed to setup demo environment")
                return False
            
            # Run pipeline steps
            steps = [
                ("Object Tracking", self.run_object_tracking),
                ("Segmentation", self.run_segmentation),
                ("3D Reconstruction", self.run_3d_reconstruction),
                ("Scene Graph", self.build_scene_graph),
                ("LLM Interpretation", self.run_llm_interpretation),
                ("Robot Simulation", self.run_robot_simulation),
                ("Results Export", self.export_results)
            ]
            
            for step_name, step_function in steps:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running: {step_name}")
                logger.info(f"{'='*60}")
                
                success = step_function()
                
                if not success:
                    logger.error(f"Step failed: {step_name}")
                    print(f"\n❌ Demo failed at step: {step_name}")
                    return False
                
                logger.info(f"✅ Completed: {step_name}")
            
            # Final summary
            total_time = time.time() - start_time
            logger.info(f"\n🎉 TANGRAM demo completed successfully in {total_time:.1f} seconds!")
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            print("\n⏹️  Demo stopped by user")
            return False
            
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            print(f"\n❌ Demo failed: {e}")
            return False
            
        finally:
            # Cleanup
            if self.simulator:
                self.simulator.cleanup()

def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description="TANGRAM Portfolio Demo")
    parser.add_argument("--video", "-v", help="Path to input video file")
    parser.add_argument("--gui", action="store_true", help="Show GUI visualizations")
    parser.add_argument("--thunder", action="store_true", help="Use Thunder Compute for heavy processing")
    parser.add_argument("--name", "-n", help="Experiment name for this demo")
    
    args = parser.parse_args()
    
    print("""
    🤖 TANGRAM Robotic Scene Understanding Pipeline Demo
    
    This demo showcases the complete pipeline:
    1. Video-based object detection and tracking
    2. Instance segmentation with object association  
    3. 3D scene reconstruction and triangulation
    4. Temporal scene graph construction
    5. LLM-based scene interpretation and task planning
    6. Robot simulation and task execution
    7. Comprehensive results export and evaluation
    
    Perfect for portfolio presentation! 🎯
    """)
    
    # Create and run demo
    demo = TANGRAMDemo(
        video_path=args.video,
        use_gui=args.gui,
        use_thunder=args.thunder,
        experiment_name=args.name
    )
    
    success = demo.run_complete_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("Check the exported HTML report for comprehensive results.")
    else:
        print("\n❌ Demo encountered issues. Check the logs for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())