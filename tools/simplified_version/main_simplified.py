#!/usr/bin/env python3
"""
TANGRAM Main Pipeline Controller - Simplified Version

This is the simplified main entry point for TANGRAM.
Uses the consolidated core module for streamlined functionality.

Usage:
    python main_simplified.py                    # Launch GUI
    python main_simplified.py --input video.mp4  # Process video
    python main_simplified.py --no-gui --input video.mp4  # Command line mode
"""

import argparse
import sys
import json
import cv2
from pathlib import Path

# Add project modules to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    """Main entry point for simplified TANGRAM pipeline."""
    parser = argparse.ArgumentParser(description="TANGRAM Simplified Pipeline")
    parser.add_argument("--input", "-i", help="Input video/image file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--no-gui", action="store_true", help="Force command line mode")
    parser.add_argument("--command", "-c", help="Robot command to execute")
    
    args = parser.parse_args()
    
    # Launch interactive GUI by default unless --no-gui is specified
    if not args.no_gui and not args.input:
        print("🚀 Launching TANGRAM Interactive GUI...")
        try:
            from src.tangram.gui.interactive_gui import TangramGUI
            app = TangramGUI()
            app.run()
            return 0
        except Exception as e:
            print(f"❌ GUI launch failed: {e}")
            print("💡 Use --no-gui to run in command line mode")
            return 1
    
    # Command line mode
    if not args.input:
        print("Error: --input is required for command line mode")
        print("💡 Run without --no-gui to use the interactive GUI")
        return 1
    
    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print("🚀 TANGRAM Simplified Pipeline")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Import and initialize core pipeline
    try:
        from tangram_core import TANGRAMCore
        pipeline = TANGRAMCore()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Determine if input is image or video
        input_path = Path(args.input)
        is_video = input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
        
        if is_video:
            print("📹 Processing video...")
            results = pipeline.process_video(args.input)
            
            # Save results
            results_file = output_dir / "video_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Processed {len(results)} frames")
            print(f"📁 Results saved to: {results_file}")
            
            # Get latest scene data for visualization
            if results:
                latest_scene = results[-1].get("scene_graph", {})
                
                # Create visualization
                fig = pipeline.visualize_scene(latest_scene)
                viz_file = output_dir / "scene_visualization.png"
                fig.savefig(viz_file, dpi=300, bbox_inches='tight')
                print(f"📊 Visualization saved to: {viz_file}")
                
                # Execute command if provided
                if args.command:
                    print(f"🤖 Executing command: {args.command}")
                    response = pipeline.execute_command(args.command, latest_scene)
                    print(f"Response: {response}")
                    
                    # Save command results
                    command_file = output_dir / "command_results.txt"
                    with open(command_file, 'w') as f:
                        f.write(f"Command: {args.command}\n")
                        f.write(f"Response: {response}\n")
                        f.write(f"Robot Position: {pipeline.robot.position}\n")
        
        else:
            print("🖼️ Processing image...")
            image = cv2.imread(args.input)
            if image is None:
                print("❌ Failed to load image")
                return 1
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pipeline.process_image(image_rgb)
            
            # Save results
            results_file = output_dir / "image_analysis.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"✅ Detected {len(result['detections'])} objects")
            print(f"📁 Results saved to: {results_file}")
            
            # Create visualization
            scene_data = result.get("scene_graph", {})
            fig = pipeline.visualize_scene(scene_data)
            viz_file = output_dir / "scene_visualization.png"
            fig.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {viz_file}")
            
            # Execute command if provided
            if args.command:
                print(f"🤖 Executing command: {args.command}")
                response = pipeline.execute_command(args.command, scene_data)
                print(f"Response: {response}")
        
        # Cleanup
        pipeline.cleanup()
        
        print("✅ Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        pipeline.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())