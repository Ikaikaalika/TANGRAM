#!/usr/bin/env python3
"""
TANGRAM - AI-Powered Robotic Scene Understanding
Main entry point for the TANGRAM system.

Usage:
    python tangram.py gui          # Launch interactive GUI
    python tangram.py process <video>  # Process a video through pipeline
    python tangram.py demo         # Run demonstration
"""

import sys
import time
import argparse
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def main():
    """Main entry point for TANGRAM system."""
    parser = argparse.ArgumentParser(
        description="TANGRAM - AI-Powered Robotic Scene Understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch interactive GUI')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process video through pipeline')
    process_parser.add_argument('video', help='Path to video file')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    
    args = parser.parse_args()
    
    if args.command == 'gui':
        launch_gui()
    elif args.command == 'process':
        process_video(args.video)
    elif args.command == 'demo':
        run_demo()
    else:
        parser.print_help()

def launch_gui():
    """Launch the interactive GUI."""
    print("🚀 Launching TANGRAM Safe GUI...")
    print("Using safe mode to prevent system freezing...")
    try:
        from safe_gui import SafeTangramGUI
        app = SafeTangramGUI()
        app.run()
    except ImportError as e:
        print(f"❌ Failed to import GUI: {e}")
        print("Falling back to lightweight demo...")
        run_demo()
    except Exception as e:
        print(f"❌ GUI error: {e}")
        print("Try running: python safe_gui.py")

def process_video(video_path):
    """Process a video through the TANGRAM pipeline."""
    print(f"🎥 Processing video: {video_path}")
    try:
        from main import TANGRAMPipeline
        pipeline = TANGRAMPipeline(video_path, "temp")
        results = pipeline.run_tracking()
        print("✅ Video processing complete")
        return results
    except Exception as e:
        print(f"❌ Processing error: {e}")

def run_demo():
    """Run a demonstration."""
    print("🎬 Running TANGRAM Demonstration...")
    try:
        # Try lightweight demo first (safer)
        from scripts.demos.lightweight_demo import LightweightDemo
        
        print("Starting with lightweight demo (prevents system freezing)...")
        demo = LightweightDemo()
        success = demo.run_lightweight_demo()
        
        if success and all(demo.components_ready.values()):
            print("\n🎯 All components ready! Would you like to run the full advanced demo?")
            print("⚠️  Warning: Advanced demo uses more resources and may take longer")
            
            # For now, stick with lightweight demo to avoid freezing
            print("✅ Lightweight demo completed successfully!")
        else:
            print("✅ Lightweight demo completed (some components may need setup)")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("💡 Try running: python scripts/demos/lightweight_demo.py")

if __name__ == "__main__":
    main()