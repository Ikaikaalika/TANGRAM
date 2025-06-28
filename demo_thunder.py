#!/usr/bin/env python3
"""
TANGRAM Thunder Compute Demo Script

This demo showcases the complete pipeline with Thunder Compute integration
for heavy processing tasks like SAM segmentation and COLMAP reconstruction.

Usage:
    python demo_thunder.py --thunder --video path/to/video.mp4
    python demo_thunder.py --thunder --gui  # Use sample video with GUI

Author: TANGRAM Team
License: MIT
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from thunder.thunder_integration import ThunderIntegratedDemo
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def main():
    """Main Thunder Compute demo entry point."""
    parser = argparse.ArgumentParser(description="TANGRAM Thunder Compute Demo")
    parser.add_argument("--video", "-v", help="Path to input video file")
    parser.add_argument("--thunder", action="store_true", required=True, 
                       help="Enable Thunder Compute (required for this demo)")
    parser.add_argument("--gui", action="store_true", help="Show GUI visualizations")
    parser.add_argument("--name", "-n", help="Experiment name for this demo")
    
    args = parser.parse_args()
    
    print("""
    ⚡ TANGRAM Thunder Compute Demo ⚡
    
    This demo showcases automatic Thunder Compute integration:
    
    🔄 Automatic Detection:
    • Video complexity analysis
    • Local GPU availability check
    • Thunder Compute health verification
    
    ⚡ Thunder Compute Tasks:
    • SAM segmentation for large videos
    • COLMAP 3D reconstruction
    • Heavy matrix computations
    
    💻 Local Tasks:
    • YOLO object detection/tracking
    • Scene graph construction
    • LLM interpretation
    • Robot simulation
    
    🎯 Optimized for your portfolio demonstration!
    """)
    
    # Validate Thunder Compute setup
    from config import HARDWARE_CONFIG
    if not HARDWARE_CONFIG["thunder_compute"]["enabled"]:
        print("❌ Thunder Compute is disabled in config.py")
        print("Please update your configuration with Thunder Compute details:")
        print("""
# In config.py, update:
HARDWARE_CONFIG = {
    "thunder_compute": {
        "enabled": True,
        "ssh_host": "your-thunder-instance.com",
        "ssh_user": "ubuntu", 
        "ssh_key_path": "~/.ssh/thunder_key.pem"
    }
}
        """)
        return 1
    
    # Get video path
    video_path = args.video
    if not video_path:
        # Use sample video
        from config import SAMPLE_VIDEOS_DIR
        sample_video = SAMPLE_VIDEOS_DIR / "tabletop_manipulation.mp4"
        if sample_video.exists():
            video_path = str(sample_video)
            print(f"Using sample video: {sample_video}")
        else:
            print("❌ No video file available. Please provide --video or ensure sample data exists.")
            return 1
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    print(f"🎬 Processing video: {video_path}")
    print(f"⚡ Thunder Compute: ENABLED")
    
    # Create Thunder-integrated demo
    demo = ThunderIntegratedDemo(args.name)
    
    try:
        # Test Thunder Compute connection first
        print("\n🔗 Testing Thunder Compute connection...")
        from thunder.thunder_client import ThunderComputeClient
        
        client = ThunderComputeClient()
        if client.connect():
            print("✅ Thunder Compute connection successful!")
            client.disconnect()
        else:
            print("⚠️  Thunder Compute connection failed - will use local fallback")
        
        # Run optimized pipeline
        print("\n🚀 Starting Thunder-optimized pipeline...")
        results = demo.run_thunder_optimized_pipeline(video_path)
        
        print("\n📊 Pipeline Results:")
        for step, result in results.items():
            if result:
                print(f"  ✅ {step.title()}: Success")
            else:
                print(f"  ❌ {step.title()}: Failed")
        
        # Generate report
        export_summary = demo.exporter.export_complete_report(video_path)
        
        print(f"\n🎉 Thunder Demo Complete!")
        print(f"📊 Performance Score: {export_summary['performance_score']:.1%}")
        print(f"📁 Results: {export_summary['export_directory']}")
        print(f"🌐 HTML Report: {export_summary['html_report']}")
        
        print(f"\n💼 For your portfolio:")
        print(f"1. Show the automatic Thunder/local switching")
        print(f"2. Highlight the performance optimization")
        print(f"3. Demonstrate scalability for large datasets")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Thunder demo error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())