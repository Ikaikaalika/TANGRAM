#!/usr/bin/env python3
"""
TANGRAM GUI Launcher

Simple launcher script for the TANGRAM GUI application.

Usage:
    python launch_gui.py

Author: TANGRAM Team
License: MIT
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Check required dependencies for GUI."""
    missing_deps = []
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter")
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("🚀 TANGRAM GUI Launcher")
    print("=" * 40)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print("✅ All dependencies available")
    
    # Check if GUI files exist
    gui_module = PROJECT_ROOT / "gui" / "tangram_gui.py"
    if not gui_module.exists():
        print(f"❌ GUI module not found: {gui_module}")
        return 1
    
    print("✅ GUI module found")
    
    # Launch GUI
    try:
        print("🎯 Launching TANGRAM GUI...")
        from gui.tangram_gui import main as gui_main
        return gui_main()
        
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())