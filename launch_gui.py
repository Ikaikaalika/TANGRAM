#!/usr/bin/env python3
"""
TANGRAM GUI Launcher

Simple launcher script for the TANGRAM management GUI.
Handles dependencies and provides fallback options.
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import tkinter
        return True
    except ImportError:
        print("❌ tkinter is not available")
        print("On macOS: tkinter should be included with Python")
        print("On Ubuntu/Debian: sudo apt-get install python3-tk")
        print("On CentOS/RHEL: sudo yum install tkinter")
        return False

def main():
    """Launch the TANGRAM GUI."""
    print("🚀 TANGRAM Pipeline Manager")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Launch GUI
    try:
        gui_path = Path(__file__).parent / "gui" / "tangram_manager.py"
        
        if not gui_path.exists():
            print(f"❌ GUI file not found: {gui_path}")
            return 1
        
        print("✅ Launching TANGRAM Manager...")
        subprocess.run([sys.executable, str(gui_path)], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 TANGRAM Manager closed")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ GUI failed to start: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())