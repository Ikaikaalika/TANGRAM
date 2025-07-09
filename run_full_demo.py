#!/usr/bin/env python3
"""
TANGRAM Full Demo Runner
========================

Runs all demo components automatically to showcase the complete pipeline.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tangram_integration_demo import TANGRAMDemo

def main():
    """Run the full demo automatically"""
    print("🚀 Starting TANGRAM Full Demo...")
    
    # Create demo instance
    demo = TANGRAMDemo()
    
    # Run all components
    demo.run_all_demos()
    
    print("\n🎉 Demo completed! Check the results above.")

if __name__ == "__main__":
    main()