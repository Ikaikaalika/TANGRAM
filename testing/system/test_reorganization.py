#!/usr/bin/env python3
"""
Test script to validate the reorganized codebase structure.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_imports():
    """Test that key imports work with new structure."""
    print("Testing reorganized imports...")
    
    try:
        # Test utils imports
        from src.tangram.utils.mock_data import create_mock_3d_positions
        from src.tangram.utils.logging_utils import setup_logger
        from src.tangram.utils.file_utils import save_json
        print("✅ Utils imports work")
        
        # Test mock data functionality
        mock_data = create_mock_3d_positions()
        assert isinstance(mock_data, dict)
        assert len(mock_data) > 0
        print("✅ Mock data generation works")
        
        # Test config import
        from config import HARDWARE_CONFIG, DATA_DIR
        assert DATA_DIR is not None
        print("✅ Config imports work")
        
        print("\n🎉 All core functionality tests passed!")
        print("📁 Reorganized structure is working correctly")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return True

def test_structure():
    """Test that the new directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "src/tangram/core",
        "src/tangram/gui", 
        "src/tangram/utils",
        "src/tangram/integrations",
        "examples",
        "tests",
        "docs"
    ]
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} missing")
            return False
    
    print("✅ All required directories exist")
    return True

if __name__ == "__main__":
    print("🧪 TANGRAM Reorganization Test")
    print("=" * 40)
    
    structure_ok = test_structure()
    imports_ok = test_imports()
    
    if structure_ok and imports_ok:
        print("\n🎉 Reorganization successful!")
        sys.exit(0)
    else:
        print("\n❌ Reorganization has issues")
        sys.exit(1)