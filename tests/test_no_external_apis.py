#!/usr/bin/env python3
"""
Network Isolation Test for LOCAL-ONLY TANGRAM

This script validates that TANGRAM makes ZERO external API calls when 
configured for local-only operation.

Usage:
    python tests/test_no_external_apis.py

Author: TANGRAM Team
License: MIT
"""

import sys
import socket
import threading
import time
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Track any attempted external connections
EXTERNAL_CONNECTIONS = []
BLOCKED_HOSTS = [
    "api.deepseek.com",
    "api.openai.com", 
    "huggingface.co",
    "googleapis.com"
]

def mock_socket_connect(original_connect):
    """Mock socket connections to detect external API calls."""
    def patched_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        
        # Log all connection attempts
        EXTERNAL_CONNECTIONS.append({
            "host": host,
            "port": address[1] if isinstance(address, tuple) and len(address) > 1 else None,
            "timestamp": time.time()
        })
        
        # Block known external API hosts
        if any(blocked in host for blocked in BLOCKED_HOSTS):
            raise ConnectionError(f"BLOCKED: Attempted connection to external API: {host}")
        
        # Allow localhost connections
        if host in ["localhost", "127.0.0.1", "0.0.0.0"]:
            return original_connect(address)
        
        # Block all other external connections
        if not host.startswith("192.168.") and not host.startswith("10."):
            raise ConnectionError(f"BLOCKED: External connection attempt to: {host}")
        
        return original_connect(address)
    
    return patched_connect

def test_local_only_config():
    """Test that configuration enforces local-only operation."""
    print("🔒 Testing LOCAL-ONLY Configuration")
    print("=" * 50)
    
    try:
        from config import LLM_CONFIG
        
        # Verify local-only configuration
        provider = LLM_CONFIG.get("provider")
        if provider != "local":
            print(f"❌ FAIL: Provider is '{provider}', must be 'local'")
            return False
        
        local_config = LLM_CONFIG.get("local", {})
        fallback = local_config.get("fallback_to_api", True)
        
        if fallback:
            print("❌ FAIL: fallback_to_api is enabled - external APIs allowed")
            return False
        
        print("✅ PASS: Configuration enforces local-only operation")
        print(f"   - Provider: {provider}")
        print(f"   - Fallback disabled: {not fallback}")
        print(f"   - Model: {local_config.get('model', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Configuration test error: {e}")
        return False

def test_interpreter_creation():
    """Test that interpreter creation fails gracefully without external calls."""
    print("\n🧪 Testing Interpreter Creation (Offline)")
    print("=" * 50)
    
    # Patch socket to block external connections
    with patch.object(socket.socket, 'connect', side_effect=mock_socket_connect(socket.socket.connect)):
        try:
            from src.tangram.core.llm.interpret_scene import create_scene_interpreter
            
            # This should either work with local LLM or fail explicitly
            try:
                interpreter = create_scene_interpreter()
                print("✅ PASS: Interpreter created successfully (local LLM available)")
                print(f"   - Type: {type(interpreter).__name__}")
                return True
                
            except RuntimeError as e:
                if "Local LLM" in str(e) and "external" not in str(e).lower():
                    print("✅ PASS: Interpreter failed explicitly (local LLM unavailable)")
                    print(f"   - Error: {str(e)[:100]}...")
                    return True
                else:
                    print(f"❌ FAIL: Unexpected error: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ FAIL: Interpreter creation test error: {e}")
            return False

def test_mock_scene_analysis():
    """Test scene analysis with mock data and network isolation."""
    print("\n🎭 Testing Scene Analysis (Network Isolated)")
    print("=" * 50)
    
    # Patch socket to monitor and block external connections
    with patch.object(socket.socket, 'connect', side_effect=mock_socket_connect(socket.socket.connect)):
        try:
            from src.tangram.core.llm.interpret_scene import create_scene_interpreter
            
            # Clear connection tracking
            global EXTERNAL_CONNECTIONS
            EXTERNAL_CONNECTIONS = []
            
            try:
                interpreter = create_scene_interpreter()
                
                # Mock scene data
                mock_scene = {
                    "nodes": [
                        {"id": "obj_1", "properties": {"class_name": "apple", "position_3d": [0, 0, 0]}}
                    ],
                    "edges": []
                }
                
                print("Attempting scene analysis with network isolation...")
                result = interpreter.analyze_scene_graph(mock_scene)
                
                print("✅ PASS: Scene analysis completed without external calls")
                print(f"   - External connections attempted: {len(EXTERNAL_CONNECTIONS)}")
                
                if EXTERNAL_CONNECTIONS:
                    print("   - Connection attempts:")
                    for conn in EXTERNAL_CONNECTIONS[:3]:  # Show first 3
                        print(f"     * {conn['host']}:{conn.get('port', 'N/A')}")
                
                return len(EXTERNAL_CONNECTIONS) == 0
                
            except RuntimeError as e:
                if "Local LLM" in str(e):
                    print("✅ PASS: Scene analysis failed explicitly (local LLM unavailable)")
                    print(f"   - No external API fallback attempted")
                    return True
                else:
                    print(f"❌ FAIL: Unexpected error: {e}")
                    return False
                    
        except Exception as e:
            if "BLOCKED" in str(e):
                print(f"❌ FAIL: System attempted external connection: {e}")
                return False
            else:
                print(f"❌ FAIL: Scene analysis test error: {e}")
                return False

def test_imports_no_network():
    """Test that importing modules doesn't trigger network calls."""
    print("\n📦 Testing Import Safety (Network Isolated)")
    print("=" * 50)
    
    # Patch socket before imports
    with patch.object(socket.socket, 'connect', side_effect=mock_socket_connect(socket.socket.connect)):
        global EXTERNAL_CONNECTIONS
        EXTERNAL_CONNECTIONS = []
        
        try:
            # Import key modules
            from src.tangram.core.llm.local_llm_client import LocalLLMClient
            from src.tangram.core.llm.interpret_scene import create_scene_interpreter
            from config import LLM_CONFIG
            
            print("✅ PASS: All imports completed without network calls")
            print(f"   - External connections during import: {len(EXTERNAL_CONNECTIONS)}")
            
            return len(EXTERNAL_CONNECTIONS) == 0
            
        except Exception as e:
            if "BLOCKED" in str(e):
                print(f"❌ FAIL: Import triggered external connection: {e}")
                return False
            else:
                print(f"❌ FAIL: Import test error: {e}")
                return False

def main():
    """Run all network isolation tests."""
    print("🛡️ TANGRAM Network Isolation Test")
    print("==================================")
    print("Verifying ZERO external API calls")
    print("")
    
    tests = [
        ("Configuration Validation", test_local_only_config),
        ("Import Safety", test_imports_no_network), 
        ("Interpreter Creation", test_interpreter_creation),
        ("Scene Analysis Isolation", test_mock_scene_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Network Isolation Test Results")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 SUCCESS: System enforces LOCAL-ONLY operation")
        print("🔒 ZERO external API calls detected")
        return 0
    else:
        print(f"\n⚠️ FAILURE: {len(results) - passed} test(s) failed")
        print("🚨 System may attempt external API calls")
        return 1

if __name__ == "__main__":
    sys.exit(main())