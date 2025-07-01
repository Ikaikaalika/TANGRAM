#!/usr/bin/env python3
"""
Test script for local LLM integration on Thunder Compute.

This script tests the local DeepSeek setup and integration with TANGRAM.

Usage:
    python tests/test_local_llm.py

Author: TANGRAM Team  
License: MIT
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_local_llm_availability():
    """Test if local LLM client can be imported and initialized."""
    print("🧪 Testing Local LLM Availability")
    print("=" * 40)
    
    try:
        from src.tangram.core.llm.local_llm_client import LocalLLMClient
        print("✅ Local LLM client import successful")
        
        # Test initialization
        client = LocalLLMClient(model_name="deepseek-r1:latest")
        print(f"✅ Client initialized")
        print(f"   - Model: {client.model_name}")
        print(f"   - Base URL: {client.base_url}")
        print(f"   - Available: {client.is_available}")
        
        return client.is_available
        
    except Exception as e:
        print(f"❌ Local LLM client test failed: {e}")
        return False

def test_local_inference():
    """Test local inference capabilities."""
    print("\n🚀 Testing Local Inference")
    print("=" * 40)
    
    try:
        from src.tangram.core.llm.local_llm_client import LocalLLMClient
        
        client = LocalLLMClient()
        if not client.is_available:
            print("⚠️  Local LLM not available, skipping inference test")
            return False
        
        # Simple test prompt
        prompt = "Analyze this scene: A red apple is on a table. Generate one robot task."
        system_prompt = "You are a robotic scene understanding assistant. Be concise."
        
        print(f"Test prompt: {prompt[:50]}...")
        
        response = client.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=100,
            temperature=0.7
        )
        
        print("✅ Local inference successful!")
        print(f"   - Model: {response['model']}")
        print(f"   - Source: {response['source']}")
        print(f"   - Generation time: {response.get('generation_time', 'N/A'):.2f}s")
        print(f"   - Tokens: {response.get('usage', {}).get('total_tokens', 'N/A')}")
        print(f"   - Response preview: {response['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Local inference test failed: {e}")
        return False

def test_scene_interpreter():
    """Test the local DeepSeek scene interpreter."""
    print("\n🎭 Testing Scene Interpreter")
    print("=" * 40)
    
    try:
        from src.tangram.core.llm.interpret_scene import create_scene_interpreter
        
        # Create interpreter (should use local if available)
        interpreter = create_scene_interpreter()
        
        print(f"✅ Scene interpreter created: {type(interpreter).__name__}")
        
        # Create mock scene graph data
        mock_scene_data = {
            "nodes": [
                {
                    "id": "obj_1",
                    "properties": {
                        "class_name": "apple",
                        "position_3d": [0.1, 0.2, 0.0],
                        "track_id": 1
                    }
                },
                {
                    "id": "obj_2", 
                    "properties": {
                        "class_name": "cup",
                        "position_3d": [0.3, 0.1, 0.0],
                        "track_id": 2
                    }
                }
            ],
            "edges": [
                {
                    "source": "obj_1",
                    "target": "obj_2",
                    "properties": {
                        "relation": "near",
                        "confidence": 0.8
                    }
                }
            ]
        }
        
        print("Testing scene analysis with mock data...")
        result = interpreter.analyze_scene_graph(mock_scene_data)
        
        print("✅ Scene analysis successful!")
        print(f"   - Objects: {len(result.get('objects', []))}")
        print(f"   - Tasks: {len(result.get('task_sequence', []))}")
        print(f"   - Model used: {result.get('model_used', 'Unknown')}")
        print(f"   - Source: {result.get('source', 'Unknown')}")
        
        if 'scene_analysis' in result:
            print(f"   - Analysis preview: {result['scene_analysis'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Scene interpreter test failed: {e}")
        return False

def test_configuration():
    """Test LLM configuration."""
    print("\n⚙️  Testing Configuration")
    print("=" * 40)
    
    try:
        from config import LLM_CONFIG
        
        print("Current LLM configuration:")
        print(f"   - Provider: {LLM_CONFIG.get('provider')}")
        
        if LLM_CONFIG.get('provider') == 'local':
            local_config = LLM_CONFIG.get('local', {})
            print(f"   - Local enabled: {local_config.get('enabled')}")
            print(f"   - Host: {local_config.get('host')}")
            print(f"   - Port: {local_config.get('port')}")
            print(f"   - Model: {local_config.get('model')}")
            print(f"   - Fallback to API: {local_config.get('fallback_to_api')}")
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🌩️ TANGRAM Local LLM Integration Test")
    print("=====================================\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Local LLM Availability", test_local_llm_availability), 
        ("Local Inference", test_local_inference),
        ("Scene Interpreter", test_scene_interpreter)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Local LLM integration is ready.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())