#!/usr/bin/env python3
"""
TANGRAM Lightweight Demo

A streamlined version that avoids system freezing:
- Minimal GUI initialization
- Progressive loading with status updates
- Graceful fallbacks for all components
- Safe resource management

Author: TANGRAM Team
License: MIT
"""

import sys
import time
import os
from pathlib import Path
import threading
import signal

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class LightweightDemo:
    """Lightweight demo that won't freeze the system"""
    
    def __init__(self):
        self.should_stop = False
        self.components_ready = {
            'vision': False,
            'robot': False,
            'llm': False
        }
        
        # Set up graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Demo interrupted by user")
        self.should_stop = True
        sys.exit(0)
    
    def run_lightweight_demo(self):
        """Run a lightweight version of the demo"""
        
        print("🚀 TANGRAM Lightweight Demo")
        print("=" * 40)
        print("This demo will test components progressively...")
        print("Press Ctrl+C at any time to stop safely.\n")
        
        # Step 1: Test basic imports
        if not self._test_basic_imports():
            return False
        
        # Step 2: Test computer vision (without heavy models)
        if not self._test_vision_lightweight():
            return False
        
        # Step 3: Test robot simulation (minimal)
        if not self._test_robot_lightweight():
            return False
        
        # Step 4: Test LLM (connection only)
        if not self._test_llm_lightweight():
            return False
        
        # Step 5: Run minimal pipeline
        if not self._run_minimal_pipeline():
            return False
        
        print("\n🎉 Lightweight demo completed successfully!")
        print("📋 Component Status:")
        for component, status in self.components_ready.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component.capitalize()}: {'Ready' if status else 'Not Available'}")
        
        return True
    
    def _test_basic_imports(self):
        """Test basic Python imports without heavy dependencies"""
        
        print("🔍 Testing basic imports...")
        
        try:
            import numpy as np
            print("  ✅ NumPy available")
            
            import json
            print("  ✅ JSON available")
            
            import threading
            print("  ✅ Threading available")
            
            # Test project structure
            if (PROJECT_ROOT / "src").exists():
                print("  ✅ Project structure valid")
            else:
                print("  ❌ Project structure invalid")
                return False
            
            return True
            
        except ImportError as e:
            print(f"  ❌ Import error: {e}")
            return False
    
    def _test_vision_lightweight(self):
        """Test computer vision components without loading heavy models"""
        
        print("\n🔍 Testing computer vision (lightweight)...")
        
        try:
            # Test OpenCV import
            import cv2
            print(f"  ✅ OpenCV {cv2.__version__} available")
            
            # Test basic image operations
            test_image = cv2.imread(str(PROJECT_ROOT / "data" / "raw_videos" / "demo_video.mp4"))
            if test_image is not None:
                print("  ✅ Video file readable")
            else:
                print("  ⚠️  No test video found, will create synthetic data")
            
            # Test synthetic object detection (without YOLO)
            synthetic_detections = [
                {
                    'class_name': 'orange',
                    'confidence': 0.85,
                    'bbox': [100, 150, 50, 50],
                    'center_3d': (0.2, 0.3, 1.1)
                },
                {
                    'class_name': 'cup',
                    'confidence': 0.92,
                    'bbox': [200, 180, 40, 60],
                    'center_3d': (0.4, 0.1, 1.1)
                }
            ]
            print(f"  ✅ Synthetic detection ready: {len(synthetic_detections)} objects")
            
            self.components_ready['vision'] = True
            return True
            
        except ImportError as e:
            print(f"  ❌ Vision test failed: {e}")
            print("  💡 Install with: pip install opencv-python")
            return False
    
    def _test_robot_lightweight(self):
        """Test robot simulation without GUI"""
        
        print("\n🦾 Testing robot simulation (lightweight)...")
        
        try:
            # Try PyBullet import
            import pybullet as p
            print(f"  ✅ PyBullet available")
            
            # Test headless connection
            physics_client = p.connect(p.DIRECT)  # No GUI
            print("  ✅ PyBullet headless connection successful")
            
            # Test basic physics setup
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1./240.)
            print("  ✅ Physics parameters set")
            
            # Test simple object creation
            plane_id = p.loadURDF("plane.urdf")
            print("  ✅ Ground plane loaded")
            
            # Test box creation (simplified robot)
            box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            robot_id = p.createMultiBody(baseMass=1.0, 
                                       baseCollisionShapeIndex=box_collision,
                                       baseVisualShapeIndex=box_visual,
                                       basePosition=[0, 0, 1])
            print("  ✅ Simple robot created")
            
            # Test simulation step
            for _ in range(10):
                p.stepSimulation()
            print("  ✅ Simulation stepping works")
            
            # Cleanup
            p.disconnect()
            print("  ✅ PyBullet cleanup successful")
            
            self.components_ready['robot'] = True
            return True
            
        except ImportError as e:
            print(f"  ❌ Robot test failed: {e}")
            print("  💡 Install with: pip install pybullet")
            return False
        except Exception as e:
            print(f"  ❌ Robot simulation error: {e}")
            return False
    
    def _test_llm_lightweight(self):
        """Test LLM connection without heavy inference"""
        
        print("\n🧠 Testing LLM connection (lightweight)...")
        
        try:
            # Test if Ollama is available
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("  ✅ Ollama service accessible")
                
                # Check if DeepSeek model is available
                if 'deepseek' in result.stdout.lower():
                    print("  ✅ DeepSeek model found")
                    self.components_ready['llm'] = True
                else:
                    print("  ⚠️  DeepSeek model not found")
                    print("  💡 Install with: ollama pull deepseek-r1:7b")
            else:
                print("  ❌ Ollama not accessible")
                print("  💡 Install Ollama and run: ollama serve")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("  ❌ Ollama service timeout")
            return False
        except FileNotFoundError:
            print("  ❌ Ollama not installed")
            print("  💡 Install from: https://ollama.ai")
            return False
        except Exception as e:
            print(f"  ❌ LLM test error: {e}")
            return False
    
    def _run_minimal_pipeline(self):
        """Run a minimal end-to-end pipeline"""
        
        print("\n🔄 Running minimal pipeline...")
        
        try:
            # Step 1: Create synthetic scene data
            print("  📊 Creating synthetic scene...")
            scene_objects = [
                {'name': 'orange', 'position': (0.2, 0.3, 1.1), 'confidence': 0.85},
                {'name': 'cup', 'position': (0.4, 0.1, 1.1), 'confidence': 0.92},
                {'name': 'book', 'position': (-0.1, 0.2, 1.1), 'confidence': 0.78}
            ]
            print(f"    ✅ {len(scene_objects)} objects in scene")
            
            # Step 2: Generate task plan
            print("  🧠 Generating task plan...")
            task_steps = [
                "1. Analyze scene layout and object positions",
                "2. Plan approach path to first object (orange)",
                "3. Execute pick motion for orange",
                "4. Move to designated pile location",
                "5. Place orange at pile center",
                "6. Repeat for remaining objects",
                "7. Verify task completion"
            ]
            
            for i, step in enumerate(task_steps, 1):
                print(f"    Step {i}: {step.split('. ')[1]}")
                time.sleep(0.2)  # Simulate processing time
            
            print("    ✅ Task plan generated")
            
            # Step 3: Simulate execution
            print("  🦾 Simulating robot execution...")
            
            execution_results = []
            for obj in scene_objects:
                print(f"    🔄 Processing {obj['name']}...")
                
                # Simulate pick and place
                time.sleep(0.5)
                
                result = {
                    'object': obj['name'],
                    'action': 'pick_and_place',
                    'success': True,
                    'duration': 2.5
                }
                execution_results.append(result)
                print(f"    ✅ {obj['name']} processed successfully")
            
            # Step 4: Generate results
            print("  📊 Generating results...")
            
            success_rate = sum(1 for r in execution_results if r['success']) / len(execution_results)
            total_time = sum(r['duration'] for r in execution_results)
            
            print(f"    📈 Success rate: {success_rate:.1%}")
            print(f"    ⏱️  Total time: {total_time:.1f}s")
            print(f"    🎯 Objects processed: {len(execution_results)}")
            
            # Save results
            output_dir = PROJECT_ROOT / "results" / "lightweight_demo"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_data = {
                'scene_objects': scene_objects,
                'task_steps': task_steps,
                'execution_results': execution_results,
                'metrics': {
                    'success_rate': success_rate,
                    'total_time': total_time,
                    'objects_processed': len(execution_results)
                }
            }
            
            import json
            with open(output_dir / "lightweight_demo_results.json", 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"    💾 Results saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Pipeline error: {e}")
            return False

def main():
    """Main function"""
    
    print("🔧 TANGRAM Lightweight Demo")
    print("Designed to avoid system freezing and test components safely.")
    print()
    
    demo = LightweightDemo()
    
    try:
        success = demo.run_lightweight_demo()
        
        if success:
            print("\n💡 Next Steps:")
            print("1. If all components are ready, try: python tangram.py demo")
            print("2. If components failed, install missing dependencies")
            print("3. For GUI demo: python src/tangram/gui/interactive_gui.py")
            return 0
        else:
            print("\n❌ Demo encountered issues. Check the error messages above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())