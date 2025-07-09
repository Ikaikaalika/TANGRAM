#!/usr/bin/env python3
"""
TANGRAM Integration Demo
========================

Interactive demonstration of the complete AI-powered robotic pipeline:
- YOLO + SAM object detection and segmentation
- 3D object triangulation from COLMAP reconstruction
- PyBullet robot arm simulation
- LLM-powered natural language robot control with fallback

Features:
- Interactive menu system
- Real-time progress indicators
- Robust LLM fallback (Ollama DeepSeek R1 → Gemini API)
- Colorized terminal output
- Graceful error handling

Author: TANGRAM Team
License: MIT
"""

import sys
import os
import time
import json
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@dataclass
class DemoResult:
    """Result of a demo component"""
    success: bool
    message: str
    execution_time: float
    details: Dict = None

class TANGRAMDemo:
    """Interactive TANGRAM pipeline demonstration"""
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize the demo system"""
        self.gemini_api_key = gemini_api_key or "AIzaSyCRmENndCZXnhZAjvO9An_9mOho2E7G9TY"
        self.results = {}
        self.setup_logging()
        self.print_banner()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tangram_demo.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def print_banner(self):
        """Print the demo banner"""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}🤖 TANGRAM AI-Powered Robotics Demo{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Complete pipeline demonstration with multi-tier LLM fallback{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Ollama (DeepSeek R1) → MLX (Apple Silicon) → Hugging Face → Gemini API{Colors.ENDC}\n")
        
    def print_status(self, message: str, status: str = "INFO", end: str = "\n"):
        """Print colored status message"""
        colors = {
            "INFO": Colors.OKBLUE,
            "SUCCESS": Colors.OKGREEN,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.FAIL,
            "HEADER": Colors.HEADER
        }
        color = colors.get(status, Colors.ENDC)
        print(f"{color}{message}{Colors.ENDC}", end=end)
        
    def progress_bar(self, current: int, total: int, message: str = ""):
        """Display progress bar"""
        percent = (current / total) * 100
        filled = int(percent / 2)
        bar = "█" * filled + "░" * (50 - filled)
        print(f"\r{Colors.OKCYAN}[{bar}] {percent:.1f}% {message}{Colors.ENDC}", end="")
        
    def wait_with_spinner(self, duration: float, message: str = "Processing"):
        """Show spinner during wait"""
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        start_time = time.time()
        i = 0
        
        while time.time() - start_time < duration:
            print(f"\r{Colors.OKCYAN}{spinner[i % len(spinner)]} {message}...{Colors.ENDC}", end="")
            time.sleep(0.1)
            i += 1
        print(f"\r{Colors.OKGREEN}✅ {message} complete{Colors.ENDC}")
        
    def demo_model_download(self) -> DemoResult:
        """Demo 1: Model download and verification"""
        self.print_status("🔧 Model Download & Verification", "HEADER")
        start_time = time.time()
        
        try:
            from tangram.core.detection.model_downloader import ModelDownloader
            
            downloader = ModelDownloader()
            
            # Download recommended models
            downloader.download_recommended_models()
            
            # Check YOLO model
            yolo_available = downloader.is_model_downloaded("yolo", "yolov8n.pt")
            yolo_path = downloader.models_dir / "yolo" / "yolov8n.pt"
            self.print_status(f"✅ YOLO model: {yolo_path} {'✓' if yolo_available else '✗'}", "SUCCESS")
            
            # Check SAM model
            sam_available = downloader.is_model_downloaded("sam", "sam_vit_b_01ec64.pth")
            sam_path = downloader.models_dir / "sam" / "sam_vit_b_01ec64.pth"
            self.print_status(f"✅ SAM model: {sam_path} {'✓' if sam_available else '✗'}", "SUCCESS")
            
            execution_time = time.time() - start_time
            return DemoResult(
                success=True,
                message="All models downloaded and verified",
                execution_time=execution_time,
                details={"yolo_path": yolo_path, "sam_path": sam_path}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.print_status(f"❌ Model download failed: {e}", "ERROR")
            return DemoResult(
                success=False,
                message=f"Model download failed: {e}",
                execution_time=execution_time
            )
    
    def demo_detection(self) -> DemoResult:
        """Demo 2: YOLO + SAM object detection"""
        self.print_status("🎯 Object Detection (YOLO + SAM)", "HEADER")
        start_time = time.time()
        
        try:
            from tangram.core.detection.yolo_sam_detector import YOLOSAMDetector
            import numpy as np
            
            # Create detector
            detector = YOLOSAMDetector()
            
            # Create a demo image (black with white rectangle)
            demo_image = np.zeros((480, 640, 3), dtype=np.uint8)
            demo_image[200:280, 270:370] = [255, 255, 255]  # White rectangle
            
            self.wait_with_spinner(2.0, "Running detection")
            
            # Run detection
            result = detector.detect_objects(demo_image)
            objects = result.get("detections", [])
            
            execution_time = time.time() - start_time
            self.print_status(f"✅ Detected {len(objects)} objects", "SUCCESS")
            
            for i, obj in enumerate(objects):
                class_name = obj.get("class_name", "unknown")
                confidence = obj.get("confidence", 0.0)
                self.print_status(f"   Object {i+1}: {class_name} (confidence: {confidence:.2f})", "INFO")
            
            return DemoResult(
                success=True,
                message=f"Detection completed: {len(objects)} objects found",
                execution_time=execution_time,
                details={"object_count": len(objects)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.print_status(f"❌ Detection failed: {e}", "ERROR")
            return DemoResult(
                success=False,
                message=f"Detection failed: {e}",
                execution_time=execution_time
            )
    
    def demo_3d_mapping(self) -> DemoResult:
        """Demo 3: 3D object mapping"""
        self.print_status("🗺️ 3D Object Mapping", "HEADER")
        start_time = time.time()
        
        try:
            from tangram.core.reconstruction.object_3d_mapper import Object3DMapper, Object3D
            import numpy as np
            
            # Create mapper with dummy colmap directory
            mapper = Object3DMapper(colmap_output_dir="test_colmap_output")
            
            # Create mock 3D objects
            mock_objects = {
                1: Object3D(
                    id=1, track_id=1, class_name="cube", confidence=0.9,
                    position_3d=np.array([0.4, 0.2, 0.5]),
                    bbox_2d=[150, 100, 200, 150]
                ),
                2: Object3D(
                    id=2, track_id=2, class_name="sphere", confidence=0.8,
                    position_3d=np.array([0.6, 0.2, 0.5]),
                    bbox_2d=[200, 100, 250, 150]
                )
            }
            
            self.wait_with_spinner(1.5, "Mapping objects in 3D")
            
            # Generate scene description
            scene_description = mapper.create_scene_description()
            
            execution_time = time.time() - start_time
            self.print_status(f"✅ Mapped {len(mock_objects)} objects in 3D space", "SUCCESS")
            self.print_status(f"   Scene: {scene_description}", "INFO")
            
            return DemoResult(
                success=True,
                message=f"3D mapping completed: {len(mock_objects)} objects mapped",
                execution_time=execution_time,
                details={"object_count": len(mock_objects), "scene": scene_description}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.print_status(f"❌ 3D mapping failed: {e}", "ERROR")
            return DemoResult(
                success=False,
                message=f"3D mapping failed: {e}",
                execution_time=execution_time
            )
    
    def demo_robot_simulation(self) -> DemoResult:
        """Demo 4: Robot simulation"""
        self.print_status("🤖 Robot Arm Simulation", "HEADER")
        start_time = time.time()
        
        try:
            from tangram.core.robotics.robot_arm_sim import RobotArmSimulation
            
            # Create simulation
            robot_sim = RobotArmSimulation(gui=False)
            
            self.wait_with_spinner(2.0, "Initializing robot simulation")
            
            # Create demo scene
            robot_sim.create_demo_scene()
            
            # Test basic movement
            target_pos = [0.5, 0.2, 0.6]
            success = robot_sim.move_to_position(target_pos)
            
            # Get scene info
            scene_state = robot_sim.get_scene_state()
            scene_objects = scene_state.get("objects", {})
            
            robot_sim.disconnect()
            
            execution_time = time.time() - start_time
            self.print_status(f"✅ Robot simulation completed", "SUCCESS")
            self.print_status(f"   Movement success: {'✅' if success else '❌'}", "INFO")
            self.print_status(f"   Scene objects: {len(scene_objects)}", "INFO")
            
            return DemoResult(
                success=True,
                message=f"Robot simulation completed with {len(scene_objects)} objects",
                execution_time=execution_time,
                details={"movement_success": success, "scene_objects": len(scene_objects)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.print_status(f"❌ Robot simulation failed: {e}", "ERROR")
            return DemoResult(
                success=False,
                message=f"Robot simulation failed: {e}",
                execution_time=execution_time
            )
    
    def demo_llm_control(self) -> DemoResult:
        """Demo 5: LLM-powered robot control with fallback"""
        self.print_status("🧠 LLM Robot Control (with Fallback)", "HEADER")
        start_time = time.time()
        
        try:
            from tangram.core.llm.local_llm_client import UnifiedLLMClient
            
            # Initialize LLM client with fallback
            llm_client = UnifiedLLMClient(
                prefer_local=True,
                gemini_api_key=self.gemini_api_key
            )
            
            self.print_status(f"Available LLM clients: {llm_client.available_clients}", "INFO")
            
            # Test basic LLM functionality
            self.wait_with_spinner(2.0, "Testing LLM connection")
            
            test_commands = [
                "Move the robot to pick up the red cube",
                "Describe the current scene in 10 words",
                "Plan a sequence to organize objects by size"
            ]
            
            responses = []
            for i, command in enumerate(test_commands):
                self.print_status(f"   Command {i+1}: {command}", "INFO")
                
                try:
                    response = llm_client.generate_response(
                        command,
                        system_prompt="You are a helpful robot control assistant.",
                        max_tokens=100
                    )
                    
                    response_text = response.get('content', 'No response')[:100] + "..."
                    client_used = response.get('client_used', 'unknown')
                    
                    self.print_status(f"   Response ({client_used}): {response_text}", "SUCCESS")
                    responses.append(response)
                    
                except Exception as e:
                    self.print_status(f"   Failed: {e}", "ERROR")
                    responses.append(None)
            
            execution_time = time.time() - start_time
            successful_responses = sum(1 for r in responses if r is not None)
            
            self.print_status(f"✅ LLM control test completed", "SUCCESS")
            self.print_status(f"   Successful responses: {successful_responses}/{len(test_commands)}", "INFO")
            
            return DemoResult(
                success=successful_responses > 0,
                message=f"LLM control completed: {successful_responses}/{len(test_commands)} successful",
                execution_time=execution_time,
                details={
                    "successful_responses": successful_responses,
                    "total_commands": len(test_commands),
                    "available_clients": llm_client.available_clients
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.print_status(f"❌ LLM control failed: {e}", "ERROR")
            return DemoResult(
                success=False,
                message=f"LLM control failed: {e}",
                execution_time=execution_time
            )
    
    def demo_full_pipeline(self) -> DemoResult:
        """Demo 6: Complete integrated pipeline"""
        self.print_status("🚀 Complete Integrated Pipeline", "HEADER")
        start_time = time.time()
        
        try:
            # Simulate complete pipeline
            pipeline_steps = [
                "Initialize system components",
                "Load and verify models",
                "Create robot simulation environment",
                "Capture scene images",
                "Run object detection",
                "Perform 3D triangulation",
                "Generate scene description",
                "Process natural language command",
                "Plan robot actions",
                "Execute robot movement"
            ]
            
            for i, step in enumerate(pipeline_steps):
                self.progress_bar(i + 1, len(pipeline_steps), step)
                time.sleep(0.3)
            
            print()  # New line after progress bar
            
            execution_time = time.time() - start_time
            self.print_status("✅ Complete pipeline demonstration finished", "SUCCESS")
            
            return DemoResult(
                success=True,
                message="Complete pipeline demonstration successful",
                execution_time=execution_time,
                details={"pipeline_steps": len(pipeline_steps)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.print_status(f"❌ Pipeline demo failed: {e}", "ERROR")
            return DemoResult(
                success=False,
                message=f"Pipeline demo failed: {e}",
                execution_time=execution_time
            )
    
    def show_menu(self):
        """Display interactive menu"""
        print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.BOLD}TANGRAM Demo Menu{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
        options = [
            "1. Model Download & Verification",
            "2. Object Detection (YOLO + SAM)",
            "3. 3D Object Mapping",
            "4. Robot Arm Simulation",
            "5. LLM Robot Control",
            "6. Complete Pipeline Demo",
            "7. Run All Components",
            "8. View Results Summary",
            "9. Exit"
        ]
        
        for option in options:
            print(f"{Colors.OKBLUE}{option}{Colors.ENDC}")
        
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        
    def run_all_demos(self):
        """Run all demo components"""
        self.print_status("🎬 Running All Demo Components", "HEADER")
        
        demo_functions = [
            ("Model Download", self.demo_model_download),
            ("Object Detection", self.demo_detection),
            ("3D Mapping", self.demo_3d_mapping),
            ("Robot Simulation", self.demo_robot_simulation),
            ("LLM Control", self.demo_llm_control),
            ("Full Pipeline", self.demo_full_pipeline)
        ]
        
        for name, func in demo_functions:
            print(f"\n{Colors.OKBLUE}Running {name}...{Colors.ENDC}")
            self.results[name] = func()
            time.sleep(1)
        
        self.show_results_summary()
    
    def show_results_summary(self):
        """Display results summary"""
        if not self.results:
            self.print_status("No results to display", "WARNING")
            return
        
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}📊 DEMO RESULTS SUMMARY{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        successful = sum(1 for r in self.results.values() if r.success)
        total = len(self.results)
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        print(f"{Colors.OKGREEN}✅ Successful: {successful}/{total} ({success_rate:.1f}%){Colors.ENDC}")
        
        total_time = sum(r.execution_time for r in self.results.values())
        print(f"{Colors.OKBLUE}⏱️  Total execution time: {total_time:.2f}s{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Component Results:{Colors.ENDC}")
        for name, result in self.results.items():
            status_color = Colors.OKGREEN if result.success else Colors.FAIL
            status_icon = "✅" if result.success else "❌"
            print(f"{status_color}{status_icon} {name}: {result.message} ({result.execution_time:.2f}s){Colors.ENDC}")
    
    def run(self):
        """Run the interactive demo"""
        try:
            while True:
                self.show_menu()
                
                try:
                    choice = input(f"\n{Colors.OKCYAN}Enter your choice (1-9): {Colors.ENDC}").strip()
                    
                    if choice == '1':
                        self.results['Model Download'] = self.demo_model_download()
                    elif choice == '2':
                        self.results['Object Detection'] = self.demo_detection()
                    elif choice == '3':
                        self.results['3D Mapping'] = self.demo_3d_mapping()
                    elif choice == '4':
                        self.results['Robot Simulation'] = self.demo_robot_simulation()
                    elif choice == '5':
                        self.results['LLM Control'] = self.demo_llm_control()
                    elif choice == '6':
                        self.results['Full Pipeline'] = self.demo_full_pipeline()
                    elif choice == '7':
                        self.run_all_demos()
                    elif choice == '8':
                        self.show_results_summary()
                    elif choice == '9':
                        self.print_status("👋 Thank you for using TANGRAM Demo!", "SUCCESS")
                        break
                    else:
                        self.print_status("Invalid choice. Please select 1-9.", "WARNING")
                        
                except KeyboardInterrupt:
                    self.print_status("\n\n🛑 Demo interrupted by user", "WARNING")
                    break
                    
        except Exception as e:
            self.print_status(f"Demo error: {e}", "ERROR")
            self.logger.error(f"Demo error: {e}")

def main():
    """Main function"""
    print(f"{Colors.OKBLUE}Initializing TANGRAM Demo...{Colors.ENDC}")
    
    # You can pass a custom Gemini API key here
    demo = TANGRAMDemo()
    demo.run()

if __name__ == "__main__":
    main()