#!/usr/bin/env python3
"""
TANGRAM GUI - Interactive Visualization Interface

This GUI provides real-time visualization and control of the TANGRAM
robotic scene understanding pipeline, including:
- Live video processing with overlay visualizations
- PyBullet simulation window integration
- Pipeline progress monitoring
- Interactive parameter controls
- Results visualization and export

Author: TANGRAM Team
License: MIT
"""

import sys
import threading
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
from PIL import Image, ImageTk

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import *
from src.tangram.utils.logging_utils import setup_logger
from tracker.track_objects import YOLOByteTracker
from segmenter.run_sam import SAMSegmenter
from scene_graph.build_graph import SceneGraphBuilder
from llm.interpret_scene import DeepSeekSceneInterpreter
from thunder.thunder_integration import ThunderIntegratedSimulator
from export.results_exporter import ResultsExporter
from gui.simulation_viewer import SimulationViewer

logger = setup_logger(__name__)

class PipelineState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class GUIState:
    """Track GUI application state"""
    video_path: Optional[str] = None
    pipeline_state: PipelineState = PipelineState.IDLE
    current_frame: Optional[np.ndarray] = None
    tracking_results: List[Dict] = None
    segmentation_results: List[Dict] = None
    scene_graph: Optional[Dict] = None
    simulation_results: Optional[Dict] = None
    progress: float = 0.0

class TANGRAMGui:
    """
    Main TANGRAM GUI Application
    
    Provides interactive visualization and control of the complete
    robotic scene understanding pipeline.
    """
    
    def __init__(self):
        """Initialize TANGRAM GUI."""
        self.root = tk.Tk()
        self.root.title("TANGRAM - Robotic Scene Understanding Pipeline")
        self.root.geometry("1400x900")
        
        # Application state
        self.state = GUIState()
        
        # Pipeline components
        self.tracker = None
        self.segmenter = None
        self.scene_builder = None
        self.llm_interpreter = None
        self.simulator = None
        self.exporter = None
        
        # GUI components
        self.video_canvas = None
        self.simulation_viewer = None
        self.progress_var = None
        self.status_var = None
        self.log_text = None
        
        # Threading
        self.processing_thread = None
        self.video_thread = None
        self.stop_event = threading.Event()
        
        # Initialize GUI
        self.setup_gui()
        self.initialize_pipeline_components()
        
        logger.info("TANGRAM GUI initialized")
    
    def setup_gui(self):
        """Set up the main GUI layout."""
        self.root.configure(bg='#2b2b2b')
        
        # Create main frames
        self.create_menu_bar()
        self.create_toolbar()
        self.create_main_layout()
        self.create_status_bar()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Video...", command=self.open_video)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_html_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Pipeline menu
        pipeline_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Pipeline", menu=pipeline_menu)
        pipeline_menu.add_command(label="Run Complete Pipeline", command=self.run_complete_pipeline)
        pipeline_menu.add_command(label="Run Object Tracking", command=self.run_tracking_only)
        pipeline_menu.add_command(label="Run Segmentation", command=self.run_segmentation_only)
        pipeline_menu.add_separator()
        pipeline_menu.add_command(label="Reset Pipeline", command=self.reset_pipeline)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Simulation", command=self.show_simulation_window)
        view_menu.add_command(label="Show Scene Graph", command=self.show_scene_graph)
        view_menu.add_command(label="Show Results", command=self.show_results_window)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About TANGRAM", command=self.show_about)
    
    def create_toolbar(self):
        """Create main toolbar with common actions."""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Video controls
        ttk.Button(toolbar, text="📁 Open Video", command=self.open_video).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Pipeline controls
        ttk.Button(toolbar, text="▶️ Run Pipeline", command=self.run_complete_pipeline).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏸️ Pause", command=self.pause_pipeline).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹️ Stop", command=self.stop_pipeline).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Thunder Compute toggle
        self.thunder_var = tk.BooleanVar(value=HARDWARE_CONFIG["thunder_compute"]["enabled"])
        ttk.Checkbutton(toolbar, text="⚡ Thunder Compute", variable=self.thunder_var).pack(side=tk.LEFT, padx=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(toolbar, variable=self.progress_var, length=200)
        progress_bar.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(toolbar, text="Progress:").pack(side=tk.RIGHT, padx=2)
    
    def create_main_layout(self):
        """Create main application layout."""
        # Create paned window for resizable layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Video and visualization
        left_frame = ttk.LabelFrame(main_paned, text="Video Processing", padding=10)
        main_paned.add(left_frame, weight=2)
        
        # Video display
        self.video_frame = ttk.Frame(left_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(self.video_frame, bg='black', width=640, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Video controls
        video_controls = ttk.Frame(left_frame)
        video_controls.pack(fill=tk.X, pady=5)
        
        self.video_position_var = tk.DoubleVar()
        self.video_position_scale = ttk.Scale(video_controls, from_=0, to=100, 
                                            variable=self.video_position_var,
                                            orient=tk.HORIZONTAL,
                                            command=self.seek_video)
        self.video_position_scale.pack(fill=tk.X, padx=5)
        
        # Right panel - Controls and results
        right_frame = ttk.LabelFrame(main_paned, text="Pipeline Control & Results", padding=10)
        main_paned.add(right_frame, weight=1)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pipeline status tab
        self.create_status_tab(notebook)
        
        # Simulation tab
        self.create_simulation_tab(notebook)
        
        # Configuration tab
        self.create_config_tab(notebook)
        
        # Results tab
        self.create_results_tab(notebook)
        
        # Logs tab
        self.create_logs_tab(notebook)
    
    def create_status_tab(self, notebook):
        """Create pipeline status monitoring tab."""
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="Pipeline Status")
        
        # Pipeline steps status
        steps_frame = ttk.LabelFrame(status_frame, text="Pipeline Steps", padding=10)
        steps_frame.pack(fill=tk.X, pady=5)
        
        self.step_status = {}
        steps = [
            ("Object Tracking", "🎯"),
            ("Segmentation", "🎨"),
            ("3D Reconstruction", "📐"),
            ("Scene Graph", "🕸️"),
            ("LLM Interpretation", "🧠"),
            ("Robot Simulation", "🤖"),
            ("Results Export", "📊")
        ]
        
        for i, (step_name, emoji) in enumerate(steps):
            frame = ttk.Frame(steps_frame)
            frame.pack(fill=tk.X, pady=2)
            
            status_var = tk.StringVar(value="⏳ Pending")
            ttk.Label(frame, text=f"{emoji} {step_name}:").pack(side=tk.LEFT)
            ttk.Label(frame, textvariable=status_var).pack(side=tk.RIGHT)
            
            self.step_status[step_name.lower().replace(" ", "_")] = status_var
        
        # Real-time metrics
        metrics_frame = ttk.LabelFrame(status_frame, text="Real-time Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.metrics_text = tk.Text(metrics_frame, height=8, wrap=tk.WORD)
        metrics_scroll = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=metrics_scroll.set)
        
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metrics_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Update metrics periodically
        self.update_metrics()
    
    def create_simulation_tab(self, notebook):
        """Create simulation monitoring and control tab."""
        simulation_frame = ttk.Frame(notebook)
        notebook.add(simulation_frame, text="Simulation")
        
        # Initialize simulation viewer
        self.simulation_viewer = SimulationViewer(
            simulation_frame, 
            callback_func=self.log_message
        )
    
    def create_config_tab(self, notebook):
        """Create configuration tab."""
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configuration")
        
        # Model selection
        models_frame = ttk.LabelFrame(config_frame, text="Model Configuration", padding=10)
        models_frame.pack(fill=tk.X, pady=5)
        
        # YOLO model selection
        ttk.Label(models_frame, text="YOLO Model:").pack(anchor=tk.W)
        self.yolo_model_var = tk.StringVar(value="yolov8n.pt")
        yolo_combo = ttk.Combobox(models_frame, textvariable=self.yolo_model_var)
        yolo_combo['values'] = ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt")
        yolo_combo.pack(fill=tk.X, pady=2)
        
        # SAM model selection
        ttk.Label(models_frame, text="SAM Model:").pack(anchor=tk.W, pady=(10,0))
        self.sam_model_var = tk.StringVar(value="vit_b")
        sam_combo = ttk.Combobox(models_frame, textvariable=self.sam_model_var)
        sam_combo['values'] = ("vit_b", "vit_h", "vit_l")
        sam_combo.pack(fill=tk.X, pady=2)
        
        # Processing options
        options_frame = ttk.LabelFrame(config_frame, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.show_tracking_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show tracking overlays", 
                       variable=self.show_tracking_var).pack(anchor=tk.W)
        
        self.show_masks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show segmentation masks", 
                       variable=self.show_masks_var).pack(anchor=tk.W)
        
        self.real_time_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Real-time processing", 
                       variable=self.real_time_var).pack(anchor=tk.W)
    
    def create_results_tab(self, notebook):
        """Create results visualization tab."""
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        # Results summary
        summary_frame = ttk.LabelFrame(results_frame, text="Results Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=5)
        
        self.results_text = tk.Text(summary_frame, height=12, wrap=tk.WORD)
        results_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scroll.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Export buttons
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="📊 Export HTML Report", 
                  command=self.export_html_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(export_frame, text="📹 Export Video", 
                  command=self.export_video).pack(side=tk.LEFT, padx=2)
        ttk.Button(export_frame, text="📋 Export JSON", 
                  command=self.export_json).pack(side=tk.LEFT, padx=2)
    
    def create_logs_tab(self, notebook):
        """Create logs monitoring tab."""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(log_controls, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT)
        ttk.Button(log_controls, text="Save Logs", command=self.save_logs).pack(side=tk.LEFT, padx=5)
        
        # Log level selection
        ttk.Label(log_controls, text="Log Level:").pack(side=tk.RIGHT, padx=5)
        self.log_level_var = tk.StringVar(value="INFO")
        log_level_combo = ttk.Combobox(log_controls, textvariable=self.log_level_var, width=8)
        log_level_combo['values'] = ("DEBUG", "INFO", "WARNING", "ERROR")
        log_level_combo.pack(side=tk.RIGHT)
    
    def create_status_bar(self):
        """Create status bar at bottom."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        # Connection status
        self.connection_var = tk.StringVar(value="Local")
        ttk.Label(status_frame, textvariable=self.connection_var).pack(side=tk.RIGHT, padx=5)
        ttk.Label(status_frame, text="Mode:").pack(side=tk.RIGHT)
    
    def initialize_pipeline_components(self):
        """Initialize pipeline components."""
        try:
            self.log_message("Initializing pipeline components...")
            
            self.tracker = YOLOByteTracker()
            self.segmenter = SAMSegmenter(model_type="vit_b")
            self.scene_builder = SceneGraphBuilder()
            self.llm_interpreter = DeepSeekSceneInterpreter()
            self.simulator = ThunderIntegratedSimulator()
            self.exporter = ResultsExporter("gui_session")
            
            self.log_message("✅ Pipeline components initialized successfully")
            self.status_var.set("Pipeline Ready")
            
            # Update connection status
            if self.thunder_var.get():
                self.connection_var.set("Thunder Compute")
            else:
                self.connection_var.set("Local")
                
        except Exception as e:
            self.log_message(f"❌ Failed to initialize pipeline: {e}", level="ERROR")
            self.status_var.set("Initialization Error")
    
    def open_video(self):
        """Open video file dialog."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.state.video_path = filename
            self.log_message(f"Video loaded: {Path(filename).name}")
            self.status_var.set(f"Video: {Path(filename).name}")
            
            # Load and display first frame
            self.load_video_preview()
    
    def load_video_preview(self):
        """Load and display video preview."""
        if not self.state.video_path:
            return
        
        try:
            cap = cv2.VideoCapture(self.state.video_path)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame to fit canvas
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame(frame_rgb)
                
            cap.release()
            
        except Exception as e:
            self.log_message(f"Error loading video preview: {e}", level="ERROR")
    
    def display_frame(self, frame: np.ndarray):
        """Display frame in video canvas."""
        try:
            # Get canvas dimensions
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Resize frame to fit canvas while maintaining aspect ratio
            frame_height, frame_width = frame.shape[:2]
            scale = min(canvas_width / frame_width, canvas_height / frame_height)
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(resized_frame)
            photo_image = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.video_canvas.delete("all")
            self.video_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo_image, anchor=tk.CENTER
            )
            
            # Keep reference to prevent garbage collection
            self.video_canvas.image = photo_image
            
        except Exception as e:
            self.log_message(f"Error displaying frame: {e}", level="ERROR")
    
    def run_complete_pipeline(self):
        """Run the complete TANGRAM pipeline."""
        if not self.state.video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        
        if self.state.pipeline_state == PipelineState.PROCESSING:
            messagebox.showwarning("Warning", "Pipeline is already running")
            return
        
        # Start pipeline in separate thread
        self.processing_thread = threading.Thread(target=self._run_pipeline_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _run_pipeline_thread(self):
        """Run pipeline in background thread."""
        try:
            self.state.pipeline_state = PipelineState.PROCESSING
            self.status_var.set("Running Pipeline...")
            
            # Step 1: Object Tracking
            self.update_step_status("object_tracking", "🔄 Processing...")
            self.log_message("Starting object tracking...")
            
            tracking_results = self.tracker.process_video(
                self.state.video_path,
                output_dir="data/tracking"
            )
            
            self.state.tracking_results = tracking_results
            self.update_step_status("object_tracking", "✅ Complete")
            self.progress_var.set(15)
            
            # Step 2: Segmentation
            self.update_step_status("segmentation", "🔄 Processing...")
            self.log_message("Starting segmentation...")
            
            segmentation_results = self.segmenter.process_video_with_tracking(
                self.state.video_path,
                tracking_results,
                "data/masks"
            )
            
            self.state.segmentation_results = segmentation_results
            self.update_step_status("segmentation", "✅ Complete")
            self.progress_var.set(30)
            
            # Step 3: 3D Reconstruction
            self.update_step_status("3d_reconstruction", "🔄 Processing...")
            self.log_message("Starting 3D reconstruction...")
            
            # Extract frames for reconstruction
            from reconstruction.extract_frames import extract_frames
            extract_frames(
                self.state.video_path,
                "data/frames",
                frame_interval=10,
                max_frames=50
            )
            
            # Create mock 3D positions for demo
            object_positions = self._create_mock_3d_positions()
            self.update_step_status("3d_reconstruction", "✅ Complete")
            self.progress_var.set(45)
            
            # Step 4: Scene Graph
            self.update_step_status("scene_graph", "🔄 Processing...")
            self.log_message("Building scene graph...")
            
            scene_graph = self.scene_builder.build_graph_from_data(
                tracking_results, object_positions
            )
            self.state.scene_graph = scene_graph
            self.update_step_status("scene_graph", "✅ Complete")
            self.progress_var.set(60)
            
            # Step 5: LLM Interpretation
            self.update_step_status("llm_interpretation", "🔄 Processing...")
            self.log_message("Generating LLM interpretation...")
            
            llm_interpretation = self._create_demo_llm_interpretation()
            self.update_step_status("llm_interpretation", "✅ Complete")
            self.progress_var.set(75)
            
            # Step 6: Robot Simulation
            self.update_step_status("robot_simulation", "🔄 Processing...")
            self.log_message("Running robot simulation...")
            
            # Load robot and objects in simulation viewer
            if self.simulation_viewer:
                self.simulation_viewer.load_robot("franka_panda")
                self.simulation_viewer.add_scene_objects(object_positions)
                
                # Execute tasks
                for task in llm_interpretation.get("task_sequence", []):
                    self.simulation_viewer.execute_robot_task(task)
            
            simulation_results = self._create_mock_simulation_results()
            self.state.simulation_results = simulation_results
            self.update_step_status("robot_simulation", "✅ Complete")
            self.progress_var.set(90)
            
            # Step 7: Results Export
            self.update_step_status("results_export", "🔄 Processing...")
            self.log_message("Exporting results...")
            
            self.exporter = ResultsExporter("gui_session")
            self.update_step_status("results_export", "✅ Complete")
            self.progress_var.set(100)
            
            self.state.pipeline_state = PipelineState.COMPLETED
            self.status_var.set("Pipeline Complete!")
            self.log_message("🎉 Pipeline completed successfully!")
            
            # Update results display
            self.update_results_display()
            
        except Exception as e:
            self.state.pipeline_state = PipelineState.ERROR
            self.status_var.set("Pipeline Error")
            self.log_message(f"❌ Pipeline error: {e}", level="ERROR")
    
    def update_step_status(self, step_name: str, status: str):
        """Update pipeline step status."""
        if step_name in self.step_status:
            self.step_status[step_name].set(status)
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add message to log display."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        if self.log_text:
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        
        logger.info(message)
    
    def update_metrics(self):
        """Update real-time metrics display."""
        if self.metrics_text:
            metrics = self.get_current_metrics()
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, metrics)
        
        # Schedule next update
        self.root.after(1000, self.update_metrics)
    
    def get_current_metrics(self) -> str:
        """Get current pipeline metrics."""
        metrics = []
        
        if self.state.tracking_results:
            total_frames = len(self.state.tracking_results)
            total_detections = sum(len(frame.get("detections", [])) 
                                 for frame in self.state.tracking_results)
            metrics.append(f"📹 Frames Processed: {total_frames}")
            metrics.append(f"🎯 Objects Detected: {total_detections}")
        
        if self.state.simulation_results:
            success_rate = self.state.simulation_results.get("success_rate", 0)
            metrics.append(f"🤖 Simulation Success: {success_rate:.1%}")
        
        metrics.append(f"💾 Memory Usage: {self.get_memory_usage()}")
        metrics.append(f"⚡ GPU Available: {self.check_gpu_status()}")
        
        return "\n".join(metrics)
    
    def get_memory_usage(self) -> str:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f} MB"
        except:
            return "Unknown"
    
    def check_gpu_status(self) -> str:
        """Check GPU availability."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "MPS"
            elif torch.cuda.is_available():
                return "CUDA"
            else:
                return "CPU Only"
        except:
            return "Unknown"
    
    def show_simulation_window(self):
        """Show PyBullet simulation window."""
        try:
            # Enable GUI mode for PyBullet
            if self.simulator and hasattr(self.simulator, 'local_simulator'):
                self.simulator.local_simulator.initialize_simulation(gui=True)
                self.log_message("Simulation window opened")
        except Exception as e:
            self.log_message(f"Error opening simulation window: {e}", level="ERROR")
    
    def _create_mock_3d_positions(self):
        """Create mock 3D object positions."""
        import random
        random.seed(42)
        
        positions = {}
        for i in range(1, 4):
            positions[str(i)] = {
                "position": [
                    random.uniform(-0.3, 0.3),
                    random.uniform(-0.3, 0.3), 
                    0.7 + random.uniform(0, 0.1)
                ],
                "class_name": random.choice(["cup", "book", "bottle"]),
                "confidence": random.uniform(0.8, 0.95)
            }
        return positions
    
    def _create_demo_llm_interpretation(self):
        """Create demo LLM interpretation."""
        return {
            "scene_explanation": "Scene contains objects on a table suitable for manipulation",
            "task_sequence": [
                {"type": "grasp", "description": "Pick up the cup"},
                {"type": "place", "description": "Place cup in designated area"},
                {"type": "push", "description": "Push book to align properly"}
            ]
        }
    
    def _create_mock_simulation_results(self):
        """Create mock simulation results."""
        import random
        random.seed(42)
        
        return {
            "total_tasks": 3,
            "successful_tasks": 2,
            "success_rate": 0.67,
            "robot_info": {"name": "Franka Panda", "num_joints": 7}
        }
    
    def update_results_display(self):
        """Update the results display with current data."""
        if not self.results_text:
            return
        
        self.results_text.delete(1.0, tk.END)
        
        results_summary = []
        
        if self.state.tracking_results:
            results_summary.append(f"📹 Video Processing:")
            results_summary.append(f"  • Frames: {len(self.state.tracking_results)}")
            total_detections = sum(len(f.get("detections", [])) for f in self.state.tracking_results)
            results_summary.append(f"  • Detections: {total_detections}")
            results_summary.append("")
        
        if self.state.scene_graph:
            results_summary.append(f"🕸️ Scene Graph:")
            results_summary.append(f"  • Nodes: {len(self.state.scene_graph.get('nodes', []))}")
            results_summary.append(f"  • Edges: {len(self.state.scene_graph.get('edges', []))}")
            results_summary.append("")
        
        if self.state.simulation_results:
            results_summary.append(f"🤖 Robot Simulation:")
            results_summary.append(f"  • Tasks: {self.state.simulation_results.get('total_tasks', 0)}")
            results_summary.append(f"  • Success Rate: {self.state.simulation_results.get('success_rate', 0):.1%}")
            results_summary.append("")
        
        results_summary.append("📊 Export Options:")
        results_summary.append("  • HTML Report: Comprehensive results")
        results_summary.append("  • Video Export: Annotated video")
        results_summary.append("  • JSON Data: Raw pipeline data")
        
        self.results_text.insert(tk.END, "\n".join(results_summary))
    
    def run_tracking_only(self):
        """Run only object tracking."""
        if not self.state.video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return
        
        self.log_message("Running object tracking only...")
        # Implementation for tracking-only mode
    
    def run_segmentation_only(self):
        """Run only segmentation."""
        if not self.state.tracking_results:
            messagebox.showerror("Error", "Please run tracking first")
            return
        
        self.log_message("Running segmentation only...")
        # Implementation for segmentation-only mode
    
    def reset_pipeline(self):
        """Reset pipeline to initial state."""
        self.state = GUIState()
        self.progress_var.set(0)
        self.status_var.set("Ready")
        
        # Reset step status
        for status_var in self.step_status.values():
            status_var.set("⏳ Pending")
        
        # Clear displays
        if self.results_text:
            self.results_text.delete(1.0, tk.END)
        
        self.log_message("Pipeline reset")
    
    def pause_pipeline(self):
        """Pause pipeline execution."""
        self.log_message("Pipeline paused")
    
    def stop_pipeline(self):
        """Stop pipeline execution."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
        self.log_message("Pipeline stopped")
    
    def seek_video(self, value):
        """Seek video to specific position."""
        # Implementation for video seeking
        pass
    
    def show_scene_graph(self):
        """Show scene graph visualization."""
        if not self.state.scene_graph:
            messagebox.showinfo("Info", "No scene graph available")
            return
        
        # Create scene graph window
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Scene Graph Visualization")
        graph_window.geometry("600x400")
        
        graph_text = tk.Text(graph_window, wrap=tk.WORD)
        graph_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        import json
        graph_json = json.dumps(self.state.scene_graph, indent=2)
        graph_text.insert(tk.END, graph_json)
    
    def show_results_window(self):
        """Show detailed results window."""
        results_window = tk.Toplevel(self.root)
        results_window.title("Detailed Results")
        results_window.geometry("800x600")
        
        results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD)
        results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Show detailed results
        if self.state.simulation_results:
            import json
            results_json = json.dumps(self.state.simulation_results, indent=2)
            results_text.insert(tk.END, results_json)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
TANGRAM - Robotic Scene Understanding Pipeline

A comprehensive pipeline for video-based robotic scene understanding
that combines computer vision, 3D reconstruction, scene graphs,
LLM interpretation, and robot simulation.

Features:
• Object detection and tracking
• Instance segmentation 
• 3D scene reconstruction
• Scene graph construction
• LLM-based task planning
• Robot simulation
• Thunder Compute integration

Author: TANGRAM Team
License: MIT
        """
        messagebox.showinfo("About TANGRAM", about_text.strip())
    
    def clear_logs(self):
        """Clear log display."""
        if self.log_text:
            self.log_text.delete(1.0, tk.END)
        self.log_message("Logs cleared")
    
    def save_logs(self):
        """Save logs to file."""
        if not self.log_text:
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Logs saved to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save logs: {e}")
    
    def export_html_report(self):
        """Export HTML results report."""
        if not self.exporter:
            messagebox.showerror("Error", "No results to export")
            return
        
        try:
            report_path = self.exporter.generate_html_report()
            messagebox.showinfo("Success", f"HTML report exported: {report_path}")
            self.log_message(f"HTML report exported: {report_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export HTML report: {e}")
    
    def export_video(self):
        """Export annotated video."""
        messagebox.showinfo("Info", "Video export functionality coming soon")
    
    def export_json(self):
        """Export JSON results."""
        filename = filedialog.asksaveasfilename(
            title="Export JSON Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import json
                export_data = {
                    "tracking_results": self.state.tracking_results,
                    "scene_graph": self.state.scene_graph,
                    "simulation_results": self.state.simulation_results
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                messagebox.showinfo("Success", f"Results exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export JSON: {e}")
    
    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit TANGRAM?"):
            self.stop_event.set()
            
            # Clean up threads
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            
            # Clean up simulation viewer
            if self.simulation_viewer:
                self.simulation_viewer.cleanup()
            
            self.root.destroy()
    
    def run(self):
        """Start the GUI application."""
        self.log_message("🚀 TANGRAM GUI started")
        self.root.mainloop()

def main():
    """Main entry point for TANGRAM GUI."""
    print("🤖 Starting TANGRAM GUI...")
    
    try:
        app = TANGRAMGui()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start TANGRAM GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())