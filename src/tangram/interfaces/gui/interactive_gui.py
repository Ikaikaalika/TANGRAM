#!/usr/bin/env python3
"""
TANGRAM Interactive GUI - Clean Version

A complete GUI application that allows users to:
1. Upload videos and see object detection
2. View 3D environment reconstruction 
3. Chat with LLM for robot commands
4. Control robot arms in virtual 3D space
"""

import sys
import threading
import json
import time
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.tangram.pipeline.planning.llm.local_llm_client import LocalLLMClient
from src.tangram.utils.logging_utils import setup_logger, log_function_call

class TangramGUI:
    def __init__(self):
        # Set up logging first
        self.logger = setup_logger(__name__, "tangram_gui.log")
        self.logger.info("=" * 60)
        self.logger.info("TANGRAM GUI INITIALIZATION STARTED")
        self.logger.info("=" * 60)
        
        self.root = tk.Tk()
        self.root.title("TANGRAM Interactive - Video to 3D Robot Control")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        self.logger.debug("Main window configured")
        
        # State
        self.current_media = None
        self.media_type = None  # 'video' or 'image'
        self.scene_objects = []
        self.robot_pos = {'x': 0, 'y': 3, 'z': 1.5}
        self.llm_client = None
        self.logger.debug("GUI state variables initialized")
        
        # Camera state
        self.camera = None
        self.camera_window = None
        self.is_recording = False
        self.camera_thread = None
        self.recorded_frames = []
        self.logger.debug("Camera state variables initialized")
        
        # Initialize LLM
        self.logger.info("Initializing LLM client...")
        try:
            self.llm_client = LocalLLMClient()
            self.logger.info("✅ LLM client initialized successfully")
            print("✅ LLM Ready")
        except Exception as e:
            self.logger.error(f"❌ LLM client initialization failed: {e}")
            print("❌ LLM Failed")
        
        self.logger.info("Creating GUI components...")
        self.create_gui()
        self.logger.info("✅ TANGRAM GUI initialization completed successfully")
    
    def create_gui(self):
        # Header
        header = tk.Frame(self.root, bg='#34495e', height=60)
        header.pack(fill=tk.X, padx=10, pady=5)
        header.pack_propagate(False)
        
        tk.Label(header, text="TANGRAM Interactive", font=('Helvetica', 20, 'bold'), 
                fg='white', bg='#34495e').pack(side=tk.LEFT, padx=20, pady=15)
        
        # Button frame for multiple upload options
        button_frame = tk.Frame(header, bg='#34495e')
        button_frame.pack(side=tk.RIGHT, padx=20, pady=15)
        
        tk.Button(button_frame, text="📁 Upload Media", font=('Helvetica', 12, 'bold'),
                 bg='#3498db', fg='white', command=self.upload_media,
                 padx=15, pady=8).pack(side=tk.RIGHT, padx=(0,10))
        
        tk.Button(button_frame, text="📷 Camera", font=('Helvetica', 12, 'bold'),
                 bg='#e74c3c', fg='white', command=self.open_camera,
                 padx=15, pady=8).pack(side=tk.RIGHT, padx=(0,10))
        
        # Main content with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Media Analysis
        self.create_media_tab(notebook)
        
        # Tab 2: Environment Creation
        self.create_environment_tab(notebook)
        
        # Tab 3: 3D Environment
        self.create_3d_tab(notebook)
        
        # Tab 4: Scene Graph
        self.create_scene_graph_tab(notebook)
        
        # Tab 5: Robot Control
        self.create_llm_tab(notebook)
    
    def create_media_tab(self, parent):
        frame = tk.Frame(parent, bg='#2c3e50')
        parent.add(frame, text="📷 Media Analysis")
        
        # Top control bar with prominent analyze button
        control_bar = tk.Frame(frame, bg='#e74c3c', height=80)
        control_bar.pack(fill=tk.X, padx=10, pady=(10,5))
        control_bar.pack_propagate(False)
        
        tk.Label(control_bar, text="📷 MEDIA ANALYSIS", 
                font=('Helvetica', 16, 'bold'), fg='white', bg='#e74c3c').pack(side=tk.LEFT, padx=20, pady=20)
        
        # Prominent Analyze button at the top
        self.process_btn = tk.Button(control_bar, text="🔄 ANALYZE MEDIA", 
                                    font=('Helvetica', 14, 'bold'),
                                    bg='#95a5a6', fg='#7f8c8d', command=self.process_media,
                                    padx=30, pady=15, state=tk.DISABLED,
                                    relief=tk.RAISED, bd=3)
        self.logger.debug("ANALYZE MEDIA button created in disabled state")
        self.process_btn.pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Media display area
        media_frame = tk.Frame(frame, bg='#34495e')
        media_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(media_frame, text="Media with Object Detection", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        self.media_canvas = tk.Canvas(media_frame, bg='black', width=800, height=600)
        self.media_canvas.pack(pady=10)
        
        # Media info and button status
        info_frame = tk.Frame(media_frame, bg='#34495e')
        info_frame.pack(pady=5)
        
        self.media_info = tk.Label(info_frame, text="No media loaded", 
                                  font=('Helvetica', 10), fg='#bdc3c7', bg='#34495e')
        self.media_info.pack()
        
        self.button_status = tk.Label(info_frame, text="Button: DISABLED", 
                                     font=('Helvetica', 9), fg='#e74c3c', bg='#34495e')
        self.button_status.pack()
        
        # Detection info
        tk.Label(media_frame, text="Detected Objects:", 
                font=('Helvetica', 12, 'bold'), fg='white', bg='#34495e').pack(pady=(20,5))
        
        self.detection_text = scrolledtext.ScrolledText(media_frame, height=8, 
                                                       font=('Courier', 10))
        self.detection_text.pack(fill=tk.X, padx=20, pady=5)
        
        # Add detection tips
        tips_frame = tk.Frame(media_frame, bg='#34495e')
        tips_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(tips_frame, text="💡 For best detection results, use photos with:", 
                font=('Helvetica', 10, 'bold'), fg='#f39c12', bg='#34495e').pack(anchor=tk.W)
        
        tips_text = "• People, vehicles, animals\n• Electronics (laptop, phone, TV)\n• Food items (apple, cup, bottle)\n• Furniture (chair, table, bed)\n• Books, bags, everyday objects"
        tk.Label(tips_frame, text=tips_text, 
                font=('Helvetica', 9), fg='#ecf0f1', bg='#34495e', justify=tk.LEFT).pack(anchor=tk.W, padx=20)
    
    def create_3d_tab(self, parent):
        frame = tk.Frame(parent, bg='#2c3e50')
        parent.add(frame, text="🌍 3D Environment")
        
        # 3D visualization
        self.fig = plt.figure(figsize=(12, 8), facecolor='#2c3e50')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#34495e')
        
        self.canvas_3d = FigureCanvasTkAgg(self.fig, frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize 3D scene
        self.setup_3d_scene()
        
        # Controls
        controls = tk.Frame(frame, bg='#34495e')
        controls.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(controls, text="Reset View", command=self.reset_view,
                 bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Top View", command=self.top_view,
                 bg='#3498db', fg='white').pack(side=tk.LEFT, padx=5)
    
    def create_environment_tab(self, parent):
        frame = tk.Frame(parent, bg='#2c3e50')
        parent.add(frame, text="🏗️ Environment Creation")
        
        # Split into process view and progress
        left_panel = tk.Frame(frame, bg='#34495e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        right_panel = tk.Frame(frame, bg='#34495e', width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=10)
        right_panel.pack_propagate(False)
        
        # Environment creation visualization (left)
        tk.Label(left_panel, text="3D Environment Creation Process", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        # Canvas for step-by-step visualization
        self.env_canvas = tk.Canvas(left_panel, bg='#2c3e50', width=800, height=600)
        self.env_canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Process steps display
        steps_frame = tk.Frame(left_panel, bg='#34495e')
        steps_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(steps_frame, text="Creation Steps:", 
                font=('Helvetica', 12, 'bold'), fg='white', bg='#34495e').pack(anchor=tk.W)
        
        self.steps_text = scrolledtext.ScrolledText(steps_frame, height=8, 
                                                   font=('Courier', 10),
                                                   bg='#2c3e50', fg='#00ff00')
        self.steps_text.pack(fill=tk.X, pady=5)
        
        # Control panel (right)
        tk.Label(right_panel, text="Environment Controls", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        # Process control buttons
        control_frame = tk.Frame(right_panel, bg='#34495e')
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_env_btn = tk.Button(control_frame, text="▶️ Build 3D Scene", 
                                      font=('Helvetica', 11, 'bold'),
                                      bg='#27ae60', fg='white', 
                                      command=self.start_environment_creation)
        self.start_env_btn.pack(fill=tk.X, pady=2)
        
        self.pause_env_btn = tk.Button(control_frame, text="⏸️ Pause", 
                                      font=('Helvetica', 11, 'bold'),
                                      bg='#f39c12', fg='white',
                                      command=self.pause_environment_creation,
                                      state=tk.DISABLED)
        self.pause_env_btn.pack(fill=tk.X, pady=2)
        
        self.reset_env_btn = tk.Button(control_frame, text="🔄 Reset", 
                                      font=('Helvetica', 11, 'bold'),
                                      bg='#e74c3c', fg='white',
                                      command=self.reset_environment_creation)
        self.reset_env_btn.pack(fill=tk.X, pady=2)
        
        # Progress indicators
        progress_frame = tk.LabelFrame(right_panel, text="Progress", 
                                     fg='white', bg='#34495e')
        progress_frame.pack(fill=tk.X, pady=10)
        
        # Overall progress
        tk.Label(progress_frame, text="Overall Progress:", 
                fg='white', bg='#34495e').pack(anchor=tk.W, padx=5)
        self.overall_progress = ttk.Progressbar(progress_frame, length=250)
        self.overall_progress.pack(fill=tk.X, padx=5, pady=2)
        
        # Current step progress
        tk.Label(progress_frame, text="Current Step:", 
                fg='white', bg='#34495e').pack(anchor=tk.W, padx=5, pady=(10,0))
        self.step_progress = ttk.Progressbar(progress_frame, length=250)
        self.step_progress.pack(fill=tk.X, padx=5, pady=2)
        
        # Status indicators
        status_frame = tk.LabelFrame(right_panel, text="Status", 
                                   fg='white', bg='#34495e')
        status_frame.pack(fill=tk.X, pady=10)
        
        self.env_status_labels = {}
        creation_steps = [
            ('frame_extraction', 'Media Processing'),
            ('object_detection', 'Object Detection'),
            ('depth_estimation', 'Depth Estimation'),
            ('3d_reconstruction', '3D Reconstruction'),
            ('scene_mapping', 'Scene Mapping')
        ]
        
        for step_id, step_name in creation_steps:
            frame = tk.Frame(status_frame, bg='#34495e')
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            tk.Label(frame, text=step_name, 
                    font=('Helvetica', 10), 
                    fg='white', bg='#34495e').pack(side=tk.LEFT)
            
            status_label = tk.Label(frame, text="⏳ Pending", 
                                  font=('Helvetica', 10, 'bold'),
                                  fg='gray', bg='#34495e')
            status_label.pack(side=tk.RIGHT)
            self.env_status_labels[step_id] = status_label
        
        # Visualization options
        viz_frame = tk.LabelFrame(right_panel, text="Visualization", 
                                fg='white', bg='#34495e')
        viz_frame.pack(fill=tk.X, pady=10)
        
        self.show_frames_var = tk.BooleanVar(value=True)
        tk.Checkbutton(viz_frame, text="Show Frames", 
                      variable=self.show_frames_var, 
                      fg='white', bg='#34495e').pack(anchor=tk.W, padx=5)
        
        self.show_detections_var = tk.BooleanVar(value=True)
        tk.Checkbutton(viz_frame, text="Show Detections", 
                      variable=self.show_detections_var,
                      fg='white', bg='#34495e').pack(anchor=tk.W, padx=5)
        
        self.show_depth_var = tk.BooleanVar(value=True)
        tk.Checkbutton(viz_frame, text="Show Depth Map", 
                      variable=self.show_depth_var,
                      fg='white', bg='#34495e').pack(anchor=tk.W, padx=5)
        
        # Initialize environment creation state
        self.env_creation_active = False
        self.env_creation_paused = False
        self.env_creation_step = 0
        self.total_env_steps = len(creation_steps)
    
    def create_scene_graph_tab(self, parent):
        frame = tk.Frame(parent, bg='#2c3e50')
        parent.add(frame, text="🕸️ Scene Graph")
        
        # Split into graph view and timeline
        top_panel = tk.Frame(frame, bg='#34495e')
        top_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        bottom_panel = tk.Frame(frame, bg='#34495e', height=150)
        bottom_panel.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        bottom_panel.pack_propagate(False)
        
        # Scene graph visualization (top)
        tk.Label(top_panel, text="Temporal Spatial Scene Graph", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        # Create matplotlib figure for scene graph
        self.graph_fig = plt.figure(figsize=(12, 8), facecolor='#2c3e50')
        self.graph_ax = self.graph_fig.add_subplot(111)
        self.graph_ax.set_facecolor('#34495e')
        
        self.canvas_graph = FigureCanvasTkAgg(self.graph_fig, top_panel)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Timeline controls (bottom)
        timeline_frame = tk.Frame(bottom_panel, bg='#34495e')
        timeline_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(timeline_frame, text="Timeline Controls", 
                font=('Helvetica', 12, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        # Timeline slider
        timeline_control_frame = tk.Frame(timeline_frame, bg='#34495e')
        timeline_control_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(timeline_control_frame, text="Time:", 
                fg='white', bg='#34495e').pack(side=tk.LEFT)
        
        self.timeline_var = tk.DoubleVar(value=0)
        self.timeline_scale = tk.Scale(timeline_control_frame, 
                                     from_=0, to=100, 
                                     orient=tk.HORIZONTAL,
                                     variable=self.timeline_var,
                                     command=self.update_scene_graph_time,
                                     bg='#34495e', fg='white',
                                     highlightbackground='#34495e')
        self.timeline_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.time_label = tk.Label(timeline_control_frame, text="0.0s", 
                                  fg='white', bg='#34495e')
        self.time_label.pack(side=tk.RIGHT)
        
        # Playback controls
        playback_frame = tk.Frame(timeline_frame, bg='#34495e')
        playback_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(playback_frame, text="⏮️", command=self.graph_first_frame,
                 bg='#3498db', fg='white').pack(side=tk.LEFT, padx=2)
        tk.Button(playback_frame, text="⏪", command=self.graph_prev_frame,
                 bg='#3498db', fg='white').pack(side=tk.LEFT, padx=2)
        
        self.graph_play_btn = tk.Button(playback_frame, text="▶️", 
                                       command=self.toggle_graph_playback,
                                       bg='#27ae60', fg='white')
        self.graph_play_btn.pack(side=tk.LEFT, padx=2)
        
        tk.Button(playback_frame, text="⏩", command=self.graph_next_frame,
                 bg='#3498db', fg='white').pack(side=tk.LEFT, padx=2)
        tk.Button(playback_frame, text="⏭️", command=self.graph_last_frame,
                 bg='#3498db', fg='white').pack(side=tk.LEFT, padx=2)
        
        # Graph options
        options_frame = tk.Frame(playback_frame, bg='#34495e')
        options_frame.pack(side=tk.RIGHT)
        
        self.show_relations_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Show Relations", 
                      variable=self.show_relations_var,
                      command=self.update_scene_graph_display,
                      fg='white', bg='#34495e').pack(side=tk.LEFT, padx=5)
        
        self.show_temporal_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Show Temporal", 
                      variable=self.show_temporal_var,
                      command=self.update_scene_graph_display,
                      fg='white', bg='#34495e').pack(side=tk.LEFT, padx=5)
        
        # Initialize scene graph state
        self.scene_graph_data = []
        self.graph_playing = False
        self.graph_frame_rate = 10  # FPS for playback
        self.current_graph_time = 0
        
        # Initialize empty scene graph
        self.setup_scene_graph()
    
    def setup_scene_graph(self):
        """Initialize the scene graph visualization"""
        # Clear any existing graph
        self.graph_ax.clear()
        
        # Set up the axes for scene graph display
        self.graph_ax.set_facecolor('#2c3e50')
        self.graph_ax.set_title("Scene Graph - No Data", color='white', fontsize=12)
        
        # Remove axes for cleaner look
        self.graph_ax.set_xticks([])
        self.graph_ax.set_yticks([])
        
        # Initial drawing
        self.canvas_graph.draw()
    
    def create_llm_tab(self, parent):
        frame = tk.Frame(parent, bg='#2c3e50')
        parent.add(frame, text="🧠 Robot Control")
        
        # Split into chat and robot control
        left_panel = tk.Frame(frame, bg='#34495e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        right_panel = tk.Frame(frame, bg='#34495e')  
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Chat interface (left)
        tk.Label(left_panel, text="Chat with DeepSeek R1", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(left_panel, height=20, 
                                                     font=('Helvetica', 11),
                                                     bg='#2c3e50', fg='white')
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Input area
        input_frame = tk.Frame(left_panel, bg='#34495e')
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.command_entry = tk.Entry(input_frame, font=('Helvetica', 12))
        self.command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.command_entry.bind('<Return>', self.send_command)
        
        tk.Button(input_frame, text="Send", command=self.send_command,
                 bg='#27ae60', fg='white', font=('Helvetica', 12, 'bold')).pack(side=tk.RIGHT)
        
        # Quick commands
        quick_frame = tk.Frame(left_panel, bg='#34495e')
        quick_frame.pack(fill=tk.X, padx=10, pady=5)
        
        commands = [
            "Pick up the red object",
            "Move all objects to center", 
            "Stack objects in a pile",
            "Go to home position"
        ]
        
        for cmd in commands:
            tk.Button(quick_frame, text=cmd, 
                     command=lambda c=cmd: self.quick_command(c),
                     bg='#9b59b6', fg='white', font=('Helvetica', 9)).pack(fill=tk.X, pady=1)
        
        # Robot status (right)
        tk.Label(right_panel, text="Robot Status", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        # Position display
        pos_frame = tk.LabelFrame(right_panel, text="Position", fg='white', bg='#34495e')
        pos_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pos_labels = {}
        for coord in ['X', 'Y', 'Z']:
            frame = tk.Frame(pos_frame, bg='#34495e')
            frame.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(frame, text=f"{coord}:", fg='white', bg='#34495e').pack(side=tk.LEFT)
            self.pos_labels[coord] = tk.Label(frame, text="0.00", 
                                            font=('Courier', 11, 'bold'),
                                            fg='#3498db', bg='#34495e')
            self.pos_labels[coord].pack(side=tk.RIGHT)
        
        # Manual controls
        manual_frame = tk.LabelFrame(right_panel, text="Manual Control", 
                                    fg='white', bg='#34495e')
        manual_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Movement buttons
        for axis in ['X', 'Y', 'Z']:
            axis_frame = tk.Frame(manual_frame, bg='#34495e')
            axis_frame.pack(fill=tk.X, pady=2)
            
            tk.Button(axis_frame, text=f"{axis}-", 
                     command=lambda a=axis.lower(): self.move_robot(a, -0.5),
                     bg='#e67e22', fg='white').pack(side=tk.LEFT, padx=2)
            tk.Button(axis_frame, text=f"{axis}+", 
                     command=lambda a=axis.lower(): self.move_robot(a, 0.5),
                     bg='#e67e22', fg='white').pack(side=tk.RIGHT, padx=2)
        
        # Action log
        tk.Label(right_panel, text="Action Log", 
                font=('Helvetica', 12, 'bold'), fg='white', bg='#34495e').pack(pady=(10,5))
        
        self.action_log = scrolledtext.ScrolledText(right_panel, height=10, 
                                                   font=('Courier', 9),
                                                   bg='#2c3e50', fg='#00ff00')
        self.action_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    @log_function_call()
    def upload_media(self):
        self.logger.info("Opening media upload dialog")
        file_path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=[
                ("Media files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.logger.info(f"User selected media file: {file_path}")
            self.current_media = file_path
            self.determine_media_type()
            
            # Enable the analyze button with extra logging
            try:
                self.process_btn.config(state=tk.NORMAL, bg='#27ae60', fg='white', text='🔄 ANALYZE MEDIA')
                self.button_status.config(text="Button: ENABLED ✅", fg='#27ae60')
                self.logger.info("✅ ANALYZE MEDIA button enabled and styled")
                print("✅ ANALYZE MEDIA button is now ENABLED!")
            except Exception as e:
                self.logger.error(f"❌ Failed to enable button: {e}")
            
            self.display_media_frame()
            print(f"📁 Media loaded: {Path(file_path).name}")
        else:
            self.logger.info("User cancelled media selection")
    
    def determine_media_type(self):
        if not self.current_media:
            return
        
        ext = Path(self.current_media).suffix.lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            self.media_type = 'video'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            self.media_type = 'image'
        else:
            self.media_type = 'unknown'
    
    def display_media_frame(self):
        if not self.current_media:
            return
        
        if self.media_type == 'video':
            cap = cv2.VideoCapture(self.current_media)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (800, 600))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                self.media_photo = ImageTk.PhotoImage(img)
                
                self.media_canvas.delete("all")
                self.media_canvas.create_image(400, 300, image=self.media_photo)
                
                # Update info
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                self.media_info.config(text=f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
            
            cap.release()
            
        elif self.media_type == 'image':
            # Load and display image
            img = cv2.imread(self.current_media)
            if img is not None:
                # Get original dimensions
                h, w = img.shape[:2]
                
                # Resize maintaining aspect ratio
                max_size = 800
                if w > h:
                    new_w = max_size
                    new_h = int(h * max_size / w)
                else:
                    new_h = max_size
                    new_w = int(w * max_size / h)
                
                img = cv2.resize(img, (new_w, new_h))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                self.media_photo = ImageTk.PhotoImage(img_pil)
                
                self.media_canvas.delete("all")
                self.media_canvas.create_image(400, 300, image=self.media_photo)
                
                # Update info
                self.media_info.config(text=f"Image: {w}x{h} pixels")
    
    @log_function_call()
    def process_media(self):
        if not self.current_media:
            self.logger.warning("Process media called but no media loaded")
            return
        
        self.logger.info(f"Starting media processing for: {self.current_media}")
        self.logger.debug(f"Media type: {self.media_type}")
        self.process_btn.config(text="Processing...", state=tk.DISABLED)
        
        # Run in background
        self.logger.debug("Starting background processing thread")
        thread = threading.Thread(target=self._process_media_bg)
        thread.daemon = True
        thread.start()
    
    def _process_media_bg(self):
        try:
            self.logger.info("Starting background media processing")
            
            # Use direct YOLO detection instead of full pipeline to avoid LLM issues
            from src.tangram.pipeline.perception.tracker.track_objects import YOLOByteTracker
            
            tracker = YOLOByteTracker()
            
            # Process media directly
            results = tracker.process_media(self.current_media, "data/tracking")
            self.logger.info(f"Detection completed, found {len(results)} frames")
            
            if results and len(results) > 0:
                # Extract detections from first frame
                detections = results[0].get('detections', [])
                self.logger.info(f"Found {len(detections)} objects in media")
                
                # Store results and update GUI
                self.scene_objects = detections
                self.root.after(0, self._processing_done)
            else:
                self.logger.warning("No detection results returned")
                self.scene_objects = []
                self.root.after(0, self._processing_done)
            
        except Exception as e:
            self.logger.error(f"Media processing failed: {e}")
            print(f"❌ Processing error: {e}")
            self.root.after(0, lambda: self.process_btn.config(text="🔄 Analyze Media", state=tk.NORMAL))
    
    def _processing_done(self):
        self.logger.info("Processing completed, updating GUI")
        self.process_btn.config(text="✅ Analysis Complete", state=tk.NORMAL)
        
        # Update displays with stored results
        if hasattr(self, 'scene_objects') and self.scene_objects:
            detections = self.scene_objects
            self.logger.info(f"Updating GUI with {len(detections)} detected objects")
            
            # Update displays
            self.update_detection_display(detections)
            self.display_detection_overlay()
            self.update_3d_scene()
            
            self.log_action(f"✅ Detected {len(detections)} objects")
        else:
            self.logger.info("No objects detected, updating GUI accordingly")
            self.scene_objects = []
            self.update_detection_display([])
            self.log_action("No objects detected")
    
    def update_detection_display(self, detections):
        self.detection_text.delete(1.0, tk.END)
        
        if not detections:
            self.detection_text.insert(tk.END, "❌ No objects detected\\n\\n")
            self.detection_text.insert(tk.END, "Possible reasons:\\n")
            self.detection_text.insert(tk.END, "• Image contains no recognizable objects\\n")
            self.detection_text.insert(tk.END, "• Objects too small or unclear\\n") 
            self.detection_text.insert(tk.END, "• Confidence below 30% threshold\\n")
            self.detection_text.insert(tk.END, "• Try photos with common objects listed above\\n")
            self.logger.info("No objects detected in current media")
            return
        
        self.detection_text.insert(tk.END, f"✅ Detected {len(detections)} objects:\\n\\n")
        
        for i, det in enumerate(detections):
            confidence = det.get('confidence', 0)
            bbox = det.get('bbox', [0, 0, 0, 0])
            self.detection_text.insert(tk.END, 
                f"Object {i+1}: {det.get('class_name', 'unknown')}\\n"
                f"  Confidence: {confidence:.1%} ({confidence:.3f})\\n"
                f"  Position: x={bbox[0]:.0f}, y={bbox[1]:.0f}\\n"
                f"  Size: {bbox[2]:.0f}×{bbox[3]:.0f}px\\n\\n")
            
            self.logger.debug(f"Detected {det.get('class_name')}: confidence={confidence:.3f}, bbox={bbox}")
    
    def display_detection_overlay(self):
        """Display media with detection overlays"""
        if not self.current_media or not self.scene_objects:
            return
        
        # This method would overlay detection boxes on the media
        # For now, just redisplay the original media
        self.display_media_frame()
    
    @log_function_call()
    def open_camera(self):
        """Open camera window for capturing video or photos"""
        self.logger.info("Opening camera window")
        if self.camera_window is not None:
            self.logger.debug("Camera window already open, bringing to front")
            self.camera_window.lift()
            return
        
        # Create camera window
        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title("TANGRAM Camera")
        self.camera_window.geometry("800x700")
        self.camera_window.configure(bg='#2c3e50')
        self.camera_window.protocol("WM_DELETE_WINDOW", self.close_camera)
        
        # Camera preview frame
        preview_frame = tk.Frame(self.camera_window, bg='#34495e')
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(preview_frame, text="Camera Preview", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(preview_frame, bg='black', width=640, height=480)
        self.camera_canvas.pack(pady=10)
        
        # Camera status
        self.camera_status = tk.Label(preview_frame, text="Initializing camera...", 
                                     fg='white', bg='#34495e', font=('Helvetica', 12))
        self.camera_status.pack(pady=5)
        
        # Control buttons
        control_frame = tk.Frame(self.camera_window, bg='#2c3e50')
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Photo capture button
        self.photo_btn = tk.Button(control_frame, text="📸 Take Photo", 
                                  font=('Helvetica', 12, 'bold'),
                                  bg='#27ae60', fg='white', 
                                  command=self.capture_photo,
                                  padx=20, pady=10)
        self.photo_btn.pack(side=tk.LEFT, padx=10)
        
        # Video recording button
        self.record_btn = tk.Button(control_frame, text="🎥 Start Recording", 
                                   font=('Helvetica', 12, 'bold'),
                                   bg='#e74c3c', fg='white', 
                                   command=self.toggle_recording,
                                   padx=20, pady=10)
        self.record_btn.pack(side=tk.LEFT, padx=10)
        
        # Recording status
        self.recording_status = tk.Label(control_frame, text="", 
                                        fg='#e74c3c', bg='#2c3e50', 
                                        font=('Helvetica', 12, 'bold'))
        self.recording_status.pack(side=tk.LEFT, padx=20)
        
        # Close button
        tk.Button(control_frame, text="❌ Close Camera", 
                 font=('Helvetica', 12, 'bold'),
                 bg='#95a5a6', fg='white', 
                 command=self.close_camera,
                 padx=20, pady=10).pack(side=tk.RIGHT, padx=10)
        
        # Initialize camera
        self.init_camera()
    
    def init_camera(self):
        """Initialize the camera"""
        try:
            self.camera = cv2.VideoCapture(0)  # Default camera
            if not self.camera.isOpened():
                self.camera_status.config(text="❌ Could not open camera")
                return
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_status.config(text="✅ Camera ready")
            
            # Start camera preview
            self.start_camera_preview()
            
        except Exception as e:
            self.camera_status.config(text=f"❌ Camera error: {e}")
    
    def start_camera_preview(self):
        """Start the camera preview"""
        if self.camera is None or not self.camera.isOpened():
            return
        
        def update_preview():
            if self.camera_window is None:
                return
            
            ret, frame = self.camera.read()
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Update canvas
                self.camera_canvas.delete("all")
                self.camera_canvas.create_image(320, 240, image=photo)
                self.camera_canvas.image = photo  # Keep a reference
                
                # Record frame if recording
                if self.is_recording:
                    self.recorded_frames.append(frame.copy())
            
            # Schedule next update
            if self.camera_window is not None:
                self.camera_window.after(33, update_preview)  # ~30 FPS
        
        update_preview()
    
    def capture_photo(self):
        """Capture a single photo"""
        if self.camera is None or not self.camera.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        ret, frame = self.camera.read()
        if not ret:
            messagebox.showerror("Error", "Could not capture photo")
            return
        
        # Save photo
        from pathlib import Path
        import time
        
        timestamp = int(time.time())
        photo_path = Path("data/raw_videos") / f"photo_{timestamp}.jpg"
        photo_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(photo_path), frame)
        
        # Update GUI with captured photo
        self.current_media = str(photo_path)
        self.media_type = 'image'
        
        # Display in main window
        self.display_media_frame()
        self.media_info.config(text=f"📸 Captured: {photo_path.name}")
        
        self.log_action(f"📸 Photo captured: {photo_path.name}")
        messagebox.showinfo("Success", f"Photo saved as {photo_path.name}")
    
    def toggle_recording(self):
        """Start or stop video recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start video recording"""
        if self.camera is None or not self.camera.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        self.is_recording = True
        self.recorded_frames = []
        
        # Update UI
        self.record_btn.config(text="⏹️ Stop Recording", bg='#c0392b')
        self.recording_status.config(text="🔴 RECORDING")
        self.photo_btn.config(state='disabled')
        
        self.log_action("🎥 Video recording started")
    
    def stop_recording(self):
        """Stop video recording and save"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Update UI
        self.record_btn.config(text="🎥 Start Recording", bg='#e74c3c')
        self.recording_status.config(text="💾 Saving...")
        self.photo_btn.config(state='normal')
        
        # Save video in background
        thread = threading.Thread(target=self._save_video)
        thread.daemon = True
        thread.start()
    
    def _save_video(self):
        """Save recorded video"""
        if not self.recorded_frames:
            self.root.after(0, lambda: self.recording_status.config(text="❌ No frames to save"))
            return
        
        try:
            from pathlib import Path
            import time
            
            timestamp = int(time.time())
            video_path = Path("data/raw_videos") / f"video_{timestamp}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Video writer settings
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (640, 480)
            
            out = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)
            
            # Write frames
            for frame in self.recorded_frames:
                resized_frame = cv2.resize(frame, frame_size)
                out.write(resized_frame)
            
            out.release()
            
            # Update main GUI
            self.root.after(0, lambda: self._video_saved(video_path))
            
        except Exception as e:
            self.root.after(0, lambda: self.recording_status.config(text=f"❌ Save error: {e}"))
    
    def _video_saved(self, video_path):
        """Called when video is successfully saved"""
        self.recording_status.config(text="✅ Video saved")
        
        # Update main GUI with recorded video
        self.current_media = str(video_path)
        self.media_type = 'video'
        
        # Display in main window
        self.display_media_frame()
        self.media_info.config(text=f"🎥 Recorded: {video_path.name}")
        
        self.log_action(f"🎥 Video saved: {video_path.name}")
        
        # Clear recorded frames to free memory
        self.recorded_frames.clear()
        
        # Show success message
        messagebox.showinfo("Success", f"Video saved as {video_path.name}")
    
    def close_camera(self):
        """Close camera and clean up"""
        if self.is_recording:
            self.stop_recording()
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        if self.camera_window is not None:
            self.camera_window.destroy()
            self.camera_window = None
        
        self.log_action("📷 Camera closed")
    
    def setup_3d_scene(self):
        self.ax.clear()
        self.ax.set_xlim(-2, 8)
        self.ax.set_ylim(-1, 7) 
        self.ax.set_zlim(0, 4)
        self.ax.set_xlabel('X (meters)', color='white')
        self.ax.set_ylabel('Y (meters)', color='white')
        self.ax.set_zlabel('Z (meters)', color='white')
        self.ax.set_title('3D Virtual Environment', color='white', fontsize=14)
        
        # Table
        xx, yy = np.meshgrid(np.linspace(1, 7, 10), np.linspace(1, 5, 10))
        zz = np.ones_like(xx) * 1.0
        self.ax.plot_surface(xx, yy, zz, alpha=0.3, color='brown')
        
        self.canvas_3d.draw()
    
    def update_3d_scene(self):
        self.setup_3d_scene()
        
        # Add detected objects
        if self.scene_objects:
            for obj in self.scene_objects:
                bbox = obj.get('bbox', [0, 0, 100, 100])
                x_2d = bbox[0] + bbox[2]/2
                y_2d = bbox[1] + bbox[3]/2
                
                # Convert to 3D coordinates
                x_3d = 2 + (x_2d / 800) * 4
                y_3d = 2 + (y_2d / 600) * 2
                z_3d = 1.1
                
                class_name = obj.get('class_name', 'object')
                if 'orange' in class_name:
                    color = 'orange'
                elif 'frisbee' in class_name:
                    color = 'green'
                else:
                    color = 'blue'
                
                self.ax.scatter([x_3d], [y_3d], [z_3d], c=color, s=300, alpha=0.8)
                self.ax.text(x_3d, y_3d, z_3d + 0.2, class_name, fontsize=10)
        
        # Draw robot arm
        self.draw_robot_arm()
        
        self.canvas_3d.draw()
    
    def draw_robot_arm(self):
        base = [0, 3, 1]
        x, y, z = self.robot_pos['x'], self.robot_pos['y'], self.robot_pos['z']
        
        # Simple 3-segment arm
        joint1 = [base[0] + x*0.3, base[1] + (y-3)*0.2, base[2] + 0.5]
        joint2 = [base[0] + x*0.6, base[1] + (y-3)*0.4, base[2] + 1.0]
        end_effector = [base[0] + x*0.8, base[1] + (y-3)*0.6, z]
        
        points = [base, joint1, joint2, end_effector]
        
        # Draw segments
        for i in range(len(points) - 1):
            self.ax.plot3D([points[i][0], points[i+1][0]], 
                          [points[i][1], points[i+1][1]], 
                          [points[i][2], points[i+1][2]], 
                          'k-', linewidth=4)
        
        # Draw joints
        for point in points:
            self.ax.scatter([point[0]], [point[1]], [point[2]], c='red', s=100)
    
    def send_command(self, event=None):
        command = self.command_entry.get().strip()
        if command:
            self.command_entry.delete(0, tk.END)
            self.process_llm_command(command)
    
    def quick_command(self, command):
        self.process_llm_command(command)
    
    def process_llm_command(self, command):
        self.chat_display.insert(tk.END, f"🧑 You: {command}\\n\\n")
        self.chat_display.insert(tk.END, "🤖 DeepSeek: Thinking...\\n\\n")
        self.chat_display.see(tk.END)
        
        # Process in background
        thread = threading.Thread(target=self._llm_process, args=(command,))
        thread.daemon = True
        thread.start()
    
    def _llm_process(self, command):
        if not self.llm_client:
            self.root.after(0, lambda: self._llm_response("❌ LLM not available"))
            return
        
        try:
            # Build comprehensive context with scene graph
            context = self._build_scene_context()
            
            prompt = f"{context}\\n\\nHuman: {command}\\n\\nProvide a detailed robot action plan with specific coordinates and steps:"
            
            response = self.llm_client.generate_response(prompt)
            self.root.after(0, lambda: self._llm_response(response))
            
            # Execute actions
            self.root.after(0, lambda: self._execute_actions(response, command))
            
        except Exception as e:
            self.root.after(0, lambda: self._llm_response(f"❌ Error: {e}"))
    
    def _build_scene_context(self):
        """Build comprehensive scene context for LLM"""
        context = "SCENE ANALYSIS:\\n"
        
        # Media information
        if self.current_media:
            media_name = Path(self.current_media).name
            context += f"Media: {media_name} ({self.media_type})\\n"
        
        # Detected objects with spatial information
        if self.scene_objects:
            context += f"Objects detected: {len(self.scene_objects)}\\n"
            for i, obj in enumerate(self.scene_objects):
                bbox = obj.get('bbox', [0, 0, 100, 100])
                x_center = bbox[0] + bbox[2]/2
                y_center = bbox[1] + bbox[3]/2
                
                # Convert to approximate 3D coordinates
                x_3d = 2 + (x_center / 800) * 4
                y_3d = 2 + (y_center / 600) * 2
                z_3d = 1.1  # On table surface
                
                context += f"  - {obj.get('class_name', 'unknown')} at ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f}) confidence={obj.get('confidence', 0):.2f}\\n"
        else:
            context += "No objects detected.\\n"
        
        # Robot state
        context += f"\\nROBOT STATE:\\n"
        context += f"Position: X={self.robot_pos['x']:.1f}, Y={self.robot_pos['y']:.1f}, Z={self.robot_pos['z']:.1f}\\n"
        context += f"Workspace: X(0-8), Y(0-6), Z(0-4)\\n"
        context += f"Table surface at Z=1.0\\n"
        
        # Available actions
        context += f"\\nAVAILABLE ACTIONS:\\n"
        context += f"- Move to coordinates: move_to(x, y, z)\\n"
        context += f"- Pick up object: pick_up(object_name)\\n"
        context += f"- Place object: place_at(x, y, z)\\n"
        context += f"- Go to home: go_home()\\n"
        
        return context
    
    def _llm_response(self, response):
        # Replace "thinking" with actual response
        content = self.chat_display.get(1.0, tk.END)
        content = content.replace("🤖 DeepSeek: Thinking...", f"🤖 DeepSeek: {response}")
        
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.insert(1.0, content)
        self.chat_display.see(tk.END)
    
    def _execute_actions(self, response, command):
        """Execute actions based on LLM response and command"""
        actions_taken = []
        
        # Enhanced action parsing with coordinate extraction
        response_lower = response.lower()
        command_lower = command.lower()
        
        # Parse specific coordinates from response
        import re
        coord_pattern = r'move.*?to.*?(\d+\.?\d*)[,\s]+(\d+\.?\d*)[,\s]+(\d+\.?\d*)'
        coord_match = re.search(coord_pattern, response_lower)
        
        if coord_match:
            x, y, z = float(coord_match.group(1)), float(coord_match.group(2)), float(coord_match.group(3))
            # Validate coordinates are within workspace
            x = max(0, min(8, x))
            y = max(0, min(6, y))
            z = max(0, min(4, z))
            
            self.robot_pos.update({'x': x, 'y': y, 'z': z})
            actions_taken.append(f"Moved to coordinates ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # Fallback to keyword-based actions
        elif "center" in command_lower or "middle" in command_lower:
            self.robot_pos.update({'x': 4, 'y': 3, 'z': 1.5})
            actions_taken.append("Moved to center position")
        
        elif "home" in command_lower:
            self.robot_pos.update({'x': 0, 'y': 3, 'z': 1.5})
            actions_taken.append("Returned to home position")
        
        elif "pick" in command_lower and self.scene_objects:
            # Try to pick up the first detected object
            obj = self.scene_objects[0]
            obj_name = obj.get('class_name', 'object')
            
            # Move to object position
            bbox = obj.get('bbox', [0, 0, 100, 100])
            x_3d = 2 + ((bbox[0] + bbox[2]/2) / 800) * 4
            y_3d = 2 + ((bbox[1] + bbox[3]/2) / 600) * 2
            z_3d = 1.1
            
            self.robot_pos.update({'x': x_3d, 'y': y_3d, 'z': z_3d})
            actions_taken.append(f"Moved to {obj_name} at ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})")
            actions_taken.append(f"Picked up {obj_name}")
        
        elif "stack" in command_lower or "pile" in command_lower:
            self.robot_pos.update({'x': 4, 'y': 3, 'z': 2.0})
            actions_taken.append("Moved to stacking position")
        
        # Object-specific actions
        elif any(obj.get('class_name', '').lower() in command_lower for obj in self.scene_objects):
            # Find the mentioned object
            for obj in self.scene_objects:
                obj_name = obj.get('class_name', '').lower()
                if obj_name in command_lower:
                    bbox = obj.get('bbox', [0, 0, 100, 100])
                    x_3d = 2 + ((bbox[0] + bbox[2]/2) / 800) * 4
                    y_3d = 2 + ((bbox[1] + bbox[3]/2) / 600) * 2
                    z_3d = 1.1
                    
                    self.robot_pos.update({'x': x_3d, 'y': y_3d, 'z': z_3d})
                    actions_taken.append(f"Moved to {obj_name} at ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})")
                    break
        
        # Update displays
        if actions_taken:
            self.update_robot_display()
            self.update_3d_scene()
            
            for action in actions_taken:
                self.log_action(action)
        else:
            self.log_action("Command understood but no physical actions taken")
    
    def move_robot(self, axis, delta):
        if axis in self.robot_pos:
            self.robot_pos[axis] += delta
            self.update_robot_display()
            self.update_3d_scene()
            self.log_action(f"Manual move: {axis.upper()} {delta:+.1f}")
    
    def update_robot_display(self):
        for coord, label in self.pos_labels.items():
            value = self.robot_pos[coord.lower()]
            label.config(text=f"{value:.2f}")
    
    def log_action(self, action):
        timestamp = time.strftime("%H:%M:%S")
        self.action_log.insert(tk.END, f"[{timestamp}] {action}\\n")
        self.action_log.see(tk.END)
    
    def reset_view(self):
        self.ax.view_init(elev=20, azim=45)
        self.canvas_3d.draw()
    
    def top_view(self):
        self.ax.view_init(elev=90, azim=0)
        self.canvas_3d.draw()
    
    def start_environment_creation(self):
        """Start the environment creation process"""
        if self.env_creation_active:
            return
        
        self.env_creation_active = True
        media_type = getattr(self, 'media_type', 'media')
        self.log_action(f"Starting 3D environment creation from {media_type}")
        
        # Update button states
        self.start_env_btn.config(state='disabled')
        self.pause_env_btn.config(state='normal')
        self.reset_env_btn.config(state='normal')
        
        # Start creation process in background
        thread = threading.Thread(target=self._environment_creation_process)
        thread.daemon = True
        thread.start()
    
    def pause_environment_creation(self):
        """Pause the environment creation process"""
        self.env_creation_active = False
        self.log_action("3D environment creation paused")
        
        # Update button states
        self.start_env_btn.config(state='normal')
        self.pause_env_btn.config(state='disabled')
    
    def reset_environment_creation(self):
        """Reset the environment creation process"""
        self.env_creation_active = False
        self.log_action("3D environment creation reset")
        
        # Reset progress bars
        self.overall_progress['value'] = 0
        self.step_progress['value'] = 0
        
        # Reset status labels
        for label in self.env_status_labels.values():
            label.config(text="⏳ Pending", fg='#95a5a6')
        
        # Clear steps text
        self.steps_text.delete(1.0, tk.END)
        
        # Update button states
        self.start_env_btn.config(state='normal')
        self.pause_env_btn.config(state='disabled')
        self.reset_env_btn.config(state='disabled')
    
    def _environment_creation_process(self):
        """Background environment creation process"""
        # Determine media type for appropriate step descriptions
        is_video = self.media_type == 'video' if hasattr(self, 'media_type') else True
        
        if is_video:
            steps = [
                ("frame_extraction", "Extracting frames from video"),
                ("object_detection", "Running object detection"),
                ("depth_estimation", "Estimating depth information"),
                ("3d_reconstruction", "Reconstructing 3D scene"),
                ("scene_mapping", "Building scene graph")
            ]
        else:
            steps = [
                ("frame_extraction", "Loading image data"),
                ("object_detection", "Running object detection"),
                ("depth_estimation", "Estimating depth information"),
                ("3d_reconstruction", "Creating 3D scene layout"),
                ("scene_mapping", "Building scene graph")
            ]
        
        total_steps = len(steps)
        
        for i, (step_id, step_desc) in enumerate(steps):
            if not self.env_creation_active:
                break
            
            # Update overall progress
            progress = (i / total_steps) * 100
            self.root.after(0, lambda p=progress: self.overall_progress.config(value=p))
            
            # Update status
            self.root.after(0, lambda sid=step_id: self.env_status_labels[sid].config(
                text="🔄 Processing", fg='#f39c12'))
            
            # Add step to log
            self.root.after(0, lambda desc=step_desc: self.steps_text.insert(tk.END, f"Starting: {desc}\\n"))
            
            # Simulate step processing
            for j in range(10):
                if not self.env_creation_active:
                    break
                
                step_progress = (j / 10) * 100
                self.root.after(0, lambda sp=step_progress: self.step_progress.config(value=sp))
                time.sleep(0.5)
            
            # Mark step complete
            self.root.after(0, lambda sid=step_id: self.env_status_labels[sid].config(
                text="✅ Complete", fg='#27ae60'))
            
            # Update step log
            self.root.after(0, lambda desc=step_desc: self.steps_text.insert(tk.END, f"Completed: {desc}\\n"))
        
        # Final completion
        if self.env_creation_active:
            self.root.after(0, lambda: self.overall_progress.config(value=100))
            self.root.after(0, lambda: self.step_progress.config(value=100))
            self.root.after(0, lambda: self.log_action("3D environment creation completed successfully"))
        
        self.env_creation_active = False
    
    def graph_first_frame(self):
        """Go to first frame of scene graph"""
        self.timeline_var.set(0)
        self.update_scene_graph_display()
    
    def graph_prev_frame(self):
        """Go to previous frame of scene graph"""
        current = self.timeline_var.get()
        if current > 0:
            self.timeline_var.set(current - 1)
            self.update_scene_graph_display()
    
    def graph_next_frame(self):
        """Go to next frame of scene graph"""
        current = self.timeline_var.get()
        max_time = self.timeline_scale['to']
        if current < max_time:
            self.timeline_var.set(current + 1)
            self.update_scene_graph_display()
    
    def graph_last_frame(self):
        """Go to last frame of scene graph"""
        max_time = self.timeline_scale['to']
        self.timeline_var.set(max_time)
        self.update_scene_graph_display()
    
    def toggle_graph_playback(self):
        """Toggle playback of scene graph"""
        if hasattr(self, 'graph_playing') and self.graph_playing:
            self.graph_playing = False
            self.graph_play_btn.config(text="▶️")
        else:
            self.graph_playing = True
            self.graph_play_btn.config(text="⏸️")
            self._start_graph_playback()
    
    def _start_graph_playback(self):
        """Start automatic playback of scene graph"""
        if hasattr(self, 'graph_playing') and self.graph_playing:
            current = self.timeline_var.get()
            max_time = self.timeline_scale['to']
            
            if current < max_time:
                self.timeline_var.set(current + 0.5)
                self.update_scene_graph_display()
                self.root.after(500, self._start_graph_playback)
            else:
                self.graph_playing = False
                self.graph_play_btn.config(text="▶️")
    
    def update_scene_graph_time(self, value=None):
        """Update scene graph display when timeline changes"""
        if value is not None:
            self.timeline_var.set(float(value))
        self.update_scene_graph_display()
    
    def update_scene_graph_display(self):
        """Update the scene graph visualization"""
        current_time = self.timeline_var.get()
        self.time_label.config(text=f"{current_time:.1f}s")
        
        # Clear the graph
        self.graph_ax.clear()
        
        # Draw a simple scene graph based on current time
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes based on detected objects
        if self.scene_objects:
            for i, obj in enumerate(self.scene_objects):
                obj_name = obj.get('class_name', f'object_{i}')
                G.add_node(obj_name)
        
        # Add some temporal relationships
        if len(G.nodes()) > 1:
            nodes = list(G.nodes())
            for i in range(len(nodes) - 1):
                if current_time > i:  # Show relationships progressively
                    G.add_edge(nodes[i], nodes[i + 1])
        
        # Draw the graph
        if G.nodes():
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=self.graph_ax, with_labels=True, 
                   node_color='lightblue', node_size=1000, 
                   font_size=8, font_weight='bold')
        
        self.graph_ax.set_title(f"Scene Graph at t={current_time:.1f}s", 
                               color='white', fontsize=12)
        self.graph_ax.set_facecolor('#2c3e50')
        self.canvas_graph.draw()
    
    def run(self):
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        # Clean up camera if open
        if self.camera is not None:
            self.close_camera()
        
        # Destroy main window
        self.root.destroy()

def main():
    print("🚀 Starting TANGRAM Interactive GUI...")
    app = TangramGUI()
    app.run()

if __name__ == "__main__":
    main()