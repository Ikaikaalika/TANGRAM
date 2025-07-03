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

from src.tangram.core.llm.local_llm_client import LocalLLMClient

class TangramGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TANGRAM Interactive - Video to 3D Robot Control")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # State
        self.current_video = None
        self.scene_objects = []
        self.robot_pos = {'x': 0, 'y': 3, 'z': 1.5}
        self.llm_client = None
        
        # Initialize LLM
        try:
            self.llm_client = LocalLLMClient()
            print("✅ LLM Ready")
        except:
            print("❌ LLM Failed")
        
        self.create_gui()
    
    def create_gui(self):
        # Header
        header = tk.Frame(self.root, bg='#34495e', height=60)
        header.pack(fill=tk.X, padx=10, pady=5)
        header.pack_propagate(False)
        
        tk.Label(header, text="TANGRAM Interactive", font=('Helvetica', 20, 'bold'), 
                fg='white', bg='#34495e').pack(side=tk.LEFT, padx=20, pady=15)
        
        tk.Button(header, text="📁 Upload Video", font=('Helvetica', 12, 'bold'),
                 bg='#3498db', fg='white', command=self.upload_video,
                 padx=15, pady=8).pack(side=tk.RIGHT, padx=20, pady=15)
        
        # Main content with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Video & Detection
        self.create_video_tab(notebook)
        
        # Tab 2: 3D Environment
        self.create_3d_tab(notebook)
        
        # Tab 3: LLM Control
        self.create_llm_tab(notebook)
    
    def create_video_tab(self, parent):
        frame = tk.Frame(parent, bg='#2c3e50')
        parent.add(frame, text="📹 Video Analysis")
        
        # Video display area
        video_frame = tk.Frame(frame, bg='#34495e')
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(video_frame, text="Video with Object Detection", 
                font=('Helvetica', 14, 'bold'), fg='white', bg='#34495e').pack(pady=5)
        
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=800, height=600)
        self.video_canvas.pack(pady=10)
        
        # Detection info
        tk.Label(video_frame, text="Detected Objects:", 
                font=('Helvetica', 12, 'bold'), fg='white', bg='#34495e').pack(pady=(20,5))
        
        self.detection_text = scrolledtext.ScrolledText(video_frame, height=10, 
                                                       font=('Courier', 10))
        self.detection_text.pack(fill=tk.X, padx=20, pady=5)
        
        # Process button
        self.process_btn = tk.Button(video_frame, text="🔄 Analyze Video", 
                                    font=('Helvetica', 12, 'bold'),
                                    bg='#e74c3c', fg='white', command=self.process_video,
                                    padx=20, pady=10, state=tk.DISABLED)
        self.process_btn.pack(pady=10)
    
    def create_3d_tab(self, parent):
        frame = tk.Frame(parent, bg='#2c3e50')
        parent.add(frame, text="🏗️ 3D Environment")
        
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
    
    def upload_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_video = file_path
            self.process_btn.config(state=tk.NORMAL)
            self.display_video_frame()
            print(f"📁 Video loaded: {Path(file_path).name}")
    
    def display_video_frame(self):
        if not self.current_video:
            return
        
        cap = cv2.VideoCapture(self.current_video)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame, (800, 600))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self.video_photo = ImageTk.PhotoImage(img)
            
            self.video_canvas.delete("all")
            self.video_canvas.create_image(400, 300, image=self.video_photo)
        
        cap.release()
    
    def process_video(self):
        if not self.current_video:
            return
        
        self.process_btn.config(text="Processing...", state=tk.DISABLED)
        
        # Run in background
        thread = threading.Thread(target=self._process_video_bg)
        thread.daemon = True
        thread.start()
    
    def _process_video_bg(self):
        try:
            # Import TANGRAM pipeline
            from main import TANGRAMPipeline
            
            pipeline = TANGRAMPipeline(self.current_video, "temp")
            results = pipeline.run_tracking()
            
            if results:
                self.root.after(0, self._processing_done)
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            self.root.after(0, lambda: self.process_btn.config(text="🔄 Analyze Video", state=tk.NORMAL))
    
    def _processing_done(self):
        self.process_btn.config(text="✅ Analysis Complete", state=tk.NORMAL)
        
        # Load results
        try:
            with open("data/tracking/tracking_results.json") as f:
                results = json.load(f)
            
            if results and len(results) > 0:
                detections = results[0].get('detections', [])
                self.scene_objects = detections
                
                # Update displays
                self.update_detection_display(detections)
                self.display_detection_video()
                self.update_3d_scene()
                
                self.log_action(f"✅ Detected {len(detections)} objects")
        
        except Exception as e:
            print(f"❌ Results loading error: {e}")
    
    def update_detection_display(self, detections):
        self.detection_text.delete(1.0, tk.END)
        self.detection_text.insert(tk.END, f"Detected {len(detections)} objects:\\n\\n")
        
        for i, det in enumerate(detections):
            self.detection_text.insert(tk.END, 
                f"Object {i+1}: {det.get('class_name', 'unknown')} "
                f"(confidence: {det.get('confidence', 0):.2f})\\n")
    
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
            # Build context
            context = "Current scene: "
            if self.scene_objects:
                obj_names = [obj.get('class_name', 'object') for obj in self.scene_objects]
                context += f"{len(self.scene_objects)} objects detected: {', '.join(obj_names)}. "
            else:
                context += "No objects detected. "
            
            context += f"Robot at position X={self.robot_pos['x']:.1f}, Y={self.robot_pos['y']:.1f}, Z={self.robot_pos['z']:.1f}."
            
            prompt = f"{context}\\n\\nHuman: {command}\\n\\nProvide a brief robot action plan:"
            
            response = self.llm_client.generate_response(prompt)
            self.root.after(0, lambda: self._llm_response(response))
            
            # Execute actions
            self.root.after(0, lambda: self._execute_actions(response, command))
            
        except Exception as e:
            self.root.after(0, lambda: self._llm_response(f"❌ Error: {e}"))
    
    def _llm_response(self, response):
        # Replace "thinking" with actual response
        content = self.chat_display.get(1.0, tk.END)
        content = content.replace("🤖 DeepSeek: Thinking...", f"🤖 DeepSeek: {response}")
        
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.insert(1.0, content)
        self.chat_display.see(tk.END)
    
    def _execute_actions(self, response, command):
        # Simple action parsing
        actions_taken = []
        
        if "center" in command.lower() or "middle" in command.lower():
            self.robot_pos.update({'x': 4, 'y': 3, 'z': 1.5})
            actions_taken.append("Moved to center position")
        
        if "home" in command.lower():
            self.robot_pos.update({'x': 0, 'y': 3, 'z': 1.5})
            actions_taken.append("Returned to home position")
        
        if "pick" in command.lower():
            actions_taken.append("Gripper closed - object picked")
        
        if "stack" in command.lower() or "pile" in command.lower():
            self.robot_pos.update({'x': 4, 'y': 3, 'z': 2.0})
            actions_taken.append("Moved to stacking position")
        
        # Update displays
        if actions_taken:
            self.update_robot_display()
            self.update_3d_scene()
            
            for action in actions_taken:
                self.log_action(action)
    
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
    
    def run(self):
        self.root.mainloop()

def main():
    print("🚀 Starting TANGRAM Interactive GUI...")
    app = TangramGUI()
    app.run()

if __name__ == "__main__":
    main()