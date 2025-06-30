#!/usr/bin/env python3
"""
TANGRAM Manager GUI

A comprehensive graphical interface for managing TANGRAM pipeline training,
simulation, and Thunder Compute resources.

Features:
- Thunder Compute instance management
- Pipeline execution and monitoring
- Training data management
- Simulation visualization
- Real-time progress tracking

Author: TANGRAM Team
License: MIT
"""

import sys
import os
import json
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import HARDWARE_CONFIG, DATA_DIR, RESULTS_DIR

class ThunderComputePanel:
    """Thunder Compute management panel."""
    
    def __init__(self, parent, manager):
        self.parent = parent
        self.manager = manager
        self.frame = ttk.LabelFrame(parent, text="Thunder Compute Management", padding="10")
        self.instance_data = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup Thunder Compute UI components."""
        # Status display
        self.status_frame = ttk.Frame(self.frame)
        self.status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.status_frame, text="Status:").pack(side=tk.LEFT)
        self.status_label = ttk.Label(self.status_frame, text="Checking...", foreground="orange")
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Instance list
        self.tree_frame = ttk.Frame(self.frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for instances
        columns = ("ID", "Status", "Address", "GPU", "vCPUs", "RAM", "Mode")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons with better alignment
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Group buttons by function with consistent spacing
        button_config = {'width': 12, 'padding': (5, 3)}
        
        ttk.Button(self.button_frame, text="Refresh", command=self.refresh_instances, **button_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(self.button_frame, text="Create Instance", command=self.create_instance_dialog, **button_config).pack(side=tk.LEFT, padx=(0, 15))
        
        # Instance control buttons grouped together
        ttk.Button(self.button_frame, text="Start", command=self.start_instance, **button_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.button_frame, text="Stop", command=self.stop_instance, **button_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.button_frame, text="Delete", command=self.delete_instance, **button_config).pack(side=tk.LEFT, padx=(0, 5))
        
        # Auto-refresh
        self.refresh_instances()
        self.auto_refresh()
    
    def auto_refresh(self):
        """Auto-refresh instance status every 30 seconds."""
        self.refresh_instances()
        self.parent.after(30000, self.auto_refresh)
    
    def refresh_instances(self):
        """Refresh Thunder Compute instance list."""
        try:
            env = os.environ.copy()
            env['TNR_DISABLE_COMPLETION'] = '1'
            result = subprocess.run(['tnr', 'status', '--no-wait'], 
                                  capture_output=True, text=True, timeout=10, env=env)
            
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            if result.returncode == 0:
                self.status_label.config(text="Connected", foreground="green")
                
                # Parse instance data
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '│' in line and any(char.isdigit() for char in line):
                        parts = [p.strip() for p in line.split('│') if p.strip()]
                        if len(parts) >= 8 and parts[0] != 'ID':
                            instance_id = parts[0]
                            if instance_id != '--':
                                self.tree.insert('', 'end', values=parts[:7])
                                self.instance_data[instance_id] = {
                                    'status': parts[1],
                                    'address': parts[2],
                                    'gpu': parts[4],
                                    'vcpus': parts[5],
                                    'ram': parts[6]
                                }
            else:
                self.status_label.config(text="Error", foreground="red")
                
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)[:20]}...", foreground="red")
    
    def get_selected_instance(self):
        """Get currently selected instance ID."""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            return item['values'][0]
        return None
    
    def create_instance_dialog(self):
        """Show enhanced create instance dialog with sliders and visual controls."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Create Thunder Compute Instance")
        dialog.geometry("500x650")
        dialog.resizable(False, False)
        
        # Make dialog modal
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (650 // 2)
        dialog.geometry(f"500x650+{x}+{y}")
        
        # Main frame with padding
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Create New Thunder Compute Instance", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Instance Configuration", padding="15")
        config_frame.pack(fill=tk.X, pady=(0, 20))
        
        # GPU selection with visual cards
        gpu_frame = ttk.LabelFrame(config_frame, text="GPU Type", padding="10")
        gpu_frame.pack(fill=tk.X, pady=(0, 15))
        
        gpu_var = tk.StringVar(value="t4")
        
        # GPU option cards
        gpu_options = [
            {"value": "t4", "name": "NVIDIA T4", "vram": "16GB", "desc": "Best for most ML workloads", "color": "#4CAF50"},
            {"value": "a100", "name": "NVIDIA A100", "vram": "40GB", "desc": "High-performance computing", "color": "#FF9800"},
            {"value": "a100xl", "name": "NVIDIA A100 XL", "vram": "80GB", "desc": "Largest models available", "color": "#F44336"}
        ]
        
        gpu_buttons = []
        for i, gpu in enumerate(gpu_options):
            card_frame = ttk.Frame(gpu_frame)
            card_frame.pack(fill=tk.X, pady=2)
            
            def make_gpu_selector(value):
                return lambda: gpu_var.set(value) or self.update_gpu_selection(gpu_buttons, value)
            
            radio = ttk.Radiobutton(card_frame, text="", variable=gpu_var, value=gpu["value"], 
                                   command=make_gpu_selector(gpu["value"]))
            radio.pack(side=tk.LEFT, padx=(0, 10))
            
            info_frame = ttk.Frame(card_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            name_label = ttk.Label(info_frame, text=f"{gpu['name']} ({gpu['vram']})", 
                                  font=('Arial', 10, 'bold'))
            name_label.pack(anchor=tk.W)
            
            desc_label = ttk.Label(info_frame, text=gpu['desc'], foreground="gray")
            desc_label.pack(anchor=tk.W)
            
            gpu_buttons.append((radio, card_frame, gpu["value"]))
        
        # vCPU selection with slider
        vcpu_frame = ttk.LabelFrame(config_frame, text="vCPUs & RAM", padding="10")
        vcpu_frame.pack(fill=tk.X, pady=(0, 15))
        
        vcpu_var = tk.IntVar(value=4)
        ram_var = tk.StringVar(value="32GB")
        
        vcpu_control_frame = ttk.Frame(vcpu_frame)
        vcpu_control_frame.pack(fill=tk.X)
        
        ttk.Label(vcpu_control_frame, text="vCPUs:").pack(side=tk.LEFT)
        vcpu_value_label = ttk.Label(vcpu_control_frame, text="4", font=('Arial', 12, 'bold'))
        vcpu_value_label.pack(side=tk.RIGHT)
        
        # Create RAM label first
        ram_label = ttk.Label(vcpu_frame, text="RAM: 32GB", foreground="gray")
        ram_label.pack(pady=(5, 0))
        
        def update_vcpu(val):
            vcpus = int(float(val))
            ram = vcpus * 8
            vcpu_value_label.config(text=str(vcpus))
            ram_label.config(text=f"RAM: {ram}GB")
            vcpu_var.set(vcpus)
            ram_var.set(f"{ram}GB")
        
        vcpu_scale = ttk.Scale(vcpu_frame, from_=4, to=32, orient=tk.HORIZONTAL, 
                              command=update_vcpu, length=400)
        vcpu_scale.set(4)
        vcpu_scale.pack(fill=tk.X, pady=(5, 0))
        
        # Scale tick marks
        tick_frame = ttk.Frame(vcpu_frame)
        tick_frame.pack(fill=tk.X)
        for i, cpu in enumerate([4, 8, 16, 32]):
            ttk.Label(tick_frame, text=str(cpu), font=('Arial', 8)).place(relx=i/3, anchor=tk.CENTER)
        
        # Mode selection with toggle buttons
        mode_frame = ttk.LabelFrame(config_frame, text="Instance Mode", padding="15")
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        mode_var = tk.StringVar(value="prototyping")
        
        mode_buttons_frame = ttk.Frame(mode_frame)
        mode_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Create buttons with consistent sizing and styling
        button_style = {'width': 20, 'padding': (10, 8)}
        
        proto_button = ttk.Button(mode_buttons_frame, text="Prototyping\n(Cost Optimized)", 
                                 command=lambda: self.set_mode(mode_var, "prototyping", proto_button, prod_button),
                                 **button_style)
        proto_button.pack(side=tk.LEFT, padx=(0, 15), expand=True, fill=tk.X)
        
        prod_button = ttk.Button(mode_buttons_frame, text="Production\n(High Reliability)", 
                                command=lambda: self.set_mode(mode_var, "production", proto_button, prod_button),
                                **button_style)
        prod_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Set initial mode button state
        self.set_mode(mode_var, "prototyping", proto_button, prod_button)
        
        # Quick presets
        preset_frame = ttk.LabelFrame(config_frame, text="Quick Presets", padding="15")
        preset_frame.pack(fill=tk.X, pady=(0, 15))
        
        preset_buttons_frame = ttk.Frame(preset_frame)
        preset_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        def apply_preset(gpu, vcpus, mode, template="none"):
            gpu_var.set(gpu)
            vcpu_var.set(vcpus)
            vcpu_scale.set(vcpus)
            mode_var.set(mode)
            template_var.set(template)
            update_vcpu(vcpus)
            self.set_mode(mode_var, mode, proto_button, prod_button)
        
        # Consistent preset button styling
        preset_style = {'width': 15, 'padding': (8, 6)}
        
        ttk.Button(preset_buttons_frame, text="Budget\n(T4, 4 vCPUs)", 
                  command=lambda: apply_preset("t4", 4, "prototyping"), **preset_style).pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)
        ttk.Button(preset_buttons_frame, text="Balanced\n(A100, 8 vCPUs)", 
                  command=lambda: apply_preset("a100", 8, "prototyping"), **preset_style).pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)
        ttk.Button(preset_buttons_frame, text="High-End\n(A100XL, 16 vCPUs)", 
                  command=lambda: apply_preset("a100xl", 16, "production"), **preset_style).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Template selection
        template_frame = ttk.LabelFrame(config_frame, text="Pre-installed Template (Optional)", padding="10")
        template_frame.pack(fill=tk.X)
        
        template_var = tk.StringVar(value="none")
        
        template_options = [
            {"value": "none", "name": "None", "desc": "Clean Ubuntu environment"},
            {"value": "ollama", "name": "Ollama", "desc": "LLM server environment"},
            {"value": "comfy-ui", "name": "ComfyUI", "desc": "AI image generation"},
            {"value": "webui-forge", "name": "WebUI Forge", "desc": "Stable Diffusion interface"}
        ]
        
        for template in template_options:
            template_radio = ttk.Radiobutton(template_frame, text=f"{template['name']} - {template['desc']}", 
                                           variable=template_var, value=template["value"])
            template_radio.pack(anchor=tk.W, pady=2)
        
        # Cost estimation
        cost_frame = ttk.LabelFrame(main_frame, text="Estimated Cost", padding="10")
        cost_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.cost_label = ttk.Label(cost_frame, text="Calculating...", font=('Arial', 11))
        self.cost_label.pack()
        
        # Update cost estimation when values change
        def update_cost():
            gpu_type = gpu_var.get()
            vcpus = vcpu_var.get()
            mode = mode_var.get()
            
            # Rough cost estimates (these would be from Thunder Compute pricing)
            gpu_costs = {"t4": 0.50, "a100": 1.50, "a100xl": 3.00}
            vcpu_cost = (vcpus - 4) * 0.10  # Additional vCPUs
            mode_multiplier = 1.2 if mode == "production" else 1.0
            
            hourly_cost = (gpu_costs.get(gpu_type, 0.50) + vcpu_cost) * mode_multiplier
            
            self.cost_label.config(text=f"~${hourly_cost:.2f}/hour ({gpu_type.upper()}, {vcpus} vCPUs, {mode})")
            
            # Schedule next update
            dialog.after(500, update_cost)
        
        update_cost()
        
        # Progress and status
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        progress.pack(fill=tk.X, pady=(0, 5))
        
        status_label = ttk.Label(progress_frame, text="Ready to create instance")
        status_label.pack()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def create_instance():
            """Create the instance with selected configuration."""
            progress.start()
            status_label.config(text="Creating instance...")
            create_button.config(state=tk.DISABLED)
            
            def create_thread():
                try:
                    cmd = ['tnr', 'create', '--gpu', gpu_var.get(), '--vcpus', str(vcpu_var.get()), '--mode', mode_var.get()]
                    if template_var.get() != "none":
                        cmd.extend(['--template', template_var.get()])
                    
                    env = os.environ.copy()
                    env['TNR_DISABLE_COMPLETION'] = '1'
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
                    
                    dialog.after(0, lambda: progress.stop())
                    dialog.after(0, lambda: create_button.config(state=tk.NORMAL))
                    
                    if result.returncode == 0:
                        dialog.after(0, lambda: status_label.config(text="Instance created successfully! ✅"))
                        dialog.after(0, lambda: self.refresh_instances())
                        dialog.after(2000, dialog.destroy)
                    else:
                        error_msg = result.stderr.strip()
                        if len(error_msg) > 60:
                            error_msg = error_msg[:60] + "..."
                        dialog.after(0, lambda: status_label.config(text=f"Error: {error_msg}"))
                        
                except Exception as e:
                    error_msg = str(e)
                    if len(error_msg) > 60:
                        error_msg = error_msg[:60] + "..."
                    dialog.after(0, lambda: status_label.config(text=f"Error: {error_msg}"))
                    dialog.after(0, lambda: create_button.config(state=tk.NORMAL))
            
            threading.Thread(target=create_thread, daemon=True).start()
        
        # Create and Cancel buttons
        create_button = ttk.Button(button_frame, text="Create Instance", command=create_instance, 
                                  style="Accent.TButton")
        create_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def update_gpu_selection(self, gpu_buttons, selected_value):
        """Update GPU selection visual feedback."""
        # Update visual feedback for selected GPU card
        for gpu_info in [
            {"name": "T4", "value": "t4", "memory": "16GB", "desc": "Cost-effective for most ML workloads"},
            {"name": "A100", "value": "a100", "memory": "40GB", "desc": "High-performance for large models"},  
            {"name": "A100 XL", "value": "a100xl", "memory": "80GB", "desc": "Maximum performance"}
        ]:
            if gpu_info["value"] in gpu_buttons:
                button = gpu_buttons[gpu_info["value"]]
                if gpu_info["value"] == selected_value:
                    button.config(relief='sunken', background='#e3f2fd')
                else:
                    button.config(relief='raised', background='SystemButtonFace')
    
    def set_mode(self, mode_var, mode, proto_button, prod_button):
        """Set instance mode and update button states."""
        mode_var.set(mode)
        
        # Configure button styling for toggle effect
        selected_style = {'relief': 'sunken', 'background': '#0078d4', 'foreground': 'white'}
        normal_style = {'relief': 'raised', 'background': 'SystemButtonFace', 'foreground': 'black'}
        
        if mode == "prototyping":
            proto_button.config(**selected_style)
            prod_button.config(**normal_style)
        else:
            proto_button.config(**normal_style)
            prod_button.config(**selected_style)
    
    def start_instance(self):
        """Start selected instance."""
        instance_id = self.get_selected_instance()
        if not instance_id:
            messagebox.showwarning("No Selection", "Please select an instance to start.")
            return
        
        self.run_instance_command("start", instance_id)
    
    def stop_instance(self):
        """Stop selected instance."""
        instance_id = self.get_selected_instance()
        if not instance_id:
            messagebox.showwarning("No Selection", "Please select an instance to stop.")
            return
        
        if messagebox.askyesno("Confirm Stop", f"Stop instance {instance_id}?"):
            self.run_instance_command("stop", instance_id)
    
    def delete_instance(self):
        """Delete selected instance."""
        instance_id = self.get_selected_instance()
        if not instance_id:
            messagebox.showwarning("No Selection", "Please select an instance to delete.")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Permanently delete instance {instance_id}?\nThis cannot be undone."):
            self.run_instance_command("delete", instance_id)
    
    def run_instance_command(self, command, instance_id):
        """Run Thunder Compute command on instance."""
        def run_command():
            try:
                env = os.environ.copy()
                env['TNR_DISABLE_COMPLETION'] = '1'
                result = subprocess.run(['tnr', command, instance_id], 
                                      capture_output=True, text=True, timeout=60, env=env)
                
                if result.returncode == 0:
                    self.parent.after(0, lambda: self.refresh_instances())
                    self.parent.after(0, lambda: messagebox.showinfo("Success", f"Instance {instance_id} {command} successful"))
                else:
                    self.parent.after(0, lambda: messagebox.showerror("Error", f"Failed to {command} instance: {result.stderr}"))
                    
            except Exception as e:
                self.parent.after(0, lambda: messagebox.showerror("Error", f"Command failed: {str(e)}"))
        
        threading.Thread(target=run_command, daemon=True).start()


class PipelinePanel:
    """Pipeline execution and monitoring panel."""
    
    def __init__(self, parent, manager):
        self.parent = parent
        self.manager = manager
        self.frame = ttk.LabelFrame(parent, text="Pipeline Execution", padding="10")
        self.current_process = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup pipeline UI components."""
        # Input selection
        input_frame = ttk.Frame(self.frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Input Video:").pack(side=tk.LEFT)
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_var, width=50)
        self.input_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side=tk.LEFT)
        
        # Pipeline mode selection
        mode_frame = ttk.Frame(self.frame)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="full")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, 
                                values=["full", "track", "segment", "reconstruct", "graph", "llm", "simulate"],
                                state="readonly", width=15)
        mode_combo.pack(side=tk.LEFT, padx=(10, 20))
        
        # Thunder Compute option
        self.use_thunder_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode_frame, text="Use Thunder Compute", variable=self.use_thunder_var).pack(side=tk.LEFT)
        
        # GUI option for simulation
        self.gui_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_frame, text="Show Simulation GUI", variable=self.gui_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Goal input
        goal_frame = ttk.Frame(self.frame)
        goal_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(goal_frame, text="Goal:").pack(side=tk.LEFT)
        self.goal_var = tk.StringVar(value="Clear the table")
        goal_entry = ttk.Entry(goal_frame, textvariable=self.goal_var, width=40)
        goal_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # Progress and status
        progress_frame = ttk.Frame(self.frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="Start Pipeline", command=self.start_pipeline)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_pipeline, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="View Results", command=self.view_results).pack(side=tk.LEFT)
        
        # Output log
        log_frame = ttk.LabelFrame(self.frame, text="Output Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_input(self):
        """Browse for input video file."""
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ],
            initialdir=DATA_DIR / "raw_videos"
        )
        if filename:
            self.input_var.set(filename)
    
    def log_message(self, message):
        """Add message to log."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def start_pipeline(self):
        """Start the TANGRAM pipeline."""
        input_file = self.input_var.get().strip()
        if not input_file:
            messagebox.showerror("Error", "Please select an input video file.")
            return
        
        if not Path(input_file).exists():
            messagebox.showerror("Error", "Input file does not exist.")
            return
        
        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Update UI state
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.status_label.config(text="Starting pipeline...")
        
        # Build command
        cmd = [sys.executable, str(PROJECT_ROOT / "main.py")]
        cmd.extend(["--input", input_file])
        cmd.extend(["--mode", self.mode_var.get()])
        cmd.extend(["--goal", self.goal_var.get()])
        
        if self.gui_var.get():
            cmd.append("--gui")
        
        # Set Thunder Compute environment
        env = os.environ.copy()
        if self.use_thunder_var.get():
            # Update config to enable Thunder Compute
            config_update = {
                "TANGRAM_USE_THUNDER": "true",
                "PYTHONPATH": str(PROJECT_ROOT)
            }
            env.update(config_update)
        
        self.log_message(f"Starting pipeline: {' '.join(cmd)}")
        
        def run_pipeline():
            """Run pipeline in separate thread."""
            try:
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    cwd=str(PROJECT_ROOT)
                )
                
                # Read output in real-time
                for line in iter(self.current_process.stdout.readline, ''):
                    if line:
                        self.parent.after(0, lambda msg=line.strip(): self.log_message(msg))
                
                # Wait for completion
                return_code = self.current_process.wait()
                
                # Update UI
                self.parent.after(0, self.pipeline_finished, return_code)
                
            except Exception as e:
                self.parent.after(0, lambda: self.log_message(f"Error: {str(e)}"))
                self.parent.after(0, lambda: self.pipeline_finished(1))
        
        # Start pipeline thread
        threading.Thread(target=run_pipeline, daemon=True).start()
    
    def stop_pipeline(self):
        """Stop the running pipeline."""
        if self.current_process:
            self.current_process.terminate()
            self.log_message("Pipeline stopped by user")
            self.pipeline_finished(1)
    
    def pipeline_finished(self, return_code):
        """Handle pipeline completion."""
        self.progress.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if return_code == 0:
            self.status_label.config(text="Pipeline completed successfully!")
            self.log_message("✅ Pipeline completed successfully!")
            messagebox.showinfo("Success", "Pipeline completed successfully!")
        else:
            self.status_label.config(text="Pipeline failed")
            self.log_message("❌ Pipeline failed")
            messagebox.showerror("Error", "Pipeline execution failed. Check the log for details.")
        
        self.current_process = None
    
    def view_results(self):
        """Open results directory."""
        results_dir = RESULTS_DIR
        if results_dir.exists():
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', str(results_dir)])
            elif sys.platform.startswith('win'):  # Windows
                subprocess.run(['explorer', str(results_dir)])
            else:  # Linux
                subprocess.run(['xdg-open', str(results_dir)])
        else:
            messagebox.showinfo("No Results", "No results directory found. Run the pipeline first.")


class TANGRAMManager:
    """Main TANGRAM management application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_main_window()
        self.setup_ui()
    
    def setup_main_window(self):
        """Setup main window properties."""
        self.root.title("TANGRAM Pipeline Manager")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Configure style
        style = ttk.Style()
        if 'aqua' in style.theme_names():
            style.theme_use('aqua')  # macOS native look
        elif 'clam' in style.theme_names():
            style.theme_use('clam')
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (800 // 2)
        self.root.geometry(f"1200x800+{x}+{y}")
    
    def setup_ui(self):
        """Setup the main UI."""
        # Menu bar
        self.setup_menu()
        
        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Thunder Compute tab
        thunder_frame = ttk.Frame(self.notebook)
        self.notebook.add(thunder_frame, text="Thunder Compute")
        self.thunder_panel = ThunderComputePanel(thunder_frame, self)
        self.thunder_panel.frame.pack(fill=tk.BOTH, expand=True)
        
        # Pipeline tab
        pipeline_frame = ttk.Frame(self.notebook)
        self.notebook.add(pipeline_frame, text="Pipeline Execution")
        self.pipeline_panel = PipelinePanel(pipeline_frame, self)
        self.pipeline_panel.frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.setup_status_bar()
    
    def setup_menu(self):
        """Setup application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Video...", command=self.open_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Configuration", command=self.show_config)
        tools_menu.add_command(label="View Logs", command=self.view_logs)
        tools_menu.add_separator()
        tools_menu.add_command(label="Thunder CLI", command=self.open_thunder_cli)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # Thunder status
        self.thunder_status = ttk.Label(self.status_bar, text="Thunder: Checking...", foreground="orange")
        self.thunder_status.pack(side=tk.RIGHT)
        
        self.update_status()
    
    def update_status(self):
        """Update status bar information."""
        # Check Thunder Compute status
        try:
            env = os.environ.copy()
            env['TNR_DISABLE_COMPLETION'] = '1'
            result = subprocess.run(['tnr', 'status', '--no-wait'], 
                                  capture_output=True, text=True, timeout=5, env=env)
            if result.returncode == 0:
                self.thunder_status.config(text="Thunder: Connected", foreground="green")
            else:
                self.thunder_status.config(text="Thunder: Error", foreground="red")
        except:
            self.thunder_status.config(text="Thunder: Not Available", foreground="gray")
        
        # Schedule next update
        self.root.after(60000, self.update_status)  # Update every minute
    
    def open_video(self):
        """Open video file dialog."""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.pipeline_panel.input_var.set(filename)
            self.notebook.select(1)  # Switch to pipeline tab
    
    def show_config(self):
        """Show configuration dialog."""
        messagebox.showinfo("Configuration", "Configuration editor coming soon!\n\nFor now, edit config.py directly.")
    
    def view_logs(self):
        """Open logs directory."""
        logs_dir = PROJECT_ROOT / "results" / "logs"
        if logs_dir.exists():
            if sys.platform.startswith('darwin'):
                subprocess.run(['open', str(logs_dir)])
            elif sys.platform.startswith('win'):
                subprocess.run(['explorer', str(logs_dir)])
            else:
                subprocess.run(['xdg-open', str(logs_dir)])
        else:
            messagebox.showinfo("No Logs", "No logs directory found.")
    
    def open_thunder_cli(self):
        """Open Thunder CLI in terminal."""
        if sys.platform.startswith('darwin'):
            subprocess.run(['open', '-a', 'Terminal', '--args', 'tnr', 'status'])
        elif sys.platform.startswith('win'):
            subprocess.run(['cmd', '/c', 'start', 'cmd', '/k', 'tnr status'])
        else:
            subprocess.run(['x-terminal-emulator', '-e', 'tnr status'])
    
    def show_about(self):
        """Show about dialog."""
        about_text = """TANGRAM Pipeline Manager
        
A comprehensive system for video-based robotic scene understanding.

Features:
• Object tracking and segmentation
• 3D scene reconstruction
• Scene graph generation
• LLM-based task planning
• Robot simulation
• Thunder Compute integration

Version: 1.0.0
Author: TANGRAM Team
License: MIT"""
        
        messagebox.showinfo("About TANGRAM", about_text)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = TANGRAMManager()
    app.run()


if __name__ == "__main__":
    main()