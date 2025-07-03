#!/usr/bin/env python3
"""
Safe GUI Launcher for TANGRAM

A lightweight GUI that progressively loads components to avoid system freezing.
"""

import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class SafeTangramGUI:
    """Safe TANGRAM GUI that won't freeze the system"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TANGRAM - Safe Mode")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        self.components_status = {
            'opencv': 'pending',
            'pybullet': 'pending', 
            'ollama': 'pending'
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        
        # Header
        header = tk.Frame(self.root, bg='#34495e', height=80)
        header.pack(fill=tk.X, padx=10, pady=5)
        header.pack_propagate(False)
        
        tk.Label(header, text="TANGRAM Safe Mode", 
                font=('Helvetica', 20, 'bold'), 
                fg='white', bg='#34495e').pack(pady=20)
        
        # Status panel
        status_frame = tk.Frame(self.root, bg='#2c3e50')
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(status_frame, text="Component Status", 
                font=('Helvetica', 14, 'bold'), 
                fg='white', bg='#2c3e50').pack(pady=5)
        
        # Status indicators
        self.status_labels = {}
        components = [
            ('opencv', 'Computer Vision (OpenCV)'),
            ('pybullet', 'Robot Simulation (PyBullet)'),
            ('ollama', 'AI Language Model (Ollama)')
        ]
        
        for comp_id, comp_name in components:
            frame = tk.Frame(status_frame, bg='#34495e')
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(frame, text=comp_name, 
                    font=('Helvetica', 12), 
                    fg='white', bg='#34495e').pack(side=tk.LEFT)
            
            status_label = tk.Label(frame, text="⏳ Checking...", 
                                  font=('Helvetica', 12, 'bold'),
                                  fg='yellow', bg='#34495e')
            status_label.pack(side=tk.RIGHT)
            self.status_labels[comp_id] = status_label
        
        # Log area
        tk.Label(status_frame, text="System Log", 
                font=('Helvetica', 14, 'bold'), 
                fg='white', bg='#2c3e50').pack(pady=(20,5))
        
        self.log_text = scrolledtext.ScrolledText(
            status_frame, height=15, 
            font=('Courier', 10),
            bg='#1a1a1a', fg='#00ff00'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(button_frame, text="Check Components", 
                 command=self.check_components,
                 bg='#3498db', fg='white', 
                 font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Run Lightweight Demo", 
                 command=self.run_lightweight_demo,
                 bg='#27ae60', fg='white',
                 font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Clear Log", 
                 command=self.clear_log,
                 bg='#e67e22', fg='white',
                 font=('Helvetica', 12, 'bold')).pack(side=tk.RIGHT, padx=5)
        
        # Initial log message
        self.log("🚀 TANGRAM Safe Mode initialized")
        self.log("Click 'Check Components' to test system readiness")
        
    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the log"""
        self.log_text.delete(1.0, tk.END)
    
    def update_status(self, component, status, message):
        """Update component status"""
        self.components_status[component] = status
        
        if status == 'success':
            self.status_labels[component].config(text="✅ Ready", fg='#00ff00')
        elif status == 'warning':
            self.status_labels[component].config(text="⚠️ Warning", fg='#ffaa00')
        elif status == 'error':
            self.status_labels[component].config(text="❌ Error", fg='#ff0000')
        else:
            self.status_labels[component].config(text="⏳ Checking...", fg='yellow')
        
        self.log(message)
    
    def check_components(self):
        """Check all components in a separate thread"""
        self.log("🔍 Starting component check...")
        
        # Run in background to avoid freezing
        thread = threading.Thread(target=self._check_components_background)
        thread.daemon = True
        thread.start()
    
    def _check_components_background(self):
        """Background component checking"""
        
        # Check OpenCV
        try:
            import cv2
            self.root.after(0, lambda: self.update_status(
                'opencv', 'success', f"✅ OpenCV {cv2.__version__} loaded successfully"
            ))
        except ImportError:
            self.root.after(0, lambda: self.update_status(
                'opencv', 'error', "❌ OpenCV not found. Install with: pip install opencv-python"
            ))
        
        time.sleep(0.5)
        
        # Check PyBullet
        try:
            import pybullet as p
            # Test connection without GUI
            client = p.connect(p.DIRECT)
            p.disconnect(client)
            self.root.after(0, lambda: self.update_status(
                'pybullet', 'success', "✅ PyBullet loaded and tested successfully"
            ))
        except ImportError:
            self.root.after(0, lambda: self.update_status(
                'pybullet', 'error', "❌ PyBullet not found. Install with: pip install pybullet"
            ))
        except Exception as e:
            self.root.after(0, lambda: self.update_status(
                'pybullet', 'warning', f"⚠️ PyBullet loaded but test failed: {e}"
            ))
        
        time.sleep(0.5)
        
        # Check Ollama
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                if 'deepseek' in result.stdout.lower():
                    self.root.after(0, lambda: self.update_status(
                        'ollama', 'success', "✅ Ollama running with DeepSeek model"
                    ))
                else:
                    self.root.after(0, lambda: self.update_status(
                        'ollama', 'warning', "⚠️ Ollama running but DeepSeek model not found"
                    ))
            else:
                self.root.after(0, lambda: self.update_status(
                    'ollama', 'error', "❌ Ollama not responding"
                ))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.root.after(0, lambda: self.update_status(
                'ollama', 'error', "❌ Ollama not installed or not running"
            ))
        
        # Final summary
        self.root.after(0, lambda: self.log("🔍 Component check completed"))
        
        ready_count = sum(1 for status in self.components_status.values() if status == 'success')
        self.root.after(0, lambda: self.log(f"📊 {ready_count}/3 components ready"))
    
    def run_lightweight_demo(self):
        """Run the lightweight demo"""
        self.log("🎬 Starting lightweight demo...")
        
        thread = threading.Thread(target=self._run_demo_background)
        thread.daemon = True
        thread.start()
    
    def _run_demo_background(self):
        """Run demo in background"""
        try:
            from scripts.demos.lightweight_demo import LightweightDemo
            
            self.root.after(0, lambda: self.log("🔄 Initializing lightweight demo..."))
            
            demo = LightweightDemo()
            
            # Redirect demo output to our log
            import io
            import contextlib
            
            old_stdout = sys.stdout
            
            # Capture demo output
            string_buffer = io.StringIO()
            with contextlib.redirect_stdout(string_buffer):
                success = demo.run_lightweight_demo()
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Add captured output to log
            demo_output = string_buffer.getvalue()
            for line in demo_output.split('\n'):
                if line.strip():
                    self.root.after(0, lambda l=line: self.log(l))
            
            if success:
                self.root.after(0, lambda: self.log("🎉 Lightweight demo completed successfully!"))
            else:
                self.root.after(0, lambda: self.log("❌ Demo completed with issues"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log(f"❌ Demo error: {e}"))
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🔧 Starting TANGRAM Safe GUI...")
    
    try:
        app = SafeTangramGUI()
        app.run()
    except Exception as e:
        print(f"❌ GUI error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())