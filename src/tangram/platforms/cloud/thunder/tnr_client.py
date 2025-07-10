#!/usr/bin/env python3
"""
Thunder Compute CLI Client for TANGRAM Pipeline

Modern wrapper around the tnr CLI for seamless integration with TANGRAM.
Handles instance management, file transfers, and remote execution.

Author: TANGRAM Team
License: MIT
"""

import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

from config import HARDWARE_CONFIG
from utils.logging_utils import setup_logger, log_function_call

logger = setup_logger(__name__)

class ThunderComputeManager:
    """
    Modern Thunder Compute client using the tnr CLI.
    
    Provides seamless integration with Thunder Compute for TANGRAM pipeline
    components requiring heavy computation.
    """
    
    def __init__(self):
        """Initialize Thunder Compute manager."""
        self.config = HARDWARE_CONFIG["thunder_compute"]
        self.cli_command = self.config.get("cli_command", "tnr")
        self.instance_id = None
        self.instance_address = None
        
        if not self.config["enabled"]:
            logger.warning("Thunder Compute is disabled in configuration")
            return
            
        logger.info("Initialized Thunder Compute manager")
    
    def _run_tnr_command(self, args: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a tnr CLI command."""
        cmd = [self.cli_command] + args
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=120  # 2 minute timeout for most commands
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            raise
        except FileNotFoundError:
            logger.error(f"Thunder CLI not found: {self.cli_command}")
            raise
    
    @log_function_call()
    def ensure_instance(self) -> bool:
        """Ensure a Thunder Compute instance is available."""
        if not self.config["enabled"]:
            return False
        
        # Check existing instances
        status_result = self._run_tnr_command(["status", "--no-wait"])
        
        if status_result.returncode != 0:
            logger.error("Failed to get instance status")
            return False
        
        # Parse status output to find running instances
        lines = status_result.stdout.strip().split('\n')
        
        # Look for running instances in the table
        for line in lines:
            if '│' in line and 'running' in line.lower():
                # Extract instance ID (first column after │)
                parts = [p.strip() for p in line.split('│') if p.strip()]
                if parts and parts[0].isdigit():
                    self.instance_id = parts[0]
                    if len(parts) > 2:
                        self.instance_address = parts[2] if parts[2] != '--' else None
                    logger.info(f"Found running instance: {self.instance_id}")
                    return True
        
        # No running instance found, create one if auto-create is enabled
        if self.config.get("auto_create_instance", False):
            return self._create_instance()
        
        logger.warning("No running instances found and auto-create is disabled")
        return False
    
    def _create_instance(self) -> bool:
        """Create a new Thunder Compute instance."""
        logger.info("Creating new Thunder Compute instance...")
        
        instance_config = self.config.get("instance_config", {})
        
        create_args = ["create"]
        
        # Add GPU configuration
        if "gpu" in instance_config:
            create_args.extend(["--gpu", instance_config["gpu"]])
        
        # Add vCPU configuration
        if "vcpus" in instance_config:
            create_args.extend(["--vcpus", str(instance_config["vcpus"])])
        
        # Add mode configuration
        if "mode" in instance_config:
            create_args.extend(["--mode", instance_config["mode"]])
        
        create_result = self._run_tnr_command(create_args)
        
        if create_result.returncode != 0:
            logger.error(f"Failed to create instance: {create_result.stderr}")
            return False
        
        # Extract instance ID from output
        output = create_result.stdout.strip()
        for line in output.split('\n'):
            if 'instance' in line.lower() and any(char.isdigit() for char in line):
                # Try to extract instance ID
                words = line.split()
                for word in words:
                    if word.isdigit():
                        self.instance_id = word
                        break
                if self.instance_id:
                    break
        
        if not self.instance_id:
            # Default to instance 0 if we can't parse the ID
            self.instance_id = "0"
        
        logger.info(f"Created instance {self.instance_id}, waiting for it to start...")
        
        # Wait for instance to be running
        return self._wait_for_instance_running()
    
    def _wait_for_instance_running(self, timeout: int = 300) -> bool:
        """Wait for instance to be in running state."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_result = self._run_tnr_command(["status", "--no-wait"])
            
            if status_result.returncode == 0:
                lines = status_result.stdout.strip().split('\n')
                
                for line in lines:
                    if '│' in line and self.instance_id in line and 'running' in line.lower():
                        # Extract address
                        parts = [p.strip() for p in line.split('│') if p.strip()]
                        if len(parts) > 2 and parts[2] != '--':
                            self.instance_address = parts[2]
                        logger.info(f"Instance {self.instance_id} is running at {self.instance_address}")
                        return True
            
            logger.info(f"Waiting for instance {self.instance_id} to start...")
            time.sleep(10)
        
        logger.error(f"Instance {self.instance_id} did not start within {timeout} seconds")
        return False
    
    @log_function_call()
    def upload_files(self, local_paths: List[Path], remote_dir: str = "/tmp/tangram_data") -> bool:
        """Upload files to Thunder Compute instance."""
        if not self.instance_id:
            logger.error("No instance available for file upload")
            return False
        
        try:
            # Create remote directory
            mkdir_cmd = f"mkdir -p {remote_dir}"
            connect_result = self._run_tnr_command([
                "connect", self.instance_id, "--", "bash", "-c", mkdir_cmd
            ])
            
            if connect_result.returncode != 0:
                logger.error(f"Failed to create remote directory: {connect_result.stderr}")
                return False
            
            # Upload each file
            for local_path in local_paths:
                local_path = Path(local_path)
                remote_path = f"{self.instance_id}:{remote_dir}/{local_path.name}"
                
                logger.info(f"Uploading {local_path} → {remote_path}")
                
                scp_result = self._run_tnr_command([
                    "scp", str(local_path), remote_path
                ])
                
                if scp_result.returncode != 0:
                    logger.error(f"Failed to upload {local_path}: {scp_result.stderr}")
                    return False
            
            logger.info("File upload completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return False
    
    @log_function_call()
    def download_files(self, remote_path: str, local_dir: Path) -> bool:
        """Download files from Thunder Compute instance."""
        if not self.instance_id:
            logger.error("No instance available for file download")
            return False
        
        try:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            remote_full_path = f"{self.instance_id}:{remote_path}"
            
            logger.info(f"Downloading {remote_full_path} → {local_dir}")
            
            scp_result = self._run_tnr_command([
                "scp", "-r", remote_full_path, str(local_dir)
            ])
            
            if scp_result.returncode != 0:
                logger.error(f"Failed to download files: {scp_result.stderr}")
                return False
            
            logger.info("File download completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"File download failed: {e}")
            return False
    
    @log_function_call()
    def execute_remote_command(self, command: str, timeout: int = 300) -> tuple:
        """Execute command on Thunder Compute instance."""
        if not self.instance_id:
            logger.error("No instance available for command execution")
            return 1, "", "No instance available"
        
        try:
            logger.debug(f"Executing remote command: {command}")
            
            result = subprocess.run([
                self.cli_command, "connect", self.instance_id, 
                "--", "bash", "-c", command
            ], capture_output=True, text=True, timeout=timeout)
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"Remote command timed out after {timeout} seconds")
            return 1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Remote command execution failed: {e}")
            return 1, "", str(e)
    
    @log_function_call()
    def run_sam_segmentation(self, video_path: Path, tracking_results: List[Dict],
                           local_output_dir: Path) -> bool:
        """Run SAM segmentation on Thunder Compute."""
        logger.info("Running SAM segmentation on Thunder Compute...")
        
        if not self.ensure_instance():
            return False
        
        try:
            remote_dir = "/tmp/tangram_sam"
            
            # Prepare files for upload
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Copy video file
                temp_video = temp_dir / video_path.name
                shutil.copy2(video_path, temp_video)
                
                # Save tracking results
                tracking_file = temp_dir / "tracking_results.json"
                with open(tracking_file, 'w') as f:
                    json.dump(tracking_results, f)
                
                # Upload files
                if not self.upload_files([temp_video, tracking_file], remote_dir):
                    return False
                
                # Create and upload processing script
                sam_script = self._create_sam_script(
                    video_file=f"{remote_dir}/{video_path.name}",
                    tracking_file=f"{remote_dir}/tracking_results.json",
                    output_dir=f"{remote_dir}/output"
                )
                
                script_file = temp_dir / "run_sam.py"
                with open(script_file, 'w') as f:
                    f.write(sam_script)
                
                if not self.upload_files([script_file], remote_dir):
                    return False
            
            # Execute SAM segmentation
            logger.info("Executing SAM segmentation on Thunder Compute...")
            exit_code, stdout, stderr = self.execute_remote_command(
                f"cd {remote_dir} && python run_sam.py",
                timeout=1800  # 30 minutes
            )
            
            if exit_code != 0:
                logger.error(f"SAM segmentation failed: {stderr}")
                return False
            
            # Download results
            local_output_dir.mkdir(parents=True, exist_ok=True)
            success = self.download_files(f"{remote_dir}/output", local_output_dir)
            
            # Cleanup remote files
            self.execute_remote_command(f"rm -rf {remote_dir}")
            
            return success
            
        except Exception as e:
            logger.error(f"Thunder SAM segmentation failed: {e}")
            return False
    
    def _create_sam_script(self, video_file: str, tracking_file: str, output_dir: str) -> str:
        """Create SAM processing script for Thunder Compute."""
        return f'''#!/usr/bin/env python3
import sys
import subprocess
import json
import os
from pathlib import Path

def install_packages():
    """Install required packages."""
    packages = [
        "torch", "torchvision", "opencv-python", "numpy", "Pillow",
        "git+https://github.com/facebookresearch/segment-anything.git"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {{package}}: {{e}}")
            return False
    return True

def download_sam_model():
    """Download SAM model."""
    import urllib.request
    
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    model_path = "sam_vit_b_01ec64.pth"
    
    if not os.path.exists(model_path):
        print("Downloading SAM model...")
        urllib.request.urlretrieve(model_url, model_path)
    
    return model_path

def main():
    """Run SAM segmentation."""
    print("Installing packages...")
    if not install_packages():
        return 1
    
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    import cv2
    import numpy as np
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {{device}}")
    
    # Load SAM model
    model_path = download_sam_model()
    sam = sam_model_registry["vit_b"](checkpoint=model_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Load tracking data
    with open("{tracking_file}", 'r') as f:
        tracking_results = json.load(f)
    
    os.makedirs("{output_dir}", exist_ok=True)
    
    # Process video
    cap = cv2.VideoCapture("{video_file}")
    all_masks = []
    
    for frame_idx, tracking_data in enumerate(tracking_results):
        print(f"Processing frame {{frame_idx}}/{{len(tracking_results)}}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        
        frame_masks = {{
            "frame_id": tracking_data["frame_id"],
            "masks": []
        }}
        
        for detection in tracking_data.get("detections", []):
            bbox = detection["bbox"]
            track_id = detection["track_id"]
            
            # Convert to SAM format
            x, y, w, h = bbox
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            input_box = np.array([x1, y1, x2, y2])
            
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                if len(masks) > 0:
                    mask_filename = f"frame_{{frame_idx}}_mask_{{track_id}}.png"
                    mask_path = os.path.join("{output_dir}", mask_filename)
                    mask_img = (masks[0] * 255).astype(np.uint8)
                    cv2.imwrite(mask_path, mask_img)
                    
                    frame_masks["masks"].append({{
                        "track_id": track_id,
                        "score": float(scores[0]),
                        "bbox": bbox,
                        "class_name": detection["class_name"],
                        "mask_file": mask_filename
                    }})
                    
            except Exception as e:
                print(f"Error segmenting object {{track_id}}: {{e}}")
                continue
        
        all_masks.append(frame_masks)
    
    cap.release()
    
    # Save results
    with open(os.path.join("{output_dir}", "segmentation_results.json"), 'w') as f:
        json.dump(all_masks, f, indent=2)
    
    print(f"SAM segmentation completed! Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    def cleanup_instance(self):
        """Stop the Thunder Compute instance."""
        if self.instance_id:
            logger.info(f"Stopping instance {self.instance_id}")
            self._run_tnr_command(["stop", self.instance_id])

def main():
    """Test Thunder Compute manager."""
    manager = ThunderComputeManager()
    
    if manager.ensure_instance():
        print("✅ Thunder Compute instance ready!")
        
        # Test basic connectivity
        exit_code, stdout, stderr = manager.execute_remote_command("echo 'Hello from Thunder Compute!'")
        if exit_code == 0:
            print(f"✅ Remote execution successful: {stdout.strip()}")
        else:
            print(f"❌ Remote execution failed: {stderr}")
    else:
        print("❌ Failed to set up Thunder Compute instance")

if __name__ == "__main__":
    main()