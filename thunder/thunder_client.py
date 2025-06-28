#!/usr/bin/env python3
"""
Thunder Compute Client for TANGRAM Pipeline

This module provides seamless integration with Thunder Compute for offloading
heavy processing tasks like SAM segmentation and COLMAP reconstruction.

Features:
- Automatic data sync to/from Thunder Compute
- Remote job execution with progress monitoring
- Result retrieval and local integration
- Error handling and retry logic

Author: TANGRAM Team
License: MIT
"""

import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import paramiko
import zipfile
import tempfile

from config import HARDWARE_CONFIG
from utils.logging_utils import setup_logger, log_function_call
from utils.file_utils import ensure_directory

logger = setup_logger(__name__)

class ThunderComputeClient:
    """
    Client for executing heavy processing tasks on Thunder Compute.
    
    Handles data transfer, remote execution, and result retrieval
    for compute-intensive pipeline components.
    """
    
    def __init__(self):
        """Initialize Thunder Compute client."""
        self.config = HARDWARE_CONFIG["thunder_compute"]
        self.ssh_host = self.config["ssh_host"]
        self.ssh_user = self.config["ssh_user"]
        self.remote_data_dir = self.config["remote_data_dir"]
        self.ssh_key_path = Path(self.config.get("ssh_key_path", "~/.ssh/id_rsa")).expanduser()
        
        # SSH client
        self.ssh_client = None
        self.sftp_client = None
        
        if not self.config["enabled"]:
            logger.warning("Thunder Compute is disabled in configuration")
            return
            
        logger.info(f"Initialized Thunder Compute client for {self.ssh_host}")
    
    @log_function_call()
    def connect(self) -> bool:
        """Establish SSH connection to Thunder Compute."""
        if not self.config["enabled"]:
            return False
            
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect using SSH key
            self.ssh_client.connect(
                hostname=self.ssh_host,
                username=self.ssh_user,
                key_filename=str(self.ssh_key_path),
                timeout=30
            )
            
            # Create SFTP client for file transfers
            self.sftp_client = self.ssh_client.open_sftp()
            
            # Create remote data directory
            self._run_remote_command(f"mkdir -p {self.remote_data_dir}")
            
            logger.info("Connected to Thunder Compute successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Thunder Compute: {e}")
            return False
    
    def disconnect(self):
        """Close connection to Thunder Compute."""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        
        logger.info("Disconnected from Thunder Compute")
    
    def _run_remote_command(self, command: str, timeout: int = 300) -> tuple:
        """Execute command on remote Thunder Compute instance."""
        if not self.ssh_client:
            raise RuntimeError("Not connected to Thunder Compute")
        
        logger.debug(f"Running remote command: {command}")
        
        stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
        
        # Wait for completion
        exit_status = stdout.channel.recv_exit_status()
        
        stdout_data = stdout.read().decode('utf-8')
        stderr_data = stderr.read().decode('utf-8')
        
        if exit_status != 0:
            logger.error(f"Remote command failed: {stderr_data}")
        
        return exit_status, stdout_data, stderr_data
    
    @log_function_call()
    def upload_data(self, local_paths: List[Path], remote_subdir: str = "") -> str:
        """
        Upload data files to Thunder Compute.
        
        Args:
            local_paths: List of local file/directory paths to upload
            remote_subdir: Subdirectory on remote system
            
        Returns:
            Remote directory path containing uploaded data
        """
        if not self.sftp_client:
            raise RuntimeError("Not connected to Thunder Compute")
        
        remote_dir = f"{self.remote_data_dir}/{remote_subdir}" if remote_subdir else self.remote_data_dir
        
        # Create remote directory
        self._run_remote_command(f"mkdir -p {remote_dir}")
        
        for local_path in local_paths:
            local_path = Path(local_path)
            
            if local_path.is_file():
                remote_file = f"{remote_dir}/{local_path.name}"
                logger.info(f"Uploading {local_path} → {remote_file}")
                self.sftp_client.put(str(local_path), remote_file)
                
            elif local_path.is_dir():
                # Create archive for directory upload
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                    with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file_path in local_path.rglob('*'):
                            if file_path.is_file():
                                arcname = file_path.relative_to(local_path.parent)
                                zipf.write(file_path, arcname)
                    
                    # Upload archive
                    remote_archive = f"{remote_dir}/{local_path.name}.zip"
                    logger.info(f"Uploading directory {local_path} as {remote_archive}")
                    self.sftp_client.put(temp_zip.name, remote_archive)
                    
                    # Extract on remote
                    self._run_remote_command(f"cd {remote_dir} && unzip -q {local_path.name}.zip")
                    self._run_remote_command(f"rm {remote_archive}")
                    
                    # Clean up local temp file
                    Path(temp_zip.name).unlink()
        
        logger.info(f"Data upload completed to {remote_dir}")
        return remote_dir
    
    @log_function_call()
    def download_results(self, remote_path: str, local_dir: Path) -> bool:
        """
        Download results from Thunder Compute.
        
        Args:
            remote_path: Remote file or directory path
            local_dir: Local directory to save results
            
        Returns:
            True if download successful
        """
        if not self.sftp_client:
            raise RuntimeError("Not connected to Thunder Compute")
        
        ensure_directory(local_dir)
        
        try:
            # Check if remote path exists
            exit_status, _, _ = self._run_remote_command(f"test -e {remote_path}")
            if exit_status != 0:
                logger.error(f"Remote path does not exist: {remote_path}")
                return False
            
            # Check if it's a directory or file
            exit_status, _, _ = self._run_remote_command(f"test -d {remote_path}")
            
            if exit_status == 0:  # Directory
                # Create archive on remote
                archive_name = f"{remote_path}.tar.gz"
                self._run_remote_command(f"tar -czf {archive_name} -C {Path(remote_path).parent} {Path(remote_path).name}")
                
                # Download archive
                local_archive = local_dir / Path(archive_name).name
                logger.info(f"Downloading {archive_name} → {local_archive}")
                self.sftp_client.get(archive_name, str(local_archive))
                
                # Extract locally
                subprocess.run(['tar', '-xzf', str(local_archive), '-C', str(local_dir)], check=True)
                local_archive.unlink()  # Remove archive
                
                # Clean up remote archive
                self._run_remote_command(f"rm {archive_name}")
                
            else:  # File
                local_file = local_dir / Path(remote_path).name
                logger.info(f"Downloading {remote_path} → {local_file}")
                self.sftp_client.get(remote_path, str(local_file))
            
            logger.info("Results download completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download results: {e}")
            return False
    
    @log_function_call()
    def run_sam_segmentation(self, video_path: Path, tracking_results: List[Dict],
                           local_output_dir: Path) -> bool:
        """
        Run SAM segmentation on Thunder Compute.
        
        Args:
            video_path: Local path to input video
            tracking_results: Tracking results from YOLO
            local_output_dir: Local directory for results
            
        Returns:
            True if segmentation successful
        """
        logger.info("Running SAM segmentation on Thunder Compute...")
        
        if not self.connect():
            return False
        
        try:
            # Create temporary files for upload
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Save tracking results
                tracking_file = temp_dir / "tracking_results.json"
                with open(tracking_file, 'w') as f:
                    json.dump(tracking_results, f)
                
                # Upload data
                remote_dir = self.upload_data([video_path, tracking_file], "sam_job")
                
                # Create remote processing script
                remote_script = f"{remote_dir}/run_sam.py"
                sam_script = self._create_sam_script(
                    video_file=f"{remote_dir}/{video_path.name}",
                    tracking_file=f"{remote_dir}/tracking_results.json",
                    output_dir=f"{remote_dir}/sam_output"
                )
                
                # Upload script
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(sam_script)
                    temp_script = Path(f.name)
                
                self.sftp_client.put(str(temp_script), remote_script)
                temp_script.unlink()
                
                # Run SAM segmentation
                logger.info("Executing SAM segmentation on Thunder Compute...")
                exit_status, stdout, stderr = self._run_remote_command(
                    f"cd {remote_dir} && python run_sam.py",
                    timeout=1800  # 30 minutes
                )
                
                if exit_status != 0:
                    logger.error(f"SAM segmentation failed: {stderr}")
                    return False
                
                # Download results
                success = self.download_results(f"{remote_dir}/sam_output", local_output_dir)
                
                # Cleanup remote data
                self._run_remote_command(f"rm -rf {remote_dir}")
                
                return success
                
        except Exception as e:
            logger.error(f"Thunder Compute SAM segmentation failed: {e}")
            return False
            
        finally:
            self.disconnect()
    
    def _create_sam_script(self, video_file: str, tracking_file: str, output_dir: str) -> str:
        """Create remote SAM processing script."""
        return f"""#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Install required packages if needed
import subprocess
try:
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
    import cv2
    import numpy as np
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "torch", "torchvision", "opencv-python", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "git+https://github.com/facebookresearch/segment-anything.git"])
    
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
    import cv2
    import numpy as np

def download_sam_model():
    \"\"\"Download SAM model if not present.\"\"\"
    import urllib.request
    
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    model_path = "sam_vit_b_01ec64.pth"
    
    if not os.path.exists(model_path):
        print("Downloading SAM model...")
        urllib.request.urlretrieve(model_url, model_path)
    
    return model_path

def run_sam_segmentation():
    \"\"\"Run SAM segmentation with tracking data.\"\"\"
    
    # Setup
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
    
    # Create output directory
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
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        
        frame_masks = {{
            "frame_id": tracking_data["frame_id"],
            "masks": []
        }}
        
        for detection in tracking_data.get("detections", []):
            bbox = detection["bbox"]
            track_id = detection["track_id"]
            
            # Convert YOLO format to SAM format
            x, y, w, h = bbox
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            input_box = np.array([x1, y1, x2, y2])
            
            try:
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                if len(masks) > 0:
                    # Save mask as image
                    mask_filename = f"frame_{{frame_idx}}_mask_{{track_id}}.png"
                    mask_path = os.path.join("{output_dir}", mask_filename)
                    mask_img = (masks[0] * 255).astype(np.uint8)
                    cv2.imwrite(mask_path, mask_img)
                    
                    mask_data = {{
                        "track_id": track_id,
                        "score": float(scores[0]),
                        "bbox": bbox,
                        "class_name": detection["class_name"],
                        "mask_file": mask_filename
                    }}
                    frame_masks["masks"].append(mask_data)
                    
            except Exception as e:
                print(f"Error segmenting object {{track_id}}: {{e}}")
                continue
        
        all_masks.append(frame_masks)
    
    cap.release()
    
    # Save results
    with open(os.path.join("{output_dir}", "segmentation_results.json"), 'w') as f:
        json.dump(all_masks, f, indent=2)
    
    print(f"SAM segmentation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    run_sam_segmentation()
"""
    
    @log_function_call()
    def run_colmap_reconstruction(self, frames_dir: Path, local_output_dir: Path) -> bool:
        """
        Run COLMAP 3D reconstruction on Thunder Compute.
        
        Args:
            frames_dir: Local directory containing extracted frames
            local_output_dir: Local directory for reconstruction results
            
        Returns:
            True if reconstruction successful
        """
        logger.info("Running COLMAP reconstruction on Thunder Compute...")
        
        if not self.connect():
            return False
        
        try:
            # Upload frames
            remote_dir = self.upload_data([frames_dir], "colmap_job")
            
            # Create COLMAP script
            colmap_script = self._create_colmap_script(
                frames_dir=f"{remote_dir}/{frames_dir.name}",
                output_dir=f"{remote_dir}/colmap_output"
            )
            
            # Upload and run script
            remote_script = f"{remote_dir}/run_colmap.sh"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(colmap_script)
                temp_script = Path(f.name)
            
            self.sftp_client.put(str(temp_script), remote_script)
            temp_script.unlink()
            
            # Make script executable and run
            self._run_remote_command(f"chmod +x {remote_script}")
            logger.info("Executing COLMAP reconstruction on Thunder Compute...")
            
            exit_status, stdout, stderr = self._run_remote_command(
                f"cd {remote_dir} && ./run_colmap.sh",
                timeout=3600  # 1 hour
            )
            
            if exit_status != 0:
                logger.error(f"COLMAP reconstruction failed: {stderr}")
                return False
            
            # Download results
            success = self.download_results(f"{remote_dir}/colmap_output", local_output_dir)
            
            # Cleanup
            self._run_remote_command(f"rm -rf {remote_dir}")
            
            return success
            
        except Exception as e:
            logger.error(f"Thunder Compute COLMAP reconstruction failed: {e}")
            return False
            
        finally:
            self.disconnect()
    
    def _create_colmap_script(self, frames_dir: str, output_dir: str) -> str:
        """Create remote COLMAP processing script."""
        return f"""#!/bin/bash
set -e

echo "Setting up COLMAP environment..."

# Install COLMAP if needed
if ! command -v colmap &> /dev/null; then
    echo "Installing COLMAP..."
    sudo apt-get update
    sudo apt-get install -y colmap
fi

echo "Running COLMAP reconstruction..."
echo "Input: {frames_dir}"
echo "Output: {output_dir}"

# Create output directory
mkdir -p {output_dir}

# Feature extraction
echo "Step 1: Feature extraction..."
colmap feature_extractor \\
    --database_path {output_dir}/database.db \\
    --image_path {frames_dir} \\
    --ImageReader.single_camera 1 \\
    --ImageReader.camera_model PINHOLE \\
    --SiftExtraction.max_image_size 1600

# Feature matching
echo "Step 2: Feature matching..."
colmap exhaustive_matcher \\
    --database_path {output_dir}/database.db \\
    --SiftMatching.guided_matching 1

# Structure from Motion
echo "Step 3: Structure from Motion..."
mkdir -p {output_dir}/sparse
colmap mapper \\
    --database_path {output_dir}/database.db \\
    --image_path {frames_dir} \\
    --output_path {output_dir}/sparse \\
    --Mapper.ba_refine_focal_length 0 \\
    --Mapper.ba_refine_principal_point 0

# Export to text format
echo "Step 4: Exporting results..."
if [ -d "{output_dir}/sparse/0" ]; then
    colmap model_converter \\
        --input_path {output_dir}/sparse/0 \\
        --output_path {output_dir}/sparse/0 \\
        --output_type TXT
    echo "COLMAP reconstruction completed successfully!"
else
    echo "Error: Sparse reconstruction failed"
    exit 1
fi
"""

def main():
    """Test Thunder Compute client."""
    client = ThunderComputeClient()
    
    if client.connect():
        print("✅ Thunder Compute connection successful!")
        client.disconnect()
    else:
        print("❌ Thunder Compute connection failed")

if __name__ == "__main__":
    main()