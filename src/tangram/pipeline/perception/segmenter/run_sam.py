#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import json
import os
from typing import List, Dict, Any, Tuple
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from config import SAM_CONFIG

class SAMSegmenter:
    def __init__(self, model_type: str = None, checkpoint_path: str = None):
        # Use config defaults if not provided
        self.model_type = model_type or SAM_CONFIG["model_type"]
        
        # SAM has issues with MPS on Apple Silicon, use CPU for stability
        if torch.backends.mps.is_available():
            print("Warning: SAM may have compatibility issues with MPS, using CPU for stability")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint_path = checkpoint_path or self._get_default_checkpoint()
        self.sam_model = None
        self.mask_generator = None
        self.predictor = None
        
    def _get_default_checkpoint(self) -> str:
        """Download default SAM checkpoint if not provided"""
        checkpoints = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth", 
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        checkpoint_name = checkpoints[self.model_type]
        checkpoint_path = f"models/{checkpoint_name}"
        
        if not os.path.exists(checkpoint_path):
            os.makedirs("models", exist_ok=True)
            print(f"Downloading SAM checkpoint: {checkpoint_name}")
            # Note: In production, download from official SAM repository
            
        return checkpoint_path
        
    def load_model(self):
        """Load SAM model and initialize generators"""
        print(f"Loading SAM model ({self.model_type}) on {self.device}")
        
        if os.path.exists(self.checkpoint_path):
            try:
                self.sam_model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
                self.sam_model.to(device=self.device)
                
                # Initialize automatic mask generator with conservative settings
                self.mask_generator = SamAutomaticMaskGenerator(
                    self.sam_model,
                    points_per_side=SAM_CONFIG["points_per_side"],
                    pred_iou_thresh=SAM_CONFIG["pred_iou_thresh"],
                    stability_score_thresh=SAM_CONFIG["stability_score_thresh"],
                    crop_n_layers=0,  # Disable cropping to save memory
                    min_mask_region_area=100,
                )
                
                # Initialize predictor for prompted segmentation
                self.predictor = SamPredictor(self.sam_model)
                
                print("SAM model loaded successfully")
            except Exception as e:
                print(f"Error loading SAM model: {e}")
                print("SAM segmentation will be disabled")
                self.sam_model = None
                self.mask_generator = None
                self.predictor = None
        else:
            print(f"SAM checkpoint not found at {self.checkpoint_path}")
            print("Please download SAM checkpoints from: https://github.com/facebookresearch/segment-anything")
    
    def segment_from_tracking(self, image: np.ndarray, tracking_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate segmentation masks for tracked objects
        """
        if self.predictor is None:
            self.load_model()
            
        if self.predictor is None:
            print("SAM model not available, skipping segmentation")
            return {"frame_id": tracking_data["frame_id"], "masks": []}
            
        try:
            self.predictor.set_image(image)
        except Exception as e:
            print(f"Error setting image for SAM predictor: {e}")
            return {"frame_id": tracking_data["frame_id"], "masks": []}
        
        frame_masks = {
            "frame_id": tracking_data["frame_id"],
            "masks": []
        }
        
        for detection in tracking_data["detections"]:
            bbox = detection["bbox"]
            track_id = detection["track_id"]
            
            # Convert YOLO format (x, y, w, h) to SAM format (x1, y1, x2, y2)
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, image.shape[1] - 1))
            y1 = max(0, min(y1, image.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, image.shape[1]))
            y2 = max(y1 + 1, min(y2, image.shape[0]))
            
            # Use bounding box as prompt for SAM
            input_box = np.array([x1, y1, x2, y2])
            
            try:
                with torch.no_grad():  # Save memory during inference
                    masks, scores, logits = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                
                if len(masks) > 0:
                    mask_data = {
                        "track_id": track_id,
                        "mask": masks[0].astype(np.uint8),
                        "score": float(scores[0]),
                        "bbox": bbox,
                        "class_name": detection["class_name"]
                    }
                    frame_masks["masks"].append(mask_data)
                    
            except Exception as e:
                print(f"Error segmenting object {track_id}: {e}")
                continue
                
        return frame_masks
    
    def auto_segment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Automatic segmentation without prompts
        """
        if self.mask_generator is None:
            self.load_model()
            
        if self.mask_generator is None:
            return []
            
        masks = self.mask_generator.generate(image)
        return masks
    
    def process_video_with_tracking(self, video_path: str, tracking_results: List[Dict], 
                                   output_dir: str = "data/processing/segmentation"):
        """
        Process video frames and generate masks for tracked objects
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        all_masks = []
        
        for frame_idx, tracking_data in enumerate(tracking_results):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate masks for this frame
            frame_masks = self.segment_from_tracking(frame_rgb, tracking_data)
            all_masks.append(frame_masks)
            
            # Save individual masks
            self._save_frame_masks(frame_masks, output_dir, frame_idx)
            
        cap.release()
        
        # Save complete mask data
        masks_file = os.path.join(output_dir, "segmentation_results.json")
        with open(masks_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_masks = []
            for frame_data in all_masks:
                json_frame = {
                    "frame_id": frame_data["frame_id"],
                    "masks": []
                }
                for mask_data in frame_data["masks"]:
                    json_mask = {
                        "track_id": mask_data["track_id"],
                        "score": mask_data["score"],
                        "bbox": mask_data["bbox"],
                        "class_name": mask_data["class_name"],
                        "mask_file": f"frame_{frame_data['frame_id']}_mask_{mask_data['track_id']}.png"
                    }
                    json_frame["masks"].append(json_mask)
                json_masks.append(json_frame)
            
            json.dump(json_masks, f, indent=2)
            
        print(f"Segmentation results saved to {masks_file}")
        return all_masks
    
    def _save_frame_masks(self, frame_masks: Dict, output_dir: str, frame_idx: int):
        """Save individual mask images"""
        for mask_data in frame_masks["masks"]:
            mask = mask_data["mask"]
            track_id = mask_data["track_id"]
            
            # Save mask as PNG
            mask_filename = f"frame_{frame_idx}_mask_{track_id}.png"
            mask_path = os.path.join(output_dir, mask_filename)
            
            # Convert boolean mask to 0-255 grayscale
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_img)
    
    def visualize_masks(self, image: np.ndarray, masks_data: Dict) -> np.ndarray:
        """Visualize masks overlaid on image"""
        result = image.copy()
        
        for mask_data in masks_data["masks"]:
            mask = mask_data["mask"]
            track_id = mask_data["track_id"]
            
            # Create colored overlay
            color = np.random.randint(0, 255, 3)
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            
            # Blend with original image
            result = cv2.addWeighted(result, 0.7, colored_mask, 0.3, 0)
            
            # Add track ID label
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                center_x, center_y = int(np.mean(x)), int(np.mean(y))
                cv2.putText(result, f"ID:{track_id}", (center_x, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result

def main():
    print("SAM Segmentation Module")
    
    # Example usage
    segmenter = SAMSegmenter(model_type="vit_b")  # Use smaller model for faster inference
    
    # Load tracking results if available
    tracking_file = "data/processing/tracking/tracking_results.json"
    if os.path.exists(tracking_file):
        print("Loading tracking results...")
        with open(tracking_file, 'r') as f:
            tracking_results = json.load(f)
            
        video_path = "data/inputs/samples/tabletop_manipulation.mp4"
        if os.path.exists(video_path):
            print("Processing video with SAM segmentation...")
            segmenter.process_video_with_tracking(video_path, tracking_results)
        else:
            print("Sample video not found")
    else:
        print("Tracking results not found. Run tracking first.")

if __name__ == "__main__":
    main()