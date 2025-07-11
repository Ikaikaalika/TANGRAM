#!/usr/bin/env python3

import numpy as np
import cv2
import json
import os
from typing import List, Tuple, Dict, Any
import open3d as o3d
import sqlite3

class PointTriangulator:
    def __init__(self):
        self.camera_matrices = []
        self.camera_poses = []
        self.distortion_coeffs = []
        self.colmap_db_path = None
        self.sparse_model_path = None
    
    def load_colmap_results(self, colmap_output_dir: str):
        """Load camera poses and intrinsics from COLMAP output"""
        self.sparse_model_path = os.path.join(colmap_output_dir, "sparse", "0")
        self.colmap_db_path = os.path.join(colmap_output_dir, "database.db")
        
        if os.path.exists(self.sparse_model_path):
            print(f"Loading COLMAP sparse reconstruction from {self.sparse_model_path}")
            
            # Read cameras.txt
            cameras_file = os.path.join(self.sparse_model_path, "cameras.txt")
            if os.path.exists(cameras_file):
                self.camera_matrices = self._read_colmap_cameras(cameras_file)
            
            # Read images.txt
            images_file = os.path.join(self.sparse_model_path, "images.txt")
            if os.path.exists(images_file):
                self.camera_poses = self._read_colmap_images(images_file)
            
            print(f"Loaded {len(self.camera_matrices)} cameras and {len(self.camera_poses)} poses")
        else:
            print(f"COLMAP sparse model not found at {self.sparse_model_path}")
    
    def _read_colmap_cameras(self, cameras_file: str) -> List[np.ndarray]:
        """Read camera intrinsics from COLMAP cameras.txt"""
        cameras = []
        with open(cameras_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                    
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                
                if model == "PINHOLE":
                    fx, fy, cx, cy = map(float, parts[4:8])
                    K = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])
                    cameras.append(K)
                    
        return cameras
    
    def _read_colmap_images(self, images_file: str) -> List[Dict[str, Any]]:
        """Read camera poses from COLMAP images.txt"""
        poses = []
        with open(images_file, 'r') as f:
            lines = f.readlines()
            
        for i in range(0, len(lines), 2):  # Every other line contains pose info
            line = lines[i].strip()
            if line.startswith('#') or not line:
                continue
                
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]
            
            # Convert quaternion to rotation matrix
            R = self._quat_to_rotation_matrix([qw, qx, qy, qz])
            t = np.array([tx, ty, tz]).reshape(3, 1)
            
            # Create pose matrix
            pose = np.hstack([R, t])
            
            poses.append({
                'image_id': image_id,
                'camera_id': camera_id,
                'image_name': image_name,
                'pose': pose,
                'R': R,
                't': t
            })
            
        return poses
    
    def _quat_to_rotation_matrix(self, quat: List[float]) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        qw, qx, qy, qz = quat
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def triangulate_object_positions(self, mask_data: List[Dict], 
                                   output_dir: str = "data/outputs/point_clouds") -> Dict[str, Any]:
        """
        Triangulate 3D positions of tracked objects using their masks and camera poses
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Group mask data by track_id
        objects_by_id = {}
        for frame_data in mask_data:
            frame_id = frame_data["frame_id"]
            
            for mask_info in frame_data["masks"]:
                track_id = mask_info["track_id"]
                
                if track_id not in objects_by_id:
                    objects_by_id[track_id] = {
                        "observations": [],
                        "class_name": mask_info["class_name"]
                    }
                
                # Calculate mask centroid as 2D observation
                mask_file = os.path.join("data/processing/segmentation", mask_info["mask_file"])
                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Find mask centroid
                        moments = cv2.moments(mask)
                        if moments["m00"] != 0:
                            cx = moments["m10"] / moments["m00"]
                            cy = moments["m01"] / moments["m00"]
                            
                            objects_by_id[track_id]["observations"].append({
                                "frame_id": frame_id,
                                "point_2d": [cx, cy],
                                "bbox": mask_info["bbox"]
                            })
        
        # Triangulate 3D positions for each object
        object_3d_positions = {}
        
        for track_id, obj_data in objects_by_id.items():
            observations = obj_data["observations"]
            
            if len(observations) >= 2:  # Need at least 2 views for triangulation
                points_3d = self._triangulate_object_points(observations)
                
                if points_3d is not None:
                    object_3d_positions[track_id] = {
                        "position": points_3d.tolist(),
                        "class_name": obj_data["class_name"],
                        "num_observations": len(observations)
                    }
        
        # Save 3D positions
        output_file = os.path.join(output_dir, "object_3d_positions.json")
        with open(output_file, 'w') as f:
            json.dump(object_3d_positions, f, indent=2)
        
        print(f"3D object positions saved to {output_file}")
        return object_3d_positions
    
    def _triangulate_object_points(self, observations: List[Dict]) -> np.ndarray:
        """Triangulate 3D point from multiple 2D observations"""
        if len(self.camera_poses) < 2 or len(observations) < 2:
            return None
        
        # Use first two observations for triangulation
        obs1, obs2 = observations[0], observations[1]
        
        # Get corresponding camera poses
        if obs1["frame_id"] < len(self.camera_poses) and obs2["frame_id"] < len(self.camera_poses):
            pose1 = self.camera_poses[obs1["frame_id"]]
            pose2 = self.camera_poses[obs2["frame_id"]]
            
            # Get camera matrices
            if len(self.camera_matrices) > 0:
                K = self.camera_matrices[0]  # Assume same camera
                
                # Project matrices
                P1 = K @ pose1["pose"]
                P2 = K @ pose2["pose"]
                
                # 2D points
                pt1 = np.array(obs1["point_2d"]).reshape(2, 1)
                pt2 = np.array(obs2["point_2d"]).reshape(2, 1)
                
                # Triangulate
                points_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
                
                # Convert to 3D
                if points_4d[3, 0] != 0:
                    points_3d = points_4d[:3, 0] / points_4d[3, 0]
                    return points_3d
        
        return None
    
    def export_point_cloud(self, object_positions: Dict[str, Any], 
                          output_file: str = "data/outputs/point_clouds/scene_point_cloud.ply"):
        """Export 3D object positions as point cloud"""
        points = []
        colors = []
        
        for track_id, obj_data in object_positions.items():
            position = obj_data["position"]
            points.append(position)
            
            # Assign color based on object class
            color = self._get_class_color(obj_data["class_name"])
            colors.append(color)
        
        if len(points) > 0:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            # Save point cloud
            o3d.io.write_point_cloud(output_file, pcd)
            print(f"Point cloud saved to {output_file}")
        
        return points, colors
    
    def _get_class_color(self, class_name: str) -> List[float]:
        """Get RGB color for object class"""
        colors = {
            "person": [1.0, 0.0, 0.0],
            "bottle": [0.0, 1.0, 0.0],
            "cup": [0.0, 0.0, 1.0],
            "bowl": [1.0, 1.0, 0.0],
            "apple": [1.0, 0.0, 1.0],
            "book": [0.0, 1.0, 1.0]
        }
        return colors.get(class_name, [0.5, 0.5, 0.5])

def main():
    print("3D Point Triangulation Module")
    
    triangulator = PointTriangulator()
    
    # Load COLMAP results if available
    colmap_dir = "data/outputs/point_clouds"
    if os.path.exists(os.path.join(colmap_dir, "sparse")):
        triangulator.load_colmap_results(colmap_dir)
        
        # Load mask data
        mask_file = "data/processing/segmentation/segmentation_results.json"
        if os.path.exists(mask_file):
            with open(mask_file, 'r') as f:
                mask_data = json.load(f)
            
            print("Triangulating object positions...")
            object_3d = triangulator.triangulate_object_positions(mask_data)
            
            if object_3d:
                triangulator.export_point_cloud(object_3d)
        else:
            print("Mask data not found. Run segmentation first.")
    else:
        print("COLMAP results not found. Run COLMAP reconstruction first.")

if __name__ == "__main__":
    main()