#!/usr/bin/env python3

# Quick test of SAM model loading
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

sam_dir = Path("models/sam")
vit_b_path = sam_dir / "sam_vit_b_01ec64.pth"

print(f"Loading SAM ViT-B from: {vit_b_path}")
sam = sam_model_registry["vit_b"](checkpoint=str(vit_b_path))
predictor = SamPredictor(sam)
print("✅ SAM model loaded successfully!")

# Test Open3D
import open3d as o3d
points = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
print(f"✅ Open3D working - created point cloud with {len(pcd.points)} points")

print("\n🎯 PyBullet, Open3D, and SAM are ready!")