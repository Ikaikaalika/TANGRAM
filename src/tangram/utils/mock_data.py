#!/usr/bin/env python3
"""
Mock data generation utilities for TANGRAM pipeline.

Used for testing and demo purposes when real data is not available.
"""

# numpy not needed for mock data generation


def create_mock_3d_positions():
    """Create mock 3D positions for testing purposes."""
    return {
        "frame_000001": [
            {"object_id": 1, "class": "apple", "x": 0.15, "y": 0.12, "z": 0.05},
            {"object_id": 2, "class": "bottle", "x": -0.08, "y": 0.20, "z": 0.12},
            {"object_id": 3, "class": "bowl", "x": 0.05, "y": -0.15, "z": 0.03}
        ],
        "frame_000015": [
            {"object_id": 1, "class": "apple", "x": 0.14, "y": 0.13, "z": 0.05},
            {"object_id": 2, "class": "bottle", "x": -0.07, "y": 0.19, "z": 0.12},
            {"object_id": 3, "class": "bowl", "x": 0.06, "y": -0.14, "z": 0.03}
        ],
        "frame_000030": [
            {"object_id": 1, "class": "apple", "x": 0.12, "y": 0.15, "z": 0.05},
            {"object_id": 2, "class": "bottle", "x": -0.05, "y": 0.18, "z": 0.12},
            {"object_id": 3, "class": "bowl", "x": 0.08, "y": -0.12, "z": 0.03}
        ]
    }


def create_mock_tracking_data():
    """Create mock tracking data for testing purposes."""
    return {
        "tracks": [
            {
                "track_id": 1,
                "class": "apple",
                "frames": [
                    {"frame": 1, "bbox": [120, 80, 160, 120], "confidence": 0.95},
                    {"frame": 15, "bbox": [122, 82, 162, 122], "confidence": 0.93},
                    {"frame": 30, "bbox": [125, 85, 165, 125], "confidence": 0.91}
                ]
            },
            {
                "track_id": 2,
                "class": "bottle", 
                "frames": [
                    {"frame": 1, "bbox": [200, 60, 240, 140], "confidence": 0.88},
                    {"frame": 15, "bbox": [202, 62, 242, 142], "confidence": 0.87},
                    {"frame": 30, "bbox": [205, 65, 245, 145], "confidence": 0.85}
                ]
            }
        ]
    }


def create_mock_scene_graph():
    """Create mock scene graph for testing purposes."""
    return {
        "objects": [
            {"id": 1, "class": "apple", "position": [0.15, 0.12, 0.05]},
            {"id": 2, "class": "bottle", "position": [-0.08, 0.20, 0.12]},
            {"id": 3, "class": "bowl", "position": [0.05, -0.15, 0.03]}
        ],
        "relationships": [
            {"type": "near", "object1": 1, "object2": 3, "confidence": 0.8},
            {"type": "above", "object1": 2, "object2": 3, "confidence": 0.9}
        ],
        "scene_description": "A tabletop scene with an apple, bottle, and bowl"
    }