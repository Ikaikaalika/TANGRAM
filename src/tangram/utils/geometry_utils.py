"""
Geometry and spatial utilities for the TANGRAM pipeline.

Provides 3D geometry calculations, coordinate transformations, and spatial analysis.
"""

import numpy as np
import math
from typing import List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

def compute_distance_3d(point1: Union[List, np.ndarray], 
                       point2: Union[List, np.ndarray]) -> float:
    """
    Compute Euclidean distance between two 3D points.
    
    Args:
        point1: First point [x, y, z]
        point2: Second point [x, y, z]
        
    Returns:
        Distance in 3D space
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.linalg.norm(p1 - p2)

def compute_distance_2d(point1: Union[List, np.ndarray], 
                       point2: Union[List, np.ndarray]) -> float:
    """
    Compute distance in 2D plane (ignoring Z coordinate).
    
    Args:
        point1: First point [x, y, z] or [x, y]
        point2: Second point [x, y, z] or [x, y]
        
    Returns:
        Distance in XY plane
    """
    p1 = np.array(point1)[:2]
    p2 = np.array(point2)[:2]
    return np.linalg.norm(p1 - p2)

def transform_coordinates(points: np.ndarray, 
                         rotation_matrix: np.ndarray,
                         translation: np.ndarray) -> np.ndarray:
    """
    Apply rigid transformation to 3D points.
    
    Args:
        points: Array of 3D points (N x 3)
        rotation_matrix: 3x3 rotation matrix
        translation: 3D translation vector
        
    Returns:
        Transformed points
    """
    if points.ndim == 1:
        points = points.reshape(1, -1)
        
    transformed = (rotation_matrix @ points.T).T + translation
    return transformed

def quaternion_to_rotation_matrix(quaternion: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        quaternion: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    q = np.array(quaternion)
    q = q / np.linalg.norm(q)  # Normalize
    
    w, x, y, z = q
    
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    return rotation_matrix

def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
        
    return x, y, z

def compute_bounding_box_3d(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D bounding box for a set of points.
    
    Args:
        points: Array of 3D points (N x 3)
        
    Returns:
        Tuple of (min_point, max_point) defining the bounding box
    """
    if points.size == 0:
        return np.zeros(3), np.zeros(3)
        
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    
    return min_point, max_point

def point_in_bounding_box(point: np.ndarray,
                         min_point: np.ndarray, 
                         max_point: np.ndarray) -> bool:
    """
    Check if point is inside 3D bounding box.
    
    Args:
        point: 3D point to test
        min_point: Minimum corner of bounding box
        max_point: Maximum corner of bounding box
        
    Returns:
        True if point is inside bounding box
    """
    return np.all(point >= min_point) and np.all(point <= max_point)

def compute_centroid(points: np.ndarray) -> np.ndarray:
    """
    Compute centroid of 3D points.
    
    Args:
        points: Array of 3D points (N x 3)
        
    Returns:
        Centroid point [x, y, z]
    """
    if points.size == 0:
        return np.zeros(3)
        
    return np.mean(points, axis=0)

def project_point_to_plane(point: np.ndarray,
                          plane_normal: np.ndarray,
                          plane_point: np.ndarray) -> np.ndarray:
    """
    Project a 3D point onto a plane.
    
    Args:
        point: 3D point to project
        plane_normal: Normal vector of the plane
        plane_point: Any point on the plane
        
    Returns:
        Projected point on the plane
    """
    # Normalize plane normal
    normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Vector from plane point to target point
    vec_to_point = point - plane_point
    
    # Distance from point to plane
    distance = np.dot(vec_to_point, normal)
    
    # Project point onto plane
    projected_point = point - distance * normal
    
    return projected_point

def compute_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two 3D vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians
    """
    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Compute angle using dot product
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle

def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to a set of 3D points using least squares.
    
    Args:
        points: Array of 3D points (N x 3)
        
    Returns:
        Tuple of (plane_normal, plane_point)
    """
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane")
        
    # Compute centroid
    centroid = compute_centroid(points)
    
    # Center the points
    centered_points = points - centroid
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(centered_points)
    
    # Normal is the last column of V (or last row of Vt)
    normal = Vt[-1]
    
    return normal, centroid

def compute_convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Compute 2D convex hull using Graham scan algorithm.
    
    Args:
        points: Array of 2D points (N x 2)
        
    Returns:
        Array of hull points in counter-clockwise order
    """
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    points = np.array(points)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    if len(points) <= 1:
        return points
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Remove last point of each half because it's repeated
    return np.array(lower[:-1] + upper[:-1])

def interpolate_3d_trajectory(waypoints: np.ndarray, 
                             num_points: int = 100) -> np.ndarray:
    """
    Interpolate smooth 3D trajectory through waypoints.
    
    Args:
        waypoints: Array of 3D waypoints (N x 3)
        num_points: Number of interpolated points
        
    Returns:
        Smooth trajectory points
    """
    if len(waypoints) < 2:
        return waypoints
        
    from scipy.interpolate import interp1d
    
    # Parameter values for waypoints
    t_waypoints = np.linspace(0, 1, len(waypoints))
    
    # Parameter values for interpolated points
    t_interp = np.linspace(0, 1, num_points)
    
    # Interpolate each dimension separately
    trajectory = np.zeros((num_points, 3))
    
    for dim in range(3):
        f = interp1d(t_waypoints, waypoints[:, dim], kind='cubic')
        trajectory[:, dim] = f(t_interp)
        
    return trajectory