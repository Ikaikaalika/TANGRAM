"""
Perception Pipeline

Computer vision components for object detection, tracking, and segmentation.

Components:
- detection: YOLO-based object detection
- tracker: Multi-object tracking with ByteTrack
- segmenter: SAM-based object segmentation
- computer_vision: Advanced reconstruction algorithms
"""

# Import key classes for easy access
try:
    from .detection.yolo_sam_detector import YOLOSAMDetector
except ImportError:
    pass

try:
    from .tracker.track_objects import YOLOByteTracker
except ImportError:
    pass

try:
    from .segmenter.run_sam import SAMSegmenter
except ImportError:
    pass

__all__ = [
    "YOLOSAMDetector",
    "YOLOByteTracker", 
    "SAMSegmenter"
]