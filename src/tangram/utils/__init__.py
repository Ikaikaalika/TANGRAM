"""
TANGRAM Utilities Package

Common utility functions used across the robotic scene understanding pipeline.
"""

from .logging_utils import setup_logger, log_function_call
from .file_utils import save_json, load_json, ensure_directory

# Optional imports that may require additional dependencies
try:
    from .geometry_utils import compute_distance_3d, transform_coordinates
    _geometry_utils_available = True
except ImportError:
    _geometry_utils_available = False

try:
    from .video_utils import extract_video_info, validate_video_file
    _video_utils_available = True
except ImportError:
    _video_utils_available = False

__all__ = [
    'setup_logger', 'log_function_call',
    'save_json', 'load_json', 'ensure_directory'
]

if _geometry_utils_available:
    __all__.extend(['compute_distance_3d', 'transform_coordinates'])

if _video_utils_available:
    __all__.extend(['extract_video_info', 'validate_video_file'])