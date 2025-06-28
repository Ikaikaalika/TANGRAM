"""
Video processing utilities for the TANGRAM pipeline.

Provides video validation, information extraction, and processing helpers.
"""

import cv2
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def extract_video_info(video_path: Union[str, Path]) -> Optional[Dict]:
    """
    Extract comprehensive information from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information or None if failed
    """
    try:
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
            
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return None
            
        # Extract video properties
        info = {
            'path': str(video_path),
            'filename': video_path.name,
            'size_mb': video_path.stat().st_size / (1024 * 1024),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': 0,
            'codec': None
        }
        
        # Calculate duration
        if info['fps'] > 0:
            info['duration_seconds'] = info['frame_count'] / info['fps']
            
        # Try to get codec information
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        if fourcc:
            info['codec'] = ''.join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
            
        cap.release()
        
        logger.info(f"Video info extracted: {info['frame_count']} frames, "
                   f"{info['duration_seconds']:.1f}s, {info['width']}x{info['height']}")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to extract video info from {video_path}: {e}")
        return None

def validate_video_file(video_path: Union[str, Path], 
                       min_duration: float = 1.0,
                       max_duration: float = 300.0,
                       min_resolution: Tuple[int, int] = (320, 240)) -> bool:
    """
    Validate video file for pipeline processing.
    
    Args:
        video_path: Path to video file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds  
        min_resolution: Minimum resolution (width, height)
        
    Returns:
        True if video is valid for processing
    """
    info = extract_video_info(video_path)
    
    if not info:
        return False
        
    # Check duration
    if info['duration_seconds'] < min_duration:
        logger.warning(f"Video too short: {info['duration_seconds']:.1f}s < {min_duration}s")
        return False
        
    if info['duration_seconds'] > max_duration:
        logger.warning(f"Video too long: {info['duration_seconds']:.1f}s > {max_duration}s")
        return False
        
    # Check resolution
    if info['width'] < min_resolution[0] or info['height'] < min_resolution[1]:
        logger.warning(f"Resolution too low: {info['width']}x{info['height']} < "
                      f"{min_resolution[0]}x{min_resolution[1]}")
        return False
        
    # Check frame rate
    if info['fps'] < 10:
        logger.warning(f"Frame rate too low: {info['fps']} fps")
        return False
        
    logger.info(f"Video validation passed: {video_path}")
    return True

def estimate_processing_time(video_info: Dict) -> Dict[str, float]:
    """
    Estimate processing time for different pipeline stages.
    
    Args:
        video_info: Video information dictionary
        
    Returns:
        Dictionary with estimated processing times in seconds
    """
    frame_count = video_info['frame_count']
    resolution_factor = (video_info['width'] * video_info['height']) / (1920 * 1080)
    
    estimates = {
        'frame_extraction': frame_count * 0.01,  # ~10ms per frame
        'object_tracking': frame_count * 0.1 * resolution_factor,  # YOLO inference
        'segmentation': frame_count * 0.5 * resolution_factor,  # SAM inference  
        'colmap_sparse': frame_count * 0.2,  # Feature extraction/matching
        'colmap_dense': frame_count * 2.0 if frame_count < 100 else frame_count * 5.0,
        'scene_graph': 5.0,  # Graph construction
        'llm_interpretation': 10.0,  # LLM API call
        'simulation': 30.0  # Robot simulation
    }
    
    estimates['total'] = sum(estimates.values())
    
    return estimates

def get_video_thumbnail(video_path: Union[str, Path], 
                       frame_number: int = 0,
                       output_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Extract a thumbnail image from video.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract (0 = first frame)
        output_path: Output path for thumbnail (auto-generated if None)
        
    Returns:
        Path to saved thumbnail or None if failed
    """
    try:
        video_path = Path(video_path)
        
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_thumbnail.jpg"
        else:
            output_path = Path(output_path)
            
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
            
        # Seek to specified frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Cannot read frame {frame_number} from {video_path}")
            return None
            
        # Save thumbnail
        success = cv2.imwrite(str(output_path), frame)
        
        if success:
            logger.info(f"Thumbnail saved: {output_path}")
            return output_path
        else:
            logger.error(f"Failed to save thumbnail: {output_path}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to extract thumbnail from {video_path}: {e}")
        return None

def check_video_codec_support() -> Dict[str, bool]:
    """
    Check which video codecs are supported by OpenCV.
    
    Returns:
        Dictionary mapping codec names to support status
    """
    codecs = {
        'H264': cv2.VideoWriter_fourcc(*'H264'),
        'XVID': cv2.VideoWriter_fourcc(*'XVID'),
        'MJPG': cv2.VideoWriter_fourcc(*'MJPG'),
        'MP4V': cv2.VideoWriter_fourcc(*'MP4V')
    }
    
    support = {}
    
    for name, fourcc in codecs.items():
        # Try to create a temporary video writer
        test_path = '/tmp/test_codec.mp4'
        writer = cv2.VideoWriter(test_path, fourcc, 30.0, (640, 480))
        
        if writer.isOpened():
            support[name] = True
            writer.release()
            # Clean up test file
            try:
                Path(test_path).unlink()
            except:
                pass
        else:
            support[name] = False
            
    return support