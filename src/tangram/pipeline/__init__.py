"""
TANGRAM Pipeline Module

Organized pipeline components for enterprise-grade robotic scene understanding.

Pipeline Structure:
- perception: Object detection, tracking, segmentation
- understanding: 3D reconstruction, scene graphs
- planning: LLM-based task planning and reasoning
- execution: Robot control and motion planning
"""

# Import submodules (but not all contents to avoid conflicts)
from . import perception
from . import understanding
from . import planning
from . import execution

__all__ = [
    "perception",
    "understanding", 
    "planning",
    "execution"
]