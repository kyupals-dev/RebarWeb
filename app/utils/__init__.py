"""
AI Phases Utils Package
Common utilities and constants shared across all phases
"""

from .constants import *
from .common import *
from .validators import *

__all__ = [
    # Constants
    'PHASE_NAMES',
    'CLASS_NAMES', 
    'DEFAULT_COLORS',
    'DEFAULT_THRESHOLDS',
    
    # Common utilities
    'safe_get',
    'ensure_directory',
    'get_file_size_mb',
    'format_timestamp',
    
    # Validators
    'validate_image_path',
    'validate_image_array',
    'validate_bounding_box',
    'validate_detection_data'
]
