"""
Common utility functions shared across all phases
"""

import os
import numpy as np
from datetime import datetime
from typing import Any, Optional, Dict, List

def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default"""
    return data.get(key, default)

def ensure_directory(directory_path: str) -> bool:
    """Ensure directory exists, create if necessary"""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
        return True
    except Exception as e:
        print(f"Failed to create directory {directory_path}: {e}")
        return False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format timestamp for logging and display"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

def calculate_polygon_area(points: List[List[float]]) -> float:
    """Calculate area of polygon using shoelace formula"""
    if len(points) < 3:
        return 0.0
    
    try:
        # Convert to numpy array
        pts = np.array(points)
        
        # Shoelace formula
        x = pts[:, 0]
        y = pts[:, 1]
        
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(points)] - x[(i + 1) % len(points)] * y[i] 
                            for i in range(len(points))))
        
        return float(area)
    except Exception:
        return 0.0

def interpolate_linear(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Linear interpolation between two points"""
    if x1 == x2:
        return y1
    
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

def parse_distance_from_filename(filename: str) -> Optional[float]:
    """Extract distance from filename patterns like 'image_180cm_optimal.jpg'"""
    import re
    
    try:
        # Look for patterns like "180cm", "180.5cm", etc.
        patterns = [
            r'(\d+(?:\.\d+)?)cm',
            r'(\d+(?:\.\d+)?)_cm',
            r'dist(\d+(?:\.\d+)?)',
            r'distance_?(\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                distance = float(match.group(1))
                return distance
        
        return None
        
    except (ValueError, AttributeError):
        return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def create_safe_filename(base_name: str, extension: str = '.jpg') -> str:
    """Create a safe filename with timestamp"""
    # Remove any unsafe characters
    safe_name = "".join(c for c in base_name if c.isalnum() or c in ('_', '-'))
    
    # Add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    
    return f"{safe_name}_{timestamp}{extension}"

def log_array_info(array: np.ndarray, name: str = "Array") -> None:
    """Log information about numpy array"""
    if array is None:
        print(f"{name}: None")
        return
    
    print(f"{name}: shape={array.shape}, dtype={array.dtype}, "
          f"min={np.min(array):.3f}, max={np.max(array):.3f}, "
          f"mean={np.mean(array):.3f}")

def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

def flatten_list(nested_list: List[List]) -> List:
    """Flatten a nested list"""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat

def round_to_precision(value: float, precision: int = 2) -> float:
    """Round value to specified decimal places"""
    return round(value, precision)

def percentage_string(value: float, total: float) -> str:
    """Create percentage string from value and total"""
    if total == 0:
        return "0.0%"
    percentage = (value / total) * 100
    return f"{percentage:.1f}%"

def time_elapsed_string(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """Create elapsed time string"""
    if end_time is None:
        end_time = datetime.now()
    
    elapsed = end_time - start_time
    total_seconds = elapsed.total_seconds()
    
    if total_seconds < 1:
        return f"{total_seconds * 1000:.0f}ms"
    elif total_seconds < 60:
        return f"{total_seconds:.2f}s"
    else:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}m {seconds:.1f}s"
