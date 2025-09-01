"""
Validation functions for AI processing phases
"""

import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from .constants import ALLOWED_IMAGE_EXTENSIONS, IMAGE_TARGET_SIZE

def validate_image_path(image_path: str) -> bool:
    """Validate image file path"""
    if not image_path:
        raise ValueError("Image path cannot be empty")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Check file extension
    _, ext = os.path.splitext(image_path.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {ext}")
    
    # Check file size (max 50MB)
    file_size = os.path.getsize(image_path)
    if file_size > 50 * 1024 * 1024:
        raise ValueError(f"Image file too large: {file_size / (1024 * 1024):.1f} MB (max 50MB)")
    
    return True

def validate_image_array(image: np.ndarray, expected_shape: Optional[Tuple] = None) -> bool:
    """Validate numpy image array"""
    if image is None:
        raise ValueError("Image array cannot be None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D array, got shape: {image.shape}")
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        raise ValueError(f"Image must have 1, 3, or 4 channels, got: {image.shape[2]}")
    
    if expected_shape and image.shape != expected_shape:
        raise ValueError(f"Expected image shape {expected_shape}, got {image.shape}")
    
    # Check data type
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(f"Unsupported image data type: {image.dtype}")
    
    return True

def validate_bounding_box(bbox: Dict) -> bool:
    """Validate bounding box dictionary"""
    if not isinstance(bbox, dict):
        raise ValueError("Bounding box must be a dictionary")
    
    required_keys = ['x_min', 'y_min', 'x_max', 'y_max', 'width', 'height']
    for key in required_keys:
        if key not in bbox:
            raise ValueError(f"Missing bounding box key: {key}")
    
    # Validate numeric values
    for key in required_keys:
        if not isinstance(bbox[key], (int, float)):
            raise ValueError(f"Bounding box {key} must be numeric, got {type(bbox[key])}")
        if bbox[key] < 0:
            raise ValueError(f"Bounding box {key} must be non-negative, got {bbox[key]}")
    
    # Validate relationships
    if bbox['x_min'] >= bbox['x_max']:
        raise ValueError(f"x_min ({bbox['x_min']}) must be less than x_max ({bbox['x_max']})")
    
    if bbox['y_min'] >= bbox['y_max']:
        raise ValueError(f"y_min ({bbox['y_min']}) must be less than y_max ({bbox['y_max']})")
    
    # Validate calculated dimensions
    expected_width = bbox['x_max'] - bbox['x_min']
    expected_height = bbox['y_max'] - bbox['y_min']
    
    if abs(bbox['width'] - expected_width) > 0.1:
        raise ValueError(f"Bounding box width inconsistent: expected {expected_width}, got {bbox['width']}")
    
    if abs(bbox['height'] - expected_height) > 0.1:
        raise ValueError(f"Bounding box height inconsistent: expected {expected_height}, got {bbox['height']}")
    
    return True

def validate_detection_data(detection: Dict) -> bool:
    """Validate single detection dictionary"""
    if not isinstance(detection, dict):
        raise ValueError("Detection must be a dictionary")
    
    required_keys = ['class_id', 'class_name', 'confidence', 'bbox']
    for key in required_keys:
        if key not in detection:
            raise ValueError(f"Missing detection key: {key}")
    
    # Validate class_id
    class_id = detection['class_id']
    if not isinstance(class_id, int) or class_id < 0:
        raise ValueError(f"class_id must be non-negative integer, got {class_id}")
    
    # Validate class_name
    class_name = detection['class_name']
    if not isinstance(class_name, str) or not class_name:
        raise ValueError(f"class_name must be non-empty string, got {class_name}")
    
    # Validate confidence
    confidence = detection['confidence']
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")
    
    # Validate bbox
    bbox = detection['bbox']
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"bbox must be list of 4 numbers, got {bbox}")
    
    for i, coord in enumerate(bbox):
        if not isinstance(coord, (int, float)):
            raise ValueError(f"bbox coordinate {i} must be numeric, got {type(coord)}")
    
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid bbox coordinates: ({x1}, {y1}, {x2}, {y2})")
    
    # Validate optional mask
    if 'mask' in detection and detection['mask'] is not None:
        mask = detection['mask']
        if not isinstance(mask, np.ndarray):
            raise ValueError("Detection mask must be numpy array")
        if len(mask.shape) != 2:
            raise ValueError(f"Detection mask must be 2D, got shape {mask.shape}")
        if mask.dtype != bool and mask.dtype != np.bool_:
            raise ValueError(f"Detection mask must be boolean, got dtype {mask.dtype}")
    
    return True

def validate_detections_list(detections: List[Dict]) -> bool:
    """Validate list of detections"""
    if not isinstance(detections, list):
        raise ValueError("Detections must be a list")
    
    for i, detection in enumerate(detections):
        try:
            validate_detection_data(detection)
        except ValueError as e:
            raise ValueError(f"Detection {i} validation failed: {str(e)}")
    
    return True

def validate_dimensions(dimensions: Dict) -> bool:
    """Validate dimensions dictionary"""
    if not isinstance(dimensions, dict):
        raise ValueError("Dimensions must be a dictionary")
    
    required_keys = ['length', 'width', 'height', 'unit', 'volume_cm3']
    for key in required_keys:
        if key not in dimensions:
            raise ValueError(f"Missing dimensions key: {key}")
    
    # Validate numeric values
    numeric_keys = ['length', 'width', 'height', 'volume_cm3']
    for key in numeric_keys:
        value = dimensions[key]
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Dimension {key} must be positive number, got {value}")
    
    # Validate unit
    unit = dimensions['unit']
    if not isinstance(unit, str) or unit not in ['cm', 'mm', 'm']:
        raise ValueError(f"Invalid unit: {unit}")
    
    # Validate volume consistency (for cm units)
    if unit == 'cm':
        expected_volume = dimensions['length'] * dimensions['width'] * dimensions['height']
        actual_volume = dimensions['volume_cm3']
        
        # Allow 1% tolerance for floating point errors
        if abs(actual_volume - expected_volume) / expected_volume > 0.01:
            raise ValueError(f"Volume inconsistent: expected {expected_volume:.1f}, got {actual_volume:.1f}")
    
    return True

def validate_cement_mixture(mixture: Dict) -> bool:
    """Validate cement mixture dictionary"""
    if not isinstance(mixture, dict):
        raise ValueError("Cement mixture must be a dictionary")
    
    required_keys = [
        'cement_ratio', 'sand_ratio', 'aggregate_ratio', 'ratio_string',
        'cement_bags', 'sand_volume_m3', 'aggregate_volume_m3'
    ]
    
    for key in required_keys:
        if key not in mixture:
            raise ValueError(f"Missing cement mixture key: {key}")
    
    # Validate ratios
    ratio_keys = ['cement_ratio', 'sand_ratio', 'aggregate_ratio']
    for key in ratio_keys:
        value = mixture[key]
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Ratio {key} must be positive number, got {value}")
    
    # Validate quantities
    quantity_keys = ['cement_bags', 'sand_volume_m3', 'aggregate_volume_m3']
    for key in quantity_keys:
        value = mixture[key]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Quantity {key} must be non-negative number, got {value}")
    
    # Validate ratio string
    ratio_string = mixture['ratio_string']
    if not isinstance(ratio_string, str) or not ratio_string:
        raise ValueError("ratio_string must be non-empty string")
    
    return True

def validate_phase_data_keys(data: Dict, required_keys: List[str], phase_name: str = "") -> bool:
    """Validate that data dictionary contains all required keys"""
    if not isinstance(data, dict):
        raise ValueError(f"{phase_name} data must be a dictionary")
    
    missing_keys = []
    for key in required_keys:
        if key not in data:
            missing_keys.append(key)
    
    if missing_keys:
        keys_str = ", ".join(missing_keys)
        raise ValueError(f"{phase_name} missing required keys: {keys_str}")
    
    return True

def validate_polygons(polygons: List[Dict]) -> bool:
    """Validate list of polygons"""
    if not isinstance(polygons, list):
        raise ValueError("Polygons must be a list")
    
    for i, polygon in enumerate(polygons):
        if not isinstance(polygon, dict):
            raise ValueError(f"Polygon {i} must be a dictionary")
        
        if 'points' not in polygon:
            raise ValueError(f"Polygon {i} missing 'points' key")
        
        points = polygon['points']
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Polygon {i} must have at least 3 points")
        
        for j, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"Polygon {i} point {j} must be [x, y] coordinates")
            
            x, y = point
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(f"Polygon {i} point {j} coordinates must be numeric")
        
        # Validate area if present
        if 'area' in polygon:
            area = polygon['area']
            if not isinstance(area, (int, float)) or area < 0:
                raise ValueError(f"Polygon {i} area must be non-negative number")
    
    return True

def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """Validate file path"""
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be non-empty string")
    
    if must_exist and not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if path is absolute or relative is valid
    if not os.path.isabs(file_path):
        # For relative paths, check if directory structure makes sense
        dirname = os.path.dirname(file_path)
        if dirname and not os.path.exists(dirname):
            if must_exist:
                raise ValueError(f"Directory does not exist: {dirname}")
    
    return True

def validate_image_size_compatibility(image_shape: Tuple, target_size: Tuple = IMAGE_TARGET_SIZE) -> bool:
    """Validate image size compatibility with target processing size"""
    if len(image_shape) < 2:
        raise ValueError(f"Image must have at least height and width dimensions")
    
    height, width = image_shape[:2]
    target_width, target_height = target_size
    
    # Check if image is reasonable size (not too small or too large)
    if width < 100 or height < 100:
        raise ValueError(f"Image too small: {width}x{height} (minimum 100x100)")
    
    if width > 5000 or height > 5000:
        raise ValueError(f"Image too large: {width}x{height} (maximum 5000x5000)")
    
    # Log compatibility info
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    
    if abs(aspect_ratio - target_aspect_ratio) > 0.5:
        print(f"⚠️  Image aspect ratio ({aspect_ratio:.2f}) differs significantly from target ({target_aspect_ratio:.2f})")
    
    return True

def validate_confidence_scores(scores: List[float]) -> bool:
    """Validate list of confidence scores"""
    if not isinstance(scores, list):
        raise ValueError("Confidence scores must be a list")
    
    for i, score in enumerate(scores):
        if not isinstance(score, (int, float)):
            raise ValueError(f"Confidence score {i} must be numeric")
        
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Confidence score {i} must be between 0.0 and 1.0, got {score}")
    
    return True
