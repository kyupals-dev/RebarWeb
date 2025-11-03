"""
Constants shared across all AI processing phases
"""

# Phase names for logging and tracking
PHASE_NAMES = [
    "Image Preparation",
    "Detectron2 Inference", 
    "Detection Validation",
    "Intersection Analysis",
    "Polygon Generation",
    "Dimension Calculation",
    "Cement Mixture Calculation",
    "Visualization Creation",
    "Auto-Save Process",
    "Response Formatting"
]

# Detectron2 model classes
CLASS_NAMES = ["back_horizontal", "front_horizontal", "front_vertical"]
NUM_CLASSES = 3

# Default colors for visualizations (BGR format for OpenCV)
DEFAULT_COLORS = {
    'back_horizontal': (128, 128, 128),   # Gray
    'front_horizontal': (0, 0, 255),     # Red  
    'front_vertical': (0, 255, 0),       # Green
    'polygon': (255, 165, 0),            # Blue
    'bounding_box': (0, 255, 255),       # Yellow
    'text_bg': (0, 0, 0),                # Black background
    'text_fg': (255, 255, 255)           # White text
}

# Default thresholds and parameters
DEFAULT_THRESHOLDS = {
    'detection_threshold': 0.2,
    'min_intersection_area': 50,
    'min_component_size': 10,
    'y_grouping_threshold': 10,
    'min_polygon_area': 100,
    'polygon_expansion': 5
}

# Image processing constants
IMAGE_TARGET_SIZE = (480, 640)  # width x height for portrait
IMAGE_EXPECTED_LANDSCAPE = (640, 480)  # width x height for landscape

# Distance and calibration constants
CALIBRATION_POINTS = {
    160: 0.2117,  # 160cm distance = 0.2117 cm/pixel
    200: 0.2822   # 200cm distance = 0.2822 cm/pixel
}

DEFAULT_DISTANCE = 180.0  # cm
MIN_DISTANCE = 100.0      # cm
MAX_DISTANCE = 300.0      # cm

# Cement mixture constants
CEMENT_RATIO = 1
SAND_RATIO = 2
AGGREGATE_RATIO = 3
CEMENT_BAG_VOLUME = 0.035  # cubic meters per bag
CONCRETE_VOLUME_FACTOR = 1.5  # multiply rebar volume
WASTE_FACTOR = 1.10  # 10% waste allowance
MIXING_EFFICIENCY = 0.95  # 5% loss during mixing

# Dimension constraints (cm)
DIMENSION_CONSTRAINTS = {
    'length': {'min': 10.0, 'max': 100.0},
    'width': {'min': 10.0, 'max': 100.0},
    'height': {'min': 50.0, 'max': 400.0}
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'mask_alpha': 0.3,      # 30% transparency for masks
    'polygon_alpha': 0.4,   # 40% transparency for polygons
    'line_thickness': 2,
    'font_scale': 0.6,
    'font_thickness': 2
}

# File extensions
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
REQUIRED_MODEL_EXTENSIONS = {'.pth', '.pkl'}

# Error messages
ERROR_MESSAGES = {
    'no_detections': 'No rebar structures detected in the image',
    'missing_classes': 'Missing required rebar classes: {}',
    'insufficient_intersections': 'Intersection area too small: {} pixels',
    'no_polygons': 'Failed to generate any polygons from centroid rows',
    'dimension_calculation_failed': 'Failed to calculate realistic dimensions',
    'cement_calculation_failed': 'Failed to calculate cement mixture',
    'visualization_failed': 'Failed to create analysis visualization',
    'save_failed': 'Failed to save analysis results'
}
