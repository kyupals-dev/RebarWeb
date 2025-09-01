"""
Phase 6: Dimension Calculation
- Measure total bounding box dimensions in pixels (total width, total height)
- Get distance from sensor or filename (default 180cm)
- Calculate pixel-to-cm conversion factor: 
  * 160cm distance = 0.2117 cm/pixel
  * 200cm distance = 0.2822 cm/pixel  
  * Linear interpolation for distances between
- Convert to real dimensions: Length = Width × factor, Width = Width × factor, Height = Height × factor
- Calculate volume = Length × Width × Height
"""

import re
import os
from .base_phase import BasePhase

class Phase06DimensionCalculation(BasePhase):
    """Phase 6: Calculate real-world dimensions from pixel measurements"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Dimension Calculation"
        
        # Calibration data points
        self.calibration_points = {
            160: 0.2117,  # 160cm distance = 0.2117 cm/pixel
            200: 0.2822   # 200cm distance = 0.2822 cm/pixel
        }
        
        self.default_distance = 180.0  # cm
        self.min_distance = 100.0      # cm
        self.max_distance = 300.0      # cm
    
    def validate_input(self, data):
        """Validate input data for Phase 6"""
        # Check polygon generation passed
        if not data.get('polygon_generation_passed', False):
            raise ValueError("Polygon generation must pass before dimension calculation")
        
        required_keys = ['bounding_box', 'processed_image']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate bounding box
        bounding_box = data['bounding_box']
        if not bounding_box or 'width' not in bounding_box or 'height' not in bounding_box:
            raise ValueError("Invalid bounding box data")
        
        return True
    
    def execute(self, data):
        """Execute Phase 6: Dimension Calculation"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        bounding_box = data['bounding_box']
        processed_image = data['processed_image']
        
        # Step 1: Get pixel dimensions from bounding box
        pixel_width = bounding_box['width']
        pixel_height = bounding_box['height']
        
        self.log(f"Pixel dimensions: {pixel_width:.1f} x {pixel_height:.1f} pixels")
        
        # Step 2: Determine distance (from filename, sensor, or default)
        distance_cm = self._determine_distance(data)
        self.log(f"Using distance: {distance_cm:.1f} cm")
        
        # Step 3: Calculate pixel-to-cm conversion factor
        pixel_to_cm_factor = self._calculate_conversion_factor(distance_cm)
        self.log(f"Pixel-to-cm factor: {pixel_to_cm_factor:.4f} cm/pixel")
        
        # Step 4: Convert pixel dimensions to real dimensions
        length_cm = pixel_width * pixel_to_cm_factor
        width_cm = pixel_width * pixel_to_cm_factor   # Square column assumption
        height_cm = pixel_height * pixel_to_cm_factor
        
        # Step 5: Apply realistic constraints
        length_cm = self._apply_dimension_constraints(length_cm, "length")
        width_cm = self._apply_dimension_constraints(width_cm, "width") 
        height_cm = self._apply_dimension_constraints(height_cm, "height")
        
        # Step 6: Calculate volume
        volume_cm3 = length_cm * width_cm * height_cm
        volume_m3 = volume_cm3 / 1_000_000  # Convert to cubic meters
        
        # Step 7: Create dimension summary
        dimensions = {
            'length': round(length_cm, 1),
            'width': round(width_cm, 1),
            'height': round(height_cm, 1),
            'unit': 'cm',
            'volume_cm3': round(volume_cm3, 1),
            'volume_m3': round(volume_m3, 6),
            'display': f"{length_cm:.0f}cm x {width_cm:.0f}cm x {height_cm:.0f}cm = {volume_cm3:.0f}cm³"
        }
        
        # Step 8: Add calculation metadata
        calculation_metadata = {
            'pixel_dimensions': {
                'width': pixel_width,
                'height': pixel_height
            },
            'distance_cm': distance_cm,
            'pixel_to_cm_factor': pixel_to_cm_factor,
            'calibration_method': self._get_calibration_method(distance_cm),
            'constraints_applied': {
                'length': length_cm != (pixel_width * pixel_to_cm_factor),
                'width': width_cm != (pixel_width * pixel_to_cm_factor),
                'height': height_cm != (pixel_height * pixel_to_cm_factor)
            }
        }
        
        # Create output data
        output_data = data.copy()
        output_data.update({
            'dimension_calculation_passed': True,
            'dimensions': dimensions,
            'calculation_metadata': calculation_metadata
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Real dimensions: {dimensions['display']}")
        self.log(f"   Volume: {volume_cm3:.1f} cm³ ({volume_m3:.6f} m³)")
        self.log(f"   Conversion factor: {pixel_to_cm_factor:.4f} cm/pixel")
        
        return output_data
    
    def _determine_distance(self, data):
        """Determine distance from various sources"""
        distance = None
        
        # Method 1: Check if distance service reading is available
        if hasattr(data, 'distance_reading') and data['distance_reading']:
            reading = data['distance_reading']
            if reading.get('success') and reading.get('distance'):
                distance = reading['distance']
                self.log(f"Distance from sensor: {distance:.1f} cm")
        
        # Method 2: Extract from filename if available
        if distance is None and 'image_path' in data:
            filename = os.path.basename(data['image_path'])
            distance = self._extract_distance_from_filename(filename)
            if distance:
                self.log(f"Distance from filename: {distance:.1f} cm")
        
        # Method 3: Use default
        if distance is None:
            distance = self.default_distance
            self.log(f"Using default distance: {distance:.1f} cm")
        
        # Validate distance range
        if distance < self.min_distance:
            self.log(f"⚠️ Distance {distance:.1f} cm too small, using minimum {self.min_distance:.1f} cm")
            distance = self.min_distance
        elif distance > self.max_distance:
            self.log(f"⚠️ Distance {distance:.1f} cm too large, using maximum {self.max_distance:.1f} cm")
            distance = self.max_distance
        
        return distance
    
    def _extract_distance_from_filename(self, filename):
        """Extract distance from filename patterns like 'image_180cm_optimal.jpg'"""
        try:
            # Look for patterns like "180cm", "180.5cm", etc.
            pattern = r'(\d+(?:\.\d+)?)cm'
            match = re.search(pattern, filename)
            
            if match:
                distance = float(match.group(1))
                return distance
            
            # Alternative patterns: "180_cm", "dist180", etc.
            alt_patterns = [
                r'(\d+(?:\.\d+)?)_cm',
                r'dist(\d+(?:\.\d+)?)',
                r'distance_?(\d+(?:\.\d+)?)'
            ]
            
            for pattern in alt_patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    distance = float(match.group(1))
                    return distance
            
            return None
            
        except (ValueError, AttributeError):
            return None
    
    def _calculate_conversion_factor(self, distance_cm):
        """Calculate pixel-to-cm conversion factor using linear interpolation"""
        calibration_distances = list(self.calibration_points.keys())
        calibration_factors = list(self.calibration_points.values())
        
        # If exact match
        if distance_cm in self.calibration_points:
            return self.calibration_points[distance_cm]
        
        # Linear interpolation between calibration points
        if distance_cm <= min(calibration_distances):
            # Extrapolate below minimum
            return calibration_factors[0]
        elif distance_cm >= max(calibration_distances):
            # Extrapolate above maximum
            return calibration_factors[-1]
        else:
            # Interpolate between points
            d1, d2 = min(calibration_distances), max(calibration_distances)
            f1, f2 = self.calibration_points[d1], self.calibration_points[d2]
            
            # Linear interpolation formula
            factor = f1 + (f2 - f1) * (distance_cm - d1) / (d2 - d1)
            return factor
    
    def _get_calibration_method(self, distance_cm):
        """Get description of calibration method used"""
        if distance_cm in self.calibration_points:
            return f"exact_calibration_{int(distance_cm)}cm"
        elif distance_cm <= min(self.calibration_points.keys()):
            return f"extrapolated_below_{min(self.calibration_points.keys())}cm"
        elif distance_cm >= max(self.calibration_points.keys()):
            return f"extrapolated_above_{max(self.calibration_points.keys())}cm"
        else:
            return f"interpolated_between_{min(self.calibration_points.keys())}-{max(self.calibration_points.keys())}cm"
    
    def _apply_dimension_constraints(self, dimension, dimension_type):
        """Apply realistic constraints to calculated dimensions"""
        # Define reasonable ranges for residential construction
        constraints = {
            'length': {'min': 10.0, 'max': 100.0},    # 10cm to 100cm
            'width': {'min': 10.0, 'max': 100.0},     # 10cm to 100cm  
            'height': {'min': 50.0, 'max': 400.0}     # 50cm to 400cm
        }
        
        if dimension_type not in constraints:
            return dimension
        
        min_val = constraints[dimension_type]['min']
        max_val = constraints[dimension_type]['max']
        
        original_dimension = dimension
        
        if dimension < min_val:
            self.log(f"⚠️ {dimension_type} {dimension:.1f}cm too small, using minimum {min_val:.1f}cm")
            dimension = min_val
        elif dimension > max_val:
            self.log(f"⚠️ {dimension_type} {dimension:.1f}cm too large, using maximum {max_val:.1f}cm")
            dimension = max_val
        
        return dimension
    
    def validate_output(self, data):
        """Validate output data from Phase 6"""
        required_keys = ['dimension_calculation_passed', 'dimensions']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # Validate dimensions structure
        dimensions = data['dimensions']
        required_dimension_keys = ['length', 'width', 'height', 'unit', 'volume_cm3', 'display']
        
        for key in required_dimension_keys:
            if key not in dimensions:
                raise ValueError(f"Missing dimension key: {key}")
        
        # Validate dimension values are positive
        for key in ['length', 'width', 'height', 'volume_cm3']:
            if dimensions[key] <= 0:
                raise ValueError(f"Dimension {key} must be positive, got {dimensions[key]}")
        
        return True
