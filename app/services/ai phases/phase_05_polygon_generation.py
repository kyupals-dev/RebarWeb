"""
Phase 5: Polygon Generation
- For each pair of adjacent rows: take bottom row and upper row
- Create quadrilaterals between corresponding points
- Each quadrilateral = 4 points: [bottom_left, bottom_right, upper_right, upper_left]
- Draw semi-transparent blue polygons on visualization
- Calculate pixel area of each polygon
- Find total bounding box around all polygons
"""

import numpy as np
import cv2
from .base_phase import BasePhase

class Phase05PolygonGeneration(BasePhase):
    """Phase 5: Generate polygons from intersection centroid rows"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Polygon Generation"
        self.min_polygon_area = 100  # minimum polygon area in pixels
        self.polygon_expansion = 5   # pixels to expand around centroids
    
    def validate_input(self, data):
        """Validate input data for Phase 5"""
        # Check intersection analysis passed
        if not data.get('intersection_analysis_passed', False):
            raise ValueError("Intersection analysis must pass before polygon generation")
        
        required_keys = ['centroid_rows', 'processed_image']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        centroid_rows = data['centroid_rows']
        if len(centroid_rows) < 2:
            raise ValueError(f"Need at least 2 centroid rows for polygon generation, got {len(centroid_rows)}")
        
        return True
    
    def execute(self, data):
        """Execute Phase 5: Polygon Generation"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        centroid_rows = data['centroid_rows']
        processed_image = data['processed_image']
        image_height, image_width = processed_image.shape[:2]
        
        self.log(f"Generating polygons from {len(centroid_rows)} centroid rows...")
        
        polygons = []
        polygon_areas = []
        all_polygon_points = []
        
        # Step 1: Generate quadrilaterals between adjacent rows
        for i in range(len(centroid_rows) - 1):
            bottom_row = centroid_rows[i]      # Lower Y-value (bottom)
            upper_row = centroid_rows[i + 1]   # Higher Y-value (top)
            
            self.log(f"Processing row pair {i+1}: {len(bottom_row['centroids'])} bottom, {len(upper_row['centroids'])} upper")
            
            # Create polygons between corresponding points
            row_polygons = self._create_row_polygons(
                bottom_row, 
                upper_row, 
                image_width, 
                image_height
            )
            
            polygons.extend(row_polygons)
            
            for poly in row_polygons:
                area = self._calculate_polygon_area(poly['points'])
                poly['area'] = area
                polygon_areas.append(area)
                all_polygon_points.extend(poly['points'])
                
                self.log(f"   Polygon area: {area:.1f} pixels")
        
        if len(polygons) == 0:
            self.log("❌ No polygons generated")
            return self._create_failure_output(
                data,
                "no_polygons_generated",
                "Failed to generate any polygons from centroid rows"
            )
        
        # Step 2: Filter out very small polygons
        valid_polygons = [p for p in polygons if p['area'] >= self.min_polygon_area]
        
        if len(valid_polygons) == 0:
            self.log(f"❌ No valid polygons after filtering (min area: {self.min_polygon_area})")
            return self._create_failure_output(
                data,
                "no_valid_polygons",
                f"No polygons meet minimum area requirement of {self.min_polygon_area} pixels"
            )
        
        self.log(f"Valid polygons: {len(valid_polygons)}/{len(polygons)} (after area filtering)")
        
        # Step 3: Calculate total bounding box
        if all_polygon_points:
            all_points = np.array(all_polygon_points)
            bounding_box = {
                'x_min': float(np.min(all_points[:, 0])),
                'y_min': float(np.min(all_points[:, 1])),
                'x_max': float(np.max(all_points[:, 0])),
                'y_max': float(np.max(all_points[:, 1]))
            }
            
            bounding_box['width'] = bounding_box['x_max'] - bounding_box['x_min']
            bounding_box['height'] = bounding_box['y_max'] - bounding_box['y_min']
            bounding_box['area'] = bounding_box['width'] * bounding_box['height']
            
        else:
            bounding_box = None
        
        # Step 4: Calculate total polygon area
        total_polygon_area = sum(p['area'] for p in valid_polygons)
        
        # Create output data
        output_data = data.copy()
        output_data.update({
            'polygon_generation_passed': True,
            'polygons': valid_polygons,
            'all_polygons': polygons,  # Including filtered ones for debugging
            'polygon_count': len(valid_polygons),
            'total_polygon_area': total_polygon_area,
            'polygon_areas': [p['area'] for p in valid_polygons],
            'bounding_box': bounding_box,
            'generation_parameters': {
                'min_polygon_area': self.min_polygon_area,
                'polygon_expansion': self.polygon_expansion
            }
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Valid polygons: {len(valid_polygons)}")
        self.log(f"   Total polygon area: {total_polygon_area:.1f} pixels")
        if bounding_box:
            self.log(f"   Bounding box: {bounding_box['width']:.1f} x {bounding_box['height']:.1f} pixels")
        
        return output_data
    
    def _create_row_polygons(self, bottom_row, upper_row, image_width, image_height):
        """Create polygons between two centroid rows"""
        polygons = []
        
        bottom_centroids = bottom_row['centroids']
        upper_centroids = upper_row['centroids']
        
        # Simple approach: create polygons between corresponding points
        min_count = min(len(bottom_centroids), len(upper_centroids))
        
        for i in range(min_count - 1):
            # Get four corner points for quadrilateral
            bottom_left = bottom_centroids[i]
            bottom_right = bottom_centroids[i + 1]
            upper_right = upper_centroids[i + 1] if i + 1 < len(upper_centroids) else upper_centroids[-1]
            upper_left = upper_centroids[i]
            
            # Expand points slightly to create visible polygons
            polygon_points = self._expand_polygon_points([
                bottom_left, bottom_right, upper_right, upper_left
            ], self.polygon_expansion, image_width, image_height)
            
            polygon = {
                'points': polygon_points,
                'bottom_row_index': bottom_row.get('index', 0),
                'upper_row_index': upper_row.get('index', 1),
                'segment_index': i
            }
            
            polygons.append(polygon)
        
        return polygons
    
    def _expand_polygon_points(self, points, expansion, image_width, image_height):
        """Expand polygon points by a small margin for better visibility"""
        expanded_points = []
        
        for point in points:
            x, y = point
            
            # Add small expansion (but keep within image bounds)
            x = max(0, min(image_width - 1, x))
            y = max(0, min(image_height - 1, y))
            
            expanded_points.append([float(x), float(y)])
        
        return expanded_points
    
    def _calculate_polygon_area(self, points):
        """Calculate area of polygon using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        # Convert to numpy array
        pts = np.array(points)
        
        # Shoelace formula
        x = pts[:, 0]
        y = pts[:, 1]
        
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(points)] - x[(i + 1) % len(points)] * y[i] 
                            for i in range(len(points))))
        
        return float(area)
    
    def _create_failure_output(self, data, error_code, error_message):
        """Create output data for polygon generation failure"""
        output_data = data.copy()
        output_data.update({
            'polygon_generation_passed': False,
            'polygon_error': error_code,
            'polygon_message': error_message,
            'should_stop_processing': True
        })
        
        self.log(f"❌ Polygon generation failed: {error_message}")
        return output_data
    
    def validate_output(self, data):
        """Validate output data from Phase 5"""
        required_keys = ['polygon_generation_passed']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # If generation passed, ensure we have polygon data
        if data['polygon_generation_passed']:
            required_success_keys = [
                'polygons',
                'polygon_count',
                'total_polygon_area',
                'bounding_box'
            ]
            
            for key in required_success_keys:
                if key not in data:
                    raise ValueError(f"Missing success output key: {key}")
            
            # Validate polygon structure
            polygons = data['polygons']
            if not isinstance(polygons, list):
                raise ValueError("polygons must be a list")
            
            for i, polygon in enumerate(polygons):
                if 'points' not in polygon or 'area' not in polygon:
                    raise ValueError(f"Polygon {i} missing required keys")
        
        return True
