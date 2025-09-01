"""
Phase 4: Intersection Analysis (Core Algorithm)
- Extract masks for front_horizontal and front_vertical
- Combine multiple instances per class into single mask
- Create intersection mask = front_horizontal AND front_vertical
- Find connected components in intersection mask
- Calculate centroid (center point) of each component
- Sort centroids by Y-coordinate (bottom → top)
- Group centroids into horizontal rows (Y-threshold = 10px)
- Sort each row by X-coordinate (right → left)
"""

import numpy as np
import cv2
from scipy import ndimage
from .base_phase import BasePhase

class Phase04IntersectionAnalysis(BasePhase):
    """Phase 4: Analyze intersections between horizontal and vertical rebar"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Intersection Analysis"
        self.y_grouping_threshold = 10  # pixels for grouping centroids into rows
        self.min_intersection_area = 50  # minimum pixels for valid intersection
        self.min_component_size = 10  # minimum connected component size
    
    def validate_input(self, data):
        """Validate input data for Phase 4"""
        # Check validation passed
        if not data.get('validation_passed', False):
            raise ValueError("Detection validation must pass before intersection analysis")
        
        required_keys = [
            'front_horizontal_detections', 
            'front_vertical_detections',
            'processed_image'
        ]
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Ensure we have detections
        if len(data['front_horizontal_detections']) == 0:
            raise ValueError("No front_horizontal detections available")
        
        if len(data['front_vertical_detections']) == 0:
            raise ValueError("No front_vertical detections available")
        
        return True
    
    def execute(self, data):
        """Execute Phase 4: Intersection Analysis"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        front_horizontal_detections = data['front_horizontal_detections']
        front_vertical_detections = data['front_vertical_detections']
        processed_image = data['processed_image']
        
        image_height, image_width = processed_image.shape[:2]
        
        # Step 1: Extract and combine masks for each class
        self.log("Step 1: Combining masks by class...")
        
        front_horizontal_mask = self._combine_masks(
            front_horizontal_detections, 
            (image_height, image_width),
            "front_horizontal"
        )
        
        front_vertical_mask = self._combine_masks(
            front_vertical_detections,
            (image_height, image_width), 
            "front_vertical"
        )
        
        # Step 2: Create intersection mask
        self.log("Step 2: Creating intersection mask...")
        intersection_mask = np.logical_and(front_horizontal_mask, front_vertical_mask)
        intersection_area = np.sum(intersection_mask)
        
        self.log(f"   Intersection area: {intersection_area} pixels")
        
        if intersection_area < self.min_intersection_area:
            self.log(f"❌ Insufficient intersection area: {intersection_area} < {self.min_intersection_area}")
            return self._create_failure_output(
                data, 
                "insufficient_intersections",
                f"Intersection area too small: {intersection_area} pixels"
            )
        
        # Step 3: Find connected components in intersection mask
        self.log("Step 3: Finding connected components...")
        
        labeled_components, num_components = ndimage.label(intersection_mask)
        
        if num_components == 0:
            self.log("❌ No connected components found in intersections")
            return self._create_failure_output(
                data,
                "no_intersection_components", 
                "No connected intersection components found"
            )
        
        self.log(f"   Found {num_components} connected components")
        
        # Step 4: Calculate centroids of each component
        self.log("Step 4: Calculating component centroids...")
        
        centroids = []
        component_info = []
        
        for component_id in range(1, num_components + 1):
            component_mask = labeled_components == component_id
            component_size = np.sum(component_mask)
            
            # Filter out tiny components
            if component_size < self.min_component_size:
                continue
            
            # Calculate centroid
            y_coords, x_coords = np.where(component_mask)
            centroid_x = float(np.mean(x_coords))
            centroid_y = float(np.mean(y_coords))
            
            centroids.append([centroid_x, centroid_y])
            component_info.append({
                'id': component_id,
                'centroid': [centroid_x, centroid_y],
                'size': int(component_size),
                'mask': component_mask
            })
            
            self.log(f"   Component {component_id}: centroid=({centroid_x:.1f}, {centroid_y:.1f}), size={component_size}")
        
        if len(centroids) == 0:
            self.log("❌ No valid centroids after filtering")
            return self._create_failure_output(
                data,
                "no_valid_centroids",
                "No valid intersection centroids found"
            )
        
        centroids = np.array(centroids)
        
        # Step 5: Sort centroids by Y-coordinate (bottom → top)
        self.log("Step 5: Sorting centroids by Y-coordinate...")
        
        # Sort by Y (bottom to top = high Y to low Y)
        y_sorted_indices = np.argsort(centroids[:, 1])[::-1]  # Reverse for bottom-to-top
        sorted_centroids = centroids[y_sorted_indices]
        sorted_components = [component_info[i] for i in y_sorted_indices]
        
        # Step 6: Group centroids into horizontal rows
        self.log("Step 6: Grouping centroids into horizontal rows...")
        
        centroid_rows = self._group_centroids_into_rows(
            sorted_centroids, 
            sorted_components,
            self.y_grouping_threshold
        )
        
        self.log(f"   Grouped into {len(centroid_rows)} horizontal rows")
        
        # Step 7: Sort each row by X-coordinate (right → left)
        self.log("Step 7: Sorting rows by X-coordinate...")
        
        for row_idx, row in enumerate(centroid_rows):
            # Sort by X-coordinate (right to left = high X to low X)
            row['centroids'] = sorted(row['centroids'], key=lambda c: c[0], reverse=True)
            row['components'] = sorted(row['components'], key=lambda c: c['centroid'][0], reverse=True)
            
            x_coords = [c[0] for c in row['centroids']]
            y_coord = row['y_center']
            self.log(f"   Row {row_idx + 1}: Y={y_coord:.1f}, X-coords={[f'{x:.1f}' for x in x_coords]}")
        
        # Create output data
        output_data = data.copy()
        output_data.update({
            'intersection_analysis_passed': True,
            'front_horizontal_mask': front_horizontal_mask,
            'front_vertical_mask': front_vertical_mask,
            'intersection_mask': intersection_mask,
            'intersection_area': int(intersection_area),
            'labeled_components': labeled_components,
            'num_components': num_components,
            'valid_components': component_info,
            'centroids': centroids.tolist(),
            'centroid_rows': centroid_rows,
            'analysis_parameters': {
                'y_grouping_threshold': self.y_grouping_threshold,
                'min_intersection_area': self.min_intersection_area,
                'min_component_size': self.min_component_size
            }
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Components: {len(component_info)}")
        self.log(f"   Rows: {len(centroid_rows)}")
        self.log(f"   Total intersection area: {intersection_area} pixels")
        
        return output_data
    
    def _combine_masks(self, detections, image_shape, class_name):
        """Combine multiple detection masks into single mask"""
        combined_mask = np.zeros(image_shape, dtype=bool)
        
        for detection in detections:
            if 'mask' in detection:
                mask = detection['mask']
                if mask.shape == image_shape:
                    combined_mask = np.logical_or(combined_mask, mask)
        
        total_area = np.sum(combined_mask)
        self.log(f"   {class_name}: {len(detections)} detections → {total_area} pixels")
        
        return combined_mask
    
    def _group_centroids_into_rows(self, centroids, components, y_threshold):
        """Group centroids into horizontal rows based on Y-coordinate similarity"""
        if len(centroids) == 0:
            return []
        
        rows = []
        used = set()
        
        for i, centroid in enumerate(centroids):
            if i in used:
                continue
            
            # Start new row with this centroid
            current_row = {
                'centroids': [centroid],
                'components': [components[i]],
                'y_center': centroid[1],
                'y_min': centroid[1],
                'y_max': centroid[1]
            }
            used.add(i)
            
            # Find other centroids within Y-threshold
            for j, other_centroid in enumerate(centroids):
                if j in used or j <= i:
                    continue
                
                y_diff = abs(centroid[1] - other_centroid[1])
                if y_diff <= y_threshold:
                    current_row['centroids'].append(other_centroid)
                    current_row['components'].append(components[j])
                    current_row['y_min'] = min(current_row['y_min'], other_centroid[1])
                    current_row['y_max'] = max(current_row['y_max'], other_centroid[1])
                    used.add(j)
            
            # Update row center
            y_values = [c[1] for c in current_row['centroids']]
            current_row['y_center'] = float(np.mean(y_values))
            current_row['count'] = len(current_row['centroids'])
            
            rows.append(current_row)
        
        # Sort rows by Y-coordinate (bottom to top)
        rows.sort(key=lambda r: r['y_center'], reverse=True)
        
        return rows
    
    def _create_failure_output(self, data, error_code, error_message):
        """Create output data for intersection analysis failure"""
        output_data = data.copy()
        output_data.update({
            'intersection_analysis_passed': False,
            'intersection_error': error_code,
            'intersection_message': error_message,
            'should_stop_processing': True
        })
        
        self.log(f"❌ Intersection analysis failed: {error_message}")
        return output_data
    
    def validate_output(self, data):
        """Validate output data from Phase 4"""
        required_keys = ['intersection_analysis_passed']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # If analysis passed, ensure we have required intersection data
        if data['intersection_analysis_passed']:
            required_success_keys = [
                'intersection_mask',
                'centroid_rows', 
                'valid_components',
                'intersection_area'
            ]
            
            for key in required_success_keys:
                if key not in data:
                    raise ValueError(f"Missing success output key: {key}")
        
        return True
