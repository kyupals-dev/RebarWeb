"""
Phase 8: Visualization Creation
- Start with original 480x640 image
- Draw Detectron2 predictions (colored masks): Green: front_vertical, Red: front_horizontal, Gray: back_horizontal
- Overlay semi-transparent blue polygons (40% opacity)
- Draw yellow bounding box around total rebar cage
- Add text annotations: Pixel dimensions, Real dimensions in cm, Analysis timestamp, Model type information
"""

import cv2
import numpy as np
from datetime import datetime
import os
from .base_phase import BasePhase

try:
    from detectron2.utils.visualizer import Visualizer, ColorMode
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

class Phase08VisualizationCreation(BasePhase):
    """Phase 8: Create enhanced visualization with analysis overlays"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Visualization Creation"
        
        # Visualization colors (BGR format for OpenCV)
        self.colors = {
            'front_vertical': (0, 255, 0),      # Green
            'front_horizontal': (0, 0, 255),    # Red  
            'back_horizontal': (128, 128, 128), # Gray
            'polygon': (255, 165, 0),           # Blue (for polygons)
            'bounding_box': (0, 255, 255),      # Yellow
            'text_bg': (0, 0, 0),               # Black background
            'text_fg': (255, 255, 255)          # White text
        }
        
        # Visualization parameters
        self.mask_alpha = 0.3  # 30% transparency for masks
        self.polygon_alpha = 0.4  # 40% transparency for polygons
        self.line_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
    
    def validate_input(self, data):
        """Validate input data for Phase 8"""
        required_keys = [
            'processed_image',
            'dimensions',
            'cement_mixture',
            'bounding_box'
        ]
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Check that we have either detectron2 data or structured detections
        if 'detectron2_instances' not in data and 'structured_detections' not in data:
            raise ValueError("Need either detectron2_instances or structured_detections for visualization")
        
        return True
    
    def execute(self, data):
        """Execute Phase 8: Visualization Creation"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        
        # Get base image
        processed_image = data['processed_image'].copy()  # BGR format
        image_height, image_width = processed_image.shape[:2]
        
        self.log(f"Creating visualization for {image_width}x{image_height} image")
        
        # Create visualization image
        viz_image = processed_image.copy()
        
        # Step 1: Draw Detectron2 predictions (colored masks)
        if DETECTRON2_AVAILABLE and data.get('detectron2_instances') is not None:
            viz_image = self._draw_detectron2_masks(viz_image, data)
        else:
            viz_image = self._draw_structured_detection_masks(viz_image, data)
        
        # Step 2: Draw semi-transparent blue polygons
        if data.get('polygons'):
            viz_image = self._draw_polygons(viz_image, data['polygons'])
        
        # Step 3: Draw yellow bounding box
        if data.get('bounding_box'):
            viz_image = self._draw_bounding_box(viz_image, data['bounding_box'])
        
        # Step 4: Add text annotations
        viz_image = self._add_text_annotations(viz_image, data)
        
        # Step 5: Save visualization
        output_path = self._save_visualization(viz_image, data)
        
        # Create output data
        output_data = data.copy()
        output_data.update({
            'visualization_creation_passed': True,
            'visualization_image': viz_image,
            'visualization_path': output_path,
            'visualization_metadata': {
                'image_size': f"{image_width}x{image_height}",
                'has_detectron2_overlay': DETECTRON2_AVAILABLE and data.get('detectron2_instances') is not None,
                'has_polygons': len(data.get('polygons', [])) > 0,
                'has_bounding_box': data.get('bounding_box') is not None,
                'colors_used': self.colors,
                'transparency_settings': {
                    'mask_alpha': self.mask_alpha,
                    'polygon_alpha': self.polygon_alpha
                }
            }
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Visualization saved: {os.path.basename(output_path)}")
        self.log(f"   Image size: {image_width}x{image_height}")
        
        return output_data
    
    def _draw_detectron2_masks(self, image, data):
        """Draw Detectron2 detection masks with proper colors"""
        try:
            instances = data['detectron2_instances']
            metadata = data.get('metadata')
            
            if instances is None or len(instances) == 0:
                return image
            
            # Create visualizer
            v = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE
            )
            
            # Draw predictions
            out = v.draw_instance_predictions(instances)
            result_image = out.get_image()[:, :, ::-1]  # Convert back to BGR
            
            # Apply transparency
            alpha = 1.0 - self.mask_alpha
            result_image = cv2.addWeighted(image, alpha, result_image, self.mask_alpha, 0)
            
            self.log("   Drew Detectron2 masks with transparency")
            return result_image
            
        except Exception as e:
            self.log(f"⚠️  Error drawing Detectron2 masks: {e}")
            return self._draw_structured_detection_masks(image, data)
    
    def _draw_structured_detection_masks(self, image, data):
        """Draw detection masks from structured detection data"""
        detections = data.get('structured_detections', [])
        
        if not detections:
            return image
        
        # Create overlay for transparency
        overlay = image.copy()
        
        for detection in detections:
            class_name = detection['class_name']
            mask = detection.get('mask')
            
            if mask is not None and class_name in self.colors:
                color = self.colors[class_name]
                
                # Apply colored mask
                overlay[mask] = color
        
        # Blend with original image for transparency
        alpha = self.mask_alpha
        result_image = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        self.log(f"   Drew {len(detections)} structured detection masks")
        return result_image
    
    def _draw_polygons(self, image, polygons):
        """Draw semi-transparent blue polygons"""
        overlay = image.copy()
        
        for polygon in polygons:
            points = polygon.get('points', [])
            if len(points) >= 3:
                # Convert points to integer array
                pts = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
                
                # Fill polygon
                cv2.fillPoly(overlay, [pts], self.colors['polygon'])
                
                # Draw outline
                cv2.polylines(image, [pts], True, self.colors['polygon'], self.line_thickness)
        
        # Blend for transparency
        alpha = self.polygon_alpha
        result_image = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        self.log(f"   Drew {len(polygons)} semi-transparent polygons")
        return result_image
    
    def _draw_bounding_box(self, image, bounding_box):
        """Draw yellow bounding box around total rebar cage"""
        x_min = int(bounding_box['x_min'])
        y_min = int(bounding_box['y_min'])
        x_max = int(bounding_box['x_max'])
        y_max = int(bounding_box['y_max'])
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 
                     self.colors['bounding_box'], self.line_thickness * 2)
        
        # Add corner markers for better visibility
        corner_size = 10
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        for corner in corners:
            cv2.circle(image, corner, corner_size, self.colors['bounding_box'], -1)
        
        self.log(f"   Drew bounding box: ({x_min},{y_min}) to ({x_max},{y_max})")
        return image
    
    def _add_text_annotations(self, image, data):
        """Add text annotations with analysis information"""
        annotations = []
        
        # Get data for annotations
        dimensions = data.get('dimensions', {})
        cement_mixture = data.get('cement_mixture', {})
        bounding_box = data.get('bounding_box', {})
        calculation_metadata = data.get('calculation_metadata', {})
        
        # Create annotation lines
        if dimensions:
            annotations.append(f"Dimensions: {dimensions.get('display', 'N/A')}")
            
        if bounding_box:
            pixel_dims = f"{bounding_box.get('width', 0):.0f}x{bounding_box.get('height', 0):.0f}px"
            annotations.append(f"Pixel Size: {pixel_dims}")
            
        if cement_mixture:
            annotations.append(f"Mix Ratio: {cement_mixture.get('ratio_string', 'N/A')}")
            
        if calculation_metadata:
            distance = calculation_metadata.get('distance_cm', 0)
            factor = calculation_metadata.get('pixel_to_cm_factor', 0)
            annotations.append(f"Distance: {distance:.0f}cm, Factor: {factor:.4f}")
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        annotations.append(f"Analysis: {timestamp}")
        
        # Add model type
        model_type = "Real Model" if data.get('model_loaded') else "Placeholder"
        annotations.append(f"Model: {model_type}")
        
        # Draw annotations
        y_start = 30
        line_height = 25
        
        for i, annotation in enumerate(annotations):
            y_pos = y_start + (i * line_height)
            
            # Draw text background
            text_size = cv2.getTextSize(annotation, self.font, self.font_scale, self.font_thickness)[0]
            cv2.rectangle(image, (10, y_pos - 20), (15 + text_size[0], y_pos + 5), 
                         self.colors['text_bg'], -1)
            
            # Draw text
            cv2.putText(image, annotation, (12, y_pos - 5), 
                       self.font, self.font_scale, self.colors['text_fg'], self.font_thickness)
        
        self.log(f"   Added {len(annotations)} text annotations")
        return image
    
    def _save_visualization(self, image, data):
        """Save visualization to file"""
        try:
            from app.utils.config import config
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            model_type = "real" if data.get('model_loaded') else "placeholder"
            filename = f'analysis_{model_type}_{timestamp}.jpg'
            
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save with high quality
            success = cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                file_size = os.path.getsize(output_path)
                self.log(f"   Saved visualization: {filename} ({file_size/1024:.1f} KB)")
                return output_path
            else:
                raise ValueError("Failed to save visualization image")
                
        except Exception as e:
            self.log(f"❌ Error saving visualization: {e}")
            # Return a fallback path
            return data.get('image_path', 'visualization_failed.jpg')
    
    def validate_output(self, data):
        """Validate output data from Phase 8"""
        required_keys = ['visualization_creation_passed', 'visualization_path']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # Check that visualization file exists
        viz_path = data['visualization_path']
        if not os.path.exists(viz_path):
            raise ValueError(f"Visualization file not found: {viz_path}")
        
        return True
