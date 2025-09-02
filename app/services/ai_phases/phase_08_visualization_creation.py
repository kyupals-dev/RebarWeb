"""
Phase 8: Visualization Creation
Creates visual overlays and analysis images
"""

import cv2
import os
from datetime import datetime
from app.utils.config import config

class Phase08VisualizationCreation:
    """Phase 8: Create visualizations"""
    
    def __init__(self):
        print("🎨 Phase 8: Visualization Creation initialized")
    
    def create_visualization(self, image, detection_result, original_path):
        """Create visualization with overlays"""
        try:
            print("🎨 Phase 8: Creating visualization...")
            
            if image is None:
                return {
                    'success': False,
                    'error': 'No image provided for visualization'
                }
            
            # Create a copy for visualization
            vis_image = image.copy()
            
            # Get detections if available
            detections = []
            if isinstance(detection_result, dict):
                detections = detection_result.get('validated_detections', [])
            
            # Add overlays
            vis_image = self._add_detection_overlays(vis_image, detections)
            vis_image = self._add_analysis_annotations(vis_image)
            
            # Save visualization
            output_path = self._save_visualization(vis_image)
            
            print(f"   ✅ Phase 8: Visualization created: {os.path.basename(output_path)}")
            
            return {
                'success': True,
                'visualization_path': output_path,
                'detections_visualized': len(detections)
            }
            
        except Exception as e:
            print(f"   ❌ Phase 8 error: {str(e)}")
            return {
                'success': False,
                'error': f'Visualization creation failed: {str(e)}'
            }
    
    def _add_detection_overlays(self, image, detections):
        """Add detection overlays to image"""
        try:
            # Create overlay
            overlay = image.copy()
            
            for i, detection in enumerate(detections):
                bbox = detection.get('bbox', [100, 50, 200, 300])
                class_name = detection.get('class_name', 'rebar')
                confidence = detection.get('confidence', 0.0)
                
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Add colored rectangle overlay
                color = self._get_class_color(class_name)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Add bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # Add label
                label = f"{class_name} ({confidence:.0%})"
                cv2.putText(image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Apply transparent overlay
            if len(detections) > 0:
                alpha = 0.3
                image = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
            
            return image
            
        except Exception as e:
            print(f"   ⚠️ Overlay error: {e}")
            return image
    
    def _add_analysis_annotations(self, image):
        """Add analysis annotations"""
        try:
            # Add title
            cv2.putText(image, "Rebar Analysis Result", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(image, f"Analyzed: {timestamp}", (10, image.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return image
            
        except Exception as e:
            print(f"   ⚠️ Annotation error: {e}")
            return image
    
    def _get_class_color(self, class_name):
        """Get color for detection class"""
        colors = {
            'front_vertical': (0, 255, 0),      # Green
            'front_horizontal': (255, 0, 0),    # Red
            'back_horizontal': (0, 0, 255),     # Blue
        }
        return colors.get(class_name, (255, 255, 0))  # Yellow default
    
    def _save_visualization(self, image):
        """Save visualization image"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'phase_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, image)
            if success:
                return output_path
            else:
                raise Exception("Failed to save visualization")
                
        except Exception as e:
            print(f"   ⚠️ Save error: {e}")
            # Return a default path
            return os.path.join(config.UPLOAD_FOLDER, 'default_visualization.jpg')
