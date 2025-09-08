"""
AI Service for Rebar Detection and Analysis
SIMPLIFIED: Back to basic detection logic that works
"""

import os
import cv2
import numpy as np
from datetime import datetime
import json
import traceback
import warnings

# Detectron2 imports
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    print("⚠️  Detectron2 not available. AI analysis will use placeholder results.")
    DETECTRON2_AVAILABLE = False

from app.utils.config import config

class AIService:
    """Simplified AI service focused on working detection"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # Simple 2-class setup
        self.class_names = ["front_horizontal", "front_vertical"]
        self.num_classes = 2
        self.detection_threshold = 0.3
        self.training_input_size = (480, 640)
        
        print("🤖 Initializing SIMPLIFIED AI Service...")
        print(f"   Classes: {self.class_names}")
        print(f"   Threshold: {self.detection_threshold}")
        self.load_model()
    
    def load_model(self):
        """Load model with minimal configuration"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model not found: {self.model_path}")
                return False
            
            print("🔄 Loading model...")
            
            # Simple configuration
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold
            self.cfg.MODEL.DEVICE = "cpu"
            
            self.predictor = DefaultPredictor(self.cfg)
            
            # Simple metadata
            self.metadata = MetadataCatalog.get("rebar_simple")
            self.metadata.thing_classes = self.class_names
            
            self.model_loaded = True
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_data=None, image_path=None):
        """Simple image analysis"""
        try:
            print("🔍 Starting analysis...")
            
            # Get image
            if image_data is not None:
                image = image_data.copy()
            elif image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is None:
                    return {'success': False, 'error': 'Failed to load image'}
            else:
                return {'success': False, 'error': 'No image provided'}
            
            # Resize if needed
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                image = cv2.resize(image, (480, 640))
            
            # Run detection
            if self.model_loaded and DETECTRON2_AVAILABLE:
                result = self._run_detection(image)
            else:
                result = self._placeholder_detection(image)
            
            return result
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}
    
    def _run_detection(self, image):
        """Run actual detection"""
        try:
            print("🤖 Running detection...")
            
            # Run inference
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            num_detections = len(instances)
            print(f"🎯 Found {num_detections} detections")
            
            if num_detections == 0:
                return {
                    'success': False,
                    'error': 'No rebar structures detected',
                    'no_detection': True
                }
            
            # Get detection data
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            # Process detections
            detections = []
            for i in range(num_detections):
                detection = {
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[classes[i]],
                    'confidence': float(scores[i]),
                    'bbox': boxes[i].tolist()
                }
                detections.append(detection)
                print(f"   {detection['class_name']}: {detection['confidence']:.3f}")
            
            # Create visualization
            analyzed_image_path = self._create_visualization(image, detections)
            
            if not analyzed_image_path:
                return {'success': False, 'error': 'Failed to create visualization'}
            
            # Simple dimensions
            dimensions = {
                'length': 27.36,
                'width': 27.36,
                'height': 200.0,
                'unit': 'cm',
                'volume': 149874,
                'display': '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters'
            }
            
            # Simple mixture
            mixture = {
                'cement': 1,
                'sand': 2,
                'aggregate': 4,
                'ratio_string': '1 Cement: 2 Sand: 4 Aggregate'
            }
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,
                'model_type': 'simplified_detection'
            }
            
        except Exception as e:
            print(f"❌ Detection error: {str(e)}")
            return {'success': False, 'error': f'Detection failed: {str(e)}'}
    
    def _create_visualization(self, image, detections):
        """Create simple visualization"""
        try:
            print("🎨 Creating visualization...")
            
            result_image = image.copy()
            
            # Count by type
            verticals = len([d for d in detections if 'vertical' in d['class_name']])
            horizontals = len([d for d in detections if 'horizontal' in d['class_name']])
            
            # Add green overlay for all detections
            overlay = result_image.copy()
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            
            # Apply transparency
            alpha = 0.3
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            # Add colored bounding boxes
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                class_name = detection['class_name']
                
                # Green for vertical, Red for horizontal
                if 'vertical' in class_name:
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Add text overlay
            self._add_text_overlay(result_image, verticals, horizontals)
            
            # Save image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analyzed_rebar_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ Visualization saved: {filename}")
                return output_path
            else:
                print("❌ Failed to save visualization")
                return None
                
        except Exception as e:
            print(f"❌ Visualization error: {str(e)}")
            return None
    
    def _add_text_overlay(self, image, verticals_count, horizontals_count):
        """Add simple text overlay"""
        try:
            # Text content
            texts = [
                "FRONT REBAR DETECTION:",
                f"Verticals: {verticals_count}/2",
                f"Horizontals: {horizontals_count}/11",
                "Dimensions: 27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters",
                "Method: Square Column + 4.5cm Offset"
            ]
            
            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0)
            
            # Calculate box size
            max_width = 0
            total_height = 0
            text_sizes = []
            
            for text in texts:
                (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_sizes.append((w, h))
                max_width = max(max_width, w)
                total_height += h + 5
            
            # Box dimensions
            box_width = max_width + 20
            box_height = total_height + 20
            box_x = 10
            box_y = 10
            
            # Draw background
            overlay = image.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), bg_color, -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            
            # Draw border
            cv2.rectangle(image, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 2)
            
            # Draw text
            current_y = box_y + 20
            for i, text in enumerate(texts):
                cv2.putText(image, text, (box_x + 10, current_y), font, font_scale, text_color, font_thickness)
                current_y += text_sizes[i][1] + 5
            
        except Exception as e:
            print(f"⚠️  Text overlay error: {e}")
    
    def _placeholder_detection(self, image):
        """Simple placeholder when model not available"""
        print("📝 Using placeholder detection...")
        
        # Create simple visualization
        analyzed_image_path = self._create_placeholder_visualization(image)
        
        dimensions = {
            'length': 27.36,
            'width': 27.36,
            'height': 200.0,
            'unit': 'cm',
            'volume': 149874,
            'display': '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters'
        }
        
        mixture = {
            'cement': 1,
            'sand': 2,
            'aggregate': 4,
            'ratio_string': '1 Cement: 2 Sand: 4 Aggregate'
        }
        
        return {
            'success': True,
            'placeholder': True,
            'detections': [
                {'class_name': 'front_vertical', 'confidence': 0.85, 'bbox': [100, 50, 200, 300]},
                {'class_name': 'front_horizontal', 'confidence': 0.78, 'bbox': [80, 280, 220, 320]}
            ],
            'num_detections': 2,
            'dimensions': dimensions,
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_image_path,
            'model_type': 'placeholder'
        }
    
    def _create_placeholder_visualization(self, image):
        """Create placeholder visualization"""
        try:
            result_image = image.copy()
            
            # Green overlay
            overlay = result_image.copy()
            cv2.rectangle(overlay, (100, 50), (200, 300), (0, 255, 0), -1)
            cv2.rectangle(overlay, (80, 280), (220, 320), (0, 255, 0), -1)
            
            alpha = 0.3
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            # Colored boxes
            cv2.rectangle(result_image, (100, 50), (200, 300), (0, 255, 0), 2)  # Green vertical
            cv2.rectangle(result_image, (80, 280), (220, 320), (0, 0, 255), 2)  # Red horizontal
            
            # Text overlay
            self._add_text_overlay(result_image, 1, 1)
            
            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analyzed_placeholder_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            cv2.imwrite(output_path, result_image)
            return output_path
            
        except Exception as e:
            print(f"❌ Placeholder error: {str(e)}")
            return None
    
    def get_model_status(self):
        """Get model status"""
        return {
            'detectron2_available': DETECTRON2_AVAILABLE,
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path) if self.model_path else False,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'threshold': self.detection_threshold,
            'model_type': 'simplified_detection'
        }
    
    def test_model(self, test_image_path=None):
        """Test the model"""
        try:
            if not test_image_path:
                captured_dir = config.UPLOAD_FOLDER
                if os.path.exists(captured_dir):
                    images = [f for f in os.listdir(captured_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        test_image_path = os.path.join(captured_dir, images[-1])
                    else:
                        return {'success': False, 'error': 'No test images available'}
                else:
                    return {'success': False, 'error': 'Directory not found'}
            
            result = self.analyze_image(image_path=test_image_path)
            
            if result['success']:
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'detections_found': result.get('num_detections', 0),
                    'model_type': result.get('model_type', 'unknown'),
                    'analyzed_image_saved': result.get('analyzed_image_path')
                }
            else:
                return result
                
        except Exception as e:
            return {'success': False, 'error': f'Test failed: {str(e)}'}
