"""
AI Service for Rebar Detection and Analysis
Integrates Detectron2 Mask R-CNN model for rebar segmentation with REAL MODEL
"""

import os
import cv2
import numpy as np
from datetime import datetime
import json
import traceback

# Detectron2 imports (will be installed later)
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
    """Handles AI model loading, inference, and rebar analysis with REAL TRAINED MODEL"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # Updated rebar classes based on your training
        self.class_names = ["front_vertical", "front_horizontal", "back_horizontal"]
        self.num_classes = 3
        
        # Updated detection threshold based on your training
        self.detection_threshold = 0.3
        
        # Training image size (480x640 portrait)
        self.training_input_size = (480, 640)  # width x height
        
        print("🤖 Initializing AI Service with REAL TRAINED MODEL...")
        print(f"   Classes: {self.class_names}")
        print(f"   Detection threshold: {self.detection_threshold}")
        print(f"   Training input size: {self.training_input_size[0]}x{self.training_input_size[1]}")
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model with REAL CONFIGURATION"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("   Please ensure model_final.pth is in the correct location")
                return False
            
            print("🔄 Loading Detectron2 configuration for REAL MODEL...")
            
            # Set up configuration matching your training
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings - REAL CONFIGURATION
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # 3 classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold  # 0.3 threshold
            self.cfg.MODEL.DEVICE = "cpu"  # Use CPU on Raspberry Pi
            
            # Input format matching your training (480x640)
            self.cfg.INPUT.MIN_SIZE_TRAIN = (640,)  # Height during training
            self.cfg.INPUT.MAX_SIZE_TRAIN = 640
            self.cfg.INPUT.MIN_SIZE_TEST = 640
            self.cfg.INPUT.MAX_SIZE_TEST = 640
            
            print("🔄 Creating predictor with REAL MODEL...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # Set up metadata for visualization with your classes
            self.metadata = MetadataCatalog.get("rebar_dataset_real")
            self.metadata.thing_classes = self.class_names
            
            # Set colors for each class (you can customize these)
            self.metadata.thing_colors = [
                (0, 255, 0),      # front_vertical - Green
                (255, 0, 0),      # front_horizontal - Red  
                (0, 0, 255),      # back_horizontal - Blue
            ]
            
            self.model_loaded = True
            print("✅ REAL AI Model loaded successfully!")
            print(f"   Model path: {self.model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Detection threshold: {self.detection_threshold}")
            print(f"   Input size: {self.training_input_size[0]}x{self.training_input_size[1]}")
            
            # Test the model with a quick inference
            test_image = np.zeros((640, 480, 3), dtype=np.uint8)  # Create test image
            try:
                test_output = self.predictor(test_image)
                print("✅ Model inference test successful!")
            except Exception as e:
                print(f"⚠️  Model inference test failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading REAL AI model: {str(e)}")
            print("   Full traceback:")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_path):
        """
        Analyze image for rebar detection using REAL TRAINED MODEL
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Analysis results with detection data
        """
        try:
            print(f"🔍 Starting REAL AI analysis of: {image_path}")
            
            # Check if image exists
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}'
                }
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image file'
                }
            
            print(f"📐 Image loaded: {image.shape} (H×W×C)")
            
            # Ensure image is the right size (480x640)
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                print(f"⚙️  Resizing image from {width}x{height} to 480x640 for model input")
                image = cv2.resize(image, (480, 640))
            
            # Use real model or placeholder
            if self.model_loaded and DETECTRON2_AVAILABLE:
                return self._analyze_with_real_model(image, image_path)
            else:
                print("⚠️  REAL MODEL not available, using placeholder")
                return self._analyze_placeholder(image, image_path)
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _analyze_with_real_model(self, image, image_path):
        """Run actual AI model analysis with REAL TRAINED MODEL"""
        try:
            print("🤖 Running REAL Detectron2 inference...")
            
            # Run inference with your trained model
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # Check if any detections
            num_detections = len(instances)
            print(f"🎯 REAL MODEL found {num_detections} detections")
            
            if num_detections == 0:
                print("❌ No rebar structures detected by REAL MODEL")
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True
                }
            
            # Extract detection data from REAL MODEL
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            # Process detections from REAL MODEL
            detections = []
            for i in range(num_detections):
                detection = {
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[classes[i]],
                    'confidence': float(scores[i]),
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'mask_area': float(np.sum(masks[i])),
                    'mask_shape': masks[i].shape
                }
                detections.append(detection)
                
                print(f"   Detection {i+1}: {detection['class_name']} ({detection['confidence']:.3f}) - Area: {detection['mask_area']:.0f}px")
            
            # Create visualization with REAL MODEL results
            analyzed_image_path = self._create_real_model_visualization(image, outputs, image_path)
            
            # Calculate dimensions from REAL MODEL detections
            dimensions = self._calculate_real_dimensions(detections, masks, image.shape)
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture(dimensions)
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,
                'original_image_path': image_path,
                'model_type': 'real_trained_model'
            }
            
        except Exception as e:
            print(f"❌ REAL MODEL inference error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'REAL MODEL inference failed: {str(e)}'
            }
    
    def _create_real_model_visualization(self, image, outputs, original_path):
        """Create visualization with REAL MODEL overlays using transparent green masks"""
        try:
            print("🎨 Creating REAL MODEL analysis visualization...")
            
            # Create visualizer with transparent green overlay
            v = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=self.metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE  # Show image with overlays
            )
            
            # Draw predictions with transparent masks
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = out.get_image()[:, :, ::-1]  # Convert back to BGR
            
            # Add transparent green overlay for better visibility
            instances = outputs["instances"].to("cpu")
            if len(instances) > 0:
                masks = instances.pred_masks.numpy()
                classes = instances.pred_classes.numpy()
                
                # Create overlay image
                overlay = image.copy()
                
                for i, (mask, class_id) in enumerate(zip(masks, classes)):
                    # Create colored mask - transparent green
                    colored_mask = np.zeros_like(image)
                    colored_mask[mask] = [0, 255, 0]  # Green color
                    
                    # Apply transparent overlay (30% opacity)
                    alpha = 0.3
                    result_image = cv2.addWeighted(result_image, 1, colored_mask, alpha, 0)
                
                # Add dimension annotations
                self._add_dimension_annotations(result_image, instances)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'real_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ REAL MODEL visualization saved: {filename}")
                return output_path
            else:
                print("❌ Failed to save REAL MODEL visualization")
                return original_path
                
        except Exception as e:
            print(f"❌ REAL MODEL visualization error: {str(e)}")
            traceback.print_exc()
            return original_path
    
    def _add_dimension_annotations(self, image, instances):
        """Add dimension text annotations to the visualization"""
        try:
            boxes = instances.pred_boxes.tensor.numpy()
            classes = instances.pred_classes.numpy()
            
            for i, (box, class_id) in enumerate(zip(boxes, classes)):
                x1, y1, x2, y2 = box
                class_name = self.class_names[class_id]
                
                # Calculate box dimensions in pixels
                width_px = x2 - x1
                height_px = y2 - y1
                
                # Add text annotation (you can improve this calculation)
                text = f"{class_name}: {width_px:.0f}x{height_px:.0f}px"
                
                # Position text above the bounding box
                text_pos = (int(x1), int(y1 - 10))
                
                # Add text with background
                cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
        except Exception as e:
            print(f"⚠️  Error adding dimension annotations: {e}")
    
    def _calculate_real_dimensions(self, detections, masks, image_shape):
        """Calculate rebar dimensions from REAL MODEL detections (0cm x 0cm x 0cm format)"""
        try:
            print("📏 Calculating dimensions from REAL MODEL detections...")
            
            if not detections:
                return {
                    'length': 0,
                    'width': 0,
                    'height': 0,
                    'unit': 'cm',
                    'volume': 0,
                    'display': '0cm x 0cm x 0cm = 0cm³',
                    'method': 'real_model_analysis'
                }
            
            # Analyze detections by class
            front_vertical = [d for d in detections if d['class_name'] == 'front_vertical']
            front_horizontal = [d for d in detections if d['class_name'] == 'front_horizontal'] 
            back_horizontal = [d for d in detections if d['class_name'] == 'back_horizontal']
            
            print(f"   Found: {len(front_vertical)} front_vertical, {len(front_horizontal)} front_horizontal, {len(back_horizontal)} back_horizontal")
            
            # Calculate dimensions based on detected rebar structures
            # This is a simplified calculation - you may need to adjust based on your specific requirements
            
            height, width, channels = image_shape
            
            # Pixel to cm conversion factor (you'll need to calibrate this based on your setup)
            # This assumes the camera is at optimal distance (160-200cm)
            pixel_to_cm = 0.1  # Rough estimate - needs calibration
            
            # Calculate length (typically from vertical rebars)
            length_cm = 0
            if front_vertical:
                max_vertical = max(front_vertical, key=lambda x: x['bbox'][3] - x['bbox'][1])
                length_px = max_vertical['bbox'][3] - max_vertical['bbox'][1]  # y2 - y1
                length_cm = length_px * pixel_to_cm
            
            # Calculate width (typically from horizontal rebars)
            width_cm = 0
            if front_horizontal:
                max_horizontal = max(front_horizontal, key=lambda x: x['bbox'][2] - x['bbox'][0])
                width_px = max_horizontal['bbox'][2] - max_horizontal['bbox'][0]  # x2 - x1
                width_cm = width_px * pixel_to_cm
            
            # Calculate height (depth estimation from front/back comparison)
            height_cm = 0
            if front_horizontal and back_horizontal:
                # Estimate depth based on difference between front and back horizontal elements
                front_area = sum(d['mask_area'] for d in front_horizontal)
                back_area = sum(d['mask_area'] for d in back_horizontal)
                depth_factor = abs(front_area - back_area) / max(front_area, back_area, 1)
                height_cm = depth_factor * 30  # Rough estimation
            else:
                # Default height if can't estimate depth
                height_cm = 25  # Standard rebar spacing
            
            # Ensure minimum realistic values
            length_cm = max(length_cm, 10)
            width_cm = max(width_cm, 10)
            height_cm = max(height_cm, 10)
            
            # Calculate volume
            volume_cm3 = length_cm * width_cm * height_cm
            
            # Create display string in requested format
            display_string = f"{length_cm:.0f}cm x {width_cm:.0f}cm x {height_cm:.0f}cm = {volume_cm3:.0f}cm³"
            
            print(f"   Calculated dimensions: {display_string}")
            
            return {
                'length': round(length_cm, 1),
                'width': round(width_cm, 1), 
                'height': round(height_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': display_string,
                'method': 'real_model_mask_analysis',
                'detection_details': {
                    'front_vertical_count': len(front_vertical),
                    'front_horizontal_count': len(front_horizontal),
                    'back_horizontal_count': len(back_horizontal),
                    'pixel_to_cm_factor': pixel_to_cm
                }
            }
            
        except Exception as e:
            print(f"❌ Error calculating REAL MODEL dimensions: {str(e)}")
            # Return safe default
            return {
                'length': 25,
                'width': 25,
                'height': 200,
                'unit': 'cm',
                'volume': 125000,
                'display': '25cm x 25cm x 200cm = 125000cm³',
                'method': 'fallback_calculation'
            }
    
    def _analyze_placeholder(self, image, image_path):
        """Generate placeholder analysis results (fallback only)"""
        print("📝 Using placeholder AI analysis (REAL MODEL not available)...")
        
        # Simulate some processing time
        import time
        time.sleep(2)
        
        # Create simple placeholder visualization
        analyzed_image_path = self._create_placeholder_visualization(image, image_path)
        
        # Placeholder dimensions in requested format
        dimensions = {
            'length': 25.4,
            'width': 25.4,
            'height': 200.0,
            'unit': 'cm',
            'volume': 101600,
            'display': '25cm x 25cm x 200cm = 101600cm³',
            'method': 'placeholder_fallback'
        }
        
        mixture = {
            'cement': 1,
            'sand': 2,
            'aggregate': 3,
            'ratio_string': '1 Cement : 2 Sand : 3 Aggregate'
        }
        
        return {
            'success': True,
            'placeholder': True,
            'detections': [
                {
                    'class_name': 'front_vertical',
                    'confidence': 0.85,
                    'bbox': [100, 50, 200, 300]
                },
                {
                    'class_name': 'front_horizontal', 
                    'confidence': 0.78,
                    'bbox': [80, 280, 220, 320]
                }
            ],
            'num_detections': 2,
            'dimensions': dimensions,
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_image_path,
            'original_image_path': image_path,
            'model_type': 'placeholder'
        }
    
    def _create_placeholder_visualization(self, image, original_path):
        """Create placeholder visualization with simple overlays"""
        try:
            print("🎨 Creating placeholder visualization...")
            
            # Copy original image
            result_image = image.copy()
            
            # Draw simple bounding boxes as placeholder
            height, width = image.shape[:2]
            
            # Draw placeholder detection boxes with transparent green overlay
            overlay = result_image.copy()
            cv2.rectangle(overlay, (100, 50), (200, 300), (0, 255, 0), -1)  # Filled green rectangle
            cv2.rectangle(overlay, (80, 280), (220, 320), (0, 255, 0), -1)  # Filled green rectangle
            
            # Apply transparency
            alpha = 0.3
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            # Add bounding box outlines
            cv2.rectangle(result_image, (100, 50), (200, 300), (0, 255, 0), 3)  # Vertical rebar
            cv2.rectangle(result_image, (80, 280), (220, 320), (255, 0, 0), 3)  # Horizontal rebar
            
            # Add labels
            cv2.putText(result_image, 'Front Vertical (85%)', (100, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, 'Front Horizontal (78%)', (80, 275), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'placeholder_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ Placeholder visualization saved: {filename}")
                return output_path
            else:
                print("❌ Failed to save placeholder visualization")
                return original_path
                
        except Exception as e:
            print(f"❌ Placeholder visualization error: {str(e)}")
            return original_path
    
    def _calculate_cement_mixture(self, dimensions):
        """Calculate cement mixture ratios based on volume"""
        print("🧮 Calculating cement mixture...")
        
        volume_cm3 = dimensions.get('volume', 0)
        volume_m3 = volume_cm3 / 1000000  # Convert cm³ to m³
        
        # Standard concrete mixture ratios for Philippine construction
        # These ratios can be adjusted based on structural requirements
        
        # Basic ratio: 1 cement : 2 sand : 3 aggregate (by volume)
        cement_ratio = 1
        sand_ratio = 2
        aggregate_ratio = 3
        
        # Calculate total volume needed (accounting for concrete around rebar)
        # Assuming concrete column with rebar cage
        concrete_volume_factor = 1.5  # 50% more concrete than rebar volume
        total_concrete_volume = volume_m3 * concrete_volume_factor
        
        # Calculate material quantities
        total_parts = cement_ratio + sand_ratio + aggregate_ratio
        cement_volume = total_concrete_volume * (cement_ratio / total_parts)
        sand_volume = total_concrete_volume * (sand_ratio / total_parts)
        aggregate_volume = total_concrete_volume * (aggregate_ratio / total_parts)
        
        # Convert to practical units (bags of cement, cubic meters of sand/aggregate)
        cement_bags = cement_volume / 0.035  # 1 bag = ~0.035 m³
        
        return {
            'cement_ratio': cement_ratio,
            'sand_ratio': sand_ratio,
            'aggregate_ratio': aggregate_ratio,
            'ratio_string': f'{cement_ratio} Cement : {sand_ratio} Sand : {aggregate_ratio} Aggregate',
            'total_concrete_volume_m3': round(total_concrete_volume, 4),
            'cement_bags': round(cement_bags, 2),
            'sand_volume_m3': round(sand_volume, 4),
            'aggregate_volume_m3': round(aggregate_volume, 4),
            'calculation_method': 'standard_philippine_mix'
        }
    
    def get_model_status(self):
        """Get current model status"""
        return {
            'detectron2_available': DETECTRON2_AVAILABLE,
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path) if self.model_path else False,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'threshold': self.detection_threshold,
            'training_input_size': self.training_input_size,
            'model_type': 'real_trained_model' if self.model_loaded else 'placeholder'
        }
    
    def test_model(self, test_image_path=None):
        """Test the REAL MODEL with a sample image"""
        try:
            if not test_image_path:
                # Use a recent captured image for testing
                captured_dir = config.UPLOAD_FOLDER
                if os.path.exists(captured_dir):
                    images = [f for f in os.listdir(captured_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        test_image_path = os.path.join(captured_dir, images[-1])  # Use most recent
                    else:
                        return {
                            'success': False,
                            'error': 'No test images available'
                        }
                else:
                    return {
                        'success': False,
                        'error': 'Captured images directory not found'
                    }
            
            print(f"🧪 Testing REAL MODEL with: {test_image_path}")
            
            # Run analysis
            result = self.analyze_image(test_image_path)
            
            if result['success']:
                model_type = result.get('model_type', 'unknown')
                print(f"✅ REAL MODEL test successful! (Model type: {model_type})")
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'detections_found': result.get('num_detections', 0),
                    'model_type': model_type,
                    'analysis_result': result
                }
            else:
                print(f"❌ REAL MODEL test failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ REAL MODEL test error: {str(e)}")
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }
