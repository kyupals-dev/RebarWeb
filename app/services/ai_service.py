"""
AI Service for Rebar Detection and Analysis
Integrates Detectron2 Mask R-CNN model for rebar segmentation
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
    """Handles AI model loading, inference, and rebar analysis"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # Rebar classes (adjust based on your training)
        self.class_names = ["front_vertical", "front_horizontal", "rebar_structure"]
        
        print("🤖 Initializing AI Service...")
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("   Please ensure model_final.pth is in the correct location")
                return False
            
            print("🔄 Loading Detectron2 configuration...")
            
            # Set up configuration (matching your training config)
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Your number of classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
            self.cfg.MODEL.DEVICE = "cpu"  # Use CPU on Raspberry Pi
            
            print("🔄 Creating predictor...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # Set up metadata for visualization
            self.metadata = MetadataCatalog.get("rebar_dataset")
            self.metadata.thing_classes = self.class_names
            
            self.model_loaded = True
            print("✅ AI Model loaded successfully!")
            print(f"   Model path: {self.model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Threshold: {self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading AI model: {str(e)}")
            print("   Full traceback:")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_path):
        """
        Analyze image for rebar detection and return results
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Analysis results with detection data
        """
        try:
            print(f"🔍 Starting AI analysis of: {image_path}")
            
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
            
            # Use real model or placeholder
            if self.model_loaded and DETECTRON2_AVAILABLE:
                return self._analyze_with_model(image, image_path)
            else:
                return self._analyze_placeholder(image, image_path)
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _analyze_with_model(self, image, image_path):
        """Run actual AI model analysis"""
        try:
            print("🤖 Running Detectron2 inference...")
            
            # Run inference
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # Check if any detections
            num_detections = len(instances)
            print(f"🎯 Found {num_detections} detections")
            
            if num_detections == 0:
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True
                }
            
            # Extract detection data
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            # Process detections
            detections = []
            for i in range(num_detections):
                detection = {
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[classes[i]],
                    'confidence': float(scores[i]),
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'mask_area': float(np.sum(masks[i]))
                }
                detections.append(detection)
                
                print(f"   Detection {i+1}: {detection['class_name']} ({detection['confidence']:.3f})")
            
            # Create visualization
            analyzed_image_path = self._create_visualization(image, outputs, image_path)
            
            # Calculate dimensions (placeholder for now)
            dimensions = self._calculate_dimensions(detections, image.shape)
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture(dimensions)
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,
                'original_image_path': image_path
            }
            
        except Exception as e:
            print(f"❌ Model inference error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Model inference failed: {str(e)}'
            }
    
    def _analyze_placeholder(self, image, image_path):
        """Generate placeholder analysis results"""
        print("📝 Using placeholder AI analysis...")
        
        # Simulate some processing time
        import time
        time.sleep(2)
        
        # Create simple placeholder visualization
        analyzed_image_path = self._create_placeholder_visualization(image, image_path)
        
        # Placeholder dimensions and mixture
        dimensions = {
            'length': 25.4,
            'width': 25.4,
            'height': 200.0,
            'unit': 'cm',
            'volume': 129.032  # cm³
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
            'original_image_path': image_path
        }
    
    def _create_visualization(self, image, outputs, original_path):
        """Create visualization with Detectron2 overlays"""
        try:
            print("🎨 Creating AI analysis visualization...")
            
            # Create visualizer
            v = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=self.metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW
            )
            
            # Draw predictions
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = out.get_image()[:, :, ::-1]  # Convert back to BGR
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ Analysis visualization saved: {filename}")
                return output_path
            else:
                print("❌ Failed to save visualization")
                return original_path
                
        except Exception as e:
            print(f"❌ Visualization error: {str(e)}")
            return original_path
    
    def _create_placeholder_visualization(self, image, original_path):
        """Create placeholder visualization with simple overlays"""
        try:
            print("🎨 Creating placeholder visualization...")
            
            # Copy original image
            result_image = image.copy()
            
            # Draw simple bounding boxes as placeholder
            height, width = image.shape[:2]
            
            # Draw placeholder detection boxes
            cv2.rectangle(result_image, (100, 50), (200, 300), (0, 255, 0), 3)  # Vertical rebar
            cv2.rectangle(result_image, (80, 280), (220, 320), (255, 0, 0), 3)  # Horizontal rebar
            
            # Add labels
            cv2.putText(result_image, 'Front Vertical (85%)', (100, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, 'Front Horizontal (78%)', (80, 275), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analysis_placeholder_{timestamp}.jpg'
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
    
    def _calculate_dimensions(self, detections, image_shape):
        """Calculate rebar dimensions from detections (placeholder implementation)"""
        # This is a placeholder - you'll need to implement actual dimension calculation
        # based on your specific requirements and calibration
        
        print("📏 Calculating dimensions (placeholder)...")
        
        # For now, return placeholder values
        # TODO: Implement real dimension calculation based on:
        # - Detection bounding boxes
        # - Camera calibration
        # - Known reference objects
        # - Pixel-to-real-world conversion
        
        return {
            'length': 25.4,    # cm
            'width': 25.4,     # cm  
            'height': 200.0,   # cm
            'unit': 'cm',
            'volume': 25.4 * 25.4 * 200.0,  # cm³
            'method': 'placeholder'
        }
    
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
            'num_classes': 3,
            'class_names': self.class_names,
            'threshold': self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST if self.cfg else 0.5
        }
    
    def test_model(self, test_image_path=None):
        """Test the model with a sample image"""
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
            
            print(f"🧪 Testing model with: {test_image_path}")
            
            # Run analysis
            result = self.analyze_image(test_image_path)
            
            if result['success']:
                print("✅ Model test successful!")
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'detections_found': result.get('num_detections', 0),
                    'analysis_result': result
                }
            else:
                print(f"❌ Model test failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ Model test error: {str(e)}")
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }
