"""
AI Service for Rebar Detection and Analysis
Integrates Detectron2 Mask R-CNN model for rebar segmentation with COMPLETE REAL MODEL IMPLEMENTATION + DEBUG
"""

import os
import cv2
import numpy as np
from datetime import datetime
import json
import traceback
import torch

if not torch.cuda.is_available():
    print("CUDA not available, forcing Detectron2 to run on CPU")

# Detectron2 imports (will be installed later)
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2 import model_zoo
    import torch
    DETECTRON2_AVAILABLE = True
except ImportError:
    print("⚠️  Detectron2 not available. AI analysis will use placeholder results.")
    DETECTRON2_AVAILABLE = False


from app.utils.config import config

class AIService:
    """Handles AI model loading, inference, and rebar analysis with COMPLETE REAL TRAINED MODEL + DEBUG"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # Updated rebar classes based on your training
        self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        self.num_classes = 3
        
        # Updated detection threshold - LOWERED for better detection on Pi
        self.detection_threshold = 0.2  # Lowered from 0.3 to 0.2
        
        # Training image size (480x640 portrait)
        self.training_input_size = (480, 640)  # width x height
        
        # Pixel-to-CM conversion calibration points
        self.calibration_160cm = 0.2117  # cm/px at 160cm distance
        self.calibration_200cm = 0.2822  # cm/px at 200cm distance
        
        print("🤖 Initializing AI Service with COMPLETE REAL TRAINED MODEL + DEBUG...")
        print(f"   Classes: {self.class_names}")
        print(f"   Detection threshold: {self.detection_threshold}")
        print(f"   Training input size: {self.training_input_size[0]}x{self.training_input_size[1]}")
        print(f"   Calibration: 160cm={self.calibration_160cm:.4f} cm/px, 200cm={self.calibration_200cm:.4f} cm/px")
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model with REAL CONFIGURATION + DEBUG"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("   Please ensure model_final.pth is in the correct location")
                return False
            
            # Check model file size
            model_size = os.path.getsize(self.model_path)
            print(f"📁 Model file size: {model_size / 1024 / 1024:.1f} MB")
            
            print("🔄 Loading Detectron2 configuration for COMPLETE REAL MODEL...")
            
            # Set up configuration matching your training
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings - REAL CONFIGURATION
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # 3 classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold  # Lowered threshold
            self.cfg.MODEL.DEVICE = "cpu"  # Use CPU on Raspberry Pi
            
            # Input format matching your training (480x640)
            self.cfg.INPUT.MIN_SIZE_TRAIN = (640,)  # Height during training
            self.cfg.INPUT.MAX_SIZE_TRAIN = 640
            self.cfg.INPUT.MIN_SIZE_TEST = 640
            self.cfg.INPUT.MAX_SIZE_TEST = 640
            
            print("🔄 Creating predictor with COMPLETE REAL MODEL...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # Set up metadata for visualization with your classes
            self.metadata = MetadataCatalog.get("rebar_dataset_real")
            self.metadata.thing_classes = self.class_names
            
            # Set colors for each class (you can customize these)
            self.metadata.thing_colors = [
                (128, 128, 128),  # back_horizontal - Gray
                (255, 0, 0),      # front_horizontal - Red  
                (0, 255, 0),      # front_vertical - Green
            ]
            
            self.model_loaded = True
            print("✅ COMPLETE REAL AI Model loaded successfully!")
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
            print(f"❌ Error loading COMPLETE REAL AI model: {str(e)}")
            print("   Full traceback:")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def get_pixel_to_cm_factor(self, distance_cm=None):
        """Get pixel-to-cm conversion factor based on distance"""
        try:
            if distance_cm is None or distance_cm <= 0:
                # Default to middle of optimal range
                distance_cm = 180
                print(f"⚠️  No distance provided, using default: {distance_cm}cm")
            
            # Clamp distance to reasonable bounds
            distance_cm = max(100, min(300, distance_cm))
            
            # Linear interpolation between calibration points
            if distance_cm <= 160:
                factor = self.calibration_160cm
            elif distance_cm >= 200:
                factor = self.calibration_200cm
            else:
                # Linear interpolation
                ratio = (distance_cm - 160) / (200 - 160)
                factor = self.calibration_160cm + ratio * (self.calibration_200cm - self.calibration_160cm)
            
            print(f"📏 Distance: {distance_cm}cm → Conversion factor: {factor:.4f} cm/px")
            return factor
            
        except Exception as e:
            print(f"⚠️  Error calculating conversion factor: {e}")
            return 0.25  # Safe default
    
    def analyze_image(self, image_path):
        """
        Analyze image for rebar detection using COMPLETE REAL TRAINED MODEL
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Analysis results with detection data
        """
        try:
            print(f"🔍 Starting COMPLETE REAL AI analysis of: {image_path}")
            
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
                return self._analyze_with_complete_real_model(image, image_path)
            else:
                print("⚠️  COMPLETE REAL MODEL not available, using placeholder")
                return self._analyze_placeholder(image, image_path)
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def debug_analyze_image(self, image_path):
        """Debug version of analyze_image with detailed logging"""
        try:
            print("=" * 80)
            print(f"🔍 DEBUG: Starting analysis of: {image_path}")
            print("=" * 80)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"❌ DEBUG: Image file not found: {image_path}")
                return {'success': False, 'error': f'Image file not found: {image_path}'}
            
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ DEBUG: Failed to load image: {image_path}")
                return {'success': False, 'error': 'Failed to load image file'}
            
            print(f"✅ DEBUG: Image loaded successfully")
            print(f"   Original size: {image.shape}")
            
            # Resize if needed
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                print(f"🔄 DEBUG: Resizing from {width}x{height} to 480x640")
                image = cv2.resize(image, (480, 640))
            
            print(f"   Final size: {image.shape}")
            
            # Check model status
            print(f"🤖 DEBUG: Model loaded: {self.model_loaded}")
            print(f"   Detectron2 available: {DETECTRON2_AVAILABLE}")
            print(f"   Model path exists: {os.path.exists(self.model_path) if self.model_path else False}")
            
            if not self.model_loaded or not DETECTRON2_AVAILABLE:
                print("⚠️  DEBUG: Using placeholder analysis")
                return self._analyze_placeholder(image, image_path)
            
            # Run model inference
            print("🔄 DEBUG: Running model inference...")
            try:
                outputs = self.predictor(image)
                instances = outputs["instances"].to("cpu")
                num_detections = len(instances)
                
                print(f"✅ DEBUG: Inference successful")
                print(f"   Total detections: {num_detections}")
                
                if num_detections == 0:
                    print("❌ DEBUG: No detections found by model")
                    return {
                        'success': False,
                        'error': 'No rebar structures detected in image',
                        'no_detection': True,
                        'debug_info': {
                            'inference_successful': True,
                            'detections_found': 0,
                            'image_shape': image.shape,
                            'detection_threshold': self.detection_threshold
                        }
                    }
                
                # Analyze detections by class
                classes = instances.pred_classes.numpy()
                scores = instances.scores.numpy()
                
                class_counts = {}
                for i, class_id in enumerate(classes):
                    class_name = self.class_names[class_id]
                    confidence = scores[i]
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = []
                    class_counts[class_name].append(confidence)
                    
                    print(f"   Detection {i+1}: {class_name} ({confidence:.3f})")
                
                # Summary by class
                for class_name, confidences in class_counts.items():
                    print(f"   {class_name}: {len(confidences)} detections (avg confidence: {np.mean(confidences):.3f})")
                
                # Check for required classes
                front_horizontal_count = len([c for c in classes if self.class_names[c] == 'front_horizontal'])
                front_vertical_count = len([c for c in classes if self.class_names[c] == 'front_vertical'])
                
                print(f"🎯 DEBUG: Required classes found:")
                print(f"   front_horizontal: {front_horizontal_count}")
                print(f"   front_vertical: {front_vertical_count}")
                
                if front_horizontal_count == 0 or front_vertical_count == 0:
                    print("❌ DEBUG: Missing required classes for intersection analysis")
                    return {
                        'success': False,
                        'error': 'No rebar structures detected in image',
                        'no_detection': True,
                        'debug_info': {
                            'inference_successful': True,
                            'detections_found': num_detections,
                            'front_horizontal_count': front_horizontal_count,
                            'front_vertical_count': front_vertical_count,
                            'class_counts': class_counts,
                            'detection_threshold': self.detection_threshold
                        }
                    }
                
                # Continue with intersection analysis
                print("🔄 DEBUG: Starting intersection analysis...")
                intersection_result = self._perform_intersection_analysis(instances, image)
                
                print(f"✅ DEBUG: Intersection analysis: {'SUCCESS' if intersection_result['success'] else 'FAILED'}")
                if not intersection_result['success']:
                    print(f"   Error: {intersection_result.get('error', 'Unknown error')}")
                    return {
                        'success': False,
                        'error': 'No rebar structures detected in image',
                        'no_detection': True,
                        'debug_info': {
                            'inference_successful': True,
                            'detections_found': num_detections,
                            'intersection_analysis_failed': True,
                            'intersection_error': intersection_result.get('error', 'Unknown')
                        }
                    }
                
                # Continue with full analysis...
                print("✅ DEBUG: Analysis completed successfully!")
                print("=" * 80)
                
                # Call the regular analysis method
                return self._analyze_with_complete_real_model(image, image_path)
                
            except Exception as inference_error:
                print(f"❌ DEBUG: Model inference failed: {str(inference_error)}")
                import traceback
                traceback.print_exc()
                return {
                    'success': False,
                    'error': f'Model inference failed: {str(inference_error)}',
                    'debug_info': {
                        'inference_successful': False,
                        'inference_error': str(inference_error)
                    }
                }
            
        except Exception as e:
            print(f"❌ DEBUG: Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}

    def test_with_actual_image(self, image_path=None):
        """Test the model with an actual captured image"""
        try:
            if not image_path:
                # Find the most recent captured image
                captured_dir = config.UPLOAD_FOLDER
                if os.path.exists(captured_dir):
                    images = [f for f in os.listdir(captured_dir) 
                             if f.endswith(('.jpg', '.jpeg', '.png')) and 'frame_capture' in f]
                    if images:
                        images.sort(key=lambda x: os.path.getmtime(os.path.join(captured_dir, x)), reverse=True)
                        image_path = os.path.join(captured_dir, images[0])
                        print(f"🧪 DEBUG: Using most recent image: {images[0]}")
                    else:
                        return {'success': False, 'error': 'No captured images found'}
                else:
                    return {'success': False, 'error': 'Upload folder not found'}
            
            return self.debug_analyze_image(image_path)
            
        except Exception as e:
            return {'success': False, 'error': f'Test failed: {str(e)}'}
    
    def _analyze_with_complete_real_model(self, image, image_path):
        """Run COMPLETE real AI model analysis with intersection detection"""
        try:
            print("🤖 Running COMPLETE REAL Detectron2 inference...")
            
            # Run inference with your trained model
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # Check if any detections
            num_detections = len(instances)
            print(f"🎯 COMPLETE REAL MODEL found {num_detections} detections")
            
            if num_detections == 0:
                print("❌ No rebar structures detected by COMPLETE REAL MODEL")
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True
                }
            
            # Extract detection data from COMPLETE REAL MODEL
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            # Process detections from COMPLETE REAL MODEL
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
            
            # **COMPLETE INTERSECTION ANALYSIS** - Following notebook exactly
            intersection_result = self._perform_intersection_analysis(instances, image)
            
            if not intersection_result['success']:
                print("❌ No rebar intersections found - No Rebar Detected")
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True
                }
            
            # Calculate dimensions from COMPLETE REAL MODEL intersections
            dimensions = self._calculate_real_dimensions_from_intersections(
                intersection_result, 
                image.shape,
                image_path  # Pass image path to get distance info
            )
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture(dimensions)
            
            # Create COMPLETE visualization with intersections
            analyzed_image_path = self._create_complete_visualization(
                image, outputs, intersection_result, image_path
            )
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,
                'original_image_path': image_path,
                'model_type': 'complete_real_trained_model',
                'intersection_analysis': intersection_result
            }
            
        except Exception as e:
            print(f"❌ COMPLETE REAL MODEL inference error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'COMPLETE REAL MODEL inference failed: {str(e)}'
            }
    
    def _perform_intersection_analysis(self, instances, image):
        """
        Perform complete intersection analysis following notebook exactly
        Steps 1-6 from your outline
        """
        try:
            print("🔍 Performing COMPLETE intersection analysis...")
            
            # Step 2: Get the predicted masks
            masks = instances.pred_masks  # Tensor of shape [num_instances, H, W]
            classes = instances.pred_classes  # Tensor of class indices
            
            # Get indices of front_vertical and front_horizontal
            # Class order: ["back_horizontal", "front_horizontal", "front_vertical"]
            front_horizontal_idx = 1
            front_vertical_idx = 2
            
            # Get masks for each class
            front_horizontal_masks = masks[classes == front_horizontal_idx]
            front_vertical_masks = masks[classes == front_vertical_idx]
            
            print(f"   Found {len(front_horizontal_masks)} front_horizontal masks")
            print(f"   Found {len(front_vertical_masks)} front_vertical masks")
            
            # If multiple instances per class, combine them into one mask per class
            if front_horizontal_masks.shape[0] > 0:
                combined_front_horizontal = torch.any(front_horizontal_masks, dim=0)
            else:
                combined_front_horizontal = torch.zeros(masks.shape[1:], dtype=torch.bool)
                
            if front_vertical_masks.shape[0] > 0:
                combined_front_vertical = torch.any(front_vertical_masks, dim=0)
            else:
                combined_front_vertical = torch.zeros(masks.shape[1:], dtype=torch.bool)
            
            # Intersection mask
            intersection_mask = combined_front_horizontal & combined_front_vertical  # logical AND
            
            # Check if we have any intersections
            if not torch.any(intersection_mask):
                print("❌ No intersections found between front_horizontal and front_vertical")
                return {'success': False, 'error': 'No intersections found'}
            
            print(f"✅ Intersection mask created with {torch.sum(intersection_mask)} pixels")
            
            # Step 3: Convert intersection mask to uint8
            intersection_uint8 = intersection_mask.numpy().astype(np.uint8)
            
            # Find connected components
            num_labels, labels_im = cv2.connectedComponents(intersection_uint8)
            
            # Collect centroids of each component
            centroids = []
            for label in range(1, num_labels):  # label 0 is background
                mask = labels_im == label
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx = int(xs.mean())
                    cy = int(ys.mean())
                    centroids.append((cx, cy, label))
            
            print(f"   Found {len(centroids)} intersection centroids")
            
            if len(centroids) == 0:
                print("❌ No intersection centroids found")
                return {'success': False, 'error': 'No intersection centroids found'}
            
            # Sort centroids by y descending (bottom → top)
            centroids_sorted_y = sorted(centroids, key=lambda c: -c[1])
            
            # Group centroids into rows using a threshold
            y_threshold = 10  # pixels
            rows = []
            current_row = []
            for c in centroids_sorted_y:
                if not current_row:
                    current_row.append(c)
                    current_y = c[1]
                elif abs(c[1] - current_y) <= y_threshold:
                    current_row.append(c)
                else:
                    rows.append(current_row)
                    current_row = [c]
                    current_y = c[1]
            if current_row:
                rows.append(current_row)
            
            # Sort each row by x descending (right → left)
            for i in range(len(rows)):
                rows[i] = sorted(rows[i], key=lambda c: -c[0])
            
            print(f"   Organized into {len(rows)} rows")
            
            # Step 4-6: Create polygons and calculate areas
            polygons = []
            all_poly_points = []
            mask_areas = []
            
            # Iterate over each row pair: bottom row -> upper row
            for i in range(len(rows) - 1):
                bottom_row = rows[i]
                upper_row = rows[i + 1]
                
                n = min(len(bottom_row), len(upper_row)) - 1
                
                for j in range(n):
                    # Get four points of quadrilateral
                    p1 = bottom_row[j][:2]
                    p2 = bottom_row[j + 1][:2]
                    p3 = upper_row[j + 1][:2]
                    p4 = upper_row[j][:2]
                    
                    poly = np.array([p1, p2, p3, p4], dtype=np.int32)
                    polygons.append(poly)
                    all_poly_points.extend(poly.tolist())
                    
                    # Calculate pixel area
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    area = cv2.countNonZero(mask)
                    mask_areas.append(area)
            
            if len(polygons) == 0:
                print("❌ No polygons created from intersections")
                return {'success': False, 'error': 'No polygons created from intersections'}
            
            # Calculate total dimensions
            all_poly_points = np.array(all_poly_points)
            x_min, y_min = all_poly_points.min(axis=0)
            x_max, y_max = all_poly_points.max(axis=0)
            total_width_px = x_max - x_min
            total_height_px = y_max - y_min
            total_area_px = sum(mask_areas)
            
            print(f"✅ Intersection analysis complete:")
            print(f"   Total width: {total_width_px} pixels")
            print(f"   Total height: {total_height_px} pixels")
            print(f"   Total area: {total_area_px} pixels²")
            print(f"   Created {len(polygons)} polygons")
            
            return {
                'success': True,
                'intersection_mask': intersection_mask,
                'centroids': centroids,
                'rows': rows,
                'polygons': polygons,
                'total_width_px': int(total_width_px),
                'total_height_px': int(total_height_px),
                'total_area_px': int(total_area_px),
                'mask_areas': mask_areas,
                'combined_front_horizontal': combined_front_horizontal,
                'combined_front_vertical': combined_front_vertical
            }
            
        except Exception as e:
            print(f"❌ Error in intersection analysis: {str(e)}")
            traceback.print_exc()
            return {'success': False, 'error': f'Intersection analysis failed: {str(e)}'}
    
    def _calculate_real_dimensions_from_intersections(self, intersection_result, image_shape, image_path):
        """
        Calculate real dimensions from intersection analysis (Step 7)
        Convert pixels to centimeters using distance-based calibration
        """
        try:
            print("📏 Calculating REAL dimensions from intersection analysis...")
            
            # Get pixel dimensions from intersection analysis
            width_px = intersection_result['total_width_px']
            height_px = intersection_result['total_height_px']
            area_px = intersection_result['total_area_px']
            
            print(f"   Pixel dimensions: {width_px}px × {height_px}px")
            print(f"   Pixel area: {area_px}px²")
            
            # Get distance for calibration (try to get from distance service)
            distance_cm = self._get_distance_from_context(image_path)
            
            # Get calibration factor
            pixel_to_cm = self.get_pixel_to_cm_factor(distance_cm)
            
            # Convert pixels to centimeters
            width_cm = width_px * pixel_to_cm
            height_cm = height_px * pixel_to_cm
            
            # For square column: length = width
            length_cm = width_cm
            
            # Calculate volume (length × width × height)
            volume_cm3 = length_cm * width_cm * height_cm
            
            # Round to reasonable precision
            length_cm = round(length_cm, 1)
            width_cm = round(width_cm, 1)
            height_cm = round(height_cm, 1)
            volume_cm3 = round(volume_cm3, 1)
            
            # Create display string in requested format
            display_string = f"{length_cm}cm x {width_cm}cm x {height_cm}cm = {volume_cm3}cm³"
            
            print(f"✅ Real dimensions calculated:")
            print(f"   Length: {length_cm}cm")
            print(f"   Width: {width_cm}cm") 
            print(f"   Height: {height_cm}cm")
            print(f"   Volume: {volume_cm3}cm³")
            print(f"   Display: {display_string}")
            print(f"   Calibration: {pixel_to_cm:.4f} cm/px at {distance_cm}cm")
            
            return {
                'length': length_cm,
                'width': width_cm, 
                'height': height_cm,
                'unit': 'cm',
                'volume': volume_cm3,
                'display': display_string,
                'method': 'complete_real_intersection_analysis',
                'calibration': {
                    'pixel_to_cm': pixel_to_cm,
                    'distance_cm': distance_cm,
                    'width_px': width_px,
                    'height_px': height_px,
                    'area_px': area_px
                }
            }
            
        except Exception as e:
            print(f"❌ Error calculating REAL dimensions: {str(e)}")
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
    
    def _get_distance_from_context(self, image_path):
        """Try to extract distance from image filename or use default"""
        try:
            # Try to extract distance from filename if it contains distance info
            filename = os.path.basename(image_path)
            
            # Look for distance pattern like "180cm" in filename
            import re
            distance_match = re.search(r'(\d+)cm', filename)
            if distance_match:
                distance = int(distance_match.group(1))
                if 100 <= distance <= 300:  # Reasonable range
                    print(f"📏 Extracted distance from filename: {distance}cm")
                    return distance
            
            # Default to optimal range center
            default_distance = 180
            print(f"📏 Using default distance: {default_distance}cm")
            return default_distance
            
        except Exception as e:
            print(f"⚠️  Error getting distance from context: {e}")
            return 180  # Safe default
    
    def _create_complete_visualization(self, image, outputs, intersection_result, original_path):
        """Create COMPLETE visualization with both detections and intersection analysis"""
        try:
            print("🎨 Creating COMPLETE MODEL analysis visualization...")
            
            # Start with original image
            result_image = image.copy()
            
            # Step 1: Draw original detections (front_vertical and front_horizontal)
            instances = outputs["instances"].to("cpu")
            
            # Create visualizer for detections
            v = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=self.metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW  # Show detections on original image
            )
            
            # Draw instance predictions
            out = v.draw_instance_predictions(instances)
            result_image = out.get_image()[:, :, ::-1]  # Convert back to BGR
            
            # Step 2: Add intersection analysis - blue polygons
            if intersection_result['success']:
                polygons = intersection_result['polygons']
                
                # Create overlay for semi-transparent blue polygons
                overlay = result_image.copy()
                alpha = 0.4  # Transparency factor
                
                for poly in polygons:
                    # Draw semi-transparent blue polygon
                    cv2.fillPoly(overlay, [poly], (255, 0, 0))  # Blue in BGR
                
                # Apply transparency
                result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
                
                # Draw polygon outlines
                for poly in polygons:
                    cv2.polylines(result_image, [poly], True, (255, 0, 0), 2)  # Blue outline
                
                # Add bounding box of total area
                total_width = intersection_result['total_width_px']
                total_height = intersection_result['total_height_px']
                
                # Find bounding box coordinates
                all_points = []
                for poly in polygons:
                    all_points.extend(poly.tolist())
                all_points = np.array(all_points)
                
                if len(all_points) > 0:
                    x_min, y_min = all_points.min(axis=0)
                    x_max, y_max = all_points.max(axis=0)
                    
                    # Draw total bounding box
                    cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)  # Yellow box
                    
                    # Add dimension text
                    text = f"Rebar Cage: {total_width}x{total_height}px"
                    cv2.putText(result_image, text, (x_min, y_min-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                print(f"✅ Added {len(polygons)} blue intersection polygons to visualization")
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'complete_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ COMPLETE MODEL visualization saved: {filename}")
                return output_path
            else:
                print("❌ Failed to save COMPLETE MODEL visualization")
                return original_path
                
        except Exception as e:
            print(f"❌ COMPLETE MODEL visualization error: {str(e)}")
            traceback.print_exc()
            return original_path
    
    def _analyze_placeholder(self, image, image_path):
        """Generate placeholder analysis results (fallback only)"""
        print("📝 Using placeholder AI analysis (COMPLETE REAL MODEL not available)...")
        
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
            'model_type': 'complete_real_trained_model' if self.model_loaded else 'placeholder',
            'calibration_160cm': self.calibration_160cm,
            'calibration_200cm': self.calibration_200cm
        }
    
    def test_model(self, test_image_path=None):
        """Test the COMPLETE REAL MODEL with a sample image"""
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
            
            print(f"🧪 Testing COMPLETE REAL MODEL with: {test_image_path}")
            
            # Run analysis
            result = self.analyze_image(test_image_path)
            
            if result['success']:
                model_type = result.get('model_type', 'unknown')
                intersection_success = result.get('intersection_analysis', {}).get('success', False)
                
                print(f"✅ COMPLETE REAL MODEL test successful! (Model type: {model_type})")
                print(f"   Intersection analysis: {'✅ SUCCESS' if intersection_success else '❌ FAILED'}")
                
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'detections_found': result.get('num_detections', 0),
                    'model_type': model_type,
                    'intersection_analysis': intersection_success,
                    'dimensions_calculated': result.get('dimensions', {}).get('display', 'N/A'),
                    'analysis_result': result
                }
            else:
                print(f"❌ COMPLETE REAL MODEL test failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ COMPLETE REAL MODEL test error: {str(e)}")
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }
