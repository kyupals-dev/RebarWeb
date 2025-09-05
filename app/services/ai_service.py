"""
Complete Updated AI Service for Rebar Detection and Analysis
SIMPLIFIED VERSION: Focus on 2 front_vertical + 11 front_horizontal detection
with 4.5cm offset calculation and cement mixture estimation
"""

import os
import cv2
import numpy as np
from datetime import datetime
import json
import traceback

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
    """
    SIMPLIFIED AI Service for Rebar Detection
    Focus: 2 front_vertical + 11 front_horizontal rebars with intersections
    """
    
    def __init__(self):
        # Target detection counts (relaxed for real-world conditions)
        self.target_front_vertical = 2    # Target: 2 front vertical rebars
        self.target_front_horizontal = 11 # Target: 11 front horizontal rebars
        self.min_front_vertical = 1      # Minimum acceptable
        self.min_front_horizontal = 5    # Minimum acceptable
        
        # Offset for cement calculation
        self.offset_cm = 4.5  # 4.5cm offset on each side
        
        # Model configuration
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        self.num_classes = 3
        
        # LOWERED detection threshold for better detection
        self.detection_threshold = 0.1  # Lowered from 0.3 to 0.1
        
        # Training image size
        self.training_input_size = (480, 640)  # width x height
        
        # Model state
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.metadata = None
        
        print("🎯 SIMPLIFIED AI Service for Rebar Detection")
        print(f"   Target: {self.target_front_vertical} verticals + {self.target_front_horizontal} horizontals")
        print(f"   Minimum: {self.min_front_vertical} verticals + {self.min_front_horizontal} horizontals")
        print(f"   Detection threshold: {self.detection_threshold} (lowered for better detection)")
        print(f"   Offset calculation: {self.offset_cm}cm per side")
        print("   📝 SIMPLIFIED: Only saves analyzed images with AI overlays")
        
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model with SIMPLIFIED configuration"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("   Please ensure model_final.pth is in the correct location")
                print("   Expected size: ~170-250 MB for Mask R-CNN model")
                return False
            
            # Check model file size
            file_size = os.path.getsize(self.model_path)
            print(f"📁 Model file size: {file_size / (1024*1024):.1f} MB")
            
            if file_size < 1024*1024:  # Less than 1MB
                print("⚠️  WARNING: Model file seems too small (expected ~170-250 MB)")
            
            print("🔄 Loading Detectron2 configuration for SIMPLIFIED detection...")
            
            # Set up configuration matching your Colab training
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings - EXACTLY like your Colab training
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # 3 classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold  # 0.1 threshold
            self.cfg.MODEL.DEVICE = "cpu"  # CPU for Raspberry Pi
            
            # Input format matching your training
            self.cfg.INPUT.MIN_SIZE_TRAIN = (640,)
            self.cfg.INPUT.MAX_SIZE_TRAIN = 640
            self.cfg.INPUT.MIN_SIZE_TEST = 640
            self.cfg.INPUT.MAX_SIZE_TEST = 640
            
            print("🔄 Creating predictor with SIMPLIFIED model...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # Set up metadata for visualization
            self.metadata = MetadataCatalog.get("rebar_dataset_simplified")
            self.metadata.thing_classes = self.class_names
            
            # Colors for each class
            self.metadata.thing_colors = [
                (128, 128, 128),  # back_horizontal - Gray
                (0, 0, 255),      # front_horizontal - Red  
                (0, 255, 0),      # front_vertical - Green
            ]
            
            self.model_loaded = True
            print("✅ SIMPLIFIED AI Model loaded successfully!")
            print(f"   Model path: {self.model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Detection threshold: {self.detection_threshold}")
            print(f"   Input size: {self.training_input_size[0]}x{self.training_input_size[1]}")
            
            # Test the model with a quick inference
            print("🧪 Testing model with dummy image...")
            test_image = np.zeros((640, 480, 3), dtype=np.uint8)
            try:
                test_output = self.predictor(test_image)
                print(f"✅ Model inference test successful! (Found {len(test_output['instances'])} detections on blank image)")
            except Exception as e:
                print(f"⚠️  Model inference test failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading SIMPLIFIED AI model: {str(e)}")
            print("   Full traceback:")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_data=None, image_path=None):
        """
        SIMPLIFIED rebar analysis focusing on your specific requirements:
        - 2 front_vertical rebars (minimum 1)
        - 11 front_horizontal rebars (minimum 5)
        - Intersection verification
        - 4.5cm offset calculation
        - Only saves analyzed images with AI overlays
        """
        try:
            print(f"🎯 Starting SIMPLIFIED rebar analysis (2V + 11H target)...")
            
            # Handle different input types
            if image_data is not None:
                print("📸 Using direct frame data from camera (no original saved)")
                image = image_data.copy()
                original_source = "camera_frame"
            elif image_path and os.path.exists(image_path):
                print(f"📁 Loading image from: {image_path}")
                image = cv2.imread(image_path)
                original_source = "file"
                if image is None:
                    return {'success': False, 'error': 'Failed to load image file'}
            else:
                return {'success': False, 'error': 'No image data or valid path provided'}
            
            print(f"📐 Image loaded: {image.shape} (H×W×C) from {original_source}")
            
            # Ensure image is the right size (480x640)
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                print(f"⚙️  Resizing image from {width}x{height} to 480x640 for model input")
                image = cv2.resize(image, (480, 640))
            
            # Use real model or placeholder
            if self.model_loaded and DETECTRON2_AVAILABLE:
                result = self._analyze_with_simplified_model(image)
            else:
                print("⚠️  REAL MODEL not available, using placeholder")
                result = self._analyze_placeholder_simplified(image)
            
            # Ensure we return only the analyzed image path
            if result['success'] and 'analyzed_image_path' in result:
                filename = os.path.basename(result['analyzed_image_path'])
                print(f"✅ SIMPLIFIED analysis complete. ONLY analyzed image saved: {filename}")
                print(f"🎯 Results: {result.get('front_vertical_count', 0)}V + {result.get('front_horizontal_count', 0)}H detected")
            
            return result
                
        except Exception as e:
            print(f"❌ SIMPLIFIED analysis error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Simplified analysis failed: {str(e)}'
            }
    
    def _analyze_with_simplified_model(self, image):
        """Run SIMPLIFIED model analysis focused on your requirements"""
        try:
            print("🤖 Running SIMPLIFIED Detectron2 inference...")
            
            # First, run debug detection to see what model finds
            self._debug_detection(image)
            
            # Run inference with your trained model
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # Check if any detections
            num_detections = len(instances)
            print(f"🎯 SIMPLIFIED MODEL found {num_detections} detections (threshold: {self.detection_threshold})")
            
            if num_detections == 0:
                print("❌ No rebar structures detected by SIMPLIFIED MODEL")
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True,
                    'debug_suggestion': 'Try better lighting or closer distance'
                }
            
            # Extract detection data
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            # Separate detections by class
            front_vertical_detections = []
            front_horizontal_detections = []
            back_horizontal_detections = []
            
            for i in range(num_detections):
                class_id = int(classes[i])
                confidence = float(scores[i])
                class_name = self.class_names[class_id]
                bbox = boxes[i].tolist()  # [x1, y1, x2, y2]
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'mask_area': float(np.sum(masks[i])),
                    'mask': masks[i]
                }
                
                if class_name == 'front_vertical':
                    front_vertical_detections.append(detection)
                elif class_name == 'front_horizontal':
                    front_horizontal_detections.append(detection)
                elif class_name == 'back_horizontal':
                    back_horizontal_detections.append(detection)
                
                print(f"   Detection {i+1}: {class_name} ({confidence:.3f}) - Area: {detection['mask_area']:.0f}px")
            
            # Count detections by class
            fv_count = len(front_vertical_detections)
            fh_count = len(front_horizontal_detections)
            bh_count = len(back_horizontal_detections)
            
            print(f"🎯 SIMPLIFIED detection summary:")
            print(f"   Front Vertical: {fv_count} (target: {self.target_front_vertical}, min: {self.min_front_vertical})")
            print(f"   Front Horizontal: {fh_count} (target: {self.target_front_horizontal}, min: {self.min_front_horizontal})")
            print(f"   Back Horizontal: {bh_count} (not used in calculation)")
            
            # Check minimum requirements (relaxed for real-world conditions)
            if fv_count < self.min_front_vertical:
                return {
                    'success': False,
                    'error': f'Need at least {self.min_front_vertical} front_vertical rebars (found {fv_count})',
                    'no_detection': True
                }
            
            if fh_count < self.min_front_horizontal:
                return {
                    'success': False,
                    'error': f'Need at least {self.min_front_horizontal} front_horizontal rebars (found {fh_count})',
                    'no_detection': True
                }
            
            # Verify intersections between front vertical and horizontal
            intersections = self._verify_intersections_simplified(front_vertical_detections, front_horizontal_detections)
            intersection_count = len(intersections)
            
            print(f"🔗 Found {intersection_count} intersections between front rebars")
            
            if intersection_count < 2:
                print(f"⚠️  Low intersection count: {intersection_count} (expected more for rebar structure)")
                # Don't fail completely, but note the issue
            
            # Calculate dimensions with your 4.5cm offset
            dimensions = self._calculate_simplified_dimensions(front_vertical_detections, front_horizontal_detections, image.shape)
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture_simplified(dimensions)
            
            # Create visualization (ONLY FILE SAVED)
            analyzed_image_path = self._create_simplified_visualization(image, front_vertical_detections, front_horizontal_detections, intersections)
            
            if not analyzed_image_path:
                return {
                    'success': False,
                    'error': 'Failed to create analyzed image visualization'
                }
            
            return {
                'success': True,
                'detections': front_vertical_detections + front_horizontal_detections,
                'num_detections': fv_count + fh_count,
                'front_vertical_count': fv_count,
                'front_horizontal_count': fh_count,
                'intersection_count': intersection_count,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,  # ONLY image saved
                'model_type': 'simplified_real_model',
                'target_achieved': {
                    'vertical': fv_count >= self.target_front_vertical,
                    'horizontal': fh_count >= self.target_front_horizontal
                }
            }
            
        except Exception as e:
            print(f"❌ SIMPLIFIED MODEL inference error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'SIMPLIFIED MODEL inference failed: {str(e)}'
            }
    
    def _debug_detection(self, image):
        """Debug method to see what model detects with very low threshold"""
        try:
            print("🔍 DEBUG: Running detection with very low threshold (0.05)...")
            
            # Create temporary config with very low threshold
            debug_cfg = self.cfg.clone()
            debug_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
            
            debug_predictor = DefaultPredictor(debug_cfg)
            debug_outputs = debug_predictor(image)
            debug_instances = debug_outputs["instances"].to("cpu")
            
            debug_count = len(debug_instances)
            print(f"🎯 DEBUG: Found {debug_count} detections with threshold 0.05")
            
            if debug_count > 0:
                debug_scores = debug_instances.scores.numpy()
                debug_classes = debug_instances.pred_classes.numpy()
                
                # Count by class
                class_counts = {name: 0 for name in self.class_names}
                
                print("📋 DEBUG: All detections (threshold 0.05):")
                for i in range(min(debug_count, 20)):  # Show max 20 detections
                    class_id = int(debug_classes[i])
                    confidence = float(debug_scores[i])
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                    class_counts[class_name] += 1
                    print(f"   {i+1}: {class_name} (confidence: {confidence:.4f})")
                
                print(f"🎯 DEBUG class counts:")
                for class_name, count in class_counts.items():
                    print(f"   {class_name}: {count}")
                
            else:
                print("❌ DEBUG: Even with very low threshold (0.05), no detections found")
                print("💡 This suggests:")
                print("   1. Model may need retraining on current image conditions")
                print("   2. Image preprocessing issues")
                print("   3. Model file may be corrupted")
                
        except Exception as e:
            print(f"❌ DEBUG detection failed: {e}")
    
    def _verify_intersections_simplified(self, verticals, horizontals):
        """Verify intersections between front vertical and horizontal rebars"""
        intersections = []
        
        for i, vertical in enumerate(verticals):
            for j, horizontal in enumerate(horizontals):
                if self._masks_intersect(vertical['mask'], horizontal['mask']):
                    intersections.append({
                        'vertical_id': i,
                        'horizontal_id': j,
                        'vertical_class': vertical['class_name'],
                        'horizontal_class': horizontal['class_name'],
                        'type': 'front_vertical_x_front_horizontal'
                    })
        
        return intersections
    
    def _masks_intersect(self, mask1, mask2):
        """Check if two masks intersect with minimum area requirement"""
        intersection = np.logical_and(mask1, mask2)
        intersection_area = np.sum(intersection)
        return intersection_area > 50  # Minimum intersection area in pixels
    
    def _calculate_simplified_dimensions(self, verticals, horizontals, image_shape):
        """
        Calculate rebar dimensions with your 4.5cm offset requirement
        Focus on square column measurement
        """
        try:
            print("📏 Calculating SIMPLIFIED dimensions with 4.5cm offset...")
            
            # Get all bounding boxes
            all_detections = verticals + horizontals
            
            if not all_detections:
                # Fallback dimensions
                return self._get_fallback_dimensions()
            
            # Find overall structure bounds
            x_coords = []
            y_coords = []
            
            for detection in all_detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            
            # Structure size in pixels
            structure_width_px = max(x_coords) - min(x_coords)
            structure_height_px = max(y_coords) - min(y_coords)
            
            print(f"   Structure bounds: {structure_width_px:.0f}px × {structure_height_px:.0f}px")
            
            # Convert to cm (calibration factor - you may need to adjust this)
            # This factor depends on your camera distance and rebar size
            pixel_to_cm = 0.12  # Approximate conversion factor
            
            structure_width_cm = structure_width_px * pixel_to_cm
            structure_height_cm = structure_height_px * pixel_to_cm
            
            print(f"   Structure size: {structure_width_cm:.1f}cm × {structure_height_cm:.1f}cm")
            
            # Add your 4.5cm offset on each side (9cm total per dimension)
            total_width_cm = structure_width_cm + (2 * self.offset_cm)
            total_height_cm = structure_height_cm + (2 * self.offset_cm)
            
            print(f"   With offset: {total_width_cm:.1f}cm × {total_height_cm:.1f}cm")
            
            # For square columns, use the larger dimension to ensure coverage
            side_length_cm = max(total_width_cm, total_height_cm)
            
            # Ensure minimum realistic size
            side_length_cm = max(side_length_cm, 20.0)
            
            # For square columns, all sides are equal
            length_cm = side_length_cm
            width_cm = side_length_cm
            height_cm = side_length_cm  # Assuming cube for simplicity
            
            # Calculate volume
            volume_cm3 = length_cm * width_cm * height_cm
            
            print(f"   Final dimensions: {length_cm:.1f}cm × {width_cm:.1f}cm × {height_cm:.1f}cm")
            print(f"   Volume: {volume_cm3:.0f}cm³")
            
            return {
                'length': round(length_cm, 1),
                'width': round(width_cm, 1),
                'height': round(height_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': f"{length_cm:.1f}cm x {width_cm:.1f}cm x {height_cm:.1f}cm = {volume_cm3:.0f}cm³",
                'offset_applied': f"{self.offset_cm}cm per side",
                'structure_size_cm': f"{structure_width_cm:.1f} x {structure_height_cm:.1f}",
                'total_size_cm': f"{total_width_cm:.1f} x {total_height_cm:.1f}",
                'method': 'simplified_rebar_analysis',
                'pixel_to_cm_factor': pixel_to_cm
            }
            
        except Exception as e:
            print(f"❌ Error calculating SIMPLIFIED dimensions: {str(e)}")
            return self._get_fallback_dimensions()
    
    def _get_fallback_dimensions(self):
        """Get fallback dimensions when calculation fails"""
        return {
            'length': 25.0,
            'width': 25.0,
            'height': 25.0,
            'unit': 'cm',
            'volume': 15625,
            'display': '25.0cm x 25.0cm x 25.0cm = 15625cm³',
            'offset_applied': f"{self.offset_cm}cm per side",
            'method': 'fallback_calculation'
        }
    
    def _calculate_cement_mixture_simplified(self, dimensions):
        """Calculate cement mixture based on volume with your specifications"""
        print("🧮 Calculating SIMPLIFIED cement mixture...")
        
        volume_cm3 = dimensions.get('volume', 0)
        volume_m3 = volume_cm3 / 1000000  # Convert cm³ to m³
        
        # Standard concrete mixture ratios for Philippine construction
        cement_ratio = 1
        sand_ratio = 2
        aggregate_ratio = 3
        
        # Calculate total volume needed (accounting for concrete around rebar)
        concrete_volume_factor = 1.5  # 50% more concrete than rebar volume
        total_concrete_volume = volume_m3 * concrete_volume_factor
        
        # Calculate material quantities
        total_parts = cement_ratio + sand_ratio + aggregate_ratio
        cement_volume = total_concrete_volume * (cement_ratio / total_parts)
        sand_volume = total_concrete_volume * (sand_ratio / total_parts)
        aggregate_volume = total_concrete_volume * (aggregate_ratio / total_parts)
        
        # Convert to practical units
        cement_bags = cement_volume / 0.035  # 1 bag ≈ 0.035 m³
        
        print(f"   Concrete volume needed: {total_concrete_volume:.4f} m³")
        print(f"   Cement bags: {cement_bags:.2f}")
        
        return {
            'cement_ratio': cement_ratio,
            'sand_ratio': sand_ratio,
            'aggregate_ratio': aggregate_ratio,
            'ratio_string': f'{cement_ratio} Cement : {sand_ratio} Sand : {aggregate_ratio} Aggregate',
            'total_concrete_volume_m3': round(total_concrete_volume, 4),
            'cement_bags': round(cement_bags, 2),
            'sand_volume_m3': round(sand_volume, 4),
            'aggregate_volume_m3': round(aggregate_volume, 4),
            'calculation_method': 'simplified_philippine_mix'
        }
    
    def _create_simplified_visualization(self, image, verticals, horizontals, intersections):
        """Create SIMPLIFIED visualization - ONLY method that saves images"""
        try:
            print("🎨 Creating SIMPLIFIED analysis visualization (ONLY FILE SAVED)...")
            
            # Create result image
            result_image = image.copy()
            
            # Draw front vertical rebars in GREEN
            for i, vertical in enumerate(verticals):
                bbox = vertical['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                confidence = vertical['confidence']
                
                # Draw thick green rectangle
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add label
                label = f"V{i+1} ({confidence:.2f})"
                cv2.putText(result_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw front horizontal rebars in RED
            for i, horizontal in enumerate(horizontals):
                bbox = horizontal['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                confidence = horizontal['confidence']
                
                # Draw red rectangle
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f"H{i+1} ({confidence:.2f})"
                cv2.putText(result_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add summary information
            summary_text = f"SIMPLIFIED: {len(verticals)}V + {len(horizontals)}H (Target: 2V + 11H)"
            cv2.putText(result_image, summary_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            intersection_text = f"Intersections: {len(intersections)}"
            cv2.putText(result_image, intersection_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            offset_text = f"Offset: {self.offset_cm}cm per side"
            cv2.putText(result_image, offset_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Generate output filename for analyzed image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analyzed_simplified_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image (THIS IS THE ONLY FILE SAVED)
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                file_size = os.path.getsize(output_path)
                saved_img = cv2.imread(output_path)
                if saved_img is not None:
                    saved_height, saved_width = saved_img.shape[:2]
                    print(f"✅ SIMPLIFIED ANALYZED IMAGE SAVED (ONLY COPY):")
                    print(f"   📁 File: {filename}")
                    print(f"   📐 Dimensions: {saved_width}x{saved_height}")
                    print(f"   💾 Size: {file_size / 1024:.1f} KB")
                    print(f"   🎯 Contains: {len(verticals)}V + {len(horizontals)}H detections with intersections")
                    return output_path
                else:
                    print("❌ Could not verify saved analyzed image")
                    return None
            else:
                print("❌ Failed to save SIMPLIFIED ANALYZED IMAGE")
                return None
                
        except Exception as e:
            print(f"❌ SIMPLIFIED visualization error: {str(e)}")
            traceback.print_exc()
            return None
    
    def _analyze_placeholder_simplified(self, image):
        """Generate SIMPLIFIED placeholder analysis results"""
        print("📝 Using SIMPLIFIED placeholder AI analysis (REAL MODEL not available)...")
        
        # Create simple placeholder visualization (ONLY FILE SAVED)
        analyzed_image_path = self._create_placeholder_visualization_simplified(image)
        
        if not analyzed_image_path:
            return {
                'success': False,
                'error': 'Failed to create placeholder visualization'
            }
        
        # Placeholder dimensions with 4.5cm offset
        dimensions = {
            'length': 29.0,  # 20cm structure + 9cm offset
            'width': 29.0,   # 20cm structure + 9cm offset
            'height': 29.0,  # Square column
            'unit': 'cm',
            'volume': 24389,  # 29³
            'display': '29.0cm x 29.0cm x 29.0cm = 24389cm³',
            'offset_applied': f'{self.offset_cm}cm per side',
            'method': 'simplified_placeholder'
        }
        
        mixture = {
            'cement_ratio': 1,
            'sand_ratio': 2,
            'aggregate_ratio': 3,
            'ratio_string': '1 Cement : 2 Sand : 3 Aggregate',
            'cement_bags': 1.05,
            'sand_volume_m3': 0.0001,
            'aggregate_volume_m3': 0.00015,
            'total_concrete_volume_m3': 0.00037
        }
        
        return {
            'success': True,
            'placeholder': True,
            'detections': [
                {
                    'class_name': 'front_vertical',
                    'confidence': 0.85,
                    'bbox': [150, 100, 170, 400]
                },
                {
                    'class_name': 'front_vertical',
                    'confidence': 0.82,
                    'bbox': [300, 100, 320, 400]
                }
            ] + [
                {
                    'class_name': 'front_horizontal',
                    'confidence': 0.75 + i*0.02,
                    'bbox': [140, 120 + i*25, 330, 130 + i*25]
                } for i in range(11)
            ],
            'num_detections': 13,
            'front_vertical_count': 2,
            'front_horizontal_count': 11,
            'intersection_count': 22,
            'dimensions': dimensions,
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_image_path,
            'model_type': 'simplified_placeholder',
            'target_achieved': {
                'vertical': True,
                'horizontal': True
            }
        }
    
    def _create_placeholder_visualization_simplified(self, image):
        """Create SIMPLIFIED placeholder visualization - ONLY method that saves placeholder images"""
        try:
            print("🎨 Creating SIMPLIFIED placeholder visualization (ONLY FILE SAVED)...")
            
            # Copy original image
            result_image = image.copy()
            
            # Draw 2 front vertical rebars (GREEN)
            cv2.rectangle(result_image, (150, 100), (170, 400), (0, 255, 0), 3)  # V1
            cv2.rectangle(result_image, (300, 100), (320, 400), (0, 255, 0), 3)  # V2
            
            # Add vertical labels
            cv2.putText(result_image, 'V1 (0.85)', (150, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, 'V2 (0.82)', (300, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw 11 front horizontal rebars (RED)
            for i in range(11):
                y = 120 + i * 25
                cv2.rectangle(result_image, (140, y), (330, y+10), (0, 0, 255), 2)
                confidence = 0.75 + i*0.02
                cv2.putText(result_image, f'H{i+1}', (335, y+8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Add transparent green overlay for intersections
            overlay = result_image.copy()
            for i in range(11):
                y = 120 + i * 25
                # Intersection with V1
                cv2.rectangle(overlay, (150, y), (170, y+10), (0, 255, 0), -1)
                # Intersection with V2  
                cv2.rectangle(overlay, (300, y), (320, y+10), (0, 255, 0), -1)
            
            # Apply transparency
            alpha = 0.3
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            # Add summary text
            cv2.putText(result_image, 'SIMPLIFIED PLACEHOLDER: 2V + 11H', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, 'Intersections: 22 (2V x 11H)', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_image, f'Offset: {self.offset_cm}cm per side', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Generate output filename for placeholder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analyzed_placeholder_simplified_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image (THIS IS THE ONLY FILE SAVED)
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                file_size = os.path.getsize(output_path)
                print(f"✅ SIMPLIFIED PLACEHOLDER ANALYZED IMAGE SAVED (ONLY COPY):")
                print(f"   📁 File: {filename}")
                print(f"   💾 Size: {file_size / 1024:.1f} KB")
                print(f"   🎯 Contains: 2V + 11H placeholder with {self.offset_cm}cm offset")
                return output_path
            else:
                print("❌ Failed to save SIMPLIFIED placeholder analyzed image")
                return None
                
        except Exception as e:
            print(f"❌ SIMPLIFIED placeholder visualization error: {str(e)}")
            return None
    
    def get_model_status(self):
        """Get current SIMPLIFIED model status"""
        return {
            'detectron2_available': DETECTRON2_AVAILABLE,
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path) if self.model_path else False,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'threshold': self.detection_threshold,
            'training_input_size': self.training_input_size,
            'model_type': 'simplified_real_model' if self.model_loaded else 'simplified_placeholder',
            'save_mode': 'analyzed_images_only',
            'target_detections': {
                'front_vertical': self.target_front_vertical,
                'front_horizontal': self.target_front_horizontal
            },
            'minimum_detections': {
                'front_vertical': self.min_front_vertical,
                'front_horizontal': self.min_front_horizontal
            },
            'offset_calculation': f'{self.offset_cm}cm per side'
        }
    
    def test_model(self, test_image_path=None):
        """Test the SIMPLIFIED MODEL with a sample image"""
        try:
            if not test_image_path:
                # Use a recent captured image for testing
                captured_dir = config.UPLOAD_FOLDER
                if os.path.exists(captured_dir):
                    images = [f for f in os.listdir(captured_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        # Sort by modification time and get most recent
                        images.sort(key=lambda x: os.path.getmtime(os.path.join(captured_dir, x)), reverse=True)
                        test_image_path = os.path.join(captured_dir, images[0])
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
            
            print(f"🧪 Testing SIMPLIFIED MODEL with: {os.path.basename(test_image_path)}")
            
            # Run simplified analysis (will save only analyzed image)
            result = self.analyze_image(image_path=test_image_path)
            
            if result['success']:
                model_type = result.get('model_type', 'unknown')
                fv_count = result.get('front_vertical_count', 0)
                fh_count = result.get('front_horizontal_count', 0)
                print(f"✅ SIMPLIFIED MODEL test successful! (Model type: {model_type})")
                print(f"   Detected: {fv_count} vertical + {fh_count} horizontal rebars")
                print("   Only analyzed image saved (no duplicates)")
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'detections_found': result.get('num_detections', 0),
                    'front_vertical_count': fv_count,
                    'front_horizontal_count': fh_count,
                    'target_achieved': result.get('target_achieved', {}),
                    'model_type': model_type,
                    'analyzed_image_saved': result.get('analyzed_image_path'),
                    'save_mode': 'analyzed_only'
                }
            else:
                print(f"❌ SIMPLIFIED MODEL test failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ SIMPLIFIED MODEL test error: {str(e)}")
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }
    
    def debug_current_detection(self, image_data=None, image_path=None):
        """
        Debug method specifically for troubleshooting detection issues
        Use this when getting "No rebar structures detected" errors
        """
        try:
            print("🔧 DEBUGGING CURRENT DETECTION ISSUES...")
            print("=" * 50)
            
            # Handle input
            if image_data is not None:
                image = image_data.copy()
                source = "camera_frame"
            elif image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
                source = "file"
                if image is None:
                    print("❌ Could not load image file")
                    return
            else:
                print("❌ No image provided for debugging")
                return
            
            print(f"📸 Debugging image from: {source}")
            print(f"📐 Original shape: {image.shape}")
            
            # Resize if needed
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                image = cv2.resize(image, (480, 640))
                print(f"📐 Resized to: {image.shape}")
            
            if not self.model_loaded:
                print("❌ Model not loaded - cannot debug detection")
                return
            
            print(f"🎯 Current detection threshold: {self.detection_threshold}")
            
            # Test with multiple thresholds
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            
            for threshold in thresholds:
                print(f"\n🔍 Testing threshold: {threshold}")
                
                # Create config with this threshold
                test_cfg = self.cfg.clone()
                test_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
                test_predictor = DefaultPredictor(test_cfg)
                
                # Run detection
                outputs = test_predictor(image)
                instances = outputs["instances"].to("cpu")
                
                detection_count = len(instances)
                print(f"   Found: {detection_count} detections")
                
                if detection_count > 0:
                    scores = instances.scores.numpy()
                    classes = instances.pred_classes.numpy()
                    
                    # Count by class
                    class_counts = {name: 0 for name in self.class_names}
                    for class_id in classes:
                        if class_id < len(self.class_names):
                            class_counts[self.class_names[class_id]] += 1
                    
                    print(f"   Class breakdown:")
                    for class_name, count in class_counts.items():
                        print(f"     {class_name}: {count}")
                    
                    # Show top detections
                    if detection_count <= 5:
                        print(f"   All detections:")
                        for i in range(detection_count):
                            class_id = int(classes[i])
                            confidence = float(scores[i])
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"unknown_{class_id}"
                            print(f"     {i+1}: {class_name} ({confidence:.4f})")
            
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   1. If detections found at lower thresholds, use threshold 0.1 or 0.05")
            print(f"   2. If no detections at any threshold, check:")
            print(f"      - Image lighting conditions")
            print(f"      - Rebar visibility and clarity")
            print(f"      - Camera distance (optimal: 160-200cm)")
            print(f"      - Model training data compatibility")
            
        except Exception as e:
            print(f"❌ Debug detection error: {e}")
            traceback.print_exc()
