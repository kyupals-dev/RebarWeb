"""
AI Service for Rebar Detection and Analysis
Integrates Detectron2 Mask R-CNN model for rebar segmentation with REAL MODEL
MODIFIED: Only saves analyzed images with AI overlays (no original duplicates)
UPDATED: Added simplified analysis with metadata support for gallery modal
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
        
        # NEW: Simplified analysis configuration
        self.offset_cm = 2.0  # Offset per side in cm
        self.target_front_vertical = 2  # Target: 2 vertical rebars
        self.target_front_horizontal = 11  # Target: 11 horizontal rebars
        
        print("🤖 Initializing AI Service with REAL TRAINED MODEL...")
        print(f"   Classes: {self.class_names}")
        print(f"   Detection threshold: {self.detection_threshold}")
        print(f"   Training input size: {self.training_input_size[0]}x{self.training_input_size[1]}")
        print(f"   Simplified targets: {self.target_front_vertical}V + {self.target_front_horizontal}H")
        print(f"   Offset applied: {self.offset_cm}cm per side")
        print("   📝 MODIFIED: Only saves analyzed images (no originals)")
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
    
    def analyze_image(self, image_data=None, image_path=None):
        """
        Analyze image for rebar detection using REAL TRAINED MODEL
        MODIFIED: Only saves analyzed images with AI overlays
        
        Args:
            image_data (numpy.ndarray): Direct frame data from camera (preferred)
            image_path (str): Path to existing image file (fallback only)
            
        Returns:
            dict: Analysis results with only analyzed_image_path
        """
        try:
            print(f"🔍 Starting REAL AI analysis (analyzed image only mode)...")
            
            # Handle different input types
            if image_data is not None:
                print("📸 Using direct frame data from camera (no original saved)")
                image = image_data.copy()
                original_source = "camera_frame"
            elif image_path and os.path.exists(image_path):
                print(f"📁 Loading image from: {image_path} (fallback mode)")
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
                result = self._analyze_with_real_model(image)
            else:
                print("⚠️  REAL MODEL not available, using placeholder")
                result = self._analyze_placeholder(image)
            
            # Ensure we return only the analyzed image path
            if result['success'] and 'analyzed_image_path' in result:
                filename = os.path.basename(result['analyzed_image_path'])
                print(f"✅ Analysis complete. ONLY analyzed image saved: {filename}")
            
            return result
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _analyze_with_real_model(self, image):
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
            
            # Separate detections by class for simplified analysis
            verticals = [d for d in detections if d['class_name'] == 'front_vertical']
            horizontals = [d for d in detections if d['class_name'] == 'front_horizontal']
            
            # Calculate intersections (simplified)
            intersections = self._calculate_intersections(verticals, horizontals)
            
            # Create visualization with REAL MODEL results (ONLY FILE SAVED)
            analyzed_image_path = self._create_simplified_visualization(image, verticals, horizontals, intersections)
            
            if not analyzed_image_path:
                return {
                    'success': False,
                    'error': 'Failed to create analyzed image visualization'
                }
            
            # Calculate dimensions from REAL MODEL detections
            dimensions = self._calculate_simplified_dimensions(verticals, horizontals, image.shape)
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture_simplified(dimensions)
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,  # ONLY image saved
                'model_type': 'simplified_real_model'
            }
            
        except Exception as e:
            print(f"❌ REAL MODEL inference error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'REAL MODEL inference failed: {str(e)}'
            }
    
    def _calculate_intersections(self, verticals, horizontals):
        """Calculate intersections between vertical and horizontal rebars"""
        intersections = []
        
        for i, vertical in enumerate(verticals):
            v_bbox = vertical['bbox']
            for j, horizontal in enumerate(horizontals):
                h_bbox = horizontal['bbox']
                
                # Check if bounding boxes intersect
                if self._boxes_intersect(v_bbox, h_bbox):
                    intersections.append({
                        'vertical_id': i,
                        'horizontal_id': j,
                        'vertical_class': vertical['class_name'],
                        'horizontal_class': horizontal['class_name'],
                        'intersection_area': self._calculate_intersection_area(v_bbox, h_bbox)
                    })
        
        return intersections
    
    def _boxes_intersect(self, bbox1, bbox2):
        """Check if two bounding boxes intersect"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _calculate_intersection_area(self, bbox1, bbox2):
        """Calculate intersection area between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        # Check if there's a valid intersection
        if x1_int < x2_int and y1_int < y2_int:
            return (x2_int - x1_int) * (y2_int - y1_int)
        
        return 0
    
    def _create_simplified_visualization(self, image, verticals, horizontals, intersections):
        """Create SIMPLIFIED visualization - ONLY method that saves images WITH METADATA"""
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
            summary_text = f"SIMPLIFIED: {len(verticals)}V + {len(horizontals)}H (Target: {self.target_front_vertical}V + {self.target_front_horizontal}H)"
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
                    
                    # NEW: Save metadata for gallery modal
                    self._save_analysis_metadata(filename, verticals, horizontals, intersections)
                    
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
    
    def _save_analysis_metadata(self, filename, verticals, horizontals, intersections):
        """Save analysis metadata for gallery modal display"""
        try:
            from app.services.image_service import ImageService
            
            # Calculate dimensions for metadata
            dimensions = self._calculate_simplified_dimensions(verticals, horizontals, (640, 480, 3))
            
            # Calculate cement mixture for metadata
            mixture = self._calculate_cement_mixture_simplified(dimensions)
            
            # Create metadata structure
            metadata = {
                'analysis_date': datetime.now().isoformat(),
                'image_filename': filename,
                'analysis_type': 'rebar_detection_simplified',
                'dimensions': {
                    'length': float(dimensions['length']),
                    'width': float(dimensions['width']),
                    'height': float(dimensions['height']),
                    'unit': dimensions['unit'],
                    'volume': float(dimensions['volume']),
                    'display': dimensions['display'],
                    'offset_applied': dimensions.get('offset_applied', f'{self.offset_cm}cm per side'),
                    'method': dimensions.get('method', 'simplified_analysis')
                },
                'cement_mixture': {
                    'cement_ratio': mixture['cement_ratio'],
                    'sand_ratio': mixture['sand_ratio'],
                    'aggregate_ratio': mixture['aggregate_ratio'],
                    'ratio_string': mixture['ratio_string'],
                    'cement_bags': float(mixture['cement_bags']),
                    'sand_volume_m3': float(mixture['sand_volume_m3']),
                    'aggregate_volume_m3': float(mixture['aggregate_volume_m3']),
                    'total_concrete_volume_m3': float(mixture['total_concrete_volume_m3'])
                },
                'detections': {
                    'count': len(verticals) + len(horizontals),
                    'front_vertical_count': len(verticals),
                    'front_horizontal_count': len(horizontals),
                    'intersection_count': len(intersections),
                    'target_achieved': {
                        'vertical': len(verticals) >= self.target_front_vertical,
                        'horizontal': len(horizontals) >= self.target_front_horizontal
                    },
                    'confidence_scores': {
                        'vertical_avg': float(sum(v['confidence'] for v in verticals) / len(verticals)) if verticals else 0.0,
                        'horizontal_avg': float(sum(h['confidence'] for h in horizontals) / len(horizontals)) if horizontals else 0.0
                    }
                },
                'model_info': {
                    'model_type': 'simplified_real_model' if self.model_loaded else 'simplified_placeholder',
                    'detection_threshold': float(self.detection_threshold),
                    'classes_detected': list(set([v['class_name'] for v in verticals] + [h['class_name'] for h in horizontals])),
                    'placeholder_mode': not self.model_loaded
                }
            }
            
            # Save metadata using image service
            image_service = ImageService()
            metadata_saved = image_service._save_image_metadata(filename, metadata)
            
            if metadata_saved:
                print(f"📖 Analysis metadata saved for gallery: {filename}")
            else:
                print(f"⚠️  Failed to save metadata for: {filename}")
                
        except Exception as e:
            print(f"❌ Error saving analysis metadata: {e}")
    
    def _calculate_simplified_dimensions(self, verticals, horizontals, image_shape):
        """Calculate simplified dimensions from detected rebars with offset"""
        try:
            print("📏 Calculating simplified dimensions with offset...")
            
            height, width, channels = image_shape
            
            # Basic pixel to cm conversion (this should be calibrated)
            pixel_to_cm = 0.05  # Rough estimate - needs proper calibration
            
            # Calculate length from vertical rebars (max span)
            length_cm = 25.0  # Default
            if len(verticals) >= 2:
                v_x_coords = [(v['bbox'][0] + v['bbox'][2]) / 2 for v in verticals]
                length_px = max(v_x_coords) - min(v_x_coords)
                length_cm = max(length_px * pixel_to_cm, 20.0)
            
            # Calculate width from horizontal rebars (max span)
            width_cm = 25.0  # Default
            if len(horizontals) >= 2:
                h_y_coords = [(h['bbox'][1] + h['bbox'][3]) / 2 for h in horizontals]
                width_px = max(h_y_coords) - min(h_y_coords)
                width_cm = max(width_px * pixel_to_cm, 20.0)
            
            # Apply offset (add to both length and width)
            length_with_offset = length_cm + (2 * self.offset_cm)
            width_with_offset = width_cm + (2 * self.offset_cm)
            
            # Height calculation (estimated from detections)
            height_cm = length_with_offset  # Assume square column
            
            # Calculate volume
            volume_cm3 = length_with_offset * width_with_offset * height_cm
            
            # Create display string
            display_string = f"{length_with_offset:.1f}cm x {width_with_offset:.1f}cm x {height_cm:.1f}cm = {volume_cm3:.0f}cm³"
            
            print(f"   Calculated: {display_string} (with {self.offset_cm}cm offset)")
            
            return {
                'length': round(length_with_offset, 1),
                'width': round(width_with_offset, 1), 
                'height': round(height_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': display_string,
                'method': 'simplified_detection_with_offset',
                'offset_applied': f'{self.offset_cm}cm per side',
                'original_length': round(length_cm, 1),
                'original_width': round(width_cm, 1)
            }
            
        except Exception as e:
            print(f"❌ Error calculating simplified dimensions: {str(e)}")
            # Return safe default with offset
            default_size = 25.0 + (2 * self.offset_cm)
            return {
                'length': default_size,
                'width': default_size,
                'height': default_size,
                'unit': 'cm',
                'volume': round(default_size ** 3, 1),
                'display': f'{default_size:.1f}cm x {default_size:.1f}cm x {default_size:.1f}cm = {default_size ** 3:.0f}cm³',
                'method': 'fallback_with_offset',
                'offset_applied': f'{self.offset_cm}cm per side'
            }
    
    def _calculate_cement_mixture_simplified(self, dimensions):
        """Calculate cement mixture ratios based on simplified volume"""
        print("🧮 Calculating simplified cement mixture...")
        
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
            'calculation_method': 'simplified_philippine_mix'
        }
    
    def _analyze_placeholder(self, image):
        """Generate placeholder analysis results (fallback only)"""
        print("📝 Using simplified placeholder AI analysis (REAL MODEL not available)...")
        
        # Simulate some processing time
        import time
        time.sleep(2)
        
        # Create simplified placeholder visualization (ONLY FILE SAVED)
        analyzed_image_path = self._create_placeholder_visualization_simplified(image)
        
        if not analyzed_image_path:
            return {
                'success': False,
                'error': 'Failed to create placeholder visualization'
            }
        
        # Placeholder dimensions with offset
        dimensions = self._calculate_simplified_dimensions([], [], image.shape)
        
        mixture = self._calculate_cement_mixture_simplified(dimensions)
        
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
                    'confidence': 0.75 + i*0.01,
                    'bbox': [140, 120 + i*25, 330, 130 + i*25]
                } for i in range(11)
            ],
            'num_detections': 13,
            'dimensions': dimensions,
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_image_path,  # ONLY saved image
            'model_type': 'simplified_placeholder'
        }
    
    def _create_placeholder_visualization_simplified(self, image):
        """Create SIMPLIFIED placeholder visualization - ONLY method that saves placeholder images WITH METADATA"""
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
            cv2.putText(result_image, f'SIMPLIFIED PLACEHOLDER: {self.target_front_vertical}V + {self.target_front_horizontal}H', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, f'Intersections: {self.target_front_vertical * self.target_front_horizontal} ({self.target_front_vertical}V x {self.target_front_horizontal}H)', (10, 60), 
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
                print(f"   🎯 Contains: {self.target_front_vertical}V + {self.target_front_horizontal}H placeholder with {self.offset_cm}cm offset")
                
                # NEW: Save placeholder metadata for gallery modal
                self._save_placeholder_metadata(filename)
                
                return output_path
            else:
                print("❌ Failed to save SIMPLIFIED placeholder analyzed image")
                return None
                
        except Exception as e:
            print(f"❌ SIMPLIFIED placeholder visualization error: {str(e)}")
            return None
    
    def _save_placeholder_metadata(self, filename):
        """Save placeholder metadata for gallery modal display"""
        try:
            from app.services.image_service import ImageService
            
            # Calculate placeholder dimensions with offset
            base_size = 25.0
            size_with_offset = base_size + (2 * self.offset_cm)
            volume = size_with_offset ** 3
            
            # Create placeholder metadata
            metadata = {
                'analysis_date': datetime.now().isoformat(),
                'image_filename': filename,
                'analysis_type': 'rebar_detection_placeholder',
                'dimensions': {
                    'length': size_with_offset,
                    'width': size_with_offset,
                    'height': size_with_offset,
                    'unit': 'cm',
                    'volume': volume,
                    'display': f'{size_with_offset:.1f}cm x {size_with_offset:.1f}cm x {size_with_offset:.1f}cm = {volume:.0f}cm³',
                    'offset_applied': f'{self.offset_cm}cm per side',
                    'method': 'placeholder_calculation'
                },
                'cement_mixture': {
                    'cement_ratio': 1,
                    'sand_ratio': 2,
                    'aggregate_ratio': 3,
                    'ratio_string': '1 Cement : 2 Sand : 3 Aggregate',
                    'cement_bags': 1.05,
                    'sand_volume_m3': 0.0001,
                    'aggregate_volume_m3': 0.00015,
                    'total_concrete_volume_m3': 0.00037
                },
                'detections': {
                    'count': self.target_front_vertical + self.target_front_horizontal,
                    'front_vertical_count': self.target_front_vertical,
                    'front_horizontal_count': self.target_front_horizontal,
                    'intersection_count': self.target_front_vertical * self.target_front_horizontal,
                    'target_achieved': {
                        'vertical': True,
                        'horizontal': True
                    },
                    'confidence_scores': {
                        'vertical_avg': 0.835,  # (0.85 + 0.82) / 2
                        'horizontal_avg': 0.785   # Average of placeholder scores
                    }
                },
                'model_info': {
                    'model_type': 'simplified_placeholder',
                    'detection_threshold': float(self.detection_threshold),
                    'classes_detected': ['front_vertical', 'front_horizontal'],
                    'placeholder_mode': True
                }
            }
            
            # Save metadata using image service
            image_service = ImageService()
            metadata_saved = image_service._save_image_metadata(filename, metadata)
            
            if metadata_saved:
                print(f"📖 Placeholder metadata saved for gallery: {filename}")
            else:
                print(f"⚠️  Failed to save placeholder metadata for: {filename}")
                
        except Exception as e:
            print(f"❌ Error saving placeholder metadata: {e}")
    
    def _create_real_model_visualization(self, image, outputs):
        """Create visualization with REAL MODEL overlays - ONLY method that saves images"""
        try:
            print("🎨 Creating REAL MODEL analysis visualization (ONLY FILE SAVED)...")
            
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
                for i, (mask, class_id) in enumerate(zip(masks, classes)):
                    # Create colored mask - transparent green
                    colored_mask = np.zeros_like(image)
                    colored_mask[mask] = [0, 255, 0]  # Green color
                    
                    # Apply transparent overlay (30% opacity)
                    alpha = 0.3
                    result_image = cv2.addWeighted(result_image, 1, colored_mask, alpha, 0)
                
                # Add dimension annotations
                self._add_dimension_annotations(result_image, instances)
            
            # Generate output filename with clear naming for analyzed image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analyzed_rebar_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image (THIS IS THE ONLY FILE SAVED)
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                # Verify saved image
                file_size = os.path.getsize(output_path)
                saved_img = cv2.imread(output_path)
                if saved_img is not None:
                    saved_height, saved_width = saved_img.shape[:2]
                    print(f"✅ ANALYZED IMAGE SAVED (ONLY COPY):")
                    print(f"   📁 File: {filename}")
                    print(f"   📐 Dimensions: {saved_width}x{saved_height}")
                    print(f"   💾 Size: {file_size / 1024:.1f} KB")
                    print(f"   🎯 Contains: AI overlays + rebar detection")
                    return output_path
                else:
                    print("❌ Could not verify saved analyzed image")
                    return None
            else:
                print("❌ Failed to save ANALYZED IMAGE")
                return None
                
        except Exception as e:
            print(f"❌ REAL MODEL visualization error: {str(e)}")
            traceback.print_exc()
            return None
    
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
        """Calculate physical dimensions from REAL MODEL detections"""
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
            
            height, width, channels = image_shape
            
            # Pixel to cm conversion factor (calibrated for optimal distance)
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
    
    def _create_placeholder_visualization(self, image):
        """Create placeholder visualization - ONLY method that saves placeholder images"""
        try:
            print("🎨 Creating placeholder visualization (ONLY FILE SAVED)...")
            
            # Copy original image
            result_image = image.copy()
            
            # Draw simple bounding boxes as placeholder with transparent green overlay
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
            
            # Generate output filename for placeholder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'analyzed_placeholder_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            # Save analyzed image (THIS IS THE ONLY FILE SAVED)
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                file_size = os.path.getsize(output_path)
                print(f"✅ PLACEHOLDER ANALYZED IMAGE SAVED (ONLY COPY):")
                print(f"   📁 File: {filename}")
                print(f"   💾 Size: {file_size / 1024:.1f} KB")
                print(f"   🎯 Contains: Placeholder overlays")
                return output_path
            else:
                print("❌ Failed to save placeholder analyzed image")
                return None
                
        except Exception as e:
            print(f"❌ Placeholder visualization error: {str(e)}")
            return None
    
    def _calculate_cement_mixture(self, dimensions):
        """Calculate cement mixture ratios based on volume"""
        print("🧮 Calculating cement mixture...")
        
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
            'model_type': 'simplified_real_model' if self.model_loaded else 'simplified_placeholder',
            'save_mode': 'analyzed_images_only',  # Status indicator
            'offset_cm': self.offset_cm,
            'targets': {
                'front_vertical': self.target_front_vertical,
                'front_horizontal': self.target_front_horizontal
            }
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
            
            print(f"🧪 Testing SIMPLIFIED MODEL with: {test_image_path}")
            
            # Run analysis (will save only analyzed image)
            result = self.analyze_image(image_path=test_image_path)
            
            if result['success']:
                model_type = result.get('model_type', 'unknown')
                print(f"✅ SIMPLIFIED MODEL test successful! (Model type: {model_type})")
                print("   Only analyzed image saved (no duplicates)")
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'detections_found': result.get('num_detections', 0),
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
