"""
AI Service for Rebar Detection and Analysis
Complete implementation with retry mechanism, auto-save, and enhanced stability
"""

import os
import cv2
import numpy as np
from datetime import datetime
import json
import traceback
import gc

# Detectron2 imports
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
    """Complete AI service with retry mechanism, auto-save, and enhanced stability"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # Updated rebar classes
        self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        self.num_classes = 3
        
        # Lowered detection threshold for better Pi performance
        self.detection_threshold = 0.2
        
        # Training image size
        self.training_input_size = (480, 640)
        
        # Calibration points
        self.calibration_160cm = 0.2117
        self.calibration_200cm = 0.2822
        
        print("🤖 Initializing Complete AI Service with retry and auto-save...")
        print(f"   Classes: {self.class_names}")
        print(f"   Detection threshold: {self.detection_threshold}")
        print(f"   Training input size: {self.training_input_size[0]}x{self.training_input_size[1]}")
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            model_size = os.path.getsize(self.model_path)
            print(f"📁 Model file size: {model_size / 1024 / 1024:.1f} MB")
            
            print("🔄 Loading Detectron2 configuration...")
            
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold
            self.cfg.MODEL.DEVICE = "cpu"
            
            # Input format
            self.cfg.INPUT.MIN_SIZE_TRAIN = (640,)
            self.cfg.INPUT.MAX_SIZE_TRAIN = 640
            self.cfg.INPUT.MIN_SIZE_TEST = 640
            self.cfg.INPUT.MAX_SIZE_TEST = 640
            
            print("🔄 Creating predictor...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # Set up metadata
            self.metadata = MetadataCatalog.get("rebar_dataset_real")
            self.metadata.thing_classes = self.class_names
            self.metadata.thing_colors = [
                (128, 128, 128),  # back_horizontal - Gray
                (255, 0, 0),      # front_horizontal - Red  
                (0, 255, 0),      # front_vertical - Green
            ]
            
            self.model_loaded = True
            print("✅ AI Model loaded successfully!")
            
            # Test inference
            test_image = np.zeros((640, 480, 3), dtype=np.uint8)
            try:
                test_output = self.predictor(test_image)
                print("✅ Model inference test successful!")
            except Exception as e:
                print(f"⚠️  Model inference test failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading AI model: {str(e)}")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def _ensure_model_stability(self):
        """Ensure model is in stable state before inference"""
        try:
            if not self.model_loaded or not self.predictor:
                return False
            
            # Clear memory
            gc.collect()
            
            # Warm up the model
            try:
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                _ = self.predictor(test_image)
            except:
                pass
            
            return True
        except Exception as e:
            print(f"⚠️ Model stability check failed: {e}")
            return False

    def analyze_image_with_retries(self, image_path, max_retries=3):
        """Analyze image with retry mechanism for consistency"""
        print(f"🔍 Starting analysis with retry mechanism (max {max_retries} attempts)")
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                # Ensure model stability
                if not self._ensure_model_stability():
                    print(f"⚠️ Model stability check failed on attempt {attempt + 1}")
                    continue
                
                # Run analysis
                result = self.analyze_image(image_path)
                
                if result['success']:
                    print(f"✅ Analysis successful on attempt {attempt + 1}")
                    
                    # AUTO-SAVE TO GALLERY
                    self._auto_save_to_gallery(result, image_path)
                    
                    return result
                
                elif result.get('no_detection', False) and attempt < max_retries - 1:
                    print(f"⚠️ No detection on attempt {attempt + 1}, retrying...")
                    import time
                    time.sleep(0.5)
                    continue
                else:
                    print(f"❌ Analysis failed on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return result
                    
            except Exception as e:
                print(f"❌ Exception on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': f'All {max_retries} attempts failed: {str(e)}'
                    }
        
        return {
            'success': False,
            'error': f'All {max_retries} attempts failed'
        }

    def _auto_save_to_gallery(self, analysis_result, original_image_path):
        """Auto-save analysis results to gallery"""
        try:
            print("💾 Auto-saving results to gallery...")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'original_image': os.path.basename(original_image_path),
                'analyzed_image': os.path.basename(analysis_result.get('analyzed_image_path', '')),
                'dimensions': analysis_result.get('dimensions', {}),
                'cement_mixture': analysis_result.get('cement_mixture', {}),
                'detections': analysis_result.get('detections', []),
                'model_type': analysis_result.get('model_type', 'unknown'),
                'analysis_successful': True
            }
            
            metadata_filename = f'analysis_metadata_{timestamp}.json'
            metadata_path = os.path.join(config.UPLOAD_FOLDER, metadata_filename)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Analysis metadata saved: {metadata_filename}")
            
            analyzed_image_path = analysis_result.get('analyzed_image_path')
            if analyzed_image_path and os.path.exists(analyzed_image_path):
                print(f"✅ Analyzed image available for gallery: {os.path.basename(analyzed_image_path)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error auto-saving to gallery: {str(e)}")
            return False

    def get_pixel_to_cm_factor(self, distance_cm=None):
        """Get pixel-to-cm conversion factor based on distance"""
        try:
            if distance_cm is None or distance_cm <= 0:
                distance_cm = 180
                print(f"⚠️ Using default distance: {distance_cm}cm")
            
            distance_cm = max(100, min(300, distance_cm))
            
            if distance_cm <= 160:
                factor = self.calibration_160cm
            elif distance_cm >= 200:
                factor = self.calibration_200cm
            else:
                ratio = (distance_cm - 160) / (200 - 160)
                factor = self.calibration_160cm + ratio * (self.calibration_200cm - self.calibration_160cm)
            
            print(f"📏 Distance: {distance_cm}cm → Conversion factor: {factor:.4f} cm/px")
            return factor
            
        except Exception as e:
            print(f"⚠️ Error calculating conversion factor: {e}")
            return 0.25

    def analyze_image(self, image_path):
        """Analyze image for rebar detection"""
        try:
            print(f"🔍 Starting AI analysis of: {image_path}")
            
            if not os.path.exists(image_path):
                return {'success': False, 'error': f'Image file not found: {image_path}'}
            
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Failed to load image file'}
            
            print(f"📐 Image loaded: {image.shape} (H×W×C)")
            
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                print(f"⚙️ Resizing image from {width}x{height} to 480x640")
                image = cv2.resize(image, (480, 640))
            
            if self.model_loaded and DETECTRON2_AVAILABLE:
                return self._analyze_with_complete_real_model(image, image_path)
            else:
                print("⚠️ REAL MODEL not available, using placeholder")
                return self._analyze_placeholder(image, image_path)
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            traceback.print_exc()
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}

    def _analyze_with_complete_real_model(self, image, image_path):
        """Run complete real AI model analysis with enhanced stability"""
        try:
            print("🤖 Running COMPLETE REAL Detectron2 inference...")
            
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            num_detections = len(instances)
            print(f"🎯 REAL MODEL found {num_detections} detections")
            
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
            class_counts = {'front_horizontal': 0, 'front_vertical': 0, 'back_horizontal': 0}
            
            for i in range(num_detections):
                detection = {
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[classes[i]],
                    'confidence': float(scores[i]),
                    'bbox': boxes[i].tolist(),
                    'mask_area': float(np.sum(masks[i])),
                    'mask_shape': masks[i].shape
                }
                detections.append(detection)
                
                class_name = detection['class_name']
                if class_name in class_counts:
                    class_counts[class_name] += 1
                
                print(f"   Detection {i+1}: {detection['class_name']} ({detection['confidence']:.3f})")
            
            print(f"   Class summary: {class_counts}")
            
            # Check for required classes
            if class_counts['front_horizontal'] == 0 or class_counts['front_vertical'] == 0:
                missing_classes = []
                if class_counts['front_horizontal'] == 0:
                    missing_classes.append('front_horizontal')
                if class_counts['front_vertical'] == 0:
                    missing_classes.append('front_vertical')
                
                print(f"❌ Missing required classes: {missing_classes}")
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True,
                    'debug_info': {
                        'missing_classes': missing_classes,
                        'found_classes': class_counts
                    }
                }
            
            # Intersection analysis
            intersection_result = self._perform_intersection_analysis(instances, image)
            
            if not intersection_result['success']:
                print("❌ No rebar intersections found")
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True,
                    'debug_info': {
                        'intersection_failed': True,
                        'intersection_error': intersection_result.get('error'),
                        'class_counts': class_counts
                    }
                }
            
            # Calculate dimensions
            dimensions = self._calculate_real_dimensions_from_intersections(
                intersection_result, image.shape, image_path
            )
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture(dimensions)
            
            # Create enhanced visualization
            analyzed_image_path = self._create_enhanced_visualization_with_metadata(
                image, outputs, intersection_result, image_path, dimensions
            )
            
            analysis_result = {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,
                'original_image_path': image_path,
                'model_type': 'complete_real_trained_model',
                'intersection_analysis': intersection_result,
                'class_counts': class_counts
            }
            
            print("✅ COMPLETE analysis successful")
            return analysis_result
            
        except Exception as e:
            print(f"❌ REAL MODEL inference error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'REAL MODEL inference failed: {str(e)}'
            }

    def _perform_intersection_analysis(self, instances, image):
        """Perform complete intersection analysis"""
        try:
            print("🔍 Performing intersection analysis...")
            
            masks = instances.pred_masks
            classes = instances.pred_classes
            
            front_horizontal_idx = 1
            front_vertical_idx = 2
            
            front_horizontal_masks = masks[classes == front_horizontal_idx]
            front_vertical_masks = masks[classes == front_vertical_idx]
            
            print(f"   Found {len(front_horizontal_masks)} front_horizontal masks")
            print(f"   Found {len(front_vertical_masks)} front_vertical masks")
            
            if front_horizontal_masks.shape[0] > 0:
                combined_front_horizontal = torch.any(front_horizontal_masks, dim=0)
            else:
                combined_front_horizontal = torch.zeros(masks.shape[1:], dtype=torch.bool)
                
            if front_vertical_masks.shape[0] > 0:
                combined_front_vertical = torch.any(front_vertical_masks, dim=0)
            else:
                combined_front_vertical = torch.zeros(masks.shape[1:], dtype=torch.bool)
            
            intersection_mask = combined_front_horizontal & combined_front_vertical
            
            if not torch.any(intersection_mask):
                print("❌ No intersections found")
                return {'success': False, 'error': 'No intersections found'}
            
            print(f"✅ Intersection mask created with {torch.sum(intersection_mask)} pixels")
            
            intersection_uint8 = intersection_mask.numpy().astype(np.uint8)
            num_labels, labels_im = cv2.connectedComponents(intersection_uint8)
            
            centroids = []
            for label in range(1, num_labels):
                mask = labels_im == label
                ys, xs = np.where(mask)
                if len(xs) > 0:
                    cx = int(xs.mean())
                    cy = int(ys.mean())
                    centroids.append((cx, cy, label))
            
            print(f"   Found {len(centroids)} intersection centroids")
            
            if len(centroids) == 0:
                return {'success': False, 'error': 'No intersection centroids found'}
            
            # Sort and group centroids
            centroids_sorted_y = sorted(centroids, key=lambda c: -c[1])
            
            y_threshold = 10
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
            
            for i in range(len(rows)):
                rows[i] = sorted(rows[i], key=lambda c: -c[0])
            
            print(f"   Organized into {len(rows)} rows")
            
            # Create polygons
            polygons = []
            all_poly_points = []
            mask_areas = []
            
            for i in range(len(rows) - 1):
                bottom_row = rows[i]
                upper_row = rows[i + 1]
                
                n = min(len(bottom_row), len(upper_row)) - 1
                
                for j in range(n):
                    p1 = bottom_row[j][:2]
                    p2 = bottom_row[j + 1][:2]
                    p3 = upper_row[j + 1][:2]
                    p4 = upper_row[j][:2]
                    
                    poly = np.array([p1, p2, p3, p4], dtype=np.int32)
                    polygons.append(poly)
                    all_poly_points.extend(poly.tolist())
                    
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 255)
                    area = cv2.countNonZero(mask)
                    mask_areas.append(area)
            
            if len(polygons) == 0:
                return {'success': False, 'error': 'No polygons created'}
            
            all_poly_points = np.array(all_poly_points)
            x_min, y_min = all_poly_points.min(axis=0)
            x_max, y_max = all_poly_points.max(axis=0)
            total_width_px = x_max - x_min
            total_height_px = y_max - y_min
            total_area_px = sum(mask_areas)
            
            print(f"✅ Intersection analysis complete:")
            print(f"   Total width: {total_width_px} pixels")
            print(f"   Total height: {total_height_px} pixels")
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
        """Calculate real dimensions from intersection analysis"""
        try:
            print("📏 Calculating real dimensions...")
            
            width_px = intersection_result['total_width_px']
            height_px = intersection_result['total_height_px']
            area_px = intersection_result['total_area_px']
            
            print(f"   Pixel dimensions: {width_px}px × {height_px}px")
            
            distance_cm = self._get_distance_from_context(image_path)
            pixel_to_cm = self.get_pixel_to_cm_factor(distance_cm)
            
            width_cm = width_px * pixel_to_cm
            height_cm = height_px * pixel_to_cm
            length_cm = width_cm  # Square column
            
            volume_cm3 = length_cm * width_cm * height_cm
            
            length_cm = round(length_cm, 1)
            width_cm = round(width_cm, 1)
            height_cm = round(height_cm, 1)
            volume_cm3 = round(volume_cm3, 1)
            
            display_string = f"{length_cm}cm x {width_cm}cm x {height_cm}cm = {volume_cm3}cm³"
            
            print(f"✅ Real dimensions calculated:")
            print(f"   Display: {display_string}")
            
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
            print(f"❌ Error calculating dimensions: {str(e)}")
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
        """Extract distance from filename or use default"""
        try:
            filename = os.path.basename(image_path)
            import re
            distance_match = re.search(r'(\d+)cm', filename)
            if distance_match:
                distance = int(distance_match.group(1))
                if 100 <= distance <= 300:
                    print(f"📏 Extracted distance from filename: {distance}cm")
                    return distance
            
            print(f"📏 Using default distance: 180cm")
            return 180
            
        except Exception as e:
            print(f"⚠️ Error getting distance: {e}")
            return 180

    def _create_enhanced_visualization_with_metadata(self, image, outputs, intersection_result, original_path, dimensions):
        """Create enhanced visualization with metadata"""
        try:
            print("🎨 Creating enhanced visualization...")
            
            result_image = self._create_complete_visualization_base(image, outputs, intersection_result)
            
            # Add dimension overlay
            if dimensions:
                height, width = result_image.shape[:2]
                
                # Background for text
                overlay = result_image.copy()
                cv2.rectangle(overlay, (10, height-100), (width-10, height-10), (0, 0, 0), -1)
                result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
                
                # Add text
                dimension_text = dimensions.get('display', 'N/A')
                cv2.putText(result_image, f"Dimensions: {dimension_text}", 
                           (20, height-70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.putText(result_image, "Model: Real Detectron2 Analysis", 
                           (20, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(result_image, timestamp_text, 
                           (20, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save enhanced image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'rebar_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ Enhanced visualization saved: {filename}")
                return output_path
            else:
                print("❌ Failed to save enhanced visualization")
                return original_path
                
        except Exception as e:
            print(f"❌ Enhanced visualization error: {str(e)}")
            return original_path

    def _create_complete_visualization_base(self, image, outputs, intersection_result):
        """Base visualization method"""
        result_image = image.copy()
        
        # Draw detections
        instances = outputs["instances"].to("cpu")
        
        v = Visualizer(
            image[:, :, ::-1],
            metadata=self.metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        
        out = v.draw_instance_predictions(instances)
        result_image = out.get_image()[:, :, ::-1]
        
        # Add intersection polygons
        if intersection_result['success']:
            polygons = intersection_result['polygons']
            
            overlay = result_image.copy()
            alpha = 0.4
            
            for poly in polygons:
                cv2.fillPoly(overlay, [poly], (255, 0, 0))
            
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            for poly in polygons:
                cv2.polylines(result_image, [poly], True, (255, 0, 0), 2)
            
            # Add bounding box
            all_points = []
            for poly in polygons:
                all_points.extend(poly.tolist())
            
            if len(all_points) > 0:
                all_points = np.array(all_points)
                x_min, y_min = all_points.min(axis=0)
                x_max, y_max = all_points.max(axis=0)
                
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
        
        return result_image

    def _analyze_placeholder(self, image, image_path):
        """Placeholder analysis for fallback"""
        print("📝 Using placeholder analysis...")
        
        import time
        time.sleep(2)
        
        analyzed_image_path = self._create_placeholder_visualization(image, image_path)
        
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
                {'class_name': 'front_vertical', 'confidence': 0.85, 'bbox': [100, 50, 200, 300]},
                {'class_name': 'front_horizontal', 'confidence': 0.78, 'bbox': [80, 280, 220, 320]}
            ],
            'num_detections': 2,
            'dimensions': dimensions,
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_image_path,
            'original_image_path': image_path,
            'model_type': 'placeholder'
        }

    def _create_placeholder_visualization(self, image, original_path):
        """Create placeholder visualization"""
        try:
            result_image = image.copy()
            
            overlay = result_image.copy()
            cv2.rectangle(overlay, (100, 50), (200, 300), (0, 255, 0), -1)
            cv2.rectangle(overlay, (80, 280), (220, 320), (0, 255, 0), -1)
            
            alpha = 0.3
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            cv2.rectangle(result_image, (100, 50), (200, 300), (0, 255, 0), 3)
            cv2.rectangle(result_image, (80, 280), (220, 320), (255, 0, 0), 3)
            
            cv2.putText(result_image, 'Front Vertical (85%)', (100, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, 'Front Horizontal (78%)', (80, 275), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'placeholder_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ Placeholder visualization saved: {filename}")
                return output_path
            else:
                return original_path
                
        except Exception as e:
            print(f"❌ Placeholder visualization error: {str(e)}")
            return original_path

    def _calculate_cement_mixture(self, dimensions):
        """Calculate cement mixture ratios based on volume"""
        print("🧮 Calculating cement mixture...")
        
        volume_cm3 = dimensions.get('volume', 0)
        volume_m3 = volume_cm3 / 1000000
        
        cement_ratio = 1
        sand_ratio = 2
        aggregate_ratio = 3
        
        concrete_volume_factor = 1.5
        total_concrete_volume = volume_m3 * concrete_volume_factor
        
        total_parts = cement_ratio + sand_ratio + aggregate_ratio
        cement_volume = total_concrete_volume * (cement_ratio / total_parts)
        sand_volume = total_concrete_volume * (sand_ratio / total_parts)
        aggregate_volume = total_concrete_volume * (aggregate_ratio / total_parts)
        
        cement_bags = cement_volume / 0.035
        
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

    def debug_analyze_image(self, image_path):
        """Debug version with detailed logging"""
        try:
            print("=" * 80)
            print(f"🔍 DEBUG: Starting analysis of: {image_path}")
            print("=" * 80)
            
            if not os.path.exists(image_path):
                return {'success': False, 'error': f'Image file not found: {image_path}'}
            
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': 'Failed to load image file'}
            
            print(f"✅ DEBUG: Image loaded successfully")
            print(f"   Original size: {image.shape}")
            
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                print(f"🔄 DEBUG: Resizing from {width}x{height} to 480x640")
                image = cv2.resize(image, (480, 640))
            
            print(f"   Final size: {image.shape}")
            print(f"🤖 DEBUG: Model loaded: {self.model_loaded}")
            print(f"   Detectron2 available: {DETECTRON2_AVAILABLE}")
            
            if not self.model_loaded or not DETECTRON2_AVAILABLE:
                print("⚠️ DEBUG: Using placeholder analysis")
                return self._analyze_placeholder(image, image_path)
            
            print("🔄 DEBUG: Running model inference...")
            try:
                outputs = self.predictor(image)
                instances = outputs["instances"].to("cpu")
                num_detections = len(instances)
                
                print(f"✅ DEBUG: Inference successful")
                print(f"   Total detections: {num_detections}")
                
                if num_detections == 0:
                    return {
                        'success': False,
                        'error': 'No rebar structures detected in image',
                        'no_detection': True,
                        'debug_info': {
                            'inference_successful': True,
                            'detections_found': 0,
                            'detection_threshold': self.detection_threshold
                        }
                    }
                
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
                
                for class_name, confidences in class_counts.items():
                    print(f"   {class_name}: {len(confidences)} detections (avg: {np.mean(confidences):.3f})")
                
                front_horizontal_count = len([c for c in classes if self.class_names[c] == 'front_horizontal'])
                front_vertical_count = len([c for c in classes if self.class_names[c] == 'front_vertical'])
                
                print(f"🎯 DEBUG: Required classes found:")
                print(f"   front_horizontal: {front_horizontal_count}")
                print(f"   front_vertical: {front_vertical_count}")
                
                if front_horizontal_count == 0 or front_vertical_count == 0:
                    print("❌ DEBUG: Missing required classes")
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
                
                print("🔄 DEBUG: Starting intersection analysis...")
                intersection_result = self._perform_intersection_analysis(instances, image)
                
                print(f"✅ DEBUG: Intersection analysis: {'SUCCESS' if intersection_result['success'] else 'FAILED'}")
                if not intersection_result['success']:
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
                
                print("✅ DEBUG: Analysis completed successfully!")
                print("=" * 80)
                
                return self._analyze_with_complete_real_model(image, image_path)
                
            except Exception as inference_error:
                print(f"❌ DEBUG: Model inference failed: {str(inference_error)}")
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
            traceback.print_exc()
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}

    def test_with_actual_image(self, image_path=None):
        """Test the model with an actual captured image"""
        try:
            if not image_path:
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
        """Test the model with a sample image"""
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
                    return {'success': False, 'error': 'Captured images directory not found'}
            
            print(f"🧪 Testing REAL MODEL with: {test_image_path}")
            
            result = self.analyze_image(test_image_path)
            
            if result['success']:
                model_type = result.get('model_type', 'unknown')
                intersection_success = result.get('intersection_analysis', {}).get('success', False)
                
                print(f"✅ REAL MODEL test successful! (Model type: {model_type})")
                
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
                return result
                
        except Exception as e:
            print(f"❌ REAL MODEL test error: {str(e)}")
            return {'success': False, 'error': f'Test failed: {str(e)}'}
