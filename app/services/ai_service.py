"""
AI Service for Rebar Detection and Analysis
Integrates Detectron2 Mask R-CNN model for rebar segmentation with REAL MODEL
Updated with 4-step intersection-based analysis
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
        """Run actual AI model analysis with 4-step rebar analysis"""
        try:
            print("🤖 Running REAL Detectron2 inference with 4-step analysis...")
            
            # STEP 1: Initial Instance Prediction
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # Check if any detections
            num_detections = len(instances)
            print(f"🎯 STEP 1: Found {num_detections} initial detections")
            
            if num_detections == 0:
                print("❌ No rebar structures detected by REAL MODEL")
                return {
                    'success': False,
                    'error': 'No rebar structures detected in image',
                    'no_detection': True
                }
            
            # Extract basic detection data
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
            
            # STEP 2: Get Intersection Points 
            print("🎯 STEP 2: Finding intersection points...")
            intersection_points = self._find_intersection_points(instances)
            
            if not intersection_points:
                print("⚠️ No intersections found, using basic detection analysis")
                # Fallback to basic dimension calculation
                dimensions = self._calculate_basic_dimensions_from_detections(detections, masks, image.shape)
                analyzed_image_path = self._create_basic_visualization(image, outputs, image_path)
            else:
                # STEP 3: Order Grid Points
                print("🎯 STEP 3: Ordering grid points...")
                ordered_points = self._order_grid_points(intersection_points, image.shape)
                
                # STEP 4: Create Quadrilaterals and Calculate Dimensions
                print("🎯 STEP 4: Creating quadrilaterals and calculating dimensions...")
                quadrilaterals, dimensions = self._create_quadrilaterals_and_dimensions(ordered_points, image.shape)
                
                # Create final visualization
                analyzed_image_path = self._create_final_visualization(image, outputs, quadrilaterals, ordered_points, image_path)
            
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
                'model_type': 'real_trained_model',
                'analysis_method': 'quadrilateral_intersection_analysis' if intersection_points else 'basic_detection_analysis'
            }
            
        except Exception as e:
            print(f"❌ REAL MODEL inference error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'REAL MODEL inference failed: {str(e)}'
            }
    
    def _find_intersection_points(self, instances):
        """STEP 2: Find intersections between front_horizontal and front_vertical"""
        try:
            pred_classes = instances.pred_classes.numpy()
            pred_masks = instances.pred_masks.numpy()
            
            # Find indices for each class
            fh_indices = [i for i, c in enumerate(pred_classes) if self.class_names[c] == 'front_horizontal']
            fv_indices = [i for i, c in enumerate(pred_classes) if self.class_names[c] == 'front_vertical']
            
            print(f"   Found {len(fh_indices)} front_horizontal and {len(fv_indices)} front_vertical instances")
            
            if not fh_indices or not fv_indices:
                print("   Missing required classes for intersection detection")
                return []
            
            # Create intersection mask
            intersection_mask = np.zeros_like(pred_masks[0], dtype=bool)
            
            for fh in fh_indices:
                for fv in fv_indices:
                    intersection_mask |= (pred_masks[fh] & pred_masks[fv])
            
            if not np.any(intersection_mask):
                print("   No intersections found")
                return []
            
            # Extract centers using contour analysis
            centers = []
            intersection_uint8 = (intersection_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(intersection_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
            
            print(f"   Extracted {len(centers)} intersection points")
            return centers
            
        except Exception as e:
            print(f"   Error finding intersections: {e}")
            return []

    def _order_grid_points(self, centers, image_shape):
        """STEP 3: Order intersection points by grid position (bottom-to-top, right-to-left)"""
        try:
            if not centers:
                return []
            
            H, W = image_shape[:2]
            row_eps = max(5, int(0.02 * H))  # 2% of image height tolerance
            
            # Sort by Y descending (bottom→top) first
            pts = sorted(centers, key=lambda p: p[1], reverse=True)
            
            # Group into rows by Y coordinate
            rows = []
            for (cx, cy) in pts:
                placed = False
                for row in rows:
                    if abs(cy - row['y_ref']) <= row_eps:
                        row['pts'].append((cx, cy))
                        # Update reference Y (running average)
                        row['y_ref'] = int((row['y_ref'] * (len(row['pts']) - 1) + cy) / len(row['pts']))
                        placed = True
                        break
                if not placed:
                    rows.append({'y_ref': cy, 'pts': [(cx, cy)]})
            
            # Sort rows bottom→top by y_ref
            rows.sort(key=lambda r: r['y_ref'], reverse=True)
            
            # Within each row: sort right→left by X descending
            ordered_points = []
            for row in rows:
                row['pts'].sort(key=lambda p: p[0], reverse=True)
                ordered_points.extend(row['pts'])
            
            print(f"   Ordered {len(ordered_points)} points into {len(rows)} rows")
            
            # Log ordered points for debugging
            for i, (cx, cy) in enumerate(ordered_points, 1):
                print(f"   Point {i}: (x={cx}, y={cy})")
            
            return ordered_points
            
        except Exception as e:
            print(f"   Error ordering points: {e}")
            return centers  # Return unordered as fallback

    def _create_quadrilaterals_and_dimensions(self, ordered_points, image_shape):
        """STEP 4: Create quadrilaterals and calculate precise dimensions"""
        try:
            if len(ordered_points) < 4:
                print("   Insufficient points for quadrilateral formation")
                return [], self._get_fallback_dimensions()
            
            H, W = image_shape[:2]
            row_eps = max(5, int(0.02 * H))
            
            # Re-group points into rows for quadrilateral formation
            rows = []
            for (cx, cy) in ordered_points:
                placed = False
                for row in rows:
                    if abs(cy - row['y_ref']) <= row_eps:
                        row['pts'].append((cx, cy))
                        row['y_ref'] = int((row['y_ref'] * (len(row['pts']) - 1) + cy) / len(row['pts']))
                        placed = True
                        break
                if not placed:
                    rows.append({'y_ref': cy, 'pts': [(cx, cy)]})
            
            # Sort rows bottom→top and points within rows right→left
            rows.sort(key=lambda r: r['y_ref'], reverse=True)
            for row in rows:
                row['pts'].sort(key=lambda p: p[0], reverse=True)
            
            # Build quadrilaterals
            quadrilaterals = []
            quad_count = 0
            
            for r in range(len(rows) - 1):  # Adjacent row pairs
                row1 = rows[r]['pts']      # Bottom row
                row2 = rows[r + 1]['pts']  # Top row
                min_len = min(len(row1), len(row2))
                
                for c in range(min_len - 1):  # Adjacent column pairs
                    # Form quadrilateral: bottom-right, bottom-left, top-left, top-right
                    p1 = row1[c]      # Bottom-right
                    p2 = row1[c + 1]  # Bottom-left  
                    p3 = row2[c + 1]  # Top-left
                    p4 = row2[c]      # Top-right
                    
                    quad = [p1, p2, p3, p4]
                    quadrilaterals.append(quad)
                    quad_count += 1
                    
                    print(f"   Quadrilateral {quad_count}: {p1} -> {p2} -> {p3} -> {p4}")
            
            if quad_count == 0:
                return quadrilaterals, self._get_fallback_dimensions()
            
            # Convert to real-world dimensions
            pixel_to_cm = self._calculate_pixel_to_cm_ratio(image_shape, ordered_points)
            
            # Calculate structure dimensions using overall bounds
            all_x = [p[0] for p in ordered_points]
            all_y = [p[1] for p in ordered_points]
            
            structure_width_px = max(all_x) - min(all_x)
            structure_height_px = max(all_y) - min(all_y)
            
            structure_width_cm = structure_width_px * pixel_to_cm
            structure_height_cm = structure_height_px * pixel_to_cm
            structure_depth_cm = 25  # Standard rebar spacing estimate
            
            volume_cm3 = structure_width_cm * structure_height_cm * structure_depth_cm
            
            dimensions = {
                'length': round(structure_width_cm, 1),
                'width': round(structure_depth_cm, 1), 
                'height': round(structure_height_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': f"{structure_width_cm:.0f}cm x {structure_depth_cm:.0f}cm x {structure_height_cm:.0f}cm = {volume_cm3:.0f}cm³",
                'method': 'quadrilateral_intersection_analysis',
                'quadrilateral_count': len(quadrilaterals),
                'pixel_to_cm_ratio': pixel_to_cm
            }
            
            print(f"   Created {len(quadrilaterals)} quadrilaterals")
            print(f"   Calculated dimensions: {dimensions['display']}")
            
            return quadrilaterals, dimensions
            
        except Exception as e:
            print(f"   Error creating quadrilaterals: {e}")
            traceback.print_exc()
            return [], self._get_fallback_dimensions()

    def _calculate_pixel_to_cm_ratio(self, image_shape, ordered_points):
        """Calculate pixel to cm conversion ratio based on typical rebar spacing"""
        try:
            if len(ordered_points) < 2:
                return 0.15  # Default fallback
            
            # Calculate average distance between consecutive points
            distances = []
            for i in range(len(ordered_points) - 1):
                p1 = ordered_points[i]
                p2 = ordered_points[i + 1]
                dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                distances.append(dist)
            
            if distances:
                avg_distance_px = np.mean(distances)
                # Assume average rebar spacing is 18cm
                estimated_spacing_cm = 18.0
                pixel_to_cm = estimated_spacing_cm / avg_distance_px
                
                # Clamp to reasonable values
                pixel_to_cm = max(0.05, min(0.5, pixel_to_cm))
                
                print(f"   Estimated pixel-to-cm ratio: {pixel_to_cm:.3f}")
                return pixel_to_cm
            else:
                return 0.15
                
        except Exception as e:
            print(f"   Error calculating pixel ratio: {e}")
            return 0.15

    def _calculate_basic_dimensions_from_detections(self, detections, masks, image_shape):
        """Fallback dimension calculation from basic detections"""
        try:
            print("Calculating dimensions from basic detections...")
            
            if not detections:
                return self._get_fallback_dimensions()
            
            # Find bounding box of all detections
            all_boxes = [d['bbox'] for d in detections]
            if not all_boxes:
                return self._get_fallback_dimensions()
            
            # Calculate overall bounds
            min_x = min(box[0] for box in all_boxes)
            min_y = min(box[1] for box in all_boxes)
            max_x = max(box[2] for box in all_boxes)
            max_y = max(box[3] for box in all_boxes)
            
            width_px = max_x - min_x
            height_px = max_y - min_y
            
            # Basic pixel to cm conversion
            pixel_to_cm = 0.15
            
            structure_width_cm = width_px * pixel_to_cm
            structure_height_cm = height_px * pixel_to_cm
            structure_depth_cm = 25
            
            volume_cm3 = structure_width_cm * structure_height_cm * structure_depth_cm
            
            dimensions = {
                'length': round(structure_width_cm, 1),
                'width': round(structure_depth_cm, 1),
                'height': round(structure_height_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': f"{structure_width_cm:.0f}cm x {structure_depth_cm:.0f}cm x {structure_height_cm:.0f}cm = {volume_cm3:.0f}cm³",
                'method': 'basic_detection_analysis'
            }
            
            print(f"   Basic dimensions: {dimensions['display']}")
            return dimensions
            
        except Exception as e:
            print(f"   Error in basic dimension calculation: {e}")
            return self._get_fallback_dimensions()

    def _get_fallback_dimensions(self):
        """Fallback dimensions when analysis fails"""
        return {
            'length': 25.4,
            'width': 25.4,
            'height': 200.0,
            'unit': 'cm',
            'volume': 101600,
            'display': '25cm x 25cm x 200cm = 101600cm³',
            'method': 'fallback_estimation'
        }

    def _create_final_visualization(self, image, outputs, quadrilaterals, ordered_points, original_path):
        """Create final visualization with quadrilateral masking"""
        try:
            print("Creating final rebar analysis visualization...")
            
            # Start with original image
            result_image = image.copy()
            
            # Create overlay for quadrilaterals
            overlay = result_image.copy()
            
            # Colors for quadrilaterals
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), 
                     (255, 255, 0), (128, 0, 255), (255, 128, 0), (0, 255, 128)]
            
            # Draw filled quadrilaterals
            for i, quad in enumerate(quadrilaterals):
                if len(quad) >= 4:
                    pts = np.array(quad, dtype=np.int32)
                    color = colors[i % len(colors)]
                    
                    # Fill polygon 
                    cv2.fillPoly(overlay, [pts], color=color)
                    
                    # Draw outline
                    cv2.polylines(result_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Add quadrilateral number
                    center_x = int(np.mean([p[0] for p in quad]))
                    center_y = int(np.mean([p[1] for p in quad]))
                    cv2.putText(overlay, f"{i+1}", (center_x, center_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Apply transparency
            alpha = 0.4
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            # Draw ordered intersection points
            for i, (cx, cy) in enumerate(ordered_points, 1):
                cv2.circle(result_image, (cx, cy), 6, (0, 0, 255), -1)
                cv2.circle(result_image, (cx, cy), 6, (255, 255, 255), 2)
                cv2.putText(result_image, f"{i}", (cx + 8, cy - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add analysis info
            info_text = f"Rebar Grid Analysis - {len(quadrilaterals)} sections detected"
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Save analyzed image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'rebar_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"Final analysis visualization saved: {filename}")
                return output_path
            else:
                print("Failed to save final visualization")
                return original_path
                
        except Exception as e:
            print(f"Final visualization error: {str(e)}")
            traceback.print_exc()
            return original_path

    def _create_basic_visualization(self, image, outputs, original_path):
        """Fallback visualization when intersection analysis fails"""
        try:
            print("Creating basic detection visualization...")
            
            # Use standard Detectron2 visualizer
            v = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=self.metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE
            )
            
            # Draw predictions
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = out.get_image()[:, :, ::-1]  # Convert back to BGR
            
            # Save analyzed image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'basic_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"Basic analysis visualization saved: {filename}")
                return output_path
            else:
                return original_path
                
        except Exception as e:
            print(f"Basic visualization error: {str(e)}")
            return original_path
    
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
            print("Creating placeholder visualization...")
            
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
                print(f"Placeholder visualization saved: {filename}")
                return output_path
            else:
                print("Failed to save placeholder visualization")
                return original_path
                
        except Exception as e:
            print(f"Placeholder visualization error: {str(e)}")
            return original_path
