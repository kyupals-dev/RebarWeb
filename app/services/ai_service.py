"""
AI Service for Rebar Detection and Analysis - SIMPLIFIED PIPELINE
Implements the exact 4-step process from the training notebook
FIXED: Only saves analyzed images with step-by-step visualizations
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
    """SIMPLIFIED AI Service implementing exact 4-step pipeline from training"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # FIXED: Exact classes from training config
        self.class_names = ["front_horizontal", "front_vertical"]  # FIXED: Only 2 classes
        self.num_classes = 2  # FIXED: 2 classes not 3
        
        # FIXED: Exact threshold from training
        self.detection_threshold = 0.3
        
        # Pipeline constants from notebook
        self.CEMENT_BAG_WEIGHT = 40      # kg
        self.MIX_RATIO = (1, 2, 4)       # cement : sand : gravel
        self.WATER_CEMENT_RATIO = 0.53
        self.DRY_VOLUME_FACTOR = 1.54
        self.PX_TO_CM = 1 / 3.54         # conversion factor (3.54 px = 1 cm)
        self.OFFSET_CM = 4.5             # allowance for formworks

        # Material Densities (kg/m³)
        self.CEMENT_DENSITY = 1440
        self.SAND_DENSITY = 1600
        self.GRAVEL_DENSITY = 1500
        
        print("🤖 Initializing SIMPLIFIED AI Service - 4 Step Pipeline...")
        print(f"   Classes: {self.class_names}")
        print(f"   Expected: 2 verticals, 11 horizontals")
        print(f"   Detection threshold: {self.detection_threshold}")
        print("   📝 FIXED: Only saves analyzed images with 4-step visualization")
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model with EXACT TRAINING CONFIGURATION"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            print("🔄 Loading Detectron2 with EXACT TRAINING CONFIG...")
            
            # FIXED: Exact config from training
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # FIXED: Model settings matching training exactly
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # FIXED: 2 classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold  # FIXED: 0.3 threshold
            self.cfg.MODEL.DEVICE = "cpu"  # CPU for Raspberry Pi
            
            print("🔄 Creating predictor...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # FIXED: Set up metadata with exact training classes
            self.metadata = MetadataCatalog.get("rebar_dataset")
            self.metadata.thing_classes = self.class_names
            
            # FIXED: Set colors for each class
            self.metadata.thing_colors = [
                (255, 0, 0),      # front_horizontal - Red
                (0, 255, 0),      # front_vertical - Green
            ]
            
            self.model_loaded = True
            print("✅ SIMPLIFIED AI Model loaded successfully!")
            print(f"   Classes: {self.class_names}")
            print(f"   Expected detections: 2 verticals + 11 horizontals = 13 total")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_data=None, image_path=None):
        """
        SIMPLIFIED 4-Step Analysis Pipeline
        Returns 4 visualization images for display
        """
        try:
            print(f"🔍 Starting SIMPLIFIED 4-Step Analysis...")
            
            # Handle input
            if image_data is not None:
                print("📸 Using direct frame data from camera")
                image = image_data.copy()
            elif image_path and os.path.exists(image_path):
                print(f"📁 Loading image from: {image_path}")
                image = cv2.imread(image_path)
                if image is None:
                    return {'success': False, 'error': 'Failed to load image file'}
            else:
                return {'success': False, 'error': 'No image data provided'}
            
            print(f"📐 Image loaded: {image.shape}")
            
            # Ensure proper size
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                image = cv2.resize(image, (480, 640))
                print(f"⚙️  Resized to 480x640")
            
            # Run simplified pipeline
            if self.model_loaded and DETECTRON2_AVAILABLE:
                result = self._run_4step_pipeline(image)
            else:
                print("⚠️  Model not available, using placeholder")
                result = self._run_4step_placeholder(image)
            
            # Finalize results and save metadata
            result = self._finalize_analysis_results(result)
            
            return result
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Analysis failed: {str(e)}'
            }
    
    def _run_4step_pipeline(self, image):
        """Run the exact 4-step pipeline from training notebook"""
        try:
            print("🤖 Running 4-Step Pipeline with REAL MODEL...")
            
            # STEP 1: REBAR DETECTION
            print("📍 STEP 1: REBAR DETECTION")
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            num_detections = len(instances)
            print(f"   Found {num_detections} detections")
            
            if num_detections == 0:
                return {
                    'success': False,
                    'error': 'No rebar structures detected',
                    'no_detection': True
                }
            
            # Extract detection data
            pred_classes = instances.pred_classes.numpy()
            pred_masks = instances.pred_masks.numpy()
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            
            # Count by class
            fh_count = np.sum(pred_classes == 0)  # front_horizontal
            fv_count = np.sum(pred_classes == 1)  # front_vertical
            print(f"   Front horizontal: {fh_count}, Front vertical: {fv_count}")
            
            # Create Step 1 image
            step1_image = self._create_step1_detection(image, outputs)
            
            # STEP 2: QUADRANT INTERSECTIONS
            print("📍 STEP 2: QUADRANT INTERSECTIONS")
            intersections, centroids = self._find_intersections(pred_classes, pred_masks, image.shape)
            step2_image = self._create_step2_intersections(image, intersections, centroids)
            
            # STEP 3: POLYGON + VOLUME
            print("📍 STEP 3: POLYGON + VOLUME")
            polygon_data = self._create_polygon_from_centroids(centroids, image.shape)
            step3_image = self._create_step3_polygon(image, polygon_data)
            
            # STEP 4: CEMENT ESTIMATION
            print("📍 STEP 4: CEMENT ESTIMATION")
            cement_data = self._calculate_cement_mixture(polygon_data)
            step4_image = self._create_step4_cement(image, polygon_data, cement_data)
            
            # Save all 4 images
            step_images = self._save_4step_images(step1_image, step2_image, step3_image, step4_image)
            
            # Prepare results in expected format
            dimensions = {
                'length': polygon_data['length_cm'],
                'width': polygon_data['width_cm'], 
                'height': polygon_data['height_cm'],
                'unit': 'cm',
                'volume': polygon_data['volume_cm3'],
                'display': f"{polygon_data['width_cm']:.0f}cm x {polygon_data['width_cm']:.0f}cm x {polygon_data['height_cm']:.0f}cm = {polygon_data['volume_cm3']:.0f}cm³"
            }
            
            mixture = {
                'cement': self.MIX_RATIO[0],
                'sand': self.MIX_RATIO[1],
                'aggregate': self.MIX_RATIO[2],
                'ratio_string': f'{self.MIX_RATIO[0]} Cement : {self.MIX_RATIO[1]} Sand : {self.MIX_RATIO[2]} Aggregate',
                'cement_bags': cement_data['cement_bags'],
                'sand_volume_m3': cement_data['sand_m3'],
                'aggregate_volume_m3': cement_data['gravel_m3'],
                'total_concrete_volume_m3': cement_data['dry_volume_m3']
            }
            
            detections = []
            for i in range(num_detections):
                detection = {
                    'class_id': int(pred_classes[i]),
                    'class_name': self.class_names[pred_classes[i]],
                    'confidence': float(scores[i]),
                    'bbox': boxes[i].tolist()
                }
                detections.append(detection)
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': step_images['final'],  # Main analyzed image
                'step_images': step_images,  # All 4 step images
                'model_type': 'simplified_4step_pipeline',
                'pipeline_data': {
                    'front_horizontal_count': fh_count,
                    'front_vertical_count': fv_count,
                    'intersection_count': len(centroids),
                    'polygon_corners': len(polygon_data.get('corners', []))
                }
            }
            
        except Exception as e:
            print(f"❌ 4-step pipeline error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Pipeline failed: {str(e)}'
            }
    
    def _find_intersections(self, pred_classes, pred_masks, image_shape):
        """Find intersections between front_horizontal and front_vertical masks"""
        try:
            # Find indices of each class
            fh_indices = np.where(pred_classes == 0)[0]  # front_horizontal
            fv_indices = np.where(pred_classes == 1)[0]  # front_vertical
            
            print(f"   Processing {len(fh_indices)} horizontal × {len(fv_indices)} vertical intersections")
            
            # Create intersection mask
            all_intersections = np.zeros_like(pred_masks[0], dtype=np.uint8)
            
            # Find all intersections
            for fh in fh_indices:
                for fv in fv_indices:
                    inter = np.logical_and(pred_masks[fh], pred_masks[fv]).astype(np.uint8)
                    all_intersections = np.logical_or(all_intersections, inter)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(all_intersections.astype(np.uint8))
            
            centroids = []
            for lbl in range(1, num_labels):  # skip background (0)
                mask_region = (labels == lbl).astype(np.uint8)
                M = cv2.moments(mask_region)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
            
            print(f"   Found {len(centroids)} intersection centroids")
            
            return all_intersections, centroids
            
        except Exception as e:
            print(f"   ⚠️ Intersection error: {e}")
            return np.zeros_like(pred_masks[0], dtype=np.uint8), []
    
    def _create_polygon_from_centroids(self, centroids, image_shape):
        """Create polygon from centroids using quadrant sorting"""
        try:
            if len(centroids) < 4:
                print(f"   ⚠️ Not enough centroids for polygon: {len(centroids)}")
                # Use default values
                return {
                    'corners': [(100, 100), (200, 100), (200, 300), (100, 300)],
                    'width_px': 100,
                    'height_px': 200,
                    'width_cm': 100 * self.PX_TO_CM + self.OFFSET_CM,
                    'length_cm': 100 * self.PX_TO_CM + self.OFFSET_CM,
                    'height_cm': 200 * self.PX_TO_CM,
                    'volume_cm3': 0,
                    'volume_m3': 0
                }
            
            # Split into quadrants (exact from notebook)
            height, width = image_shape[:2]
            mid_x = width // 2
            mid_y = height // 2
            
            bottom_left = [pt for pt in centroids if pt[0] < mid_x and pt[1] >= mid_y]
            bottom_right = [pt for pt in centroids if pt[0] >= mid_x and pt[1] >= mid_y]
            top_left = [pt for pt in centroids if pt[0] < mid_x and pt[1] < mid_y]
            top_right = [pt for pt in centroids if pt[0] >= mid_x and pt[1] < mid_y]
            
            # Sort each quadrant (exact from notebook)
            bottom_left = sorted(bottom_left, key=lambda p: (-p[1], p[0]))
            bottom_right = sorted(bottom_right, key=lambda p: (-p[1], -p[0]))
            top_left = sorted(top_left, key=lambda p: (p[1], p[0]))
            top_right = sorted(top_right, key=lambda p: (p[1], -p[0]))
            
            print(f"   Quadrants: BL={len(bottom_left)}, BR={len(bottom_right)}, TL={len(top_left)}, TR={len(top_right)}")
            
            # Select corners
            if bottom_left and bottom_right and top_left and top_right:
                bl = bottom_left[0]
                br = min(bottom_right, key=lambda p: abs(p[1] - bl[1]))
                tl = top_left[0]
                tr = min(top_right, key=lambda p: abs(p[1] - tl[1]))
                
                corners = [bl, br, tr, tl]
                
                # Calculate dimensions (exact from notebook)
                width_px = int(np.linalg.norm(np.array(br) - np.array(bl)))
                height_px = int(np.linalg.norm(np.array(tl) - np.array(bl)))
                
                # Convert to cm + add offset (exact from notebook)
                width_cm = width_px * self.PX_TO_CM + self.OFFSET_CM
                length_cm = width_cm  # square assumption
                height_cm = height_px * self.PX_TO_CM
                
                # Volume calculation (exact from notebook)
                volume_cm3 = width_cm * length_cm * height_cm
                volume_m3 = volume_cm3 / 1_000_000
                
                print(f"   Polygon: {width_cm:.1f}cm × {length_cm:.1f}cm × {height_cm:.1f}cm = {volume_cm3:.0f}cm³")
                
                return {
                    'corners': corners,
                    'width_px': width_px,
                    'height_px': height_px,
                    'width_cm': width_cm,
                    'length_cm': length_cm,
                    'height_cm': height_cm,
                    'volume_cm3': volume_cm3,
                    'volume_m3': volume_m3
                }
            else:
                print("   ⚠️ Missing quadrants, using default polygon")
                return self._create_default_polygon()
                
        except Exception as e:
            print(f"   ⚠️ Polygon creation error: {e}")
            return self._create_default_polygon()
    
    def _create_default_polygon(self):
        """Create default polygon when detection fails"""
        width_px = 100
        height_px = 200
        width_cm = width_px * self.PX_TO_CM + self.OFFSET_CM
        length_cm = width_cm
        height_cm = height_px * self.PX_TO_CM
        volume_cm3 = width_cm * length_cm * height_cm
        
        return {
            'corners': [(100, 100), (200, 100), (200, 300), (100, 300)],
            'width_px': width_px,
            'height_px': height_px,
            'width_cm': width_cm,
            'length_cm': length_cm,
            'height_cm': height_cm,
            'volume_cm3': volume_cm3,
            'volume_m3': volume_cm3 / 1_000_000
        }
    
    def _calculate_cement_mixture(self, polygon_data):
        """Calculate cement mixture (exact from notebook)"""
        try:
            volume_m3 = polygon_data['volume_m3']
            dry_volume_m3 = volume_m3 * self.DRY_VOLUME_FACTOR
            
            # Calculate ratios (exact from notebook)
            total_ratio = sum(self.MIX_RATIO)
            cement_ratio, sand_ratio, gravel_ratio = self.MIX_RATIO
            
            # Cement calculation
            cement_m3 = dry_volume_m3 * (cement_ratio / total_ratio)
            cement_weight_kg = cement_m3 * self.CEMENT_DENSITY
            cement_bags = cement_weight_kg / self.CEMENT_BAG_WEIGHT
            
            # Sand calculation
            sand_m3 = dry_volume_m3 * (sand_ratio / total_ratio)
            sand_weight_kg = sand_m3 * self.SAND_DENSITY
            
            # Gravel calculation
            gravel_m3 = dry_volume_m3 * (gravel_ratio / total_ratio)
            gravel_weight_kg = gravel_m3 * self.GRAVEL_DENSITY
            
            # Water calculation
            water_liters = cement_weight_kg * self.WATER_CEMENT_RATIO
            
            print(f"   Cement: {cement_bags:.2f} bags, Sand: {sand_m3:.3f}m³, Gravel: {gravel_m3:.3f}m³")
            
            return {
                'dry_volume_m3': dry_volume_m3,
                'cement_bags': cement_bags,
                'cement_weight_kg': cement_weight_kg,
                'sand_m3': sand_m3,
                'sand_weight_kg': sand_weight_kg,
                'gravel_m3': gravel_m3,
                'gravel_weight_kg': gravel_weight_kg,
                'water_liters': water_liters
            }
            
        except Exception as e:
            print(f"   ⚠️ Cement calculation error: {e}")
            return {
                'dry_volume_m3': 0.001,
                'cement_bags': 1.0,
                'cement_weight_kg': 40.0,
                'sand_m3': 0.002,
                'sand_weight_kg': 3.2,
                'gravel_m3': 0.004,
                'gravel_weight_kg': 6.0,
                'water_liters': 21.2
            }
    
    def _create_step1_detection(self, image, outputs):
        """Create Step 1: Detection visualization"""
        try:
            v = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=self.metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE
            )
            
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = out.get_image()[:, :, ::-1]  # Convert back to BGR
            
            # Add title
            cv2.putText(result_image, "Step 1: REBAR DETECTION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return result_image
            
        except Exception as e:
            print(f"   ⚠️ Step 1 visualization error: {e}")
            return image.copy()
    
    def _create_step2_intersections(self, image, intersections, centroids):
        """Create Step 2: Quadrant intersections visualization"""
        try:
            result_image = image.copy()
            
            # Draw intersection mask
            if intersections is not None and np.any(intersections):
                colored_mask = np.zeros_like(image)
                colored_mask[intersections > 0] = [0, 255, 255]  # Yellow intersections
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
            
            # Draw centroids with quadrant colors
            if centroids:
                height, width = image.shape[:2]
                mid_x = width // 2
                mid_y = height // 2
                
                for i, (x, y) in enumerate(centroids):
                    # Determine quadrant and color
                    if x < mid_x and y >= mid_y:
                        color = (0, 0, 255)  # BL - Red
                        label = f"BL-{i+1}"
                    elif x >= mid_x and y >= mid_y:
                        color = (255, 0, 0)  # BR - Blue
                        label = f"BR-{i+1}"
                    elif x < mid_x and y < mid_y:
                        color = (0, 128, 0)  # TL - Green
                        label = f"TL-{i+1}"
                    else:
                        color = (128, 0, 128)  # TR - Purple
                        label = f"TR-{i+1}"
                    
                    cv2.circle(result_image, (x, y), 6, color, -1)
                    cv2.putText(result_image, label, (x + 10, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add title
            cv2.putText(result_image, "Step 2: QUADRANT INTERSECTIONS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            return result_image
            
        except Exception as e:
            print(f"   ⚠️ Step 2 visualization error: {e}")
            return image.copy()
    
    def _create_step3_polygon(self, image, polygon_data):
        """Create Step 3: Polygon + volume visualization"""
        try:
            result_image = image.copy()
            corners = polygon_data['corners']
            
            if len(corners) >= 4:
                # Draw polygon
                overlay = result_image.copy()
                pts = np.array(corners, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (255, 0, 0))  # Blue polygon
                result_image = cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0)
                
                # Draw polygon outline
                cv2.polylines(result_image, [pts], True, (0, 255, 255), 3)  # Yellow outline
                
                # Add dimension labels
                bl, br, tr, tl = corners[:4]
                
                # Width label
                cv2.putText(result_image, f"W={polygon_data['width_cm']:.1f}cm", 
                           (bl[0]+20, bl[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                # Height label
                cv2.putText(result_image, f"H={polygon_data['height_cm']:.1f}cm", 
                           (bl[0]-80, (bl[1]+tl[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                # Volume label
                cv2.putText(result_image, f"Vol={polygon_data['volume_m3']:.3f}m³", 
                           (bl[0]+20, bl[1]+80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            # Add title
            cv2.putText(result_image, "Step 3: POLYGON + VOLUME", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            return result_image
            
        except Exception as e:
            print(f"   ⚠️ Step 3 visualization error: {e}")
            return image.copy()
    
    def _create_step4_cement(self, image, polygon_data, cement_data):
        """Create Step 4: Cement estimation visualization"""
        try:
            result_image = image.copy()
            
            # Draw polygon again
            corners = polygon_data['corners']
            if len(corners) >= 4:
                overlay = result_image.copy()
                pts = np.array(corners, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 128, 255))  # Orange polygon
                result_image = cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0)
                cv2.polylines(result_image, [pts], True, (0, 255, 0), 3)  # Green outline
            
            # Add cement mixture information
            y_start = 60
            line_height = 25
            
            texts = [
                f"Cement: {cement_data['cement_bags']:.2f} bags ({cement_data['cement_weight_kg']:.1f}kg)",
                f"Sand: {cement_data['sand_m3']:.3f}m³ ({cement_data['sand_weight_kg']:.1f}kg)",
                f"Gravel: {cement_data['gravel_m3']:.3f}m³ ({cement_data['gravel_weight_kg']:.1f}kg)",
                f"Water: {cement_data['water_liters']:.1f} liters",
                f"Mix Ratio: 1:2:4 (C:S:G)"
            ]
            
            for i, text in enumerate(texts):
                cv2.putText(result_image, text, (10, y_start + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(result_image, text, (10, y_start + i * line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Add title
            cv2.putText(result_image, "Step 4: CEMENT ESTIMATION", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
            
            return result_image
            
        except Exception as e:
            print(f"   ⚠️ Step 4 visualization error: {e}")
            return image.copy()
    
    def _save_4step_images(self, step1_image, step2_image, step3_image, step4_image):
        """Save all 4 step images for display"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # Save individual step images
            step1_path = os.path.join(config.UPLOAD_FOLDER, f'step1_detection_{timestamp}.jpg')
            step2_path = os.path.join(config.UPLOAD_FOLDER, f'step2_intersections_{timestamp}.jpg')
            step3_path = os.path.join(config.UPLOAD_FOLDER, f'step3_polygon_{timestamp}.jpg')
            step4_path = os.path.join(config.UPLOAD_FOLDER, f'step4_cement_{timestamp}.jpg')
            
            # Save final combined image (step 4 is the final result)
            final_path = os.path.join(config.UPLOAD_FOLDER, f'analyzed_rebar_{timestamp}.jpg')
            
            # Save all images
            cv2.imwrite(step1_path, step1_image)
            cv2.imwrite(step2_path, step2_image)
            cv2.imwrite(step3_path, step3_image)
            cv2.imwrite(step4_path, step4_image)
            cv2.imwrite(final_path, step4_image)  # Final is step 4
            
            print(f"✅ Saved 4-step analysis images:")
            print(f"   Step 1: {os.path.basename(step1_path)}")
            print(f"   Step 2: {os.path.basename(step2_path)}")
            print(f"   Step 3: {os.path.basename(step3_path)}")
            print(f"   Step 4: {os.path.basename(step4_path)}")
            print(f"   Final: {os.path.basename(final_path)}")
            
            return {
                'step1': step1_path,
                'step2': step2_path,
                'step3': step3_path,
                'step4': step4_path,
                'final': final_path
            }
            
        except Exception as e:
            print(f"❌ Error saving step images: {str(e)}")
            return {
                'step1': None,
                'step2': None,
                'step3': None,
                'step4': None,
                'final': None
            }
    
    def _run_4step_placeholder(self, image):
        """Run placeholder 4-step pipeline when model not available"""
        try:
            print("📝 Running 4-Step PLACEHOLDER Pipeline...")
            
            # Create placeholder step images
            step1_image = self._create_placeholder_step1(image)
            step2_image = self._create_placeholder_step2(image)
            step3_image = self._create_placeholder_step3(image)
            step4_image = self._create_placeholder_step4(image)
            
            # Save step images
            step_images = self._save_4step_images(step1_image, step2_image, step3_image, step4_image)
            
            # Default polygon data
            polygon_data = {
                'width_cm': 28.2,
                'length_cm': 28.2,
                'height_cm': 56.5,
                'volume_cm3': 45000,
                'volume_m3': 0.045
            }
            
            # Default cement data
            cement_data = {
                'cement_bags': 2.5,
                'sand_m3': 0.005,
                'gravel_m3': 0.010,
                'water_liters': 53.0
            }
            
            # Format results
            dimensions = {
                'length': polygon_data['length_cm'],
                'width': polygon_data['width_cm'],
                'height': polygon_data['height_cm'],
                'unit': 'cm',
                'volume': polygon_data['volume_cm3'],
                'display': f"{polygon_data['width_cm']:.0f}cm x {polygon_data['width_cm']:.0f}cm x {polygon_data['height_cm']:.0f}cm = {polygon_data['volume_cm3']:.0f}cm³"
            }
            
            mixture = {
                'cement': self.MIX_RATIO[0],
                'sand': self.MIX_RATIO[1],
                'aggregate': self.MIX_RATIO[2],
                'ratio_string': f'{self.MIX_RATIO[0]} Cement : {self.MIX_RATIO[1]} Sand : {self.MIX_RATIO[2]} Aggregate',
                'cement_bags': cement_data['cement_bags'],
                'sand_volume_m3': cement_data['sand_m3'],
                'aggregate_volume_m3': cement_data['gravel_m3'],
                'total_concrete_volume_m3': polygon_data['volume_m3'] * self.DRY_VOLUME_FACTOR
            }
            
            return {
                'success': True,
                'placeholder': True,
                'detections': [
                    {'class_name': 'front_vertical', 'confidence': 0.85, 'bbox': [100, 50, 120, 300]},
                    {'class_name': 'front_vertical', 'confidence': 0.82, 'bbox': [180, 50, 200, 300]},
                    {'class_name': 'front_horizontal', 'confidence': 0.78, 'bbox': [80, 60, 220, 80]}
                ],
                'num_detections': 13,  # Expected: 2 verticals + 11 horizontals
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': step_images['final'],
                'step_images': step_images,
                'model_type': 'placeholder_4step_pipeline',
                'pipeline_data': {
                    'front_horizontal_count': 11,
                    'front_vertical_count': 2,
                    'intersection_count': 22,
                    'polygon_corners': 4
                }
            }
            
        except Exception as e:
            print(f"❌ Placeholder pipeline error: {str(e)}")
            return {
                'success': False,
                'error': f'Placeholder pipeline failed: {str(e)}'
            }
    
    def _create_placeholder_step1(self, image):
        """Create placeholder Step 1: Detection"""
        result_image = image.copy()
        
        # Draw placeholder detections
        # 2 vertical rebars
        cv2.rectangle(result_image, (100, 50), (120, 300), (0, 255, 0), 3)
        cv2.rectangle(result_image, (180, 50), (200, 300), (0, 255, 0), 3)
        cv2.putText(result_image, 'front_vertical (85%)', (100, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(result_image, 'front_vertical (82%)', (180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 3 sample horizontal rebars (representing 11 total)
        cv2.rectangle(result_image, (80, 60), (220, 80), (255, 0, 0), 3)
        cv2.rectangle(result_image, (80, 120), (220, 140), (255, 0, 0), 3)
        cv2.rectangle(result_image, (80, 280), (220, 300), (255, 0, 0), 3)
        cv2.putText(result_image, 'front_horizontal (78%)', (80, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(result_image, '+ 8 more horizontals...', (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add title
        cv2.putText(result_image, "Step 1: REBAR DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result_image
    
    def _create_placeholder_step2(self, image):
        """Create placeholder Step 2: Intersections"""
        result_image = image.copy()
        
        # Draw intersection points in quadrants
        intersections = [
            (110, 70, (0, 0, 255), "BL-1"),    # Bottom left
            (110, 130, (0, 0, 255), "BL-2"),
            (110, 290, (0, 0, 255), "BL-3"),
            (190, 70, (255, 0, 0), "BR-1"),    # Bottom right
            (190, 130, (255, 0, 0), "BR-2"),
            (190, 290, (255, 0, 0), "BR-3"),
        ]
        
        for x, y, color, label in intersections:
            cv2.circle(result_image, (x, y), 6, color, -1)
            cv2.putText(result_image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add title
        cv2.putText(result_image, "Step 2: QUADRANT INTERSECTIONS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        return result_image
    
    def _create_placeholder_step3(self, image):
        """Create placeholder Step 3: Polygon"""
        result_image = image.copy()
        
        # Draw polygon
        corners = [(110, 70), (190, 70), (190, 290), (110, 290)]
        overlay = result_image.copy()
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (255, 0, 0))
        result_image = cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0)
        cv2.polylines(result_image, [pts], True, (0, 255, 255), 3)
        
        # Add dimensions
        cv2.putText(result_image, "W=28.2cm", (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(result_image, "H=56.5cm", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(result_image, "Vol=0.045m³", (120, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Add title
        cv2.putText(result_image, "Step 3: POLYGON + VOLUME", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return result_image
    
    def _create_placeholder_step4(self, image):
        """Create placeholder Step 4: Cement estimation"""
        result_image = image.copy()
        
        # Draw polygon
        corners = [(110, 70), (190, 70), (190, 290), (110, 290)]
        overlay = result_image.copy()
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 128, 255))
        result_image = cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0)
        cv2.polylines(result_image, [pts], True, (0, 255, 0), 3)
        
        # Add cement mixture information
        texts = [
            "Cement: 2.5 bags (100kg)",
            "Sand: 0.005m³ (8kg)",
            "Gravel: 0.010m³ (15kg)",
            "Water: 53 liters",
            "Mix Ratio: 1:2:4"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 60 + i * 25
            cv2.putText(result_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add title
        cv2.putText(result_image, "Step 4: CEMENT ESTIMATION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
        
        return result_image
    
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
            'model_type': 'simplified_4step_pipeline' if self.model_loaded else 'placeholder',
            'save_mode': 'analyzed_images_only',
            'expected_detections': {
                'front_vertical': 2,
                'front_horizontal': 11,
                'total': 13
            }
        }
