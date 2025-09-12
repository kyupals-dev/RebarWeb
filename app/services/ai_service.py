"""
AI Service for Rebar Detection and Analysis - SIMPLIFIED FOR FRONT PHASE DETECTION
Focuses on detecting 2 front_vertical and 11 front_horizontal rebars with intersections
MODIFIED: Simplified detection logic for square column measurement with 1:2:4 cement ratio
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
    """Simplified AI Service for Front Rebar Detection (2V + 11H pattern) with 1:2:4 cement ratio"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # Simplified rebar classes - only using front classes
        self.class_names = ["front_horizontal", "front_vertical"]
        self.target_classes = ["front_horizontal", "front_vertical"]  # Only these matter
        self.num_classes = 2
        
        # Detection parameters for your specific pattern
        self.detection_threshold = 0.3  # Lower threshold for better detection
        self.target_verticals = 2      # Exactly 2 front verticals
        self.target_horizontals = 11   # Exactly 11 front horizontals
        
        # Square column parameters
        self.offset_cm = 4.5  # 4.5cm offset on each side
        self.pixel_to_cm_factor = 0.25  # Calibration factor (adjust based on distance)
        
        print("🎯 Simplified AI Service initialized for Front Rebar Detection")
        print(f"   Target: {self.target_verticals} front_vertical + {self.target_horizontals} front_horizontal")
        print(f"   Offset: {self.offset_cm}cm per side")
        print(f"   Cement ratio: 1:2:4 (Class A)")
        print(f"   Detection threshold: {self.detection_threshold}")
        self.load_model()
    
    def load_model(self):
        """Load the trained model with simplified configuration"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using simplified placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            print("🔄 Loading simplified model configuration...")
            
            # Simple configuration matching your training
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold
            self.cfg.MODEL.DEVICE = "cpu"
            
            print("🔄 Creating predictor...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # Setup metadata
            self.metadata = MetadataCatalog.get("rebar_dataset_simplified")
            self.metadata.thing_classes = self.class_names
            self.metadata.thing_colors = [
                (128, 128, 128),  # back_horizontal - Gray (not used)
                (255, 0, 0),      # front_horizontal - Red
                (0, 255, 0),      # front_vertical - Green
            ]
            
            self.model_loaded = True
            print("✅ Simplified model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_data=None, image_path=None):
        """Simplified analysis focusing on 2V + 11H pattern"""
        try:
            print(f"🎯 Starting simplified front rebar analysis (2V + 11H)...")
            
            # Handle input
            if image_data is not None:
                image = image_data.copy()
                source = "camera_frame"
            elif image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
                source = "file"
                if image is None:
                    return {'success': False, 'error': 'Failed to load image'}
            else:
                return {'success': False, 'error': 'No image data provided'}
            
            print(f"📐 Image loaded: {image.shape} from {source}")
            
            # Ensure correct size
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                image = cv2.resize(image, (480, 640))
                print(f"📐 Resized to 480x640")
            
            # Run detection
            if self.model_loaded and DETECTRON2_AVAILABLE:
                result = self._analyze_with_simplified_model(image)
            else:
                result = self._analyze_simplified_placeholder(image)
            
            return result
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}
    
    def _analyze_with_simplified_model(self, image):
        """Run model with simplified detection logic"""
        try:
            print("🤖 Running simplified detection...")
            
            # Run inference
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            # Extract detections
            if len(instances) == 0:
                print("❌ No detections found")
                return {'success': False, 'error': 'No rebar structures detected', 'no_detection': True}
            
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            # Filter for front classes only
            front_detections = []
            for i in range(len(instances)):
                class_id = int(classes[i])
                class_name = self.class_names[class_id]
                confidence = float(scores[i])
                
                # Only keep front_horizontal and front_vertical
                if class_name in self.target_classes:
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': boxes[i].tolist()
                    }
                    front_detections.append(detection)
            
            if not front_detections:
                print("❌ No front rebar detections found")
                return {'success': False, 'error': 'No front rebar detected', 'no_detection': True}
            
            # Separate verticals and horizontals
            verticals = [d for d in front_detections if d['class_name'] == 'front_vertical']
            horizontals = [d for d in front_detections if d['class_name'] == 'front_horizontal']
            
            print(f"🎯 Found: {len(verticals)} verticals, {len(horizontals)} horizontals")
            
            # Filter to get best detections for target pattern
            final_verticals = self._filter_best_verticals(verticals, target_count=self.target_verticals)
            final_horizontals = self._filter_best_horizontals(horizontals, target_count=self.target_horizontals)
            
            # Verify intersections
            valid_intersections = self._verify_intersections(final_verticals, final_horizontals, image.shape)
            
            print(f"✅ Final pattern: {len(final_verticals)} verticals, {len(final_horizontals)} horizontals")
            print(f"🔍 Valid intersections: {valid_intersections}")
            
            if len(final_verticals) < 2 or len(final_horizontals) < 8:  # Minimum requirements
                print("⚠️ Insufficient detections for reliable measurement")
                # Still proceed but with warning
            
            # Calculate dimensions
            dimensions = self._calculate_simplified_dimensions(final_verticals, final_horizontals, image.shape)
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture(dimensions)
            
            # Create visualization
            analyzed_image_path = self._create_simplified_visualization(image, final_verticals, final_horizontals, dimensions)
            
            if not analyzed_image_path:
                return {'success': False, 'error': 'Failed to create visualization'}
            
            return {
                'success': True,
                'detections': final_verticals + final_horizontals,
                'num_detections': len(final_verticals) + len(final_horizontals),
                'verticals_count': len(final_verticals),
                'horizontals_count': len(final_horizontals),
                'valid_intersections': valid_intersections,
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'analyzed_image_path': analyzed_image_path,
                'model_type': 'simplified_front_detection'
            }
            
        except Exception as e:
            print(f"❌ Model inference error: {str(e)}")
            return {'success': False, 'error': f'Inference failed: {str(e)}'}
    
    def _filter_best_verticals(self, verticals, target_count=2):
        """Filter to get best vertical detections"""
        if len(verticals) <= target_count:
            return verticals
        
        # Sort by confidence and height (taller is better for verticals)
        def vertical_score(det):
            bbox = det['bbox']
            height = bbox[3] - bbox[1]  # y2 - y1
            return det['confidence'] * 0.7 + (height / 640) * 0.3  # Height normalized to image height
        
        sorted_verticals = sorted(verticals, key=vertical_score, reverse=True)
        selected = sorted_verticals[:target_count]
        
        # Sort selected by x position (left to right)
        selected.sort(key=lambda d: d['bbox'][0])  # Sort by x1
        
        print(f"📊 Selected {len(selected)} best verticals from {len(verticals)} candidates")
        return selected
    
    def _filter_best_horizontals(self, horizontals, target_count=11):
        """Filter to get best horizontal detections"""
        if len(horizontals) <= target_count:
            return horizontals
        
        # Sort by confidence and width (wider is better for horizontals)
        def horizontal_score(det):
            bbox = det['bbox']
            width = bbox[2] - bbox[0]  # x2 - x1
            return det['confidence'] * 0.7 + (width / 480) * 0.3  # Width normalized to image width
        
        sorted_horizontals = sorted(horizontals, key=horizontal_score, reverse=True)
        selected = sorted_horizontals[:target_count]
        
        # Sort selected by y position (top to bottom)
        selected.sort(key=lambda d: d['bbox'][1])  # Sort by y1
        
        print(f"📊 Selected {len(selected)} best horizontals from {len(horizontals)} candidates")
        return selected
    
    def _verify_intersections(self, verticals, horizontals, image_shape):
        """Verify that verticals and horizontals intersect (simplified check)"""
        if not verticals or not horizontals:
            return 0
        
        intersections = 0
        height, width = image_shape[:2]
        
        for v_det in verticals:
            v_bbox = v_det['bbox']
            v_x1, v_y1, v_x2, v_y2 = v_bbox
            
            for h_det in horizontals:
                h_bbox = h_det['bbox']
                h_x1, h_y1, h_x2, h_y2 = h_bbox
                
                # Check if bounding boxes intersect
                if not (v_x2 < h_x1 or h_x2 < v_x1 or v_y2 < h_y1 or h_y2 < v_y1):
                    intersections += 1
        
        return intersections
    
    def _calculate_simplified_dimensions(self, verticals, horizontals, image_shape):
        """Calculate dimensions using simplified approach"""
        try:
            print("📏 Calculating simplified dimensions...")
            
            height, width = image_shape[:2]
            
            # Default values
            length_cm = 25.0
            width_cm = 25.0
            height_cm = 200.0
            
            # Calculate from detections if available
            if verticals and len(verticals) >= 2:
                # Use vertical spacing for width
                v1_center = (verticals[0]['bbox'][0] + verticals[0]['bbox'][2]) / 2
                v2_center = (verticals[1]['bbox'][0] + verticals[1]['bbox'][2]) / 2
                width_px = abs(v2_center - v1_center)
                width_cm = max(width_px * self.pixel_to_cm_factor, 15.0)
                
                # Use vertical height for column height estimation
                v_height_px = max([det['bbox'][3] - det['bbox'][1] for det in verticals])
                height_cm = max(v_height_px * self.pixel_to_cm_factor * 2.5, 150.0)  # Scale up for full height
            
            if horizontals and len(horizontals) >= 8:
                # Use horizontal width for length
                max_h_width = max([det['bbox'][2] - det['bbox'][0] for det in horizontals])
                length_cm = max(max_h_width * self.pixel_to_cm_factor, 15.0)
            
            # For square columns, use the larger dimension for both length and width
            dimension_cm = max(length_cm, width_cm)
            length_cm = width_cm = dimension_cm
            
            # Add offset (4.5cm on each side = 9cm total per dimension)
            offset_total = self.offset_cm * 2
            final_length = length_cm + offset_total
            final_width = width_cm + offset_total
            final_height = height_cm
            
            # Calculate volume
            volume_cm3 = final_length * final_width * final_height
            
            dimensions = {
                'length': round(final_length, 1),
                'width': round(final_width, 1),
                'height': round(final_height, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': f"{final_length:.0f}cm x {final_width:.0f}cm x {final_height:.0f}cm = {volume_cm3:.0f}cm³",
                'method': 'simplified_front_detection',
                'offset_applied': offset_total,
                'raw_dimensions': {
                    'length': round(length_cm, 1),
                    'width': round(width_cm, 1),
                    'height': round(height_cm, 1)
                }
            }
            
            print(f"📐 Calculated: {dimensions['display']}")
            print(f"   Offset applied: +{offset_total}cm per side")
            
            return dimensions
            
        except Exception as e:
            print(f"❌ Dimension calculation error: {e}")
            return {
                'length': 34.0,   # 25 + 9 offset
                'width': 34.0,    # 25 + 9 offset  
                'height': 200.0,
                'unit': 'cm',
                'volume': 231200,
                'display': '34cm x 34cm x 200cm = 231200cm³',
                'method': 'fallback_with_offset'
            }
    
    def _create_simplified_visualization(self, image, verticals, horizontals, dimensions):
        """Create clear visualization with readable overlays"""
        try:
            print("🎨 Creating simplified visualization...")
            
            result_image = image.copy()
            
            # Create semi-transparent overlay
            overlay = result_image.copy()
            
            # Draw verticals in GREEN
            for i, v_det in enumerate(verticals):
                bbox = v_det['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Fill with transparent green
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                
                # Draw border
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Label
                label = f"V{i+1} ({v_det['confidence']:.2f})"
                cv2.putText(result_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw horizontals in RED
            for i, h_det in enumerate(horizontals):
                bbox = h_det['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Fill with transparent red
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                
                # Draw border
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Label (only for first few to avoid clutter)
                if i < 5:
                    label = f"H{i+1}"
                    cv2.putText(result_image, label, (x1+5, y1+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Apply transparency
            alpha = 0.25
            result_image = cv2.addWeighted(result_image, 1-alpha, overlay, alpha, 0)
            
            # Add information panel
            self._add_info_panel(result_image, len(verticals), len(horizontals), dimensions)
            
            # Save visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'simplified_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                print(f"✅ Simplified visualization saved: {filename}")
                return output_path
            else:
                print("❌ Failed to save visualization")
                return None
                
        except Exception as e:
            print(f"❌ Visualization error: {str(e)}")
            return None
    
    def _add_info_panel(self, image, v_count, h_count, dimensions):
        """Add readable info panel to image"""
        try:
            # Info panel background
            panel_height = 120
            panel_width = 400
            panel_x = 10
            panel_y = 10
            
            # Semi-transparent black background
            overlay = image.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
            
            # Add white border
            cv2.rectangle(image, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (255, 255, 255), 2)
            
            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (255, 255, 255)
            
            # Add text lines
            y_offset = panel_y + 25
            line_spacing = 20
            
            # Detection counts
            text1 = f"FRONT REBAR DETECTION:"
            cv2.putText(image, text1, (panel_x + 10, y_offset), font, font_scale, color, thickness)
            
            y_offset += line_spacing
            text2 = f"Verticals: {v_count}/{self.target_verticals} | Horizontals: {h_count}/{self.target_horizontals}"
            cv2.putText(image, text2, (panel_x + 10, y_offset), font, 0.5, color, thickness)
            
            y_offset += line_spacing
            text3 = f"Dimensions: {dimensions['display']}"
            cv2.putText(image, text3, (panel_x + 10, y_offset), font, 0.5, color, thickness)
            
            y_offset += line_spacing
            text4 = f"Method: Square Column + {self.offset_cm}cm Offset"
            cv2.putText(image, text4, (panel_x + 10, y_offset), font, 0.5, color, thickness)
            
        except Exception as e:
            print(f"⚠️ Error adding info panel: {e}")
    
    def _analyze_simplified_placeholder(self, image):
        """Generate realistic placeholder for 2V + 11H pattern with CORRECT 1:2:4 ratio"""
        print("📝 Using simplified placeholder (2V + 11H pattern)...")
        
        # Create visualization with target pattern
        analyzed_image_path = self._create_placeholder_visualization(image)
        
        if not analyzed_image_path:
            return {'success': False, 'error': 'Failed to create placeholder'}
        
        # Calculate dimensions with offset
        base_dimension = 25.0
        offset_total = self.offset_cm * 2
        final_dimension = base_dimension + offset_total
        height = 200.0
        volume = final_dimension * final_dimension * height
        
        dimensions = {
            'length': final_dimension,
            'width': final_dimension,
            'height': height,
            'unit': 'cm',
            'volume': volume,
            'display': f'{final_dimension:.0f}cm x {final_dimension:.0f}cm x {height:.0f}cm = {volume:.0f}cm³',
            'method': 'placeholder_with_offset'
        }
        
        # UPDATED: Use correct 1:2:4 mixture ratio
        mixture = {
            'cement_ratio': 1,
            'sand_ratio': 2,
            'aggregate_ratio': 4,  # CHANGED from 3 to 4
            'ratio_string': '1 Cement : 2 Sand : 4 Aggregate'  # UPDATED
        }
        
        return {
            'success': True,
            'placeholder': True,
            'verticals_count': 2,
            'horizontals_count': 11,
            'dimensions': dimensions,
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_image_path,
            'model_type': 'simplified_placeholder'
        }
    
    def _create_placeholder_visualization(self, image):
        """Create placeholder with 2V + 11H pattern"""
        try:
            result_image = image.copy()
            
            # Draw 2 vertical rebars (green)
            v1_x, v2_x = 150, 330
            v_y1, v_y2 = 50, 590
            
            for i, x in enumerate([v1_x, v2_x]):
                cv2.rectangle(result_image, (x-15, v_y1), (x+15, v_y2), (0, 255, 0), 3)
                cv2.putText(result_image, f'V{i+1}', (x-10, v_y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw 11 horizontal rebars (red)
            h_x1, h_x2 = 120, 360
            h_positions = np.linspace(80, 560, 11)
            
            for i, y in enumerate(h_positions):
                y = int(y)
                cv2.rectangle(result_image, (h_x1, y-8), (h_x2, y+8), (0, 0, 255), 2)
                if i < 5:  # Label first few
                    cv2.putText(result_image, f'H{i+1}', (h_x1+5, y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Add info panel
            dimensions = {'display': '34cm x 34cm x 200cm = 231200cm³'}
            self._add_info_panel(result_image, 2, 11, dimensions)
            
            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'simplified_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            return output_path if success else None
            
        except Exception as e:
            print(f"❌ Placeholder error: {e}")
            return None
    
    def _calculate_cement_mixture(self, dimensions):
        """Calculate cement mixture with CORRECT 1:2:4 ratio (Class A Construction)"""
        volume_cm3 = dimensions.get('volume', 0)
        volume_m3 = volume_cm3 / 1000000
        
        # UPDATED: Class A mixture ratios (1:2:4) for superstructures and columns
        cement_ratio = 1
        sand_ratio = 2  
        aggregate_ratio = 4  # CHANGED from 3 to 4
        
        # Philippine construction standards for Class A
        # 9 bags per cubic meter as per specification
        bags_per_cubic_meter = 9
        
        # Calculate quantities
        concrete_factor = 1.5  # Account for concrete around rebar
        total_concrete_volume = volume_m3 * concrete_factor
        
        # Calculate material quantities using 1:2:4 ratio
        total_parts = cement_ratio + sand_ratio + aggregate_ratio  # 1+2+4 = 7 parts
        
        # Cement calculation (9 bags per cubic meter)
        cement_bags = total_concrete_volume * bags_per_cubic_meter
        
        # Fine aggregates: 0.50 cu.m + 10% waste per cu.m (from specification)
        sand_volume_base = 0.50 * total_concrete_volume
        sand_waste_factor = 1.10  # 10% waste
        sand_volume = sand_volume_base * sand_waste_factor
        
        # Coarse aggregates: 0.77 cu.m + 5% waste per cu.m (from specification)
        aggregate_volume_base = 0.77 * total_concrete_volume
        aggregate_waste_factor = 1.05  # 5% waste
        aggregate_volume = aggregate_volume_base * aggregate_waste_factor
        
        print(f"🧮 Cement calculation (Class A - 1:2:4):")
        print(f"   Volume: {volume_cm3:.0f} cm³ = {volume_m3:.6f} m³")
        print(f"   Concrete needed: {total_concrete_volume:.6f} m³")
        print(f"   Cement: {cement_bags:.2f} bags")
        print(f"   Sand: {sand_volume:.6f} m³ (with 10% waste)")
        print(f"   Aggregate: {aggregate_volume:.6f} m³ (with 5% waste)")
        
        return {
            'cement_ratio': cement_ratio,
            'sand_ratio': sand_ratio,
            'aggregate_ratio': aggregate_ratio,
            'ratio_string': f'{cement_ratio} Cement : {sand_ratio} Sand : {aggregate_ratio} Aggregate',
            'total_concrete_volume_m3': round(total_concrete_volume, 6),
            'cement_bags': round(cement_bags, 2),
            'sand_volume_m3': round(sand_volume, 6),
            'aggregate_volume_m3': round(aggregate_volume, 6),
            'specifications': {
                'class': 'Class A (Superstructures)',
                'cement_bags_per_cubic_meter': bags_per_cubic_meter,
                'max_water_cement_ratio': 0.53,
                'min_compressive_strength_psi': 3000,
                'min_compressive_strength_mpa': 20.7,
                'max_aggregate_size_mm': 37.5,
                'slump_range_mm': '50-100',
                'sand_waste_factor': '10%',
                'aggregate_waste_factor': '5%'
            }
        }
    
    def get_model_status(self):
        """Get simplified model status"""
        return {
            'detectron2_available': DETECTRON2_AVAILABLE,
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path) if self.model_path else False,
            'target_pattern': f'{self.target_verticals}V + {self.target_horizontals}H',
            'class_names': self.class_names,
            'target_classes': self.target_classes,
            'threshold': self.detection_threshold,
            'offset_cm': self.offset_cm,
            'cement_ratio': '1:2:4 (Class A)',
            'model_type': 'simplified_front_detection',
            'save_mode': 'analyzed_images_only'
        }
    
    def test_model(self, test_image_path=None):
        """Test simplified model"""
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
                    return {'success': False, 'error': 'No test directory'}
            
            print(f"🧪 Testing simplified model with: {test_image_path}")
            
            result = self.analyze_image(image_path=test_image_path)
            
            if result['success']:
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'verticals_found': result.get('verticals_count', 0),
                    'horizontals_found': result.get('horizontals_count', 0),
                    'target_pattern': f'{self.target_verticals}V + {self.target_horizontals}H',
                    'model_type': result.get('model_type', 'unknown'),
                    'analyzed_image_saved': result.get('analyzed_image_path')
                }
            else:
                return result
                
        except Exception as e:
            print(f"❌ Test error: {str(e)}")
            return {'success': False, 'error': f'Test failed: {str(e)}'}
