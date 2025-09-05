"""
AI Service for Rebar Detection and Analysis
UPDATED: Simplified approach focusing on 2 front_vertical + 11 front_horizontal rebars
with 4.5cm offset calculation for square columns
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

class SimplifiedRebarProcessor:
    """Simplified processor for 2 vertical + 11 horizontal rebar detection"""
    
    def __init__(self):
        self.offset_cm = 4.5  # 4.5cm offset for each side
        self.target_vertical = 2
        self.target_horizontal = 11
        
    def process_detections(self, instances, image_shape):
        """Process Detectron2 outputs into simplified rebar measurements"""
        try:
            print("🔄 Processing detections with simplified logic...")
            
            if len(instances) == 0:
                return {'success': False, 'error': 'No detections found'}
            
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            # Filter for target classes
            detections = self._filter_detections(boxes, scores, classes, masks)
            
            if not detections['success']:
                return detections
            
            # Verify intersections
            intersections = self._find_intersections(detections['vertical'], detections['horizontal'])
            
            if len(intersections) < 5:
                return {
                    'success': False,
                    'error': f'Only {len(intersections)} intersections found, expected at least 5'
                }
            
            # Calculate dimensions
            dimensions = self._calculate_dimensions(
                detections['vertical'], 
                detections['horizontal'], 
                intersections, 
                image_shape
            )
            
            # Calculate cement mixture
            mixture = self._calculate_cement_mixture(dimensions)
            
            result = {
                'success': True,
                'vertical_count': len(detections['vertical']),
                'horizontal_count': len(detections['horizontal']),
                'intersection_count': len(intersections),
                'dimensions': dimensions,
                'cement_mixture': mixture,
                'detections_summary': {
                    'vertical_rebars': detections['vertical'],
                    'horizontal_rebars': detections['horizontal'],
                    'intersections': intersections
                }
            }
            
            print(f"✅ Processing complete: {result['vertical_count']}V + {result['horizontal_count']}H")
            return result
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            return {'success': False, 'error': f'Processing failed: {str(e)}'}
    
    def _filter_detections(self, boxes, scores, classes, masks):
        """Filter detections for front_vertical and front_horizontal only"""
        class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        
        vertical_rebars = []
        horizontal_rebars = []
        
        for i, (box, score, cls, mask) in enumerate(zip(boxes, scores, classes, masks)):
            class_name = class_names[cls]
            
            detection = {
                'id': i,
                'class_name': class_name,
                'confidence': float(score),
                'bbox': box.tolist(),
                'mask': mask,
                'centroid': self._get_centroid(mask)
            }
            
            if class_name == 'front_vertical':
                vertical_rebars.append(detection)
            elif class_name == 'front_horizontal':
                horizontal_rebars.append(detection)
        
        # Validate counts (allow some flexibility)
        vertical_ok = 1 <= len(vertical_rebars) <= 3
        horizontal_ok = 8 <= len(horizontal_rebars) <= 15
        
        if not vertical_ok:
            return {
                'success': False,
                'error': f'Expected 2 vertical rebars, found {len(vertical_rebars)}'
            }
        
        if not horizontal_ok:
            return {
                'success': False,
                'error': f'Expected ~11 horizontal rebars, found {len(horizontal_rebars)}'
            }
        
        print(f"   ✅ Filtered: {len(vertical_rebars)} vertical, {len(horizontal_rebars)} horizontal")
        
        return {
            'success': True,
            'vertical': vertical_rebars,
            'horizontal': horizontal_rebars
        }
    
    def _get_centroid(self, mask):
        """Get centroid of binary mask"""
        try:
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            return (0, 0)
        except:
            return (0, 0)
    
    def _find_intersections(self, vertical_rebars, horizontal_rebars):
        """Find intersections between vertical and horizontal rebars"""
        intersections = []
        
        for v_idx, vertical in enumerate(vertical_rebars):
            for h_idx, horizontal in enumerate(horizontal_rebars):
                intersection_mask = np.logical_and(vertical['mask'], horizontal['mask'])
                intersection_area = np.sum(intersection_mask)
                
                if intersection_area > 20:
                    intersection = {
                        'vertical_id': v_idx,
                        'horizontal_id': h_idx,
                        'vertical_confidence': vertical['confidence'],
                        'horizontal_confidence': horizontal['confidence'],
                        'area': float(intersection_area),
                        'centroid': self._get_centroid(intersection_mask),
                        'is_valid': intersection_area > 50
                    }
                    intersections.append(intersection)
        
        intersections.sort(key=lambda x: x['area'], reverse=True)
        print(f"   🔗 Found {len(intersections)} intersections")
        return intersections
    
    def _calculate_dimensions(self, vertical_rebars, horizontal_rebars, intersections, image_shape):
        """Calculate rebar column dimensions with 4.5cm offset"""
        try:
            print("📏 Calculating dimensions with offset...")
            
            height, width, _ = image_shape
            
            # Calculate width from vertical rebar spacing
            if len(vertical_rebars) >= 2:
                verticals_sorted = sorted(vertical_rebars, key=lambda v: v['centroid'][0])
                left_vertical = verticals_sorted[0]
                right_vertical = verticals_sorted[-1]
                
                left_edge = left_vertical['bbox'][0]
                right_edge = right_vertical['bbox'][2]
                rebar_span_px = right_edge - left_edge
            else:
                if vertical_rebars:
                    bbox = vertical_rebars[0]['bbox']
                    rebar_span_px = bbox[2] - bbox[0]
                else:
                    rebar_span_px = width * 0.3
            
            # Calculate height from horizontal rebar spacing
            if len(horizontal_rebars) >= 2:
                horizontals_sorted = sorted(horizontal_rebars, key=lambda h: h['centroid'][1])
                top_horizontal = horizontals_sorted[0]
                bottom_horizontal = horizontals_sorted[-1]
                
                top_edge = top_horizontal['bbox'][1]
                bottom_edge = bottom_horizontal['bbox'][3]
                rebar_height_px = bottom_edge - top_edge
            else:
                rebar_height_px = height * 0.8
            
            # Convert pixels to centimeters
            pixel_to_cm_factor = 0.12
            
            internal_width_cm = rebar_span_px * pixel_to_cm_factor
            internal_height_cm = rebar_height_px * pixel_to_cm_factor
            
            # Add offset for square column dimensions
            column_width_cm = internal_width_cm + (2 * self.offset_cm)
            column_length_cm = column_width_cm  # Square column
            column_height_cm = internal_height_cm
            
            # Ensure minimum realistic dimensions
            column_width_cm = max(column_width_cm, 15.0)
            column_length_cm = max(column_length_cm, 15.0)
            column_height_cm = max(column_height_cm, 50.0)
            
            # Calculate volume
            volume_cm3 = column_length_cm * column_width_cm * column_height_cm
            
            display_string = (f"{column_length_cm:.1f}cm x {column_width_cm:.1f}cm x "
                            f"{column_height_cm:.1f}cm = {volume_cm3:.0f}cm³")
            
            result = {
                'length': round(column_length_cm, 1),
                'width': round(column_width_cm, 1),
                'height': round(column_height_cm, 1),
                'unit': 'cm',
                'volume': round(volume_cm3, 1),
                'display': display_string,
                'method': 'intersection_based_with_offset',
                'offset_applied': self.offset_cm,
                'internal_dimensions': {
                    'width_cm': round(internal_width_cm, 1),
                    'height_cm': round(internal_height_cm, 1)
                },
                'pixel_measurements': {
                    'width_px': rebar_span_px,
                    'height_px': rebar_height_px,
                    'conversion_factor': pixel_to_cm_factor
                }
            }
            
            print(f"   Internal: {internal_width_cm:.1f} x {internal_height_cm:.1f} cm")
            print(f"   With offset: {display_string}")
            print(f"   Applied offset: +{self.offset_cm}cm each side")
            
            return result
            
        except Exception as e:
            print(f"❌ Dimension calculation error: {str(e)}")
            default_side = 25 + (2 * self.offset_cm)
            default_volume = default_side * default_side * 200
            
            return {
                'length': default_side,
                'width': default_side,
                'height': 200.0,
                'unit': 'cm',
                'volume': default_volume,
                'display': f'{default_side}cm x {default_side}cm x 200cm = {default_volume:.0f}cm³',
                'method': 'fallback_with_offset',
                'offset_applied': self.offset_cm
            }
    
    def _calculate_cement_mixture(self, dimensions):
        """Calculate cement mixture based on column volume"""
        try:
            volume_cm3 = dimensions.get('volume', 0)
            volume_m3 = volume_cm3 / 1000000
            
            # Standard Philippine concrete mix ratios
            cement_ratio = 1
            sand_ratio = 2
            aggregate_ratio = 3
            
            concrete_volume_factor = 1.4
            total_concrete_volume_m3 = volume_m3 * concrete_volume_factor
            
            total_parts = cement_ratio + sand_ratio + aggregate_ratio
            volume_per_part = total_concrete_volume_m3 / total_parts
            
            cement_volume_m3 = volume_per_part * cement_ratio
            sand_volume_m3 = volume_per_part * sand_ratio
            aggregate_volume_m3 = volume_per_part * aggregate_ratio
            
            cement_bags = cement_volume_m3 / 0.035
            
            result = {
                'cement_ratio': cement_ratio,
                'sand_ratio': sand_ratio,
                'aggregate_ratio': aggregate_ratio,
                'ratio_string': f'{cement_ratio} Cement : {sand_ratio} Sand : {aggregate_ratio} Aggregate',
                'cement_bags': round(cement_bags, 2),
                'sand_volume_m3': round(sand_volume_m3, 4),
                'aggregate_volume_m3': round(aggregate_volume_m3, 4),
                'total_concrete_volume_m3': round(total_concrete_volume_m3, 4),
                'column_volume_m3': round(volume_m3, 4),
                'wastage_factor': concrete_volume_factor
            }
            
            print(f"   📦 Cement needed: {result['cement_bags']} bags")
            print(f"   🏗️ Total concrete: {result['total_concrete_volume_m3']} m³")
            
            return result
            
        except Exception as e:
            print(f"❌ Cement calculation error: {str(e)}")
            return {
                'cement_ratio': 1,
                'sand_ratio': 2,
                'aggregate_ratio': 3,
                'ratio_string': '1 Cement : 2 Sand : 3 Aggregate',
                'cement_bags': 3.0,
                'sand_volume_m3': 0.0002,
                'aggregate_volume_m3': 0.0003,
                'total_concrete_volume_m3': 0.0007
            }
    
    def create_analysis_visualization(self, image, vertical_rebars, horizontal_rebars, 
                                    intersections, dimensions):
        """Create visualization showing detected rebars and measurements"""
        try:
            print("🎨 Creating analysis visualization...")
            
            result_image = image.copy()
            
            # Draw vertical rebars in green
            for i, vertical in enumerate(vertical_rebars):
                mask = vertical['mask']
                bbox = vertical['bbox']
                confidence = vertical['confidence']
                
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask] = [0, 255, 0]  # Green
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
                
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"V{i+1} ({confidence:.2f})"
                cv2.putText(result_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw horizontal rebars in red
            for i, horizontal in enumerate(horizontal_rebars):
                mask = horizontal['mask']
                bbox = horizontal['bbox']
                confidence = horizontal['confidence']
                
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask] = [0, 0, 255]  # Red
                result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
                
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                if i < 5:  # Only show first few labels to avoid clutter
                    label = f"H{i+1} ({confidence:.2f})"
                    cv2.putText(result_image, label, (x2+5, y1+15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw intersection points in yellow
            valid_intersections = [i for i in intersections if i['is_valid']]
            for i, intersection in enumerate(valid_intersections[:10]):
                cx, cy = intersection['centroid']
                cv2.circle(result_image, (cx, cy), 6, (0, 255, 255), -1)  # Yellow
                cv2.circle(result_image, (cx, cy), 8, (255, 255, 255), 2)  # White border
            
            # Add information overlay
            self._add_info_overlay(result_image, dimensions, len(vertical_rebars), 
                                 len(horizontal_rebars), len(valid_intersections))
            
            print(f"   ✅ Visualization created with {len(vertical_rebars)}V + {len(horizontal_rebars)}H")
            return result_image
            
        except Exception as e:
            print(f"❌ Visualization error: {str(e)}")
            return image
    
    def _add_info_overlay(self, image, dimensions, v_count, h_count, int_count):
        """Add information overlay to the image"""
        try:
            # Main dimensions box
            cv2.rectangle(image, (10, 10), (620, 100), (0, 0, 0), -1)
            cv2.rectangle(image, (10, 10), (620, 100), (255, 255, 255), 2)
            
            dimensions_text = dimensions['display']
            cv2.putText(image, dimensions_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            offset_text = f"Offset Applied: +{dimensions['offset_applied']}cm each side"
            cv2.putText(image, offset_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            method_text = f"Method: {dimensions['method']}"
            cv2.putText(image, method_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            
            # Detection counts box
            height = image.shape[0]
            cv2.rectangle(image, (10, height-80), (400, height-10), (0, 0, 0), -1)
            cv2.rectangle(image, (10, height-80), (400, height-10), (255, 255, 255), 2)
            
            counts_text = f"Detected: {v_count} Vertical + {h_count} Horizontal"
            cv2.putText(image, counts_text, (20, height-55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            intersections_text = f"Valid Intersections: {int_count}"
            cv2.putText(image, intersections_text, (20, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        except Exception as e:
            print(f"⚠️ Error adding info overlay: {e}")


class AIService:
    """Handles AI model loading, inference, and simplified rebar analysis"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.metadata = None
        
        # Simplified rebar classes
        self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        self.num_classes = 3
        self.detection_threshold = 0.2  # Lower threshold for better detection
        self.training_input_size = (480, 640)  # width x height
        
        # Target detection counts
        self.target_detections = {
            "front_vertical": 2,
            "front_horizontal": 11
        }
        self.offset_cm = 4.5  # 4.5cm offset for square columns
        
        print("🤖 AI Service initialized with simplified rebar detection")
        print(f"   Target: {self.target_detections}")
        print(f"   Offset: {self.offset_cm}cm per side")
        print("   📝 SAVE MODE: Only analyzed images with overlays")
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model with optimized settings"""
        try:
            if not DETECTRON2_AVAILABLE:
                print("❌ Detectron2 not available, using placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False
            
            print("🔄 Loading Detectron2 model with simplified configuration...")
            
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Optimized model settings
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold
            self.cfg.MODEL.DEVICE = "cpu"  # Raspberry Pi 5
            
            # Optimized for 480x640 input
            self.cfg.INPUT.MIN_SIZE_TEST = 640
            self.cfg.INPUT.MAX_SIZE_TEST = 640
            
            print("🔄 Creating predictor...")
            self.predictor = DefaultPredictor(self.cfg)
            
            # Set up metadata
            self.metadata = MetadataCatalog.get("simplified_rebar_dataset")
            self.metadata.thing_classes = self.class_names
            self.metadata.thing_colors = [
                (128, 128, 128),  # back_horizontal - Gray (ignored)
                (255, 0, 0),      # front_horizontal - Red
                (0, 255, 0),      # front_vertical - Green
            ]
            
            self.model_loaded = True
            print("✅ Simplified AI Model loaded successfully!")
            print(f"   Model: {self.model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Threshold: {self.detection_threshold}")
            print(f"   Input size: {self.training_input_size[0]}x{self.training_input_size[1]}")
            
            # Test model
            test_image = np.zeros((640, 480, 3), dtype=np.uint8)
            try:
                test_output = self.predictor(test_image)
                print("✅ Model inference test successful!")
            except Exception as e:
                print(f"⚠️  Model test failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_data=None, image_path=None):
        """
        Analyze image for simplified rebar detection
        SAVES: Only analyzed images with AI overlays (no originals)
        """
        try:
            print(f"🔍 Starting simplified AI analysis (analyzed image only mode)...")
            
            # Handle input
            if image_data is not None:
                print("📸 Using direct frame data (no original saved)")
                image = image_data.copy()
                source = "camera_frame"
            elif image_path and os.path.exists(image_path):
                print(f"📁 Loading image from: {image_path}")
                image = cv2.imread(image_path)
                source = "file"
                if image is None:
                    return {'success': False, 'error': 'Failed to load image file'}
            else:
                return {'success': False, 'error': 'No image data provided'}
            
            print(f"📐 Image loaded: {image.shape} from {source}")
            
            # Ensure correct size (480x640)
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                print(f"⚙️  Resizing from {width}x{height} to 480x640")
                image = cv2.resize(image, (480, 640))
            
            # Run detection
            if self.model_loaded and DETECTRON2_AVAILABLE:
                result = self._analyze_with_simplified_model(image)
            else:
                print("⚠️  Model not available, using placeholder")
                result = self._analyze_simplified_placeholder(image)
            
            if result['success'] and 'analyzed_image_path' in result:
                filename = os.path.basename(result['analyzed_image_path'])
                print(f"✅ Analysis complete. ONLY analyzed image saved: {filename}")
            
            return result
                
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
            traceback.print_exc()
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}
    
    def _analyze_with_simplified_model(self, image):
        """Run simplified rebar analysis with real trained model"""
        try:
            print("🤖 Running simplified Detectron2 inference...")
            
            # Run inference
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            num_detections = len(instances)
            print(f"🎯 Found {num_detections} total detections")
            
            if num_detections == 0:
                return {
                    'success': False,
                    'error': 'No rebar structures detected',
                    'no_detection': True
                }
            
            # Use simplified processor
            processor = SimplifiedRebarProcessor()
            results = processor.process_detections(instances, image.shape)
            
            if not results['success']:
                return {
                    'success': False,
                    'error': results['error'],
                    'no_detection': True
                }
            
            # Create visualization
            visualization_image = processor.create_analysis_visualization(
                image,
                results['detections_summary']['vertical_rebars'],
                results['detections_summary']['horizontal_rebars'], 
                results['detections_summary']['intersections'],
                results['dimensions']
            )
            
            # Save visualization (ONLY file saved)
            analyzed_image_path = self._save_visualization_image(visualization_image)
            
            if not analyzed_image_path:
                return {
                    'success': False,
                    'error': 'Failed to save analysis visualization'
                }
            
            # Format response
            response_data = {
                'success': True,
                'detections': self._format_detections_for_frontend(results),
                'num_detections': results['vertical_count'] + results['horizontal_count'],
                'dimensions': results['dimensions'],
                'cement_mixture': results['cement_mixture'],
                'analyzed_image_path': analyzed_image_path,
                'model_type': 'simplified_real_model',
                'analysis_summary': {
                    'vertical_rebars': results['vertical_count'],
                    'horizontal_rebars': results['horizontal_count'],
                    'intersections': results['intersection_count'],
                    'target_achieved': (
                        results['vertical_count'] == 2 and 
                        results['horizontal_count'] == 11
                    ),
                    'offset_applied': f"{self.offset_cm}cm each side"
                }
            }
            
            print(f"✅ Simplified analysis complete:")
            print(f"   Vertical: {results['vertical_count']}, Horizontal: {results['horizontal_count']}")
            print(f"   Intersections: {results['intersection_count']}")
            print(f"   Dimensions: {results['dimensions']['display']}")
            
            return response_data
            
        except Exception as e:
            print(f"❌ Simplified model inference error: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Simplified inference failed: {str(e)}'
            }
    
    def _format_detections_for_frontend(self, results):
        """Format detections for frontend display"""
        formatted_detections = []
        
        # Add vertical rebars
        for i, vertical in enumerate(results['detections_summary']['vertical_rebars']):
            formatted_detections.append({
                'class_id': 2,  # front_vertical class id
                'class_name': 'front_vertical',
                'confidence': vertical['confidence'],
                'bbox': vertical['bbox'],
                'rebar_id': f'V{i+1}',
                'type': 'vertical'
            })
        
        # Add horizontal rebars
        for i, horizontal in enumerate(results['detections_summary']['horizontal_rebars']):
            formatted_detections.append({
                'class_id': 1,  # front_horizontal class id  
                'class_name': 'front_horizontal',
                'confidence': horizontal['confidence'],
                'bbox': horizontal['bbox'],
                'rebar_id': f'H{i+1}',
                'type': 'horizontal'
            })
        
        return formatted_detections
    
    def _save_visualization_image(self, visualization_image):
        """Save visualization image (ONLY method that saves images)"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'simplified_rebar_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, visualization_image)
            
            if success:
                file_size = os.path.getsize(output_path)
                print(f"✅ ANALYZED IMAGE SAVED (ONLY COPY):")
                print(f"   📁 File: {filename}")
                print(f"   💾 Size: {file_size / 1024:.1f} KB")
                print(f"   🎯 Contains: Simplified rebar detection overlays")
                return output_path
            else:
                print("❌ Failed to save simplified visualization")
                return None
                
        except Exception as e:
            print(f"❌ Error saving visualization: {str(e)}")
            return None
    
    def _analyze_simplified_placeholder(self, image):
        """Simplified placeholder with realistic rebar layout"""
        print("📝 Creating simplified placeholder...")
        
        # Create realistic placeholder data
        vertical_rebars = [
            {
                'class_name': 'front_vertical',
                'confidence': 0.85,
                'bbox': [120, 50, 140, 590],
                'rebar_id': 'V1'
            },
            {
                'class_name': 'front_vertical', 
                'confidence': 0.82,
                'bbox': [340, 50, 360, 590],
                'rebar_id': 'V2'
            }
        ]
        
        horizontal_rebars = []
        for i in range(11):
            y_pos = 70 + i * 45
            horizontal_rebars.append({
                'class_name': 'front_horizontal',
                'confidence': 0.75 + (i * 0.02),
                'bbox': [100, y_pos, 380, y_pos + 10],
                'rebar_id': f'H{i+1}'
            })
        
        # Calculate dimensions with offset
        internal_width = 26.0  # cm
        final_width = internal_width + (2 * self.offset_cm)  # 35.0 cm
        volume = final_width * final_width * 200  # Square column
        
        dimensions = {
            'length': final_width,
            'width': final_width,
            'height': 200.0,
            'unit': 'cm',
            'volume': volume,
            'display': f'{final_width}cm x {final_width}cm x 200cm = {volume:.0f}cm³',
            'method': 'simplified_placeholder',
            'offset_applied': self.offset_cm
        }
        
        mixture = {
            'cement_ratio': 1,
            'sand_ratio': 2,
            'aggregate_ratio': 3,
            'ratio_string': '1 Cement : 2 Sand : 3 Aggregate',
            'cement_bags': round(volume * 1.4 / 1000000 / 6 / 0.035, 2),
            'sand_volume_m3': round(volume * 1.4 / 1000000 * 2 / 6, 4),
            'aggregate_volume_m3': round(volume * 1.4 / 1000000 * 3 / 6, 4),
            'total_concrete_volume_m3': round(volume * 1.4 / 1000000, 4)
        }
        
        # Create placeholder visualization
        analyzed_image_path = self._create_simplified_placeholder_visualization(image, dimensions)
        
        return {
            'success': True,
            'placeholder': True,
            'detections': vertical_rebars + horizontal_rebars,
            'num_detections': 13,  # 2 + 11
            'dimensions': dimensions,
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_image_path,
            'model_type': 'simplified_placeholder',
            'analysis_summary': {
                'vertical_rebars': 2,
                'horizontal_rebars': 11,
                'intersections': 22,  # 2 * 11
                'target_achieved': True,
                'offset_applied': f"{self.offset_cm}cm each side"
            }
        }
    
    def _create_simplified_placeholder_visualization(self, image, dimensions):
        """Create simplified placeholder visualization"""
        try:
            print("🎨 Creating simplified placeholder visualization...")
            
            result_image = image.copy()
            
            # Draw placeholder vertical rebars (green)
            cv2.rectangle(result_image, (120, 50), (140, 590), (0, 255, 0), 3)
            cv2.putText(result_image, "V1 (0.85)", (120, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.rectangle(result_image, (340, 50), (360, 590), (0, 255, 0), 3)
            cv2.putText(result_image, "V2 (0.82)", (340, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw placeholder horizontal rebars (red)
            for i in range(11):
                y = 70 + i * 45
                cv2.rectangle(result_image, (100, y), (380, y+10), (255, 0, 0), 3)
                if i < 5:  # Only label first few
                    cv2.putText(result_image, f"H{i+1}", (385, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw intersection points (yellow)
            for i in range(11):
                y = 75 + i * 45
                cv2.circle(result_image, (130, y), 4, (0, 255, 255), -1)  # Left intersections
                cv2.circle(result_image, (350, y), 4, (0, 255, 255), -1)  # Right intersections
            
            # Add information overlay
            cv2.rectangle(result_image, (10, 10), (620, 100), (0, 0, 0), -1)
            cv2.rectangle(result_image, (10, 10), (620, 100), (255, 255, 255), 2)
            
            cv2.putText(result_image, dimensions['display'], (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_image, f"Offset: +{dimensions['offset_applied']}cm each side (PLACEHOLDER)", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(result_image, "Method: simplified_placeholder", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            
            # Detection counts
            height = result_image.shape[0]
            cv2.rectangle(result_image, (10, height-80), (400, height-10), (0, 0, 0), -1)
            cv2.rectangle(result_image, (10, height-80), (400, height-10), (255, 255, 255), 2)
            
            cv2.putText(result_image, "Detected: 2 Vertical + 11 Horizontal", (20, height-55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, "Valid Intersections: 22", (20, height-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save placeholder visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'simplified_placeholder_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                file_size = os.path.getsize(output_path)
                print(f"✅ PLACEHOLDER ANALYZED IMAGE SAVED (ONLY COPY):")
                print(f"   📁 File: {filename}")
                print(f"   💾 Size: {file_size / 1024:.1f} KB")
                print(f"   🎯 Contains: Simplified placeholder overlays")
                return output_path
            else:
                print("❌ Failed to save placeholder visualization")
                return None
                
        except Exception as e:
            print(f"❌ Placeholder visualization error: {str(e)}")
            return None
    
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
            'model_type': 'simplified_rebar_detection',
            'target_detections': self.target_detections,
            'offset_cm': self.offset_cm,
            'save_mode': 'analyzed_images_only'
        }
    
    def test_model(self, test_image_path=None):
        """Test the simplified model with a sample image"""
        try:
            if not test_image_path:
                # Use most recent captured image
                captured_dir = config.UPLOAD_FOLDER
                if os.path.exists(captured_dir):
                    images = [f for f in os.listdir(captured_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        # Sort by modification time, get most recent
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
            
            print(f"🧪 Testing simplified model with: {test_image_path}")
            
            # Run analysis (saves only analyzed image)
            result = self.analyze_image(image_path=test_image_path)
            
            if result['success']:
                model_type = result.get('model_type', 'unknown')
                print(f"✅ Simplified model test successful! (Type: {model_type})")
                print("   Only analyzed image saved (no duplicates)")
                
                return {
                    'success': True,
                    'test_image': test_image_path,
                    'detections_found': result.get('num_detections', 0),
                    'model_type': model_type,
                    'analyzed_image_saved': result.get('analyzed_image_path'),
                    'save_mode': 'analyzed_only',
                    'analysis_summary': result.get('analysis_summary', {}),
                    'dimensions': result.get('dimensions', {}),
                    'cement_mixture': result.get('cement_mixture', {})
                }
            else:
                print(f"❌ Simplified model test failed: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"❌ Simplified model test error: {str(e)}")
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }


# Test function for validating the simplified approach
def test_simplified_rebar_detection():
    """Test function for simplified rebar detection"""
    try:
        print("🧪 Testing simplified rebar detection setup...")
        
        # Test AI Service initialization
        ai_service = AIService()
        
        print(f"Model loaded: {'✅' if ai_service.model_loaded else '❌'}")
        print(f"Detectron2 available: {'✅' if DETECTRON2_AVAILABLE else '❌'}")
        
        # Get model status
        status = ai_service.get_model_status()
        print("\n📊 Model Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Test with dummy image if model is available
        if ai_service.model_loaded:
            print("\n🖼️  Testing with dummy image...")
            test_image = np.zeros((640, 480, 3), dtype=np.uint8)
            
            # Add some fake rebar-like patterns
            cv2.rectangle(test_image, (100, 50), (120, 590), (255, 255, 255), -1)  # Vertical 1
            cv2.rectangle(test_image, (380, 50), (400, 590), (255, 255, 255), -1)  # Vertical 2
            
            for i in range(11):
                y = 70 + i * 45
                cv2.rectangle(test_image, (80, y), (420, y+15), (255, 255, 255), -1)  # Horizontals
            
            result = ai_service.analyze_image(image_data=test_image)
            
            if result['success']:
                print("✅ Dummy image test successful!")
                print(f"   Detections: {result.get('num_detections', 0)}")
                print(f"   Dimensions: {result['dimensions']['display']}")
                print(f"   Analysis: {result.get('analysis_summary', {})}")
            else:
                print(f"❌ Dummy image test failed: {result.get('error')}")
        
        print("\n✅ Simplified rebar detection test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    test_simplified_rebar_detection()
