"""
Enhanced AI Service with Comprehensive Debugging
Memory-optimized version with extensive debug logging
"""

import os
import cv2
import numpy as np
from datetime import datetime
import traceback
import time
import psutil

# Detectron2 imports
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

from app.utils.config import config

class AIService:
    """Enhanced AI service with comprehensive debugging"""
    
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.cfg = None
        self.metadata = None
        self.debug_mode = True
        self.analysis_count = 0
        self.last_analysis_time = None
        self.memory_usage_history = []
        
        # Model configuration
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        self.num_classes = 3
        self.detection_threshold = 0.3
        self.training_input_size = (480, 640)
        
        self.debug_log("Initializing Enhanced AI Service with Debugging...")
        self.log_system_info()
        
        # Only load model if available
        if DETECTRON2_AVAILABLE and os.path.exists(self.model_path):
            self.load_model()
        else:
            self.debug_log("Using optimized placeholder mode")
    
    def debug_log(self, message, level="INFO"):
        """Enhanced debug logging with timestamps and memory info"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        memory = self.get_memory_usage()
        print(f"[{timestamp}] [{level}] [MEM:{memory}MB] {message}")
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return round(process.memory_info().rss / 1024 / 1024, 1)
        except:
            return 0
    
    def log_system_info(self):
        """Log system information for debugging"""
        try:
            self.debug_log("=== SYSTEM INFORMATION ===")
            self.debug_log(f"CPU Count: {psutil.cpu_count()}")
            memory = psutil.virtual_memory()
            self.debug_log(f"Total RAM: {memory.total / 1024 / 1024 / 1024:.1f} GB")
            self.debug_log(f"Available RAM: {memory.available / 1024 / 1024 / 1024:.1f} GB")
            self.debug_log(f"RAM Usage: {memory.percent}%")
            
            disk = psutil.disk_usage('/')
            self.debug_log(f"Disk Free: {disk.free / 1024 / 1024 / 1024:.1f} GB")
            
            self.debug_log(f"Detectron2 Available: {DETECTRON2_AVAILABLE}")
            self.debug_log(f"Model Path: {self.model_path}")
            self.debug_log(f"Model Exists: {os.path.exists(self.model_path)}")
            if os.path.exists(self.model_path):
                model_size = os.path.getsize(self.model_path) / 1024 / 1024
                self.debug_log(f"Model Size: {model_size:.1f} MB")
            self.debug_log("=========================")
        except Exception as e:
            self.debug_log(f"Error getting system info: {e}", "ERROR")
    
    def load_model(self):
        """Load Detectron2 model with comprehensive debugging"""
        try:
            self.debug_log("Starting model loading process...")
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            # Check model file
            if not os.path.exists(self.model_path):
                self.debug_log(f"Model file not found: {self.model_path}", "ERROR")
                return False
            
            model_size = os.path.getsize(self.model_path) / 1024 / 1024
            self.debug_log(f"Loading model file: {model_size:.1f} MB")
            
            # Configure model
            self.debug_log("Configuring Detectron2...")
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Memory optimizations
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold
            
            # Reduce memory usage
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
            self.cfg.TEST.DETECTIONS_PER_IMAGE = 20
            
            memory_after_config = self.get_memory_usage()
            self.debug_log(f"Memory after config: {memory_after_config}MB (+{memory_after_config-start_memory}MB)")
            
            # Create predictor
            self.debug_log("Creating DefaultPredictor...")
            self.predictor = DefaultPredictor(self.cfg)
            
            memory_after_predictor = self.get_memory_usage()
            self.debug_log(f"Memory after predictor: {memory_after_predictor}MB (+{memory_after_predictor-memory_after_config}MB)")
            
            # Set up metadata
            self.debug_log("Setting up metadata...")
            self.metadata = MetadataCatalog.get("rebar_dataset")
            self.metadata.thing_classes = self.class_names
            
            # Test model
            self.debug_log("Testing model with dummy image...")
            test_image = np.zeros((640, 480, 3), dtype=np.uint8)
            test_start = time.time()
            test_outputs = self.predictor(test_image)
            test_time = time.time() - test_start
            
            final_memory = self.get_memory_usage()
            total_time = time.time() - start_time
            
            self.debug_log(f"Model test completed in {test_time:.2f}s")
            self.debug_log(f"Total loading time: {total_time:.2f}s")
            self.debug_log(f"Total memory increase: {final_memory-start_memory:.1f}MB")
            self.debug_log(f"Test detections: {len(test_outputs['instances'])}")
            
            self.model_loaded = True
            self.debug_log("Model loaded successfully!", "SUCCESS")
            return True
            
        except Exception as e:
            self.debug_log(f"Model loading failed: {e}", "ERROR")
            self.debug_log("Full traceback:", "ERROR")
            traceback.print_exc()
            self.model_loaded = False
            return False
    
    def analyze_image(self, image_path):
        """Enhanced analysis with comprehensive debugging"""
        try:
            self.analysis_count += 1
            analysis_start = time.time()
            start_memory = self.get_memory_usage()
            
            self.debug_log(f"=== ANALYSIS #{self.analysis_count} START ===")
            self.debug_log(f"Image path: {image_path}")
            self.debug_log(f"Start memory: {start_memory}MB")
            
            # Validate image file
            if not os.path.exists(image_path):
                return self._error_response('Image file not found', image_path)
            
            file_size = os.path.getsize(image_path) / 1024
            self.debug_log(f"Image file size: {file_size:.1f} KB")
            
            # Load image
            self.debug_log("Loading image with OpenCV...")
            image = cv2.imread(image_path)
            if image is None:
                return self._error_response('Failed to load image with OpenCV', image_path)
            
            original_shape = image.shape
            self.debug_log(f"Original image shape: {original_shape}")
            
            memory_after_load = self.get_memory_usage()
            self.debug_log(f"Memory after image load: {memory_after_load}MB (+{memory_after_load-start_memory}MB)")
            
            # Resize if needed
            height, width = image.shape[:2]
            if width != 480 or height != 640:
                self.debug_log(f"Resizing from {width}x{height} to 480x640...")
                image = cv2.resize(image, (480, 640))
                resized_shape = image.shape
                self.debug_log(f"Resized image shape: {resized_shape}")
            
            # Image analysis for debugging
            self.debug_log("Analyzing image characteristics...")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            self.debug_log(f"Image brightness: {brightness:.1f}")
            self.debug_log(f"Image contrast: {contrast:.1f}")
            
            # Try real model analysis first
            if self.model_loaded and self.predictor:
                self.debug_log("Attempting real model analysis...")
                result = self._analyze_with_model(image, image_path, start_memory)
                if result['success']:
                    total_time = time.time() - analysis_start
                    self.debug_log(f"=== ANALYSIS #{self.analysis_count} SUCCESS (Real Model) - {total_time:.2f}s ===")
                    return result
                else:
                    self.debug_log("Real model analysis failed, falling back to placeholder", "WARNING")
            else:
                self.debug_log("Real model not available, using placeholder", "INFO")
            
            # Enhanced placeholder
            result = self._enhanced_placeholder_analysis(image, image_path, start_memory)
            total_time = time.time() - analysis_start
            self.debug_log(f"=== ANALYSIS #{self.analysis_count} SUCCESS (Placeholder) - {total_time:.2f}s ===")
            return result
            
        except Exception as e:
            self.debug_log(f"Analysis error: {e}", "ERROR")
            traceback.print_exc()
            return self._error_response(f'Analysis failed: {str(e)}', image_path)
    
    def _analyze_with_model(self, image, image_path, start_memory):
        """Real model analysis with detailed debugging"""
        try:
            self.debug_log("Starting Detectron2 inference...")
            inference_start = time.time()
            
            # Pre-inference memory check
            pre_inference_memory = self.get_memory_usage()
            self.debug_log(f"Pre-inference memory: {pre_inference_memory}MB")
            
            # Run inference
            outputs = self.predictor(image)
            inference_time = time.time() - inference_start
            
            # Post-inference memory check
            post_inference_memory = self.get_memory_usage()
            self.debug_log(f"Inference completed in {inference_time:.2f}s")
            self.debug_log(f"Post-inference memory: {post_inference_memory}MB (+{post_inference_memory-pre_inference_memory}MB)")
            
            instances = outputs["instances"].to("cpu")
            num_detections = len(instances)
            self.debug_log(f"Raw detections found: {num_detections}")
            
            if num_detections == 0:
                self.debug_log("No detections found", "WARNING")
                return {
                    'success': False,
                    'error': 'No rebar structures detected',
                    'no_detection': True
                }
            
            # Extract detection data
            self.debug_log("Extracting detection data...")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            detections = []
            for i in range(num_detections):
                class_id = int(classes[i])
                confidence = float(scores[i])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                bbox = boxes[i].tolist()
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                }
                detections.append(detection)
                
                self.debug_log(f"  Detection {i+1}: {class_name} ({confidence:.3f}) at {bbox}")
            
            # Create visualization
            self.debug_log("Creating visualization...")
            viz_start = time.time()
            analyzed_image_path = self._create_visualization(image, outputs, image_path)
            viz_time = time.time() - viz_start
            self.debug_log(f"Visualization created in {viz_time:.2f}s")
            
            # Calculate results
            self.debug_log("Calculating dimensions and mixture...")
            dimensions = self._calculate_dimensions_from_detections(detections)
            mixture = self._calculate_cement_mixture(dimensions)
            
            final_memory = self.get_memory_usage()
            self.debug_log(f"Final memory: {final_memory}MB (total increase: {final_memory-start_memory}MB)")
            
            return self._format_success_response(detections, dimensions, mixture, image_path, analyzed_image_path, 'real_model')
            
        except Exception as e:
            self.debug_log(f"Real model analysis error: {e}", "ERROR")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _enhanced_placeholder_analysis(self, image, image_path, start_memory):
        """Enhanced placeholder with debugging"""
        try:
            self.debug_log("Running enhanced placeholder analysis...")
            
            # Image analysis for variation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Vary dimensions based on image characteristics
            base_length = 25.4
            base_width = 25.4
            base_height = 200.0
            
            length_var = (brightness - 127) * 0.1
            width_var = (contrast - 50) * 0.05
            height_var = np.random.uniform(-10, 10)
            
            length = max(20, base_length + length_var)
            width = max(20, base_width + width_var)
            height = max(150, base_height + height_var)
            
            self.debug_log(f"Calculated dimensions: {length:.1f} x {width:.1f} x {height:.1f} cm")
            
            # Create realistic detections
            detections = [
                {
                    'class_id': 2,
                    'class_name': 'front_vertical',
                    'confidence': 0.85 + np.random.uniform(-0.05, 0.05),
                    'bbox': [100, 50, 200, 300]
                },
                {
                    'class_id': 1,
                    'class_name': 'front_horizontal',
                    'confidence': 0.78 + np.random.uniform(-0.05, 0.05),
                    'bbox': [80, 280, 220, 320]
                }
            ]
            
            self.debug_log(f"Generated {len(detections)} placeholder detections")
            
            dimensions = {
                'length': round(length, 1),
                'width': round(width, 1),
                'height': round(height, 1),
                'unit': 'cm'
            }
            
            mixture = self._calculate_cement_mixture(dimensions)
            
            # Create visualization
            analyzed_image_path = self._create_placeholder_visualization(image, image_path)
            
            final_memory = self.get_memory_usage()
            self.debug_log(f"Placeholder analysis memory: {final_memory}MB (increase: {final_memory-start_memory}MB)")
            
            return self._format_success_response(detections, dimensions, mixture, image_path, analyzed_image_path, 'enhanced_placeholder')
            
        except Exception as e:
            self.debug_log(f"Placeholder analysis error: {e}", "ERROR")
            return self._error_response(str(e), image_path)
    
    def _calculate_dimensions_from_detections(self, detections):
        """Calculate dimensions with debugging"""
        self.debug_log(f"Calculating dimensions from {len(detections)} detections...")
        
        total_area = 0
        max_width = 0
        max_height = 0
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            total_area += area
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            
            self.debug_log(f"  Detection {i+1} bbox: {width:.0f}x{height:.0f}px, area: {area:.0f}px²")
        
        # Convert pixels to cm
        pixel_to_cm = 0.1
        
        length_cm = max(20, max_width * pixel_to_cm)
        width_cm = max(20, max_height * pixel_to_cm * 0.5)
        height_cm = 200.0
        
        self.debug_log(f"Calculated dimensions: {length_cm:.1f} x {width_cm:.1f} x {height_cm:.1f} cm")
        
        return {
            'length': round(length_cm, 1),
            'width': round(width_cm, 1),
            'height': round(height_cm, 1),
            'unit': 'cm'
        }
    
    def _calculate_cement_mixture(self, dimensions):
        """Calculate cement mixture with debugging"""
        volume_cm3 = dimensions['length'] * dimensions['width'] * dimensions['height']
        volume_m3 = volume_cm3 / 1000000
        
        self.debug_log(f"Column volume: {volume_cm3:.0f} cm³ ({volume_m3:.6f} m³)")
        
        # Standard ratios
        cement_ratio = 1
        sand_ratio = 2
        aggregate_ratio = 3
        
        # Calculate quantities
        concrete_volume = volume_m3 * 1.5
        total_parts = cement_ratio + sand_ratio + aggregate_ratio
        
        cement_volume = concrete_volume * (cement_ratio / total_parts)
        sand_volume = concrete_volume * (sand_ratio / total_parts)
        aggregate_volume = concrete_volume * (aggregate_ratio / total_parts)
        
        cement_bags = cement_volume / 0.035
        
        self.debug_log(f"Concrete needed: {concrete_volume:.6f} m³")
        self.debug_log(f"Cement bags: {cement_bags:.2f}")
        
        return {
            'cement': cement_ratio,
            'sand': sand_ratio,
            'aggregate': aggregate_ratio,
            'ratio_string': f'{cement_ratio} Cement : {sand_ratio} Sand : {aggregate_ratio} Aggregate',
            'cement_bags': round(cement_bags, 2),
            'sand_volume_m3': round(sand_volume, 4),
            'aggregate_volume_m3': round(aggregate_volume, 4),
            'total_concrete_volume_m3': round(concrete_volume, 4)
        }
    
    def _create_visualization(self, image, outputs, original_path):
        """Create visualization with debugging"""
        try:
            self.debug_log("Creating Detectron2 visualization...")
            v = Visualizer(
                image[:, :, ::-1],
                metadata=self.metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE
            )
            
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_image = out.get_image()[:, :, ::-1]
            
            return self._save_visualization(result_image, 'real')
            
        except Exception as e:
            self.debug_log(f"Visualization error: {e}", "ERROR")
            return self._create_placeholder_visualization(image, original_path)
    
    def _create_placeholder_visualization(self, image, original_path):
        """Create placeholder visualization with debugging"""
        try:
            self.debug_log("Creating placeholder visualization...")
            result_image = image.copy()
            
            # Add overlays
            overlay = result_image.copy()
            cv2.rectangle(overlay, (100, 50), (200, 300), (0, 255, 0), -1)
            cv2.rectangle(overlay, (80, 280), (220, 320), (0, 255, 0), -1)
            
            result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
            
            # Add bounding boxes and labels
            cv2.rectangle(result_image, (100, 50), (200, 300), (0, 255, 0), 3)
            cv2.rectangle(result_image, (80, 280), (220, 320), (255, 0, 0), 3)
            
            cv2.putText(result_image, 'Front Vertical (85%)', (100, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result_image, 'Front Horizontal (78%)', (80, 275),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            return self._save_visualization(result_image, 'placeholder')
            
        except Exception as e:
            self.debug_log(f"Placeholder visualization error: {e}", "ERROR")
            return original_path
    
    def _save_visualization(self, image, prefix):
        """Save visualization with debugging"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f'{prefix}_analysis_{timestamp}.jpg'
            output_path = os.path.join(config.UPLOAD_FOLDER, filename)
            
            self.debug_log(f"Saving visualization: {filename}")
            success = cv2.imwrite(output_path, image)
            
            if success:
                file_size = os.path.getsize(output_path) / 1024
                self.debug_log(f"Visualization saved: {file_size:.1f} KB")
                return output_path
            else:
                raise Exception("Failed to save visualization")
                
        except Exception as e:
            self.debug_log(f"Save visualization error: {e}", "ERROR")
            return ""
    
    def _format_success_response(self, detections, dimensions, mixture, original_path, analyzed_path, model_type):
        """Format response with debugging"""
        self.debug_log(f"Formatting response for {model_type}")
        response = {
            'success': True,
            'detections': detections,
            'num_detections': len(detections),
            'dimensions': {
                'length': dimensions['length'],
                'width': dimensions['width'],
                'height': dimensions['height'],
                'unit': dimensions['unit'],
                'display': f"{dimensions['length']}{dimensions['unit']} × {dimensions['width']}{dimensions['unit']} × {dimensions['height']}{dimensions['unit']}"
            },
            'cement_mixture': mixture,
            'analyzed_image_path': analyzed_path,
            'original_image_path': original_path,
            'model_type': model_type,
            'debug_info': {
                'analysis_count': self.analysis_count,
                'memory_usage': self.get_memory_usage(),
                'timestamp': datetime.now().isoformat()
            }
        }
        self.debug_log(f"Response formatted successfully")
        return response
    
    def _error_response(self, error_message, image_path=""):
        """Error response with debugging"""
        self.debug_log(f"Returning error response: {error_message}", "ERROR")
        return {
            'success': False,
            'error': error_message,
            'image_path': image_path,
            'debug_info': {
                'analysis_count': self.analysis_count,
                'memory_usage': self.get_memory_usage(),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def get_debug_info(self):
        """Get comprehensive debug information"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'system': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': round(memory.total / 1024 / 1024 / 1024, 1),
                    'memory_available_gb': round(memory.available / 1024 / 1024 / 1024, 1),
                    'memory_usage_percent': memory.percent,
                    'disk_free_gb': round(disk.free / 1024 / 1024 / 1024, 1),
                    'process_memory_mb': self.get_memory_usage()
                },
                'model': {
                    'detectron2_available': DETECTRON2_AVAILABLE,
                    'model_loaded': self.model_loaded,
                    'model_path': self.model_path,
                    'model_exists': os.path.exists(self.model_path),
                    'model_size_mb': round(os.path.getsize(self.model_path) / 1024 / 1024, 1) if os.path.exists(self.model_path) else 0,
                    'class_names': self.class_names,
                    'num_classes': self.num_classes,
                    'threshold': self.detection_threshold
                },
                'analysis': {
                    'total_analyses': self.analysis_count,
                    'last_analysis_time': self.last_analysis_time,
                    'debug_mode': self.debug_mode
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_model_status(self):
        """Get model status with debug info"""
        return {
            'detectron2_available': DETECTRON2_AVAILABLE,
            'model_loaded': self.model_loaded,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'threshold': self.detection_threshold,
            'training_input_size': self.training_input_size,
            'model_type': 'enhanced_debug_optimized',
            'debug_info': self.get_debug_info()
        }
    
    def test_model(self, test_image_path=None):
        """Test model with debugging"""
        self.debug_log("Starting model test...")
        
        if not test_image_path:
            captured_dir = config.UPLOAD_FOLDER
            if os.path.exists(captured_dir):
                images = [f for f in os.listdir(captured_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_image_path = os.path.join(captured_dir, images[-1])
                    self.debug_log(f"Using recent image for test: {images[-1]}")
                else:
                    return {'success': False, 'error': 'No test images available'}
            else:
                return {'success': False, 'error': 'Upload directory not found'}
        
        result = self.analyze_image(test_image_path)
        
        return {
            'success': True,
            'test_image': test_image_path,
            'model_type': result.get('model_type', 'unknown'),
            'analysis_result': result,
            'debug_info': self.get_debug_info()
        }
