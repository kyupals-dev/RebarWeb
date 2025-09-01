"""
Phase 2: Detectron2 Model Inference
- Load trained model_final.pth (Mask R-CNN R_50_FPN_3x)
- Run inference on 480x640 image
- Model detects 3 classes: back_horizontal, front_horizontal, front_vertical
- Apply detection threshold (0.2) to filter low-confidence detections
- Output: bounding boxes + confidence scores + segmentation masks
"""

import numpy as np
import os
from .base_phase import BasePhase

# Detectron2 imports
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

class Phase02Detectron2Inference(BasePhase):
    """Phase 2: Run Detectron2 model inference"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Detectron2 Inference"
        self.predictor = None
        self.cfg = None
        self.model_loaded = False
        
        # Model configuration
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
        self.num_classes = 3
        self.detection_threshold = 0.2
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model"""
        try:
            if not DETECTRON2_AVAILABLE:
                self.log("❌ Detectron2 not available, will use placeholder mode")
                return False
            
            if not os.path.exists(self.model_path):
                self.log(f"❌ Model file not found: {self.model_path}")
                return False
            
            self.log("🔄 Loading Detectron2 configuration...")
            
            # Set up configuration
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold
            self.cfg.MODEL.DEVICE = "cpu"  # Use CPU on Raspberry Pi
            
            # Input format (480x640)
            self.cfg.INPUT.MIN_SIZE_TRAIN = (640,)
            self.cfg.INPUT.MAX_SIZE_TRAIN = 640
            self.cfg.INPUT.MIN_SIZE_TEST = 640
            self.cfg.INPUT.MAX_SIZE_TEST = 640
            
            self.log("🔄 Creating predictor...")
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
            self.log("✅ Detectron2 model loaded successfully!")
            self.log(f"   Classes: {self.class_names}")
            self.log(f"   Threshold: {self.detection_threshold}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error loading Detectron2 model: {str(e)}")
            self.model_loaded = False
            return False
    
    def validate_input(self, data):
        """Validate input data for Phase 2"""
        required_keys = ['model_input_image']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate image format
        image = data['model_input_image']
        if image.shape != (640, 480, 3):
            raise ValueError(f"Model input image wrong shape: {image.shape}, expected (640, 480, 3)")
        
        return True
    
    def execute(self, data):
        """Execute Phase 2: Detectron2 Inference"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        image = data['model_input_image']  # RGB format
        
        # Check if model is available
        if not self.model_loaded or not DETECTRON2_AVAILABLE:
            self.log("⚠️ Using placeholder inference (model not available)")
            return self._placeholder_inference(data, image)
        
        try:
            self.log("🤖 Running Detectron2 inference...")
            
            # Run inference
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
            
            num_detections = len(instances)
            self.log(f"🎯 Found {num_detections} raw detections")
            
            if num_detections == 0:
                self.log("❌ No detections found")
                return self._create_output_data(data, outputs, instances, num_detections)
            
            # Extract detection data
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            # Log detection details
            for i in range(num_detections):
                class_name = self.class_names[classes[i]]
                confidence = scores[i]
                mask_area = float(np.sum(masks[i]))
                self.log(f"   Detection {i+1}: {class_name} ({confidence:.3f}) - Area: {mask_area:.0f}px")
            
            # Create output data
            output_data = self._create_output_data(data, outputs, instances, num_detections)
            
            self.log(f"✅ {self.phase_name} complete: {num_detections} detections")
            return output_data
            
        except Exception as e:
            self.log(f"❌ Detectron2 inference error: {str(e)}")
            # Fallback to placeholder
            return self._placeholder_inference(data, image)
    
    def _create_output_data(self, data, outputs, instances, num_detections):
        """Create standardized output data structure"""
        output_data = data.copy()
        
        # Add detectron2 outputs
        output_data.update({
            'detectron2_outputs': outputs,
            'detectron2_instances': instances,
            'num_detections': num_detections,
            'model_loaded': self.model_loaded,
            'detection_threshold': self.detection_threshold,
            'class_names': self.class_names,
            'metadata': self.metadata if hasattr(self, 'metadata') else None
        })
        
        # Extract structured detection data if we have detections
        if num_detections > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            detections = []
            for i in range(num_detections):
                detection = {
                    'class_id': int(classes[i]),
                    'class_name': self.class_names[classes[i]],
                    'confidence': float(scores[i]),
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'mask': masks[i],
                    'mask_area': float(np.sum(masks[i]))
                }
                detections.append(detection)
            
            output_data['structured_detections'] = detections
        else:
            output_data['structured_detections'] = []
        
        return output_data
    
    def _placeholder_inference(self, data, image):
        """Generate placeholder inference results"""
        self.log("📝 Generating placeholder inference results...")
        
        # Create fake detection data
        height, width = image.shape[:2]
        
        # Simulate 2 detections: 1 front_horizontal, 1 front_vertical
        fake_detections = [
            {
                'class_id': 1,  # front_horizontal
                'class_name': 'front_horizontal',
                'confidence': 0.85,
                'bbox': [100.0, 500.0, 300.0, 540.0],  # bottom horizontal bar
                'mask': np.zeros((height, width), dtype=bool),
                'mask_area': 8000.0
            },
            {
                'class_id': 2,  # front_vertical
                'class_name': 'front_vertical',
                'confidence': 0.78,
                'bbox': [150.0, 100.0, 190.0, 550.0],  # vertical bar
                'mask': np.zeros((height, width), dtype=bool),
                'mask_area': 18000.0
            }
        ]
        
        # Create fake masks
        for detection in fake_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            detection['mask'][y1:y2, x1:x2] = True
        
        output_data = data.copy()
        output_data.update({
            'detectron2_outputs': None,  # Placeholder mode
            'detectron2_instances': None,
            'num_detections': len(fake_detections),
            'model_loaded': False,
            'detection_threshold': self.detection_threshold,
            'class_names': self.class_names,
            'metadata': None,
            'structured_detections': fake_detections,
            'placeholder_mode': True
        })
        
        self.log(f"✅ Placeholder inference complete: {len(fake_detections)} fake detections")
        return output_data
    
    def validate_output(self, data):
        """Validate output data from Phase 2"""
        required_keys = ['num_detections', 'structured_detections', 'class_names']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # Validate detection structure
        detections = data['structured_detections']
        if not isinstance(detections, list):
            raise ValueError("structured_detections must be a list")
        
        # If we have detections, validate their structure
        for i, detection in enumerate(detections):
            required_detection_keys = ['class_id', 'class_name', 'confidence', 'bbox', 'mask']
            for key in required_detection_keys:
                if key not in detection:
                    raise ValueError(f"Missing detection key '{key}' in detection {i}")
        
        return True
