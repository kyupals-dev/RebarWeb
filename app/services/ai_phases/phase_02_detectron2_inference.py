"""
Phase 2: Detectron2 Inference with Better Error Handling
Runs AI model inference on prepared images
"""

import numpy as np
import os
import traceback

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

class Phase02Detectron2Inference:
    """Phase 2: Run Detectron2 inference with your trained model"""
    
    def __init__(self):
        self.predictor = None
        self.cfg = None
        self.metadata = None
        self.model_loaded = False
        
        # Match your training configuration exactly
        self.model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]  # Exact match
        self.num_classes = 3
        
        print("🤖 Phase 2: Detectron2 Inference initialized")
        self.load_model()
    
    def load_model(self):
        """Load your trained model with detailed error handling"""
        if not DETECTRON2_AVAILABLE:
            print("   ❌ Detectron2 not available")
            return False
        
        if not os.path.exists(self.model_path):
            print(f"   ❌ Model not found: {self.model_path}")
            return False
        
        try:
            print(f"   🔄 Loading model from: {self.model_path}")
            
            # Configure exactly like your training script
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Model settings matching your training
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.DEVICE = "cpu"  # CPU for Raspberry Pi
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower threshold for more detections
            
            print("   🔄 Creating predictor...")
            # Create predictor
            self.predictor = DefaultPredictor(self.cfg)
            
            # Set up metadata exactly like training
            self.metadata = MetadataCatalog.get("rebar_dataset")
            self.metadata.thing_classes = self.class_names
            
            # Test with a dummy image
            print("   🧪 Testing model with dummy image...")
            test_image = np.zeros((640, 480, 3), dtype=np.uint8)
            test_outputs = self.predictor(test_image)
            print(f"   ✅ Model test successful - got {len(test_outputs['instances'])} detections on blank image")
            
            self.model_loaded = True
            print("   ✅ Your trained model loaded successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            print("   Full traceback:")
            traceback.print_exc()
            return False
    
    def run_inference(self, prepared_image):
        """Run inference with your trained model"""
        try:
            print("🤖 Phase 2: Running Detectron2 inference...")
            
            if not self.model_loaded or not self.predictor:
                return {
                    'success': False,
                    'error': 'Model not loaded properly'
                }
            
            if prepared_image is None:
                return {
                    'success': False,
                    'error': 'No prepared image provided'
                }
            
            # Validate image
            if len(prepared_image.shape) != 3 or prepared_image.shape[2] != 3:
                return {
                    'success': False,
                    'error': f'Invalid image shape: {prepared_image.shape}'
                }
            
            print(f"   📐 Image shape: {prepared_image.shape}")
            print(f"   📊 Image dtype: {prepared_image.dtype}")
            print(f"   📈 Image range: {prepared_image.min()}-{prepared_image.max()}")
            
            # Run inference
            print("   🔄 Running model inference...")
            outputs = self.predictor(prepared_image)
            instances = outputs["instances"].to("cpu")
            
            num_detections = len(instances)
            print(f"   🎯 Found {num_detections} raw detections")
            
            if num_detections == 0:
                print("   ⚠️  No detections found")
                return {
                    'success': False,
                    'error': 'No rebar structures detected',
                    'no_detection': True
                }
            
            # Extract detection data
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            masks = instances.pred_masks.numpy()
            
            print("   📋 Processing detections...")
            detections = []
            for i in range(num_detections):
                class_id = int(classes[i])
                confidence = float(scores[i])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': boxes[i].tolist(),
                    'mask_area': float(np.sum(masks[i])),
                    'mask': masks[i]  # Keep mask for later phases
                }
                detections.append(detection)
                
                print(f"     Detection {i+1}: {class_name} ({confidence:.3f}) - Area: {detection['mask_area']:.0f}px")
            
            print(f"   ✅ Phase 2: Successfully processed {num_detections} detections")
            
            return {
                'success': True,
                'detections': detections,
                'num_detections': num_detections,
                'outputs': outputs  # Pass raw outputs for visualization
            }
            
        except Exception as e:
            print(f"   ❌ Phase 2 inference error: {e}")
            print("   Full traceback:")
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Inference failed: {str(e)}'
            }
