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
            
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
            self.cfg.MODEL.WEIGHTS = self.model_path
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_threshold
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.INPUT.MIN_SIZE_TRAIN = (640,)
            self.cfg.INPUT.MAX_SIZE_TRAIN = 640
            self.cfg.INPUT.MIN_SIZE_TEST = 640
            self.cfg.INPUT.MAX_SIZE_TEST = 640
            
            print("🔄 Creating predictor with REAL MODEL...")
            self.predictor = DefaultPredictor(self.cfg)
            
            self.metadata = MetadataCatalog.get("rebar_dataset_real")
            self.metadata.thing_classes = self.class_names
            self.metadata.thing_colors = [
                (0, 255, 0),
                (255, 0, 0),
                (0, 0, 255),
            ]
            
            self.model_loaded = True
            print("✅ REAL AI Model loaded successfully!")
            print(f"   Model path: {self.model_path}")
            print(f"   Classes: {self.class_names}")
            print(f"   Detection threshold: {self.detection_threshold}")
            print(f"   Input size: {self.training_input_size[0]}x{self.training_input_size[1]}")

            test_image = np.zeros((640, 480, 3), dtype=np.uint8)
            try:
                self.predictor(test_image)
                print("✅ Model inference test successful!")
            except Exception as e:
                print(f"⚠️  Model inference test failed: {e}")
            return True

        except Exception as e:
            print(f"❌ Error loading REAL AI model: {str(e)}")
            traceback.print_exc()
            self.model_loaded = False
            return False

    # --- keep all other methods exactly as before ---
    # (analyze_image, _analyze_with_real_model, _find_intersection_points, _order_grid_points,
    #  _create_quadrilaterals_and_dimensions, _create_final_visualization, _create_basic_visualization,
    #  _calculate_pixel_to_cm_ratio, _calculate_basic_dimensions_from_detections,
    #  _get_fallback_dimensions, _analyze_placeholder)

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
            'model_type': 'real_trained_model' if self.model_loaded else 'placeholder',
            'analysis_method': '4_step_intersection_analysis'
        }
