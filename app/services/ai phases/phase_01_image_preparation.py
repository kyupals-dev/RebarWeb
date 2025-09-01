"""
Phase 1: Image Preparation
- Camera captures 640x480 image (landscape)
- Rotate 90° clockwise → 480x640 (portrait) 
- Resize to training format → 480x640 exactly
- Convert BGR to RGB for model input
"""

import cv2
import numpy as np
import os
from .base_phase import BasePhase

class Phase01ImagePreparation(BasePhase):
    """Phase 1: Prepare image for AI model processing"""
    
    def __init__(self):
        super().__init__()
        self.phase_name = "Image Preparation"
        self.target_size = (480, 640)  # width x height for portrait
        self.expected_input_size = (640, 480)  # width x height for landscape
    
    def validate_input(self, data):
        """Validate input data for Phase 1"""
        required_keys = ['image_path']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        image_path = data['image_path']
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        return True
    
    def execute(self, data):
        """Execute Phase 1: Image Preparation"""
        self.log(f"Starting {self.phase_name}...")
        
        # Validate input
        self.validate_input(data)
        image_path = data['image_path']
        
        # Step 1: Load image
        self.log(f"Loading image: {os.path.basename(image_path)}")
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original_height, original_width = image.shape[:2]
        self.log(f"Original image size: {original_width}x{original_height} (WxH)")
        
        # Step 2: Ensure image is the right orientation and size
        # If image is already 480x640, just validate it
        if original_width == 480 and original_height == 640:
            self.log("Image already in target format 480x640")
            processed_image = image.copy()
        
        # If image is 640x480 (landscape), rotate 90° clockwise
        elif original_width == 640 and original_height == 480:
            self.log("Rotating 640x480 landscape → 480x640 portrait")
            processed_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
        # For any other size, resize to 480x640
        else:
            self.log(f"Resizing {original_width}x{original_height} → 480x640")
            processed_image = cv2.resize(image, self.target_size)
        
        # Step 3: Validate final size
        final_height, final_width = processed_image.shape[:2]
        if final_width != 480 or final_height != 640:
            raise ValueError(f"Failed to achieve target size. Got {final_width}x{final_height}, expected 480x640")
        
        # Step 4: Convert BGR to RGB for model compatibility
        self.log("Converting BGR → RGB for model input")
        rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Validate the conversion
        if rgb_image.shape != (640, 480, 3):
            raise ValueError(f"RGB conversion failed. Shape: {rgb_image.shape}, expected (640, 480, 3)")
        
        # Prepare output data
        output_data = data.copy()
        output_data.update({
            'original_image': image,  # Keep original for reference
            'processed_image': processed_image,  # BGR format for saving
            'model_input_image': rgb_image,  # RGB format for model
            'original_size': (original_width, original_height),
            'final_size': (final_width, final_height),
            'rotation_applied': original_width == 640 and original_height == 480,
            'resize_applied': not (final_width == original_width and final_height == original_height)
        })
        
        self.log(f"✅ {self.phase_name} complete:")
        self.log(f"   Original: {original_width}x{original_height}")
        self.log(f"   Final: {final_width}x{final_height}")
        self.log(f"   Rotation: {'Yes' if output_data['rotation_applied'] else 'No'}")
        self.log(f"   Resize: {'Yes' if output_data['resize_applied'] else 'No'}")
        
        return output_data
    
    def validate_output(self, data):
        """Validate output data from Phase 1"""
        required_keys = ['original_image', 'processed_image', 'model_input_image', 'final_size']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing output key: {key}")
        
        # Validate image shapes
        processed_image = data['processed_image']
        model_input_image = data['model_input_image']
        
        if processed_image.shape[:2] != (640, 480):
            raise ValueError(f"Processed image wrong size: {processed_image.shape[:2]}, expected (640, 480)")
        
        if model_input_image.shape != (640, 480, 3):
            raise ValueError(f"Model input image wrong shape: {model_input_image.shape}, expected (640, 480, 3)")
        
        # Check that final size matches target
        final_width, final_height = data['final_size']
        if final_width != 480 or final_height != 640:
            raise ValueError(f"Final size incorrect: {final_width}x{final_height}, expected 480x640")
        
        return True
