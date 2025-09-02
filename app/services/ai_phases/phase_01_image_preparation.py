"""
Phase 1: Image Preparation for AI Analysis
Handles image loading, validation, and preprocessing
"""

import cv2
import numpy as np
import os

class Phase01ImagePreparation:
    """Phase 1: Prepare images for AI analysis"""
    
    def __init__(self):
        self.target_size = (480, 640)  # width x height
        print("📝 Phase 1: Image Preparation initialized")
    
    def prepare_image(self, image_path):
        """Prepare image for AI analysis"""
        try:
            print(f"📸 Phase 1: Preparing image {os.path.basename(image_path)}")
            
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
            
            original_shape = image.shape
            
            # Resize if needed
            height, width = image.shape[:2]
            if width != self.target_size[0] or height != self.target_size[1]:
                image = cv2.resize(image, self.target_size)
            
            print("   ✅ Phase 1: Image preparation complete")
            
            return {
                'success': True,
                'prepared_image': image,
                'original_shape': original_shape,
                'target_shape': image.shape
            }
            
        except Exception as e:
            print(f"   ❌ Phase 1 error: {str(e)}")
            return {
                'success': False,
                'error': f'Image preparation failed: {str(e)}'
            }
