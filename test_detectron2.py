# test_detectron2.py
import os
import sys
sys.path.append('/home/team10/RebarWeb')

from app.services.ai_service import AIService
import cv2
import numpy as np

def test_detectron2():
    print("🧪 Testing Detectron2 installation...")
    
    # Initialize AI service
    ai_service = AIService()
    
    # Check model status
    status = ai_service.get_model_status()
    print("Model Status:", status)
    
    if ai_service.model_loaded:
        print("✅ Detectron2 loaded successfully!")
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (300, 400), (255, 255, 255), -1)
        
        # Save test image
        test_path = "/tmp/test_image.jpg"
        cv2.imwrite(test_path, test_image)
        
        # Test analysis
        result = ai_service.analyze_image(test_path)
        print("Analysis result:", result['success'])
        
        # Cleanup
        os.remove(test_path)
        
    else:
        print("❌ Detectron2 not loaded properly")

if __name__ == "__main__":
    test_detectron2()
