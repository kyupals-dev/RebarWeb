#!/usr/bin/env python3
"""
AI Model Debug Script for Rebar Detection
Run this to diagnose why the model isn't detecting rebar structures
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Add project root to path
project_root = "/home/team10/RebarWeb"
sys.path.insert(0, project_root)

def debug_ai_model():
    """Debug the AI model detection issues"""
    
    print("🔍 REBAR DETECTION DEBUG SCRIPT")
    print("=" * 50)
    
    try:
        from app.services.ai_service import AIService
        from app.utils.config import config
        
        # Initialize AI service
        print("🤖 Initializing AI service...")
        ai_service = AIService()
        
        # Check model status
        print("\n📊 AI Model Status:")
        status = ai_service.get_model_status()
        for key, value in status.items():
            icon = "✅" if value in [True, "real_trained_model"] else "❌" if value is False else "📝"
            print(f"   {icon} {key}: {value}")
        
        # Check if model file exists and get info
        model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        print(f"\n📁 Model File Analysis:")
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"   ✅ Model exists: {model_path}")
            print(f"   📦 File size: {file_size / (1024*1024):.1f} MB")
            
            # Check if file is reasonable size for a Detectron2 model
            if file_size < 10 * 1024 * 1024:  # Less than 10MB
                print(f"   ⚠️  WARNING: Model file seems small for Detectron2 ({file_size / (1024*1024):.1f} MB)")
                print("   💡 Expected size: 100-300 MB for Mask R-CNN")
            else:
                print(f"   ✅ Model file size looks reasonable")
        else:
            print(f"   ❌ Model file not found: {model_path}")
            return
        
        # Check captured images directory
        print(f"\n📂 Captured Images Analysis:")
        upload_dir = config.UPLOAD_FOLDER
        if os.path.exists(upload_dir):
            images = [f for f in os.listdir(upload_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   📊 Total images: {len(images)}")
            
            if images:
                # Get the most recent frame capture
                recent_frames = [f for f in images if f.startswith('frame_capture_')]
                if recent_frames:
                    recent_frames.sort(reverse=True)
                    latest_frame = recent_frames[0]
                    latest_path = os.path.join(upload_dir, latest_frame)
                    
                    print(f"   🆕 Latest frame: {latest_frame}")
                    
                    # Load and analyze the image
                    image = cv2.imread(latest_path)
                    if image is not None:
                        print(f"   📐 Image shape: {image.shape}")
                        print(f"   🎨 Image dtype: {image.dtype}")
                        print(f"   📊 Image range: {image.min()}-{image.max()}")
                        
                        # Check if image is too dark/bright
                        mean_brightness = np.mean(image)
                        print(f"   💡 Mean brightness: {mean_brightness:.1f} (good range: 50-200)")
                        
                        if mean_brightness < 30:
                            print("   ⚠️  Image seems too dark for detection")
                        elif mean_brightness > 220:
                            print("   ⚠️  Image seems too bright/overexposed")
                        else:
                            print("   ✅ Image brightness looks good")
                        
                        # Test with different detection thresholds
                        print(f"\n🧪 Testing Detection with Different Thresholds:")
                        
                        original_threshold = ai_service.detection_threshold
                        test_thresholds = [0.1, 0.2, 0.3, 0.5, 0.7]
                        
                        for threshold in test_thresholds:
                            print(f"   🎯 Testing threshold: {threshold}")
                            ai_service.detection_threshold = threshold
                            
                            # Update model threshold if loaded
                            if ai_service.model_loaded and ai_service.cfg:
                                ai_service.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
                                # Recreate predictor with new threshold
                                try:
                                    from detectron2.engine import DefaultPredictor
                                    ai_service.predictor = DefaultPredictor(ai_service.cfg)
                                except:
                                    pass
                            
                            # Test analysis
                            result = ai_service.analyze_image(image_path=latest_path)
                            
                            if result['success']:
                                detections = result.get('num_detections', 0)
                                print(f"      ✅ Threshold {threshold}: Found {detections} detections")
                                
                                if detections > 0:
                                    # Show detection details
                                    for i, det in enumerate(result.get('detections', [])):
                                        conf = det.get('confidence', 0)
                                        class_name = det.get('class_name', 'unknown')
                                        print(f"         Detection {i+1}: {class_name} ({conf:.3f})")
                            else:
                                error = result.get('error', 'Unknown error')
                                if 'no_detection' in error.lower() or 'no rebar' in error.lower():
                                    print(f"      ❌ Threshold {threshold}: No detections")
                                else:
                                    print(f"      ❌ Threshold {threshold}: Error - {error}")
                        
                        # Restore original threshold
                        ai_service.detection_threshold = original_threshold
                        if ai_service.model_loaded and ai_service.cfg:
                            ai_service.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = original_threshold
                    
                    else:
                        print(f"   ❌ Could not load image: {latest_path}")
                else:
                    print("   📝 No frame captures found")
            else:
                print("   📝 No images found in upload directory")
        else:
            print(f"   ❌ Upload directory not found: {upload_dir}")
        
        # Check Detectron2 configuration
        print(f"\n⚙️  Detectron2 Configuration Check:")
        if ai_service.model_loaded:
            cfg = ai_service.cfg
            print(f"   📏 Input size - Min: {cfg.INPUT.MIN_SIZE_TEST}, Max: {cfg.INPUT.MAX_SIZE_TEST}")
            print(f"   🎯 Score threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
            print(f"   🏷️  Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
            print(f"   💻 Device: {cfg.MODEL.DEVICE}")
            print(f"   🏗️  Model backbone: {cfg.MODEL.BACKBONE.NAME}")
        
        # Training data mismatch check
        print(f"\n🎓 Training Data Analysis:")
        expected_classes = ai_service.class_names
        print(f"   📚 Expected classes: {expected_classes}")
        print(f"   🔢 Number of classes: {len(expected_classes)}")
        
        print(f"\n💡 TROUBLESHOOTING SUGGESTIONS:")
        print("=" * 50)
        
        if not ai_service.model_loaded:
            print("1. ❌ MODEL NOT LOADED:")
            print("   - Check if model_final.pth exists and is the correct trained model")
            print("   - Verify Detectron2 is properly installed")
            print("   - Check for any import errors")
        
        print("2. 🎯 DETECTION THRESHOLD TOO HIGH:")
        print("   - Current threshold:", ai_service.detection_threshold)
        print("   - Try lowering to 0.1 or 0.2 for more sensitive detection")
        
        print("3. 📐 IMAGE SIZE/FORMAT ISSUES:")
        print("   - Model expects specific input size")
        print("   - Check if preprocessing is correct")
        
        print("4. 🎓 TRAINING DATA MISMATCH:")
        print("   - Verify model was trained on similar rebar images")
        print("   - Check if class names match training data")
        
        print("5. 💡 LIGHTING/IMAGE QUALITY:")
        print("   - Ensure good lighting and contrast")
        print("   - Avoid shadows or overexposure")
        print("   - Try different camera angles")
        
        print("6. 🔧 MODEL CONFIGURATION:")
        print("   - Verify model config matches training setup")
        print("   - Check if using correct model architecture")
        
        # Quick fix suggestions
        print(f"\n🚀 QUICK FIXES TO TRY:")
        print("=" * 30)
        print("1. Lower detection threshold:")
        print("   ai_service.detection_threshold = 0.1")
        
        print("2. Check image manually:")
        print(f"   Open: {latest_path if 'latest_path' in locals() else 'latest captured image'}")
        print("   Verify rebar is clearly visible")
        
        print("3. Test with placeholder mode:")
        print("   Temporarily rename model file to test pipeline")
        
        print("4. Check model file integrity:")
        print("   Try re-downloading/copying the trained model")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the RebarWeb directory and dependencies are installed")
    except Exception as e:
        print(f"❌ Debug error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ai_model()
