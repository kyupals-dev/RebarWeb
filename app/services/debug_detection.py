#!/usr/bin/env python3
"""
REBAR DETECTION DEBUG SCRIPT
Run this script to diagnose and fix your "No rebar structures detected" issue

Usage: 
cd /home/team10/RebarWeb
python3 debug_detection.py
"""

import os
import sys
import cv2
import numpy as np

# Add project to path
project_root = "/home/team10/RebarWeb"
sys.path.insert(0, project_root)

def check_environment():
    """Check if we're in the right environment"""
    print("🔍 ENVIRONMENT CHECK")
    print("=" * 40)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in project root
    if not os.path.exists("app/services/ai_service.py"):
        print("❌ Not in RebarWeb directory!")
        print("Please run: cd /home/team10/RebarWeb")
        return False
    
    print("✅ In correct RebarWeb directory")
    return True

def check_model_file():
    """Check model file"""
    print("\n📁 MODEL FILE CHECK")
    print("=" * 40)
    
    model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        size_mb = file_size / (1024 * 1024)
        print(f"✅ Model file found: {model_path}")
        print(f"📊 File size: {size_mb:.1f} MB")
        
        if size_mb < 1:
            print("⚠️  WARNING: Model file very small (expected ~170-250 MB)")
            print("   Model may be corrupted or incomplete")
            return False
        elif size_mb > 500:
            print("⚠️  WARNING: Model file very large (expected ~170-250 MB)")
        else:
            print("✅ Model file size looks reasonable")
        
        return True
    else:
        print(f"❌ Model file NOT found: {model_path}")
        print("💡 Please copy your model_final.pth to:")
        print(f"   {model_path}")
        return False

def check_detectron2():
    """Check Detectron2 installation"""
    print("\n🤖 DETECTRON2 CHECK")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        from detectron2 import __version__ as d2_version
        print(f"✅ Detectron2 version: {d2_version}")
        
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        print("✅ Detectron2 imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Detectron2 import failed: {e}")
        print("💡 Install Detectron2:")
        print("   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html")
        return False

def test_model_loading():
    """Test model loading with your configuration"""
    print("\n⚙️  MODEL LOADING TEST")
    print("=" * 40)
    
    try:
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        
        model_path = "/home/team10/RebarWeb/app/model/model_final.pth"
        
        print("🔄 Setting up configuration...")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # Low threshold
        cfg.MODEL.DEVICE = "cpu"
        
        print("🔄 Creating predictor...")
        predictor = DefaultPredictor(cfg)
        print("✅ Model loaded successfully!")
        
        # Test inference
        print("🧪 Testing inference...")
        test_image = np.zeros((640, 480, 3), dtype=np.uint8)
        outputs = predictor(test_image)
        print(f"✅ Inference test passed ({len(outputs['instances'])} detections on blank)")
        
        return True, predictor, cfg
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def find_test_image():
    """Find a test image to use"""
    print("\n📸 FINDING TEST IMAGE")
    print("=" * 40)
    
    # Check upload folder
    upload_folder = "/home/team10/RebarWeb/static/captured_images"
    
    if os.path.exists(upload_folder):
        images = [f for f in os.listdir(upload_folder) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            # Get most recent image
            image_paths = [os.path.join(upload_folder, img) for img in images]
            most_recent = max(image_paths, key=os.path.getmtime)
            
            print(f"✅ Found {len(images)} images")
            print(f"📸 Using most recent: {os.path.basename(most_recent)}")
            
            # Check image
            img = cv2.imread(most_recent)
            if img is not None:
                print(f"📐 Image shape: {img.shape}")
                return most_recent
            else:
                print("❌ Could not load most recent image")
        else:
            print("❌ No images found in upload folder")
    else:
        print(f"❌ Upload folder not found: {upload_folder}")
    
    print("💡 Please capture an image using the web interface first")
    return None

def test_detection_thresholds(predictor, cfg, image_path):
    """Test detection with multiple thresholds"""
    print(f"\n🎯 TESTING DETECTION THRESHOLDS")
    print("=" * 40)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load test image")
        return False
    
    print(f"📐 Image shape: {image.shape}")
    
    # Resize to model input size
    if image.shape[1] != 480 or image.shape[0] != 640:
        image = cv2.resize(image, (480, 640))
        print(f"📐 Resized to: {image.shape}")
    
    class_names = ["back_horizontal", "front_horizontal", "front_vertical"]
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    best_threshold = None
    best_count = 0
    
    for threshold in thresholds:
        print(f"\n🔍 Testing threshold: {threshold}")
        
        # Update config
        test_cfg = cfg.clone()
        test_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        test_predictor = DefaultPredictor(test_cfg)
        
        # Run detection
        outputs = test_predictor(image)
        instances = outputs["instances"].to("cpu")
        
        num_detections = len(instances)
        print(f"   Total detections: {num_detections}")
        
        if num_detections > 0:
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            # Count by class
            front_vertical = sum(1 for c in classes if c == 2)
            front_horizontal = sum(1 for c in classes if c == 1)
            back_horizontal = sum(1 for c in classes if c == 0)
            
            print(f"   front_vertical: {front_vertical}")
            print(f"   front_horizontal: {front_horizontal}")
            print(f"   back_horizontal: {back_horizontal}")
            
            # Track best result
            relevant_count = front_vertical + front_horizontal
            if relevant_count > best_count:
                best_count = relevant_count
                best_threshold = threshold
            
            # Show top detections
            if num_detections <= 10:
                print("   Top detections:")
                for i in range(num_detections):
                    class_id = int(classes[i])
                    confidence = float(scores[i])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
                    print(f"     {class_name}: {confidence:.4f}")
        else:
            print("   No detections found")
    
    print(f"\n📊 SUMMARY:")
    if best_threshold is not None:
        print(f"✅ Best threshold: {best_threshold} (found {best_count} relevant detections)")
        print(f"💡 RECOMMENDATION: Update your AI service threshold to {best_threshold}")
        return True, best_threshold
    else:
        print("❌ No detections found at any threshold")
        print("💡 POSSIBLE ISSUES:")
        print("   1. Model not trained on similar images")
        print("   2. Image quality/lighting issues")
        print("   3. Model file corrupted")
        print("   4. Wrong image preprocessing")
        return False, None

def create_quick_fix_code(best_threshold):
    """Generate code to fix the threshold"""
    print(f"\n🔧 QUICK FIX CODE")
    print("=" * 40)
    
    fix_code = f"""
# QUICK FIX: Update your ai_service.py

# 1. In __init__ method, change:
self.detection_threshold = {best_threshold}  # Changed from 0.3

# 2. In load_model method, change:
self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = {best_threshold}  # Changed from 0.3

# 3. Save the file and restart your application
"""
    
    print(fix_code)
    
    # Write to file
    fix_file = "/home/team10/RebarWeb/QUICK_FIX.txt"
    try:
        with open(fix_file, 'w') as f:
            f.write(fix_code)
        print(f"💾 Quick fix saved to: {fix_file}")
    except Exception as e:
        print(f"⚠️  Could not save fix file: {e}")

def main():
    """Run all diagnostic tests"""
    print("🔧 REBAR DETECTION DIAGNOSTIC TOOL")
    print("=" * 50)
    print("This tool will help fix your 'No rebar structures detected' issue")
    print("")
    
    # Check 1: Environment
    if not check_environment():
        return
    
    # Check 2: Model file
    if not check_model_file():
        print("\n❌ CRITICAL: Fix model file issue first!")
        return
    
    # Check 3: Detectron2
    if not check_detectron2():
        print("\n❌ CRITICAL: Install Detectron2 first!")
        return
    
    # Check 4: Model loading
    model_loaded, predictor, cfg = test_model_loading()
    if not model_loaded:
        print("\n❌ CRITICAL: Model loading failed!")
        return
    
    # Check 5: Test image
    test_image = find_test_image()
    if not test_image:
        print("\n⚠️  No test image - capture one and run again")
        return
    
    # Check 6: Detection testing
    detection_success, best_threshold = test_detection_thresholds(predictor, cfg, test_image)
    
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC RESULTS")
    print("=" * 50)
    
    if detection_success and best_threshold:
        print("🎉 SUCCESS! Found optimal detection threshold")
        print(f"✅ Recommended threshold: {best_threshold}")
        create_quick_fix_code(best_threshold)
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Update your ai_service.py with threshold {best_threshold}")
        print(f"2. Restart your application: python3 main.py")
        print(f"3. Test with better lighting on real rebar structure")
        
    else:
        print("❌ NO DETECTIONS FOUND")
        print("💡 TROUBLESHOOTING:")
        print("1. Check image lighting (natural daylight is best)")
        print("2. Ensure rebar structure is clearly visible")
        print("3. Try optimal distance (160-200cm)")
        print("4. Verify model was trained correctly")
        print("5. Test with real rebar structure (not tablet image)")

if __name__ == "__main__":
    main()
