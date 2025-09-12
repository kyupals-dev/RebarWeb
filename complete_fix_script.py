#!/usr/bin/env python3
"""
Complete Fix Script for Rebar Detection Issues
Applies all necessary fixes:
1. Updates AI service to use 2 classes
2. Fixes AI routes to remove mode parameter
3. Lowers detection threshold
"""

import os
import shutil
from datetime import datetime

def apply_complete_fix():
    """Apply all fixes for the rebar detection issues"""
    
    print("🚀 COMPLETE REBAR DETECTION FIX")
    print("=" * 40)
    print("Applying all fixes:")
    print("  🔧 Update AI service to 2 classes")
    print("  🔧 Fix AI routes (remove mode parameter)")
    print("  🔧 Lower detection threshold")
    print("  🔧 Update all class references")
    
    project_root = "/home/team10/RebarWeb"
    
    # Files to update
    ai_service_path = os.path.join(project_root, "app/services/ai_service.py")
    ai_routes_path = os.path.join(project_root, "app/routes/ai_routes.py")
    
    # Check files exist
    if not os.path.exists(ai_service_path):
        print(f"❌ AI service file not found: {ai_service_path}")
        return False
    
    if not os.path.exists(ai_routes_path):
        print(f"❌ AI routes file not found: {ai_routes_path}")
        return False
    
    # Create backups
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ai_service_backup = f"{ai_service_path}.backup_{timestamp}"
    ai_routes_backup = f"{ai_routes_path}.backup_{timestamp}"
    
    try:
        shutil.copy2(ai_service_path, ai_service_backup)
        shutil.copy2(ai_routes_path, ai_routes_backup)
        print(f"💾 Backups created:")
        print(f"   {ai_service_backup}")
        print(f"   {ai_routes_backup}")
    except Exception as e:
        print(f"⚠️  Could not create backups: {e}")
    
    success = True
    
    # Fix 1: Update AI Service
    print("\n🔧 Fixing AI Service...")
    if fix_ai_service(ai_service_path):
        print("   ✅ AI Service updated")
    else:
        print("   ❌ AI Service fix failed")
        success = False
    
    # Fix 2: Update AI Routes
    print("\n🔧 Fixing AI Routes...")
    if fix_ai_routes(ai_routes_path):
        print("   ✅ AI Routes updated")
    else:
        print("   ❌ AI Routes fix failed")
        success = False
    
    if success:
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("=" * 40)
        print("✅ AI service updated to 2 classes")
        print("✅ Detection threshold lowered to 0.2")
        print("✅ AI routes fixed (mode parameter removed)")
        print("✅ All class references updated")
        print("")
        print("🚀 Next steps:")
        print("1. Restart application: python3 main.py")
        print("2. Try capturing rebar image")
        print("3. Should now detect with 2-class model!")
        
    else:
        print("\n❌ SOME FIXES FAILED")
        print("Restoring from backups...")
        
        try:
            if os.path.exists(ai_service_backup):
                shutil.copy2(ai_service_backup, ai_service_path)
            if os.path.exists(ai_routes_backup):
                shutil.copy2(ai_routes_backup, ai_routes_path)
            print("🔄 Files restored from backup")
        except Exception as e:
            print(f"❌ Could not restore backups: {e}")
    
    return success

def fix_ai_service(file_path):
    """Fix the AI service file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply all AI service fixes
        fixes = [
            # Fix class names
            ('self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]',
             'self.class_names = ["front_horizontal", "front_vertical"]'),
            
            ('self.class_names = ["front_vertical", "front_horizontal", "back_horizontal"]',
             'self.class_names = ["front_horizontal", "front_vertical"]'),
            
            # Fix number of classes
            ('self.num_classes = 3', 'self.num_classes = 2'),
            
            # Fix detection threshold
            ('self.detection_threshold = 0.3', 'self.detection_threshold = 0.2'),
            
            # Fix colors (remove back_horizontal color)
            ('''self.metadata.thing_colors = [
                (0, 255, 0),      # front_vertical - Green
                (255, 0, 0),      # front_horizontal - Red  
                (0, 0, 255),      # back_horizontal - Blue
            ]''',
             '''self.metadata.thing_colors = [
                (255, 0, 0),      # front_horizontal - Red  
                (0, 255, 0),      # front_vertical - Green
            ]'''),
            
            # Remove back_horizontal references
            ("back_horizontal = [d for d in detections if d['class_name'] == 'back_horizontal']",
             "# back_horizontal removed - using 2 classes only"),
            
            # Update print statements
            ('print(f"   Found: {len(front_vertical)} front_vertical, {len(front_horizontal)} front_horizontal, {len(back_horizontal)} back_horizontal")',
             'print(f"   Found: {len(front_horizontal)} front_horizontal, {len(front_vertical)} front_vertical")'),
             
            # Update model type strings
            ('real_trained_model', 'real_trained_model_2_classes'),
            ('placeholder', 'placeholder_2_classes'),
        ]
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error fixing AI service: {e}")
        return False

def fix_ai_routes(file_path):
    """Fix the AI routes file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find and remove the mode parameter calls
        fixes = [
            # Remove mode parameter from analyze_image calls
            ('result = ai_service.analyze_image(image_data=current_frame, mode=analysis_mode)',
             'result = ai_service.analyze_image(image_data=current_frame)'),
            
            ('result = ai_service.analyze_image(image_path=fallback_image_path, mode=analysis_mode)',
             'result = ai_service.analyze_image(image_path=fallback_image_path)'),
            
            ('test_result = ai_service.analyze_image(image_data=current_frame, mode="pipeline")',
             'test_result = ai_service.analyze_image(image_data=current_frame)'),
            
            ('test_result = ai_service.analyze_image(image_path=test_image_path, mode="pipeline")',
             'test_result = ai_service.analyze_image(image_path=test_image_path)'),
            
            ('test_result = ai_service.analyze_image(image_path=test_path, mode="pipeline")',
             'test_result = ai_service.analyze_image(image_path=test_path)'),
            
            # Remove analysis_mode variable if it exists
            ('analysis_mode = data.get("analysis_mode", "pipeline")', '# analysis_mode removed'),
            ('print(f"📊 Analysis mode: {analysis_mode}")', '# Analysis mode logging removed'),
        ]
        
        for old, new in fixes:
            if old in content:
                content = content.replace(old, new)
        
        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error fixing AI routes: {e}")
        return False

def verify_fixes():
    """Verify all fixes were applied correctly"""
    
    print("\n🔍 VERIFYING ALL FIXES")
    print("=" * 30)
    
    project_root = "/home/team10/RebarWeb"
    ai_service_path = os.path.join(project_root, "app/services/ai_service.py")
    ai_routes_path = os.path.join(project_root, "app/routes/ai_routes.py")
    
    all_good = True
    
    # Check AI service
    try:
        with open(ai_service_path, 'r') as f:
            ai_content = f.read()
        
        ai_checks = [
            ('2 classes in service', 'self.num_classes = 2' in ai_content),
            ('Correct class names', '"front_horizontal", "front_vertical"' in ai_content),
            ('Lower threshold', 'self.detection_threshold = 0.2' in ai_content),
            ('No back_horizontal', 'back_horizontal' not in ai_content or '# back_horizontal removed' in ai_content)
        ]
        
        print("AI Service:")
        for check_name, check_result in ai_checks:
            icon = "✅" if check_result else "❌"
            print(f"   {icon} {check_name}")
            if not check_result:
                all_good = False
        
    except Exception as e:
        print(f"❌ Could not verify AI service: {e}")
        all_good = False
    
    # Check AI routes
    try:
        with open(ai_routes_path, 'r') as f:
            routes_content = f.read()
        
        routes_checks = [
            ('No mode parameter', 'mode=analysis_mode' not in routes_content),
            ('No mode in test', 'mode="pipeline"' not in routes_content),
            ('Clean analyze calls', 'analyze_image(image_data=current_frame)' in routes_content)
        ]
        
        print("AI Routes:")
        for check_name, check_result in routes_checks:
            icon = "✅" if check_result else "❌"
            print(f"   {icon} {check_name}")
            if not check_result:
                all_good = False
        
    except Exception as e:
        print(f"❌ Could not verify AI routes: {e}")
        all_good = False
    
    if all_good:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
    else:
        print("\n⚠️  Some verifications failed")
    
    return all_good

if __name__ == "__main__":
    print("🔧 REBAR VISTA - COMPLETE DETECTION FIX")
    print("Fixing all issues preventing rebar detection")
    print("")
    
    if apply_complete_fix():
        verify_fixes()
        print("\n📋 SUMMARY OF CHANGES:")
        print("  • Model expects 2 classes: front_horizontal, front_vertical")
        print("  • Detection threshold: 0.2 (more sensitive)")
        print("  • AI routes: mode parameter removed")
        print("  • All references updated for 2-class system")
        print("")
        print("🚀 Ready to test! Restart with: python3 main.py")
    else:
        print("\n❌ Fix failed - manual intervention needed")
        print("Check the error messages above and fix manually")
