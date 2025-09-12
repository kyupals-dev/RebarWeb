#!/usr/bin/env python3
"""
Fix Class Configuration Script
Updates the AI service to use only 2 classes as per trained model
"""

import os
import shutil
from datetime import datetime

def fix_class_configuration():
    """Fix the class configuration to match trained model"""
    
    print("🔧 FIXING CLASS CONFIGURATION")
    print("=" * 40)
    print("Updating from 3 classes to 2 classes:")
    print("  ❌ Removing: 'back_horizontal'")
    print("  ✅ Keeping: 'front_horizontal', 'front_vertical'")
    print("  🎯 Lowering threshold: 0.3 → 0.2")
    
    project_root = "/home/team10/RebarWeb"
    ai_service_path = os.path.join(project_root, "app/services/ai_service.py")
    
    if not os.path.exists(ai_service_path):
        print(f"❌ AI service file not found: {ai_service_path}")
        return False
    
    # Create backup
    backup_path = f"{ai_service_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(ai_service_path, backup_path)
        print(f"💾 Backup created: {backup_path}")
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")
    
    # Read current file
    try:
        with open(ai_service_path, 'r') as f:
            content = f.read()
        
        print("📝 Applying fixes...")
        
        # Fix 1: Update class names from 3 to 2
        old_classes = 'self.class_names = ["back_horizontal", "front_horizontal", "front_vertical"]'
        new_classes = 'self.class_names = ["front_horizontal", "front_vertical"]'
        
        if old_classes in content:
            content = content.replace(old_classes, new_classes)
            print("   ✅ Fixed class names (3 → 2 classes)")
        else:
            # Try alternative format
            old_classes_alt = 'self.class_names = ["front_vertical", "front_horizontal", "back_horizontal"]'
            if old_classes_alt in content:
                content = content.replace(old_classes_alt, new_classes)
                print("   ✅ Fixed class names (alternative format)")
        
        # Fix 2: Update number of classes
        old_num_classes = 'self.num_classes = 3'
        new_num_classes = 'self.num_classes = 2'
        
        if old_num_classes in content:
            content = content.replace(old_num_classes, new_num_classes)
            print("   ✅ Fixed number of classes (3 → 2)")
        
        # Fix 3: Lower detection threshold
        old_threshold = 'self.detection_threshold = 0.3'
        new_threshold = 'self.detection_threshold = 0.2'
        
        if old_threshold in content:
            content = content.replace(old_threshold, new_threshold)
            print("   ✅ Fixed detection threshold (0.3 → 0.2)")
        
        # Fix 4: Update colors for 2 classes only
        old_colors = '''self.metadata.thing_colors = [
                (0, 255, 0),      # front_vertical - Green
                (255, 0, 0),      # front_horizontal - Red  
                (0, 0, 255),      # back_horizontal - Blue
            ]'''
        new_colors = '''self.metadata.thing_colors = [
                (255, 0, 0),      # front_horizontal - Red  
                (0, 255, 0),      # front_vertical - Green
            ]'''
        
        if old_colors in content:
            content = content.replace(old_colors, new_colors)
            print("   ✅ Fixed color configuration (2 classes)")
        
        # Fix 5: Remove back_horizontal references in dimension calculation
        old_back_ref = "back_horizontal = [d for d in detections if d['class_name'] == 'back_horizontal']"
        if old_back_ref in content:
            content = content.replace(old_back_ref, "# back_horizontal removed - using 2 classes only")
            print("   ✅ Removed back_horizontal references")
        
        # Fix 6: Update print statements
        content = content.replace(
            'print(f"   Found: {len(front_vertical)} front_vertical, {len(front_horizontal)} front_horizontal, {len(back_horizontal)} back_horizontal")',
            'print(f"   Found: {len(front_horizontal)} front_horizontal, {len(front_vertical)} front_vertical")'
        )
        
        # Write updated file
        with open(ai_service_path, 'w') as f:
            f.write(content)
        
        print("✅ AI service file updated successfully!")
        print("")
        print("🔧 Changes applied:")
        print("   📚 Classes: front_horizontal, front_vertical (2 total)")
        print("   🎯 Threshold: 0.2 (was 0.3)")
        print("   🎨 Colors: Red, Green (2 total)")
        print("   🧮 Calculations: Updated for 2-class system")
        print("")
        print("🚀 Next steps:")
        print("1. Restart the application: python3 main.py")
        print("2. Try capturing rebar image again")
        print("3. Model should now detect with 2-class configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        
        # Restore backup if exists
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, ai_service_path)
                print(f"🔄 Restored from backup: {backup_path}")
            except:
                pass
        
        return False

def verify_changes():
    """Verify the changes were applied correctly"""
    
    print("\n🔍 VERIFYING CHANGES")
    print("=" * 25)
    
    ai_service_path = "/home/team10/RebarWeb/app/services/ai_service.py"
    
    try:
        with open(ai_service_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('2 classes', 'self.num_classes = 2' in content),
            ('Correct class names', '"front_horizontal", "front_vertical"' in content),
            ('Lower threshold', 'self.detection_threshold = 0.2' in content),
            ('No back_horizontal', 'back_horizontal' not in content or '# back_horizontal removed' in content)
        ]
        
        all_good = True
        for check_name, check_result in checks:
            icon = "✅" if check_result else "❌"
            print(f"   {icon} {check_name}")
            if not check_result:
                all_good = False
        
        if all_good:
            print("\n🎉 All changes verified successfully!")
            print("   Ready to test with 2-class model")
        else:
            print("\n⚠️  Some checks failed - manual review needed")
        
        return all_good
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 REBAR VISTA - CLASS CONFIGURATION FIX")
    print("Updating model to use 2 classes instead of 3")
    print("Matching your trained model configuration")
    print("")
    
    if fix_class_configuration():
        verify_changes()
        print("\n📋 Summary:")
        print("  • Model now expects 2 classes (front_horizontal, front_vertical)")
        print("  • Detection threshold lowered to 0.2 for better sensitivity")
        print("  • Visualization colors updated for 2 classes")
        print("  • Calculation logic adapted for 2-class system")
        print("")
        print("🚀 Restart with: python3 main.py")
    else:
        print("\n❌ Fix failed - check manually or restore from backup")
