#!/usr/bin/env python3
"""
Quick Fix for Rebar Detection Issues
This script modifies the detection threshold and tests immediately
"""

import sys
import os

# Add project root to path
project_root = "/home/team10/RebarWeb"
sys.path.insert(0, project_root)

def quick_fix_detection():
    """Quick fix for detection issues"""
    
    print("🚀 QUICK DETECTION FIX")
    print("=" * 30)
    
    try:
        # Modify the AI service file to lower the detection threshold
        ai_service_path = "/home/team10/RebarWeb/app/services/ai_service.py"
        
        if os.path.exists(ai_service_path):
            print("📝 Reading AI service file...")
            
            with open(ai_service_path, 'r') as f:
                content = f.read()
            
            # Find and replace the detection threshold
            old_threshold = "self.detection_threshold = 0.3"
            new_threshold = "self.detection_threshold = 0.1"  # Much lower threshold
            
            if old_threshold in content:
                print(f"🔧 Changing detection threshold from 0.3 to 0.1...")
                content = content.replace(old_threshold, new_threshold)
                
                # Create backup
                backup_path = ai_service_path + ".backup"
                with open(backup_path, 'w') as f:
                    f.write(content)
                print(f"💾 Backup created: {backup_path}")
                
                # Write modified file
                with open(ai_service_path, 'w') as f:
                    f.write(content)
                
                print("✅ Detection threshold lowered to 0.1")
                print("🔄 Restart the application: python3 main.py")
                
            else:
                print("⚠️  Could not find threshold setting to modify")
                print("Manual fix needed in ai_service.py")
        
        else:
            print(f"❌ AI service file not found: {ai_service_path}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def restore_threshold():
    """Restore original threshold"""
    ai_service_path = "/home/team10/RebarWeb/app/services/ai_service.py"
    backup_path = ai_service_path + ".backup"
    
    if os.path.exists(backup_path):
        print("🔄 Restoring original threshold...")
        
        with open(backup_path, 'r') as f:
            content = f.read()
        
        with open(ai_service_path, 'w') as f:
            f.write(content)
        
        print("✅ Original threshold restored")
        os.remove(backup_path)
        print("🗑️  Backup file removed")
    else:
        print("❌ No backup file found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_threshold()
    else:
        quick_fix_detection()
        print("\n📋 Usage:")
        print(f"  To fix:     python3 {sys.argv[0]}")
        print(f"  To restore: python3 {sys.argv[0]} restore")
