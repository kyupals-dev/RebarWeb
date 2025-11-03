#!/usr/bin/env python3
"""
Debug script to test AI routes import
Run this to identify the specific import error in ai_routes.py
"""

import sys
import traceback
import os

# Add the project root to Python path
project_root = "/home/team10/RebarWeb"
sys.path.insert(0, project_root)

print("🔍 Debugging AI Routes Import Issue")
print("=" * 50)
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

# Test imports step by step
print("\n1. Testing basic imports...")

try:
    from flask import Blueprint, jsonify, request
    print("✅ Flask imports successful")
except ImportError as e:
    print(f"❌ Flask import error: {e}")
    sys.exit(1)

try:
    import os
    from datetime import datetime
    print("✅ Standard library imports successful")
except ImportError as e:
    print(f"❌ Standard library import error: {e}")
    sys.exit(1)

print("\n2. Testing config import...")
try:
    from app.utils.config import config
    print("✅ Config import successful")
    print(f"   Upload folder: {config.UPLOAD_FOLDER}")
except ImportError as e:
    print(f"❌ Config import error: {e}")
    print("   This might be the issue!")
    traceback.print_exc()

print("\n3. Testing AI routes import with detailed error...")
try:
    # Try to import the entire module first
    import app.routes.ai_routes as ai_routes_module
    print("✅ AI routes module imported successfully")
    
    # Check if init_ai_routes function exists
    if hasattr(ai_routes_module, 'init_ai_routes'):
        print("✅ init_ai_routes function found in module")
        print(f"   Function type: {type(ai_routes_module.init_ai_routes)}")
    else:
        print("❌ init_ai_routes function NOT found in module")
        print(f"   Available attributes: {dir(ai_routes_module)}")
    
except Exception as e:
    print(f"❌ AI routes module import error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    # Try to identify the specific line causing the issue
    print("\n4. Trying to identify problematic import...")
    try:
        # Test each import statement from ai_routes.py individually
        print("   Testing Blueprint...")
        from flask import Blueprint, jsonify, request
        print("   ✅ Flask Blueprint import OK")
        
        print("   Testing os and datetime...")
        import os
        from datetime import datetime
        print("   ✅ Standard imports OK")
        
        print("   Testing config...")
        from app.utils.config import config
        print("   ✅ Config import OK")
        
        print("   The error must be in the ai_routes.py file execution itself")
        
    except Exception as inner_e:
        print(f"   ❌ Inner import error: {inner_e}")
        traceback.print_exc()

print("\n5. Testing specific import statement...")
try:
    from app.routes.ai_routes import init_ai_routes
    print("✅ init_ai_routes import successful!")
    print(f"   Function: {init_ai_routes}")
except ImportError as e:
    print(f"❌ Specific import error: {e}")
    print("\nThis confirms the issue. Let's check the file content...")
    
    # Check if the file exists and is readable
    ai_routes_path = os.path.join(project_root, "app", "routes", "ai_routes.py")
    print(f"\n6. File system check:")
    print(f"   File path: {ai_routes_path}")
    print(f"   File exists: {os.path.exists(ai_routes_path)}")
    
    if os.path.exists(ai_routes_path):
        try:
            with open(ai_routes_path, 'r') as f:
                content = f.read()
            print(f"   File size: {len(content)} characters")
            print(f"   Contains 'def init_ai_routes': {'def init_ai_routes' in content}")
            
            # Check for syntax errors
            print("\n7. Checking for syntax errors...")
            try:
                compile(content, ai_routes_path, 'exec')
                print("   ✅ No syntax errors found")
            except SyntaxError as se:
                print(f"   ❌ Syntax error: {se}")
                print(f"   Line {se.lineno}: {se.text}")
                
        except Exception as file_e:
            print(f"   ❌ Error reading file: {file_e}")
    else:
        print("   ❌ File does not exist!")

print("\n" + "=" * 50)
print("DIAGNOSIS COMPLETE")
print("=" * 50)
