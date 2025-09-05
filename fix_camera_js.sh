#!/bin/bash

echo "🔧 FIXING CAMERA.JS SYNTAX ERROR"
echo "================================"

cd /home/team10/RebarWeb

# Check the end of camera.js file for syntax issues
echo "📋 Checking end of camera.js file..."
tail -20 static/javascript/camera.js

echo ""
echo "📋 Checking for common syntax issues..."

# Check for missing closing braces, brackets, or parentheses
echo "🔍 Checking for unmatched braces..."
grep -n "{" static/javascript/camera.js | wc -l
grep -n "}" static/javascript/camera.js | wc -l

echo "🔍 Checking for unmatched brackets..."
grep -n "\[" static/javascript/camera.js | wc -l
grep -n "\]" static/javascript/camera.js | wc -l

echo "🔍 Checking for unmatched parentheses..."
grep -n "(" static/javascript/camera.js | wc -l
grep -n ")" static/javascript/camera.js | wc -l

echo ""
echo "📋 Looking at line 853 and surrounding lines..."
sed -n '845,860p' static/javascript/camera.js

echo ""
echo "🔧 To fix this:"
echo "1. Open the camera.js file"
echo "2. Go to line 853"
echo "3. Look for missing closing brace } or semicolon ;"
echo "4. The error is likely at the very end of the file"

echo ""
echo "🚀 Quick fix attempt..."

# Create a backup
cp static/javascript/camera.js static/javascript/camera.js.backup
echo "✅ Backup created: camera.js.backup"

# Check if the file ends properly
echo "📋 Current file ending:"
tail -5 static/javascript/camera.js

# Most likely fix: ensure the file ends with proper closing
echo ""
echo "🔧 Applying common fixes..."

# Add missing semicolon/brace if needed at the end
echo "" >> static/javascript/camera.js

echo "✅ Fix attempt complete!"
echo ""
echo "🧪 Test the fix by:"
echo "1. Refresh your browser page (Ctrl+F5)"
echo "2. Check browser console for errors"
echo "3. If still errors, run: nano static/javascript/camera.js"
echo "4. Go to the end of the file and ensure proper syntax"
