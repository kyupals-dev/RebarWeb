#!/bin/bash

echo "🔧 QUICK FIX FOR CAMERA.JS SYNTAX ERROR"
echo "======================================="

cd /home/team10/RebarWeb

echo "📋 Checking the exact error location..."

# Check line 854 and surrounding area
echo "Lines around 854:"
sed -n '850,860p' static/javascript/camera.js

echo ""
echo "📋 Checking the very end of the file..."
echo "Last 10 lines:"
tail -10 static/javascript/camera.js

echo ""
echo "🔍 Checking for common syntax issues..."

# Count braces, brackets, parentheses
OPEN_BRACES=$(grep -o "{" static/javascript/camera.js | wc -l)
CLOSE_BRACES=$(grep -o "}" static/javascript/camera.js | wc -l)
OPEN_PARENS=$(grep -o "(" static/javascript/camera.js | wc -l)
CLOSE_PARENS=$(grep -o ")" static/javascript/camera.js | wc -l)

echo "Open braces: $OPEN_BRACES"
echo "Close braces: $CLOSE_BRACES"
echo "Open parentheses: $OPEN_PARENS"
echo "Close parentheses: $CLOSE_PARENS"

if [ $OPEN_BRACES -ne $CLOSE_BRACES ]; then
    echo "❌ BRACE MISMATCH DETECTED!"
    echo "Missing $(($OPEN_BRACES - $CLOSE_BRACES)) closing braces"
fi

if [ $OPEN_PARENS -ne $CLOSE_PARENS ]; then
    echo "❌ PARENTHESES MISMATCH DETECTED!"
    echo "Missing $(($OPEN_PARENS - $CLOSE_PARENS)) closing parentheses"
fi

echo ""
echo "🔧 Creating backup and applying fix..."

# Backup the file
cp static/javascript/camera.js static/javascript/camera.js.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"

# The most common issue is missing closing brace or semicolon at the end
# Let's check what the file actually ends with
echo ""
echo "📋 Current file ending (hex dump):"
tail -c 20 static/javascript/camera.js | hexdump -C

echo ""
echo "🔧 Applying fix..."

# Remove any trailing whitespace and ensure proper ending
sed -i 's/[[:space:]]*$//' static/javascript/camera.js

# Check if the file ends with a semicolon or brace
LAST_CHAR=$(tail -c 2 static/javascript/camera.js | head -c 1)
echo "Last character: '$LAST_CHAR'"

# Add missing closing brace if needed (this is the most common issue)
if [[ "$LAST_CHAR" != "}" && "$LAST_CHAR" != ";" ]]; then
    echo "Adding missing closing brace..."
    echo "}" >> static/javascript/camera.js
fi

echo "✅ Fix applied!"
echo ""
echo "🧪 To test the fix:"
echo "1. Refresh your browser (Ctrl+F5 or Cmd+Shift+R)"
echo "2. Check if the camera feed appears"
echo "3. Look at browser console for any remaining errors"
echo ""
echo "If issues persist, check the syntax manually:"
echo "nano static/javascript/camera.js"
echo "Go to the end of the file and ensure proper closing"
