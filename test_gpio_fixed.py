#!/usr/bin/env python3
import sys

print("🔍 Testing GPIO availability...")

# Test 1: Import gpiozero
try:
    import gpiozero
    print("✅ gpiozero imported successfully")
except ImportError as e:
    print(f"❌ gpiozero import failed: {e}")
    sys.exit(1)

# Test 2: Import pin factory for Pi 5
try:
    from gpiozero import Device
    from gpiozero.pins.lgpio import LGPIOFactory
    Device.pin_factory = LGPIOFactory()
    print(f"✅ Pin factory set for Pi 5: {Device.pin_factory.__class__.__name__}")
except ImportError as e:
    print(f"⚠️  lgpio factory not available: {e}")
    print("   Trying default factory...")
except Exception as e:
    print(f"⚠️  Pin factory error: {e}")

# Test 3: Create sensor (the critical test)
try:
    from gpiozero import DistanceSensor
    print("📡 Creating HC-SR04 sensor on pins 23(TRIG) and 24(ECHO)...")
    sensor = DistanceSensor(echo=24, trigger=23)
    print("✅ HC-SR04 sensor created successfully")
    
    # Test 4: Take readings
    print("📏 Taking test readings...")
    import time
    for i in range(3):
        try:
            distance = sensor.distance * 100  # Convert to cm
            if 0 < distance < 500:
                print(f"   Reading {i+1}: {distance:.1f}cm")
            else:
                print(f"   Reading {i+1}: {distance:.1f}cm (out of range)")
            time.sleep(0.5)
        except Exception as e:
            print(f"   Reading {i+1} failed: {e}")
    
    sensor.close()
    print("✅ GPIO test completed successfully")
    
except Exception as e:
    print(f"❌ Sensor creation failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    if "busy" in str(e).lower():
        print("   🔍 GPIO pins are in use by another process")
        print("   💡 Try: pkill -f 'main.py' or restart your system")
    
    sys.exit(1)

