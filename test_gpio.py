#!/usr/bin/env python3
import sys

try:
    from gpiozero import Device, DistanceSensor
    from gpiozero.pins.lgpio import LGPIOFactory
    
    Device.pin_factory = LGPIOFactory()
    sensor = DistanceSensor(echo=24, trigger=23)
    
    print("✅ GPIO and gpiozero working correctly")
    print(f"Pin factory: {Device.pin_factory}")
    
    # Test reading
    distance = sensor.distance * 100
    print(f"Test reading: {distance:.1f}cm")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ GPIO error: {e}")
    sys.exit(1)
finally:
    try:
        sensor.close()
    except:
        pass

print("GPIO test completed successfully")
