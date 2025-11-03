from gpiozero import DistanceSensor, Device
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep

# Use lgpio factory (works on Raspberry Pi 5)
Device.pin_factory = LGPIOFactory()

# TRIG = GPIO23, ECHO = GPIO24
sensor = DistanceSensor(echo=24, trigger=23, max_distance=4)

# Calibration (simple)
calibration_offset = 0.0
calibration_factor = 1.0

try:
    while True:
        # Raw distance (0–400 cm)
        distance_cm = (sensor.distance * 400 * calibration_factor) + calibration_offset

        # Clamp to sensor's limits
        if distance_cm > 400:
            distance_cm = 400
        elif distance_cm < 0:
            distance_cm = 0

        print(f"Distance: {distance_cm:.2f} cm")
        sleep(0.5)

except KeyboardInterrupt:
    print("Measurement stopped")
