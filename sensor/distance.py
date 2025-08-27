from gpiozero import DistanceSensor
from time import sleep

# TRIG = GPIO23, ECHO = GPIO24
sensor = DistanceSensor(echo=24, trigger=23)

try:
    while True:
        print(f"Distance: {sensor.distance * 100:.2f} cm")
        sleep(1)

except KeyboardInterrupt:
    print("Measurement stopped")
