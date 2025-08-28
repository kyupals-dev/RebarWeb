import lgpio
import time
import statistics

# Pin setup
TRIG = 23
ECHO = 24

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, TRIG)
lgpio.gpio_claim_input(chip, ECHO)

def get_distance():
    """Single raw measurement in cm"""
    # Trigger pulse
    lgpio.gpio_write(chip, TRIG, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(chip, TRIG, 0)

    # Wait for echo start
    start = time.time()
    while lgpio.gpio_read(chip, ECHO) == 0:
        start = time.time()

    # Wait for echo end
    stop = time.time()
    while lgpio.gpio_read(chip, ECHO) == 1:
        stop = time.time()

    # Time difference
    elapsed = stop - start

    # Speed of sound = 34300 cm/s
    distance = (elapsed * 34300) / 2

    return distance

def get_stable_distance(samples=5, delay=0.05):
    readings = []
    for _ in range(samples):
        try:
            d = get_distance()
            if 2 <= d <= 400:
                readings.append(d)
        except:
            pass
        time.sleep(delay)

    if readings:
        return statistics.median(readings)
    else:
        return None

try:
    while True:
        distance = get_stable_distance(samples=7)
        if distance is not None:
            distance = min(distance, 400)
            print(f"Distance: {distance:.2f} cm")
        else:
            print("No valid reading")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Measurement stopped")
    lgpio.gpiochip_close(chip)
