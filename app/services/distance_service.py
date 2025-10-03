#!/usr/bin/env python3
"""
HC-SR04 Distance Service
- Max distance limited to 300 cm
- Optimal range: 198–205 cm
"""

import threading
import time
import signal
import sys
import warnings

warnings.filterwarnings("ignore")  # Suppress gpiozero warnings

# GPIO imports (only available on Raspberry Pi)
try:
    from gpiozero import DistanceSensor
    GPIO_AVAILABLE = True
except ImportError:
    print("⚠️  GPIO/gpiozero not available. Cannot run without sensor hardware.")
    GPIO_AVAILABLE = False


class DistanceService:
    """Handles HC-SR04 distance sensor for optimal positioning guidance"""

    def __init__(self):
        self.sensor = None
        self.is_running = False
        self.current_distance = None
        self.distance_lock = threading.Lock()
        self.last_error = None
        self.sensor_available = False

        # Distance thresholds (cm)
        self.min_optimal = 198
        self.max_optimal = 205

        print("📏 Initializing Distance Service...")
        self.initialize_sensor()

    def initialize_sensor(self):
        """Initialize HC-SR04 sensor"""
        try:
            if not GPIO_AVAILABLE:
                raise ImportError("GPIO not available - sensor requires Raspberry Pi hardware")

            print("🔌 Connecting to HC-SR04 sensor...")
            # Set max_distance=3.0m (limit readings to 300cm)
            self.sensor = DistanceSensor(echo=24, trigger=23, max_distance=3.0)

            # Quick test
            time.sleep(0.5)
            test_dist = self.sensor.distance * 100
            if 2 <= test_dist <= 300:
                self.sensor_available = True
                print("✅ HC-SR04 initialized successfully")
                print(f"   GPIO Pins: TRIG=23, ECHO=24")
                print(f"   Optimal range: {self.min_optimal}-{self.max_optimal}cm")
            else:
                raise Exception("Invalid initial reading")

        except Exception as e:
            print(f"❌ Sensor initialization failed: {e}")
            self.sensor_available = False
            self.last_error = str(e)

    def start_monitoring(self):
        """Start background monitoring"""
        if not self.sensor_available:
            print("❌ Cannot start monitoring - sensor not available")
            return False
            
        if not self.is_running:
            print("🚀 Starting distance monitoring...")
            self.is_running = True
            t = threading.Thread(target=self._monitoring_loop, daemon=True)
            t.start()
            return True
        return False

    def stop_monitoring(self):
        """Stop monitoring"""
        print("🛑 Stopping distance monitoring...")
        self.is_running = False
        if self.sensor:
            try:
                self.sensor.close()
            except Exception:
                pass

    def _monitoring_loop(self):
        """Background monitoring loop"""
        print("📏 Distance monitoring loop started")
        consecutive_errors = 0
        max_errors = 10

        while self.is_running:
            try:
                # Real sensor reading
                distance_m = self.sensor.distance
                distance_cm = distance_m * 100
                
                if distance_cm <= 0 or distance_cm > 300:
                    raise ValueError("Out of range")

                # Thread-safe update
                with self.distance_lock:
                    self.current_distance = round(distance_cm, 1)

                consecutive_errors = 0
                time.sleep(0.5)

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    print(f"❌ Too many errors ({max_errors}), stopping monitoring")
                    print(f"   Last error: {e}")
                    self.last_error = str(e)
                    self.is_running = False
                    break
                time.sleep(0.5)

        print("🏁 Monitoring loop stopped")

    def get_current_reading(self):
        """Return latest distance with status"""
        with self.distance_lock:
            distance = self.current_distance

        if distance is None:
            return {
                "success": False,
                "error": "No distance available",
                "status": "unavailable",
                "status_text": "SENSOR ERROR",
                "status_color": "red"
            }

        if distance < self.min_optimal:
            status, text, color = "too_close", "TOO CLOSE", "red"
        elif self.min_optimal <= distance <= self.max_optimal:
            status, text, color = "optimal", "OPTIMAL", "green"
        else:
            status, text, color = "too_far", "TOO FAR", "yellow"

        return {
            "success": True,
            "distance": distance,
            "distance_text": f"{distance:.0f}cm",
            "status": status,
            "status_text": text,
            "status_color": color,
            "optimal_range": f"{self.min_optimal}-{self.max_optimal}cm"
        }

    def get_sensor_status(self):
        """Return sensor/service state"""
        return {
            "sensor_available": self.sensor_available,
            "is_running": self.is_running,
            "gpio_available": GPIO_AVAILABLE,
            "simulation_mode": False,  # No simulation mode - always real sensor
            "last_error": self.last_error,
            "current_distance": self.current_distance,
            "optimal_range": {
                "min": self.min_optimal,
                "max": self.max_optimal,
                "unit": "cm"
            },
            "gpio_pins": {"trigger": 23, "echo": 24}
        }
    
    def test_sensor(self):
        """Test sensor functionality"""
        try:
            print("🧪 Testing HC-SR04 sensor...")
            
            if not self.sensor_available:
                return {
                    'success': False,
                    'error': 'Sensor not available'
                }
            
            # Get a few test readings
            readings = []
            for i in range(5):
                reading = self.get_current_reading()
                if reading['success']:
                    readings.append(reading['distance'])
                time.sleep(0.1)
            
            if len(readings) >= 3:
                avg_distance = sum(readings) / len(readings)
                return {
                    'success': True,
                    'test_readings': readings,
                    'average_distance': round(avg_distance, 1),
                    'readings_count': len(readings)
                }
            else:
                return {
                    'success': False,
                    'error': f'Insufficient readings: {len(readings)}/5'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Test failed: {str(e)}'
            }


if __name__ == "__main__":
    # Graceful exit
    def signal_handler(sig, frame):
        print("\n\nMeasurement stopped by user")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    service = DistanceService()
    service.start_monitoring()

    try:
        while True:
            reading = service.get_current_reading()
            if reading["success"]:
                print(f"Distance: {reading['distance_text']} ({reading['status_text']})")
            else:
                print("Sensor Error:", reading.get("error", "Unknown"))
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop_monitoring()
