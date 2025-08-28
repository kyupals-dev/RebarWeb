"""
Distance Service for HC-SR04 Ultrasonic Sensor
Handles real-time distance measurement for optimal camera positioning
"""

import threading
import time
from datetime import datetime

# GPIO imports (will be available on Raspberry Pi)
try:
    from gpiozero import DistanceSensor
    GPIO_AVAILABLE = True
except ImportError:
    print("⚠️  GPIO/gpiozero not available. Distance sensor will use simulation mode.")
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
        
        # Distance thresholds (in cm)
        self.min_optimal = 160  # 160cm minimum
        self.max_optimal = 200  # 200cm maximum
        
        # Simulation mode for testing
        self.simulation_mode = not GPIO_AVAILABLE
        self.sim_distance = 180.0  # Default simulation distance
        
        print("📏 Initializing Distance Service...")
        self.initialize_sensor()
    
    def initialize_sensor(self):
        """Initialize HC-SR04 sensor"""
        try:
            if GPIO_AVAILABLE:
                print("🔌 Connecting to HC-SR04 sensor...")
                # TRIG = GPIO23, ECHO = GPIO24 (as defined in your original script)
                self.sensor = DistanceSensor(echo=24, trigger=23)
                
                # Test sensor with a few readings
                test_readings = 0
                for i in range(5):
                    try:
                        distance = self.sensor.distance * 100  # Convert to cm
                        if 0 < distance < 1000:  # Valid range
                            test_readings += 1
                        time.sleep(0.1)
                    except Exception:
                        pass
                
                if test_readings >= 3:
                    self.sensor_available = True
                    print("✅ HC-SR04 sensor initialized successfully")
                    print(f"   GPIO Pins: TRIG=23, ECHO=24")
                    print(f"   Optimal range: {self.min_optimal}-{self.max_optimal}cm")
                else:
                    print("❌ HC-SR04 sensor test failed, using simulation")
                    self.sensor_available = False
                    self.simulation_mode = True
            else:
                print("📝 GPIO not available, using simulation mode")
                self.sensor_available = False
                self.simulation_mode = True
                
        except Exception as e:
            print(f"❌ Error initializing HC-SR04 sensor: {str(e)}")
            self.sensor_available = False
            self.simulation_mode = True
            self.last_error = str(e)
    
    def start_monitoring(self):
        """Start distance monitoring in background thread"""
        if not self.is_running:
            print("🚀 Starting distance monitoring...")
            self.is_running = True
            
            monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            monitor_thread.start()
            
            print("✅ Distance monitoring started")
            return True
        else:
            print("⚠️  Distance monitoring already running")
            return True
    
    def stop_monitoring(self):
        """Stop distance monitoring"""
        print("🛑 Stopping distance monitoring...")
        self.is_running = False
        
        # Clean up sensor
        if self.sensor:
            try:
                self.sensor.close()
            except Exception:
                pass
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in background thread)"""
        print("📏 Distance monitoring loop started")
        consecutive_errors = 0
        max_errors = 10
        
        while self.is_running:
            try:
                if self.simulation_mode:
                    # Simulation mode - vary distance for testing
                    import random
                    base_distance = 180.0
                    variation = random.uniform(-30, 30)
                    distance_cm = max(50, min(300, base_distance + variation))
                    time.sleep(0.5)  # 500ms update rate
                else:
                    # Real sensor reading
                    if self.sensor and self.sensor_available:
                        distance_m = self.sensor.distance
                        distance_cm = distance_m * 100  # Convert to cm
                        
                        # Validate reading
                        if distance_cm <= 0 or distance_cm > 500:
                            consecutive_errors += 1
                            if consecutive_errors >= max_errors:
                                print("⚠️  Too many sensor errors, switching to simulation")
                                self.simulation_mode = True
                                self.last_error = "Sensor reading errors"
                            continue
                        
                        consecutive_errors = 0  # Reset error count
                    else:
                        # Fallback to simulation if sensor not available
                        distance_cm = 180.0
                
                # Update current distance (thread-safe)
                with self.distance_lock:
                    self.current_distance = round(distance_cm, 1)
                
                # Sleep for 500ms update rate
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("🛑 Distance monitoring interrupted")
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors % 5 == 0:  # Log every 5 errors
                    print(f"⚠️  Distance sensor error: {e}")
                
                if consecutive_errors >= max_errors:
                    print("❌ Too many sensor errors, switching to simulation")
                    self.simulation_mode = True
                    self.last_error = str(e)
                
                time.sleep(0.5)
        
        print("🏁 Distance monitoring loop finished")
    
    def get_current_reading(self):
        """Get current distance reading with status"""
        try:
            with self.distance_lock:
                distance = self.current_distance
            
            if distance is None:
                return {
                    'success': False,
                    'error': 'No distance reading available',
                    'distance': None,
                    'status': 'unavailable',
                    'status_text': 'SENSOR ERROR',
                    'status_color': 'red'
                }
            
            # Determine status based on optimal range
            if distance < self.min_optimal:
                status = 'too_close'
                status_text = 'TOO CLOSE'
                status_color = 'red'
            elif self.min_optimal <= distance <= self.max_optimal:
                status = 'optimal'
                status_text = 'OPTIMAL'
                status_color = 'green'
            else:  # distance > max_optimal
                status = 'too_far'
                status_text = 'TOO FAR'
                status_color = 'yellow'
            
            return {
                'success': True,
                'distance': distance,
                'distance_text': f'{distance:.0f}cm',
                'status': status,
                'status_text': status_text,
                'status_color': status_color,
                'optimal_range': f'{self.min_optimal}-{self.max_optimal}cm',
                'simulation_mode': self.simulation_mode
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'distance': None,
                'status': 'error',
                'status_text': 'ERROR',
                'status_color': 'red'
            }
    
    def get_sensor_status(self):
        """Get detailed sensor status"""
        return {
            'sensor_available': self.sensor_available,
            'is_running': self.is_running,
            'gpio_available': GPIO_AVAILABLE,
            'simulation_mode': self.simulation_mode,
            'last_error': self.last_error,
            'current_distance': self.current_distance,
            'optimal_range': {
                'min': self.min_optimal,
                'max': self.max_optimal,
                'unit': 'cm'
            },
            'gpio_pins': {
                'trigger': 23,
                'echo': 24
            }
        }
    
    def test_sensor(self):
        """Test sensor functionality"""
        try:
            print("🧪 Testing HC-SR04 sensor...")
            
            if not self.sensor_available and not self.simulation_mode:
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
                    'readings_count': len(readings),
                    'simulation_mode': self.simulation_mode
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
