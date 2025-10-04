#!/usr/bin/env python3
"""
Distance Service for HC-SR04 Ultrasonic Sensor
Handles real-time distance measurement for optimal camera positioning
"""

import threading
import time
from datetime import datetime

# GPIO imports (required on Raspberry Pi)
try:
    from gpiozero import DistanceSensor
    GPIO_AVAILABLE = True
except ImportError:
    print("❌ GPIO/gpiozero not available. Distance sensor requires Raspberry Pi hardware.")
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
        
        print("📏 Initializing Distance Service...")
        self.initialize_sensor()
    
    def initialize_sensor(self):
        """Initialize HC-SR04 sensor"""
        try:
            if not GPIO_AVAILABLE:
                raise ImportError("GPIO not available - sensor requires Raspberry Pi hardware")
            
            print("🔌 Connecting to HC-SR04 sensor...")
            # TRIG = GPIO23, ECHO = GPIO24, max distance 3m (300cm)
            self.sensor = DistanceSensor(echo=24, trigger=23, max_distance=3.0)
            
            # Test sensor with a few readings
            print("🧪 Testing sensor...")
            test_readings = 0
            for i in range(5):
                try:
                    distance = self.sensor.distance * 100  # Convert to cm
                    if 2 < distance < 400:  # Valid range
                        test_readings += 1
                        print(f"   Test reading {i+1}: {distance:.1f}cm")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"   Test reading {i+1} failed: {e}")
            
            if test_readings >= 3:
                self.sensor_available = True
                print("✅ HC-SR04 sensor initialized successfully")
                print(f"   GPIO Pins: TRIG=23, ECHO=24")
                print(f"   Optimal range: {self.min_optimal}-{self.max_optimal}cm")
            else:
                raise Exception(f"Insufficient valid readings: {test_readings}/5")
                
        except Exception as e:
            print(f"❌ Error initializing HC-SR04 sensor: {str(e)}")
            self.sensor_available = False
            self.last_error = str(e)
    
    def start_monitoring(self):
        """Start distance monitoring in background thread"""
        if not self.sensor_available:
            print("❌ Cannot start monitoring - sensor not available")
            return False
        
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
        time.sleep(0.5)  # Give loop time to stop
        
        # Clean up sensor
        if self.sensor:
            try:
                self.sensor.close()
                print("✅ Sensor closed cleanly")
            except Exception as e:
                print(f"⚠️  Error closing sensor: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in background thread)"""
        print("📏 Distance monitoring loop started\n")
        consecutive_errors = 0
        max_errors = 10
        
        while self.is_running:
            try:
                if self.sensor:
                    try:
                        # Read distance from sensor
                        distance_m = self.sensor.distance
                        distance_cm = distance_m * 100  # Convert to cm
                        
                        # improved validation
                        if distance_cm < 5:
                            #
                            #
                            distance_cm = 5.0
                            with self.distance_lock:
                                self.current_distance = distance_cm
                            consecutive_errors = 0
                        
                        elif 5 <= distance_cm <= 400:
                            # Valid reading range
                            with self.distance_lock:
                                self.current_distance = round(distance_cm, 1)
                                consecutive_errors = 0  # Reset on success
                        
                        else:
                            # Reading too high (>400cm) - likely error
                            consecutive_errors += 1
                            if consecutive_errors % 5 == 0:
                                print(f"⚠️  Invalid reading: {distance_cm:.1f}cm (out of range)")
                                
                    except Exception as sensor_error:
                        consecutive_errors += 1
                        self.last_error = f"Sensor read error: {str(sensor_error)}"
                        
                        if consecutive_errors % 5 == 0:
                            print(f"⚠️  Sensor read error: {sensor_error}")
                        
                        # Try to recover after 5 consecutive errors
                        if consecutive_errors == 5:
                            print("🔄 Attempting sensor recovery...")
                            try:
                                self.sensor.close()
                                time.sleep(0.5)
                                self.initialize_sensor()
                                if self.sensor_available:
                                    print("✅ Sensor recovered successfully")
                                    consecutive_errors = 0
                            except Exception as reinit_error:
                                print(f"❌ Recovery failed: {reinit_error}")
                else:
                    print("❌ Sensor object is None")
                    consecutive_errors += 1
                    
                # Stop if too many consecutive errors
                if consecutive_errors >= max_errors:
                    print(f"❌ Too many errors ({consecutive_errors}/{max_errors}), stopping monitoring")
                    print(f"   Last error: {self.last_error}")
                    self.is_running = False
                    break
                    
            except Exception as e:
                consecutive_errors += 1
                self.last_error = str(e)
                print(f"⚠️  Monitoring error: {e}")
                
            # Sleep interval (500ms update rate)
            time.sleep(0.5)
        
        print("🏁 Monitoring loop stopped")
    
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
                'optimal_range': f'{self.min_optimal}-{self.max_optimal}cm'
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
