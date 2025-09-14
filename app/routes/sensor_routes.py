"""
Updated Sensor Routes to work with existing distance_service.py
FIXED: Matches your existing DistanceService method names
"""

from flask import Blueprint, jsonify, request
import traceback

# Create a Blueprint for sensor routes
sensor_bp = Blueprint('sensor', __name__)

# This will be injected when the blueprint is registered
distance_service = None

def init_sensor_routes(dist_service):
    """Initialize the sensor routes with service dependency"""
    global distance_service
    distance_service = dist_service
    print("✅ Sensor routes initialized with distance service")

def _validate_distance_service():
    """Helper function to validate distance service availability"""
    if not distance_service:
        return jsonify({
            'success': False,
            'error': 'Distance service not available'
        }), 503
    return None

@sensor_bp.route('/get-distance', methods=['GET'])
def get_distance():
    """
    Get current distance sensor reading
    FIXED: Uses your existing distance service method names
    """
    try:
        print("📏 Distance reading request received")
        
        # Validate distance service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Get current reading using your existing method
        reading = distance_service.get_current_reading()
        
        if reading and reading.get('success'):
            # Convert your format to the expected format
            result = {
                'success': True,
                'distance_cm': reading.get('distance', 0),
                'distance_text': reading.get('distance_text', '--cm'),
                'status': reading.get('status', 'unknown'),
                'status_text': reading.get('status_text', 'UNKNOWN'),
                'timestamp': reading.get('timestamp', ''),
                'simulation_mode': reading.get('simulation_mode', False),
                'optimal_range': reading.get('optimal_range', '160-200cm')
            }
            
            print(f"📏 Distance reading: {result['distance_text']} - {result['status_text']}")
            return jsonify(result)
        else:
            # Return error response
            error_msg = reading.get('error', 'Unknown error') if reading else 'No reading available'
            return jsonify({
                'success': False,
                'distance_cm': 0.0,
                'distance_text': '--cm',
                'status': 'error',
                'status_text': 'SENSOR ERROR',
                'error': error_msg,
                'simulation_mode': True
            })
            
    except Exception as e:
        error_msg = f'Distance reading failed: {str(e)}'
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'distance_cm': 0.0,
            'distance_text': '--cm',
            'status': 'error',
            'status_text': 'API ERROR'
        }), 500

@sensor_bp.route('/start-distance-monitoring', methods=['POST'])
def start_distance_monitoring():
    """
    Start distance sensor monitoring
    FIXED: Uses your existing start_monitoring method
    """
    try:
        print("🚀 Distance monitoring start request received")
        
        # Validate distance service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Start monitoring using your existing method
        result = distance_service.start_monitoring()
        
        if result:
            print("✅ Distance monitoring started")
            return jsonify({
                'success': True,
                'message': 'Distance monitoring started'
            })
        else:
            print("❌ Failed to start distance monitoring")
            return jsonify({
                'success': False,
                'error': 'Failed to start distance monitoring'
            }), 500
        
    except Exception as e:
        error_msg = f'Distance monitoring start failed: {str(e)}'
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@sensor_bp.route('/stop-distance-monitoring', methods=['POST'])
def stop_distance_monitoring():
    """
    Stop distance sensor monitoring
    FIXED: Uses your existing stop_monitoring method
    """
    try:
        print("🛑 Distance monitoring stop request received")
        
        # Validate distance service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Stop monitoring using your existing method
        distance_service.stop_monitoring()
        
        print("✅ Distance monitoring stopped")
        return jsonify({
            'success': True,
            'message': 'Distance monitoring stopped'
        })
        
    except Exception as e:
        error_msg = f'Distance monitoring stop failed: {str(e)}'
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@sensor_bp.route('/get-sensor-status', methods=['GET'])
def get_sensor_status():
    """
    Get distance sensor status and configuration
    FIXED: Uses your existing get_sensor_status method
    """
    try:
        print("📊 Sensor status request received")
        
        # Validate distance service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Get sensor status using your existing method
        status = distance_service.get_sensor_status()
        
        # Convert your format to expected format
        formatted_status = {
            'gpio_available': status.get('gpio_available', False),
            'sensor_available': status.get('sensor_available', False),
            'is_running': status.get('is_running', False),
            'simulation_mode': status.get('simulation_mode', True),
            'gpio_pins': status.get('gpio_pins', {'trigger': 23, 'echo': 24}),
            'optimal_range': status.get('optimal_range', {
                'min': 160,
                'max': 200,
                'unit': 'cm'
            }),
            'last_error': status.get('last_error'),
            'current_distance': status.get('current_distance')
        }
        
        print(f"📊 Sensor status: {'Available' if formatted_status['sensor_available'] else 'Unavailable'}")
        
        return jsonify({
            'success': True,
            'status': formatted_status
        })
        
    except Exception as e:
        error_msg = f'Sensor status request failed: {str(e)}'
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'status': {
                'gpio_available': False,
                'sensor_available': False,
                'is_running': False,
                'simulation_mode': True,
                'error': str(e)
            }
        }), 500

@sensor_bp.route('/test-distance-sensor', methods=['POST'])
def test_distance_sensor():
    """
    Test distance sensor functionality
    FIXED: Uses your existing test_sensor method
    """
    try:
        print("🧪 Distance sensor test request received")
        
        # Validate distance service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Test sensor using your existing method
        if hasattr(distance_service, 'test_sensor'):
            test_result = distance_service.test_sensor()
            
            print(f"🧪 Sensor test result: {'Success' if test_result.get('success') else 'Failed'}")
            
            return jsonify(test_result)
        else:
            # Fallback test using get_current_reading
            reading = distance_service.get_current_reading()
            
            if reading and reading.get('success'):
                return jsonify({
                    'success': True,
                    'message': 'Basic sensor test passed',
                    'current_reading': reading
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Sensor test failed - no valid reading',
                    'reading': reading
                })
        
    except Exception as e:
        error_msg = f'Distance sensor test failed: {str(e)}'
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500
