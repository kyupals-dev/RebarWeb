"""
Sensor Routes for Distance Sensor Integration
Provides API endpoints for real-time distance readings
"""

from flask import Blueprint, jsonify, request

# Create a Blueprint for sensor routes
sensor_bp = Blueprint('sensors', __name__)

# This will be injected when the blueprint is registered
distance_service = None

def init_sensor_routes(dist_service):
    """Initialize the sensor routes with service dependencies"""
    global distance_service
    distance_service = dist_service
    print("Sensor routes initialized with distance service")

def _validate_distance_service():
    """Helper function to validate distance service availability"""
    if not distance_service:
        return jsonify({
            'success': False,
            'error': 'Distance service not available'
        }), 503
    return None

@sensor_bp.route('/distance-reading', methods=['GET'])
def get_distance_reading():
    """Get current distance reading with optimal positioning status"""
    try:
        # Validate service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Get current reading
        reading = distance_service.get_current_reading()
        
        return jsonify(reading)
        
    except Exception as e:
        print(f"❌ Error getting distance reading: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Distance reading failed: {str(e)}',
            'status': 'error',
            'status_text': 'ERROR',
            'status_color': 'red'
        }), 500

@sensor_bp.route('/distance-status', methods=['GET'])
def get_distance_status():
    """Get detailed distance sensor status"""
    try:
        # Validate service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Get sensor status
        status = distance_service.get_sensor_status()
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        print(f"❌ Error getting distance status: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Status check failed: {str(e)}'
        }), 500

@sensor_bp.route('/test-distance-sensor', methods=['POST'])
def test_distance_sensor():
    """Test distance sensor functionality"""
    try:
        # Validate service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Run sensor test
        test_result = distance_service.test_sensor()
        
        if test_result['success']:
            print("✅ Distance sensor test passed")
        else:
            print(f"❌ Distance sensor test failed: {test_result.get('error')}")
        
        return jsonify(test_result)
        
    except Exception as e:
        print(f"❌ Distance sensor test error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500

@sensor_bp.route('/start-distance-monitoring', methods=['POST'])
def start_distance_monitoring():
    """Start distance monitoring service"""
    try:
        # Validate service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Start monitoring
        success = distance_service.start_monitoring()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Distance monitoring started',
                'status': 'monitoring'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start distance monitoring'
            }), 500
        
    except Exception as e:
        print(f"❌ Error starting distance monitoring: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Start monitoring failed: {str(e)}'
        }), 500

@sensor_bp.route('/stop-distance-monitoring', methods=['POST'])
def stop_distance_monitoring():
    """Stop distance monitoring service"""
    try:
        # Validate service
        validation_error = _validate_distance_service()
        if validation_error:
            return validation_error
        
        # Stop monitoring
        distance_service.stop_monitoring()
        
        return jsonify({
            'success': True,
            'message': 'Distance monitoring stopped',
            'status': 'stopped'
        })
        
    except Exception as e:
        print(f"❌ Error stopping distance monitoring: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Stop monitoring failed: {str(e)}'
        }), 500

@sensor_bp.route('/distance-health-check', methods=['GET'])
def distance_health_check():
    """Simple health check for distance service"""
    try:
        if not distance_service:
            return jsonify({
                'success': False,
                'status': 'Distance service not initialized'
            }), 503
        
        # Get current reading to test functionality
        reading = distance_service.get_current_reading()
        
        return jsonify({
            'success': True,
            'status': 'Distance service healthy',
            'is_running': distance_service.is_running,
            'sensor_available': distance_service.sensor_available,
            'simulation_mode': distance_service.simulation_mode,
            'current_reading': reading if reading['success'] else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': f'Health check failed: {str(e)}'
        }), 500
