# Improvements for app/routes/camera_routes.py

from flask import Blueprint, Response, jsonify, request
import threading
import cv2
from app.utils.config import config

# Create a Blueprint for camera routes
camera_bp = Blueprint('camera', __name__)

# This will be injected when the blueprint is registered
camera_manager = None
image_service = None

def init_camera_routes(cam_manager, img_service):
    """Initialize the camera routes with service dependencies"""
    global camera_manager, image_service
    camera_manager = cam_manager
    image_service = img_service
    print("Camera routes initialized with services")

def _validate_services():
    """Helper function to validate service availability"""
    if not camera_manager:
        return jsonify({
            'success': False,
            'error': 'Camera service not available'
        }), 503
    
    if not image_service:
        return jsonify({
            'success': False,
            'error': 'Image service not available'
        }), 503
    
    return None

@camera_bp.route('/video_feed')
def video_feed():
    """Stream video feed from camera with service validation"""
    # Check if camera manager is available
    if not camera_manager:
        print("Video feed requested but camera manager not available")
        return "Camera service not available", 503
    
    # Check camera status
    status = camera_manager.get_status()
    if not status['is_running']:
        print("Video feed requested but camera not running")
        return "Camera not running", 503
    
    def generate_frames():
        frame_count = 0
        error_count = 0
        max_errors = 10
        
        while True:
            try:
                if not camera_manager or not camera_manager.is_running:
                    print("Camera manager unavailable or stopped during streaming")
                    break
                
                current_frame = camera_manager.get_current_frame()
                
                if current_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', current_frame, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        error_count = 0  # Reset error count on success
                        frame_count += 1
                        
                        # Log every 100 frames for debugging
                        #if frame_count % 100 == 0:
                            #print(f"Streamed {frame_count} frames")
                    else:
                        error_count += 1
                        print(f"Failed to encode frame {frame_count}")
                else:
                    error_count += 1
                    if error_count % 10 == 0:  # Log every 10 errors
                        print(f"No frame available for streaming (error count: {error_count})")
                
                # If too many consecutive errors, break the stream
                if error_count >= max_errors:
                    print(f"Too many streaming errors ({error_count}), stopping stream")
                    break
                
                threading.Event().wait(1.0 / config.CAMERA_FPS)
                
            except Exception as e:
                print(f"Error in video stream: {e}")
                error_count += 1
                if error_count >= max_errors:
                    break
                threading.Event().wait(0.1)  # Brief pause on error
    
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@camera_bp.route('/capture-current-frame', methods=['POST'])
def capture_current_frame():
    """Capture the current frame from camera with validation"""
    try:
        # Validate services
        validation_error = _validate_services()
        if validation_error:
            return validation_error
        
        # Check camera status
        status = camera_manager.get_status()
        if not status['is_running']:
            return jsonify({
                'success': False,
                'error': 'Camera is not running'
            }), 400
        
        if status['last_error']:
            return jsonify({
                'success': False,
                'error': f'Camera error: {status["last_error"]}'
            }), 400
        
        current_frame = camera_manager.get_current_frame()
        
        if current_frame is not None:
            result = image_service.save_frame(current_frame, 'frame_capture')
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': 'No current frame available'
            }), 400
            
    except Exception as e:
        print(f"Error in capture_current_frame: {e}")
        return jsonify({
            'success': False,
            'error': f'Capture failed: {str(e)}'
        }), 500

@camera_bp.route('/camera-status', methods=['GET'])
def camera_status():
    """Get current camera status - useful for debugging"""
    try:
        if not camera_manager:
            return jsonify({
                'success': False,
                'error': 'Camera service not available'
            }), 503
        
        status = camera_manager.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500