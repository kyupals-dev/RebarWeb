# app/routes/camera_routes.py
# Complete camera routes with MJPEG streaming optimization and all endpoints

from flask import Blueprint, Response, jsonify, request
import threading
import cv2
import time
from app.utils.config import config

camera_bp = Blueprint('camera', __name__)

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
    """Stream video feed using MJPEG with optimizations"""
    if not camera_manager:
        print("Video feed requested but camera manager not available")
        return "Camera service not available", 503
    
    status = camera_manager.get_status()
    if not status['is_running']:
        print("Video feed requested but camera not running")
        return "Camera not running", 503
    
    def generate_frames():
        """Generate frames with adaptive quality and frame skipping"""
        frame_count = 0
        error_count = 0
        max_errors = 10
        last_frame_time = time.time()
        target_interval = 1.0 / config.CAMERA_FPS  # Target time between frames
        
        # Adaptive JPEG quality based on network conditions
        jpeg_quality = 85
        consecutive_slow_frames = 0
        
        while True:
            try:
                if not camera_manager or not camera_manager.is_running:
                    print("Camera manager unavailable or stopped during streaming")
                    break
                
                # Frame rate control - only send frame if enough time has passed
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                if elapsed < target_interval:
                    # Sleep for remaining time to maintain target FPS
                    time.sleep(target_interval - elapsed)
                    current_time = time.time()
                
                current_frame = camera_manager.get_current_frame()
                
                if current_frame is not None:
                    # Adaptive quality: reduce quality if encoding is slow
                    if consecutive_slow_frames > 3:
                        jpeg_quality = max(70, jpeg_quality - 5)
                        consecutive_slow_frames = 0
                        print(f"Reducing JPEG quality to {jpeg_quality} for better performance")
                    
                    # Encode frame with current quality setting
                    encode_start = time.time()
                    ret, buffer = cv2.imencode('.jpg', current_frame, 
                                             [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    encode_time = time.time() - encode_start
                    
                    if ret:
                        frame_bytes = buffer.tobytes()
                        
                        # Send frame in MJPEG format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' 
                               + frame_bytes + b'\r\n')
                        
                        error_count = 0
                        frame_count += 1
                        last_frame_time = current_time
                        
                        # Monitor encoding performance
                        if encode_time > target_interval:
                            consecutive_slow_frames += 1
                        else:
                            consecutive_slow_frames = max(0, consecutive_slow_frames - 1)
                        
                        # Periodic logging (every 300 frames = ~10 seconds at 30fps)
                        if frame_count % 300 == 0:
                            print(f"📹 Streamed {frame_count} frames (Quality: {jpeg_quality})")
                    else:
                        error_count += 1
                        print(f"Failed to encode frame {frame_count}")
                else:
                    error_count += 1
                    if error_count % 50 == 0:
                        print(f"No frame available (error count: {error_count})")
                
                if error_count >= max_errors:
                    print(f"Too many errors ({error_count}), stopping stream")
                    break
                
            except GeneratorExit:
                # Client disconnected - clean exit
                print(f"Client disconnected from video feed after {frame_count} frames")
                break
            except Exception as e:
                print(f"Error in video stream: {e}")
                error_count += 1
                if error_count >= max_errors:
                    break
                time.sleep(0.1)
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Connection': 'keep-alive'
        }
    )

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

@camera_bp.route('/start-camera', methods=['POST'])
def start_camera():
    """Start camera with validation"""
    try:
        validation_error = _validate_services()
        if validation_error:
            return validation_error
        
        result = camera_manager.start_camera()
        return jsonify(result)
        
    except Exception as e:
        print(f"Error starting camera: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to start camera: {str(e)}'
        }), 500

@camera_bp.route('/stop-camera', methods=['POST'])
def stop_camera():
    """Stop camera with validation"""
    try:
        validation_error = _validate_services()
        if validation_error:
            return validation_error
        
        result = camera_manager.stop_camera()
        return jsonify(result)
        
    except Exception as e:
        print(f"Error stopping camera: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to stop camera: {str(e)}'
        }), 500

@camera_bp.route('/restart-camera', methods=['POST'])
def restart_camera():
    """Restart camera with validation"""
    try:
        validation_error = _validate_services()
        if validation_error:
            return validation_error
        
        # Stop camera
        stop_result = camera_manager.stop_camera()
        if not stop_result['success']:
            return jsonify(stop_result), 500
        
        # Wait a moment
        threading.Event().wait(0.5)
        
        # Start camera
        start_result = camera_manager.start_camera()
        return jsonify(start_result)
        
    except Exception as e:
        print(f"Error restarting camera: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to restart camera: {str(e)}'
        }), 500
