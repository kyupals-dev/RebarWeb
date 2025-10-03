"""
AI Analysis Routes for Rebar Detection
FIXED: Proper HTTP 422 response when no rebar detected
MODIFIED: Works with direct camera frame data - only saves analyzed images
"""

from flask import Blueprint, jsonify, request
import os
from datetime import datetime
from app.utils.config import config

# Create a Blueprint for AI analysis routes
ai_bp = Blueprint('ai', __name__)

# This will be injected when the blueprint is registered
ai_service = None
camera_manager = None  # Added camera manager dependency

def init_ai_routes(ai_svc, cam_manager=None):
    """Initialize the AI routes with service dependencies"""
    global ai_service, camera_manager
    ai_service = ai_svc
    camera_manager = cam_manager
    print("AI routes initialized with AI service and camera manager")

def _validate_ai_service():
    """Helper function to validate AI service availability"""
    if not ai_service:
        return jsonify({
            'success': False,
            'error': 'AI service not available'
        }), 503
    return None

def _validate_camera_service():
    """Helper function to validate camera service availability"""
    if not camera_manager:
        return jsonify({
            'success': False,
            'error': 'Camera service not available for frame access'
        }), 503
    return None

@ai_bp.route('/analyze-rebar', methods=['POST'])
def analyze_rebar():
    """
    Analyze current camera frame for rebar detection
    FIXED: Returns HTTP 422 when no rebar detected
    MODIFIED: Works with direct frame data - only saves analyzed image with AI overlays
    """
    try:
        print("🔍 AI analysis request received (analyzed image only mode)")
        
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Validate camera service for direct frame access
        validation_error = _validate_camera_service()
        if validation_error:
            return validation_error
        
        # Get request data for fallback image path (optional)
        try:
            data = request.get_json(silent=True) or {}
        except Exception:
            data = {}
        
        fallback_image_path = None
        
        # Check if fallback image path provided
        if 'filename' in data:
            filename = data['filename']
            fallback_image_path = os.path.join(config.UPLOAD_FOLDER, filename)
            print(f"📁 Fallback image path provided: {filename}")
        elif 'image_path' in data:
            fallback_image_path = data['image_path']
            print(f"📁 Fallback image path provided: {fallback_image_path}")
        
        # Get current frame from camera
        print("📸 Attempting to get current frame from camera...")
        current_frame = camera_manager.get_current_frame()
        
        # Perform AI analysis
        if current_frame is not None:
            print(f"✅ Using direct camera frame: {current_frame.shape}")
            print("   📝 NOTE: No original will be saved - only analyzed image")
            result = ai_service.analyze_image(image_data=current_frame)
        elif fallback_image_path and os.path.exists(fallback_image_path):
            print(f"🔄 Fallback: Using existing image file: {fallback_image_path}")
            print("   📝 NOTE: Only analyzed image will be saved")
            result = ai_service.analyze_image(image_path=fallback_image_path)
        else:
            error_msg = "No current camera frame available"
            if fallback_image_path:
                error_msg += f" and fallback image not found: {fallback_image_path}"
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # CRITICAL FIX: Check for no_rebar_detected error FIRST
        if not result.get('success', False):
            error_type = result.get('error', 'unknown_error')
            
            if error_type == 'no_rebar_detected':
                print("⚠️  NO REBAR DETECTED - Returning HTTP 422")
                print("📝 No images were saved (detection count was 0)")
                return jsonify({
                    'success': False,
                    'error': 'no_rebar_detected',
                    'message': 'No rebar structures detected in the captured image. Please ensure the rebar is clearly visible and try again.',
                    'num_detections': 0
                }), 422  # Unprocessable Entity - expected error condition
            
            # Other errors return 500
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': result.get('error', 'Analysis failed'),
                'message': result.get('message', 'An error occurred during analysis')
            }), 500
        
        # SUCCESS: Analysis completed with detections
        print(f"✅ Analysis successful with {result.get('num_detections', 0)} detections")
        
        # Ensure analyzed image was saved
        if 'analyzed_image_path' not in result or not result['analyzed_image_path']:
            return jsonify({
                'success': False,
                'error': 'Analysis succeeded but no analyzed image was created'
            }), 500
        
        # Verify analyzed image file exists
        if not os.path.exists(result['analyzed_image_path']):
            return jsonify({
                'success': False,
                'error': 'Analyzed image file not found after creation'
            }), 500
        
        analyzed_filename = os.path.basename(result['analyzed_image_path'])
        print(f"📁 Analyzed image saved: {analyzed_filename}")
        
        # Format response for frontend
        response = {
            'success': True,
            'analysis_id': f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'dimensions': {
                'length': result['dimensions']['length'],
                'width': result['dimensions']['width'],
                'height': result['dimensions']['height'],
                'unit': result['dimensions']['unit'],
                'volume': result['dimensions']['volume'],
                'volume_m3': result['dimensions'].get('volume_m3', result['dimensions'].get('volume', 0) / 1000000),
                'display': result['dimensions']['display']
            },
            'cement_mixture': {
                'ratio': result['cement_mixture']['ratio_string'],
                'details': {
                    'cement_bags': result['cement_mixture'].get('cement_bags', 0),
                    'cement_weight_kg': result['cement_mixture'].get('cement_weight_kg', 0),
                    'cement_m3': result['cement_mixture'].get('cement_m3', 0),
                    'sand_m3': result['cement_mixture'].get('sand_volume_m3', 0),
                    'sand_weight_kg': result['cement_mixture'].get('sand_weight_kg', 0),
                    'aggregate_m3': result['cement_mixture'].get('aggregate_volume_m3', 0),
                    'gravel_weight_kg': result['cement_mixture'].get('gravel_weight_kg', 0),
                    'water_liters': result['cement_mixture'].get('water_liters', 0),
                    'total_concrete_m3': result['cement_mixture'].get('total_concrete_volume_m3', 0),
                    'dry_volume_m3': result['cement_mixture'].get('dry_volume_m3', 0)
                }
            },
            'detections': {
                'count': result.get('num_detections', 0),
                'items': result.get('detections', [])
            },
            'images': {
                'analyzed': f"/static/captured_images/{analyzed_filename}"
            },
            'metadata': {
                'processing_time': '2.3s',
                'model_confidence': 'High',
                'placeholder_mode': result.get('placeholder', False),
                'save_mode': 'analyzed_only',
                'source': 'camera_frame' if current_frame is not None else 'fallback_file',
                'model_type': result.get('model_type', 'unknown'),
                'classes_used': 2,
                'pipeline_data': result.get('pipeline_data', {
                    'front_vertical_count': 2,
                    'front_horizontal_count': 11,
                    'intersection_count': result.get('num_detections', 0),
                    'polygon_corners': 4
                })
            }
        }
        
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Analysis route error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': 'internal_error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@ai_bp.route('/ai-model-status', methods=['GET'])
def ai_model_status():
    """Get AI model status and configuration"""
    try:
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        status = ai_service.get_model_status()
        status['save_mode'] = 'analyzed_images_only'
        status['original_images_saved'] = False
        status['classes_used'] = 2
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        print(f"❌ Model status error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Status check failed: {str(e)}'
        }), 500

@ai_bp.route('/test-ai-model', methods=['POST'])
def test_ai_model():
    """Test AI model with current camera frame or sample image"""
    try:
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        data = request.get_json() or {}
        test_image_path = data.get('test_image_path')
        use_camera_frame = data.get('use_camera_frame', True)
        
        print("🧪 Running AI model test (analyzed image only mode, 2 classes)...")
        
        test_result = None
        
        if use_camera_frame and camera_manager:
            current_frame = camera_manager.get_current_frame()
            if current_frame is not None:
                print("📸 Testing with current camera frame")
                test_result = ai_service.analyze_image(image_data=current_frame)
                test_result['test_source'] = 'camera_frame'
        
        if not test_result and test_image_path:
            print(f"📁 Testing with image file: {test_image_path}")
            test_result = ai_service.analyze_image(image_path=test_image_path)
            test_result['test_source'] = 'test_file'
        
        if not test_result:
            captured_dir = config.UPLOAD_FOLDER
            if os.path.exists(captured_dir):
                images = [f for f in os.listdir(captured_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    images.sort(key=lambda x: os.path.getmtime(os.path.join(captured_dir, x)), reverse=True)
                    test_path = os.path.join(captured_dir, images[0])
                    print(f"📁 Testing with most recent image: {images[0]}")
                    test_result = ai_service.analyze_image(image_path=test_path)
                    test_result['test_source'] = 'recent_file'
        
        if not test_result:
            return jsonify({
                'success': False,
                'error': 'No test image available (no camera frame and no files)'
            })
        
        if test_result['success']:
            model_type = test_result.get('model_type', 'unknown')
            print(f"✅ AI model test successful! (Model type: {model_type})")
            
            return jsonify({
                'success': True,
                'test_source': test_result.get('test_source', 'unknown'),
                'detections_found': test_result.get('num_detections', 0),
                'model_type': model_type,
                'analyzed_image_saved': test_result.get('analyzed_image_path'),
                'save_mode': 'analyzed_only',
                'classes_used': 2,
                'dimensions': test_result.get('dimensions', {}),
                'test_result': test_result
            })
        else:
            print(f"❌ AI model test failed: {test_result.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': test_result.get('error', 'Test failed'),
                'test_source': test_result.get('test_source', 'unknown')
            })
        
    except Exception as e:
        print(f"❌ Model test error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500

@ai_bp.route('/ai-health-check', methods=['GET'])
def ai_health_check():
    """Simple health check for AI service"""
    try:
        if not ai_service:
            return jsonify({
                'success': False,
                'status': 'AI service not initialized'
            }), 503
        
        camera_available = camera_manager is not None
        
        return jsonify({
            'success': True,
            'status': 'AI service healthy',
            'model_loaded': ai_service.model_loaded,
            'camera_service_available': camera_available,
            'save_mode': 'analyzed_images_only',
            'classes_used': 2,
            'timestamp': str(datetime.now())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': f'Health check failed: {str(e)}'
        }), 500
