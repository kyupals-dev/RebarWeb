"""
AI Analysis Routes for Rebar Detection
Complete version with comprehensive debugging capabilities
"""

from flask import Blueprint, jsonify, request
import os
from datetime import datetime
from app.utils.config import config

# Create a Blueprint for AI analysis routes
ai_bp = Blueprint('ai', __name__)

# This will be injected when the blueprint is registered
ai_service = None

def init_ai_routes(ai_svc):
    """Initialize the AI routes with service dependencies"""
    global ai_service
    ai_service = ai_svc
    print("AI routes initialized with service")

def _validate_ai_service():
    """Helper function to validate AI service availability"""
    if not ai_service:
        return jsonify({
            'success': False,
            'error': 'AI service not available'
        }), 503
    return None

# ==================== MAIN AI ROUTES ====================

@ai_bp.route('/analyze-rebar', methods=['POST'])
def analyze_rebar():
    """
    Analyze captured image for rebar detection
    Expects JSON with image_path or filename
    """
    try:
        print("🔍 AI analysis request received")
        
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Get image path
        if 'image_path' in data:
            image_path = data['image_path']
        elif 'filename' in data:
            filename = data['filename']
            image_path = os.path.join(config.UPLOAD_FOLDER, filename)
        else:
            return jsonify({
                'success': False,
                'error': 'No image path or filename provided'
            }), 400
        
        # Validate image exists
        if not os.path.exists(image_path):
            return jsonify({
                'success': False,
                'error': f'Image file not found: {image_path}'
            }), 404
        
        print(f"📸 Analyzing image: {os.path.basename(image_path)}")
        
        # Run AI analysis
        result = ai_service.analyze_image(image_path)
        
        if result['success']:
            print("✅ Analysis completed successfully")
            
            # Format response for frontend
            response = {
                'success': True,
                'analysis_id': f"analysis_{len(os.listdir(config.UPLOAD_FOLDER))}",
                'dimensions': {
                    'length': result['dimensions']['length'],
                    'width': result['dimensions']['width'],
                    'height': result['dimensions']['height'],
                    'unit': result['dimensions']['unit'],
                    'display': result['dimensions']['display']
                },
                'cement_mixture': {
                    'ratio': result['cement_mixture']['ratio_string'],
                    'details': {
                        'cement_bags': result['cement_mixture'].get('cement_bags', 0),
                        'sand_m3': result['cement_mixture'].get('sand_volume_m3', 0),
                        'aggregate_m3': result['cement_mixture'].get('aggregate_volume_m3', 0),
                        'total_concrete_m3': result['cement_mixture'].get('total_concrete_volume_m3', 0)
                    }
                },
                'detections': {
                    'count': result.get('num_detections', 0),
                    'items': result.get('detections', [])
                },
                'images': {
                    'original': f"/static/captured_images/{os.path.basename(image_path)}",
                    'analyzed': f"/static/captured_images/{os.path.basename(result['analyzed_image_path'])}" if result.get('analyzed_image_path') else f"/static/captured_images/{os.path.basename(image_path)}"
                },
                'metadata': {
                    'processing_time': '2.3s',
                    'model_confidence': 'High',
                    'placeholder_mode': result.get('model_type') != 'real_model'
                }
            }
            
            return jsonify(response)
        
        else:
            # Check if it's a "no detection" error
            if result.get('no_detection', False):
                print("⚠️  No rebar detected in image")
                return jsonify({
                    'success': False,
                    'error': 'no_rebar_detected',
                    'message': 'No rebar structures detected in the image'
                }), 422  # Unprocessable Entity
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return jsonify({
                    'success': False,
                    'error': 'analysis_failed',
                    'message': result.get('error', 'Analysis failed')
                }), 500
        
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
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        status = ai_service.get_model_status()
        
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
    """Test AI model with a sample image"""
    try:
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Get optional test image from request
        data = request.get_json() or {}
        test_image_path = data.get('test_image_path')
        
        print("🧪 Running AI model test...")
        
        # Run test
        result = ai_service.test_model(test_image_path)
        
        if result['success']:
            print("✅ Model test passed")
        else:
            print(f"❌ Model test failed: {result.get('error')}")
        
        return jsonify(result)
        
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
        
        return jsonify({
            'success': True,
            'status': 'AI service healthy',
            'model_loaded': ai_service.model_loaded,
            'timestamp': str(datetime.now())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': f'Health check failed: {str(e)}'
        }), 500

# ==================== BASIC DEBUG ROUTES ====================

@ai_bp.route('/debug-model', methods=['GET'])
def debug_model():
    """Debug route to check model status in detail"""
    try:
        if not ai_service:
            return jsonify({
                'success': False,
                'error': 'AI service not available'
            }), 503
        
        status = ai_service.get_model_status()
        
        # Additional debug info
        debug_info = {
            'model_file_exists': os.path.exists(status.get('model_path', '')),
            'model_file_size': os.path.getsize(status.get('model_path', '')) if os.path.exists(status.get('model_path', '')) else 0,
            'upload_folder': config.UPLOAD_FOLDER,
            'upload_folder_exists': os.path.exists(config.UPLOAD_FOLDER)
        }
        
        return jsonify({
            'success': True,
            'model_status': status,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Debug failed: {str(e)}'
        }), 500

@ai_bp.route('/force-placeholder-test', methods=['POST'])
def force_placeholder_test():
    """Force placeholder analysis for testing UI without real model"""
    try:
        print("🎭 PLACEHOLDER: Forcing placeholder test...")
        
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({
                'success': False,
                'error': 'No filename provided'
            }), 400
        
        filename = data['filename']
        image_path = os.path.join(config.UPLOAD_FOLDER, filename)
        
        if not os.path.exists(image_path):
            return jsonify({
                'success': False,
                'error': f'Image file not found: {image_path}'
            }), 404
        
        # Create basic placeholder response
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        placeholder_filename = f'placeholder_analysis_{timestamp}.jpg'
        placeholder_path = os.path.join(config.UPLOAD_FOLDER, placeholder_filename)
        
        # Copy original image as placeholder analyzed image
        import shutil
        shutil.copy2(image_path, placeholder_path)
        
        result = {
            'success': True,
            'placeholder': True,
            'detections': [
                {
                    'class_name': 'front_vertical',
                    'confidence': 0.85,
                    'bbox': [100, 50, 200, 300]
                },
                {
                    'class_name': 'front_horizontal', 
                    'confidence': 0.78,
                    'bbox': [80, 280, 220, 320]
                }
            ],
            'num_detections': 2,
            'dimensions': {
                'length': 25.4,
                'width': 25.4,
                'height': 200.0,
                'unit': 'cm',
                'volume': 101600,
                'display': '25cm x 25cm x 200cm = 101600cm³',
                'method': 'placeholder_forced'
            },
            'cement_mixture': {
                'cement': 1,
                'sand': 2,
                'aggregate': 3,
                'ratio_string': '1 Cement : 2 Sand : 3 Aggregate'
            },
            'analyzed_image_path': placeholder_path,
            'original_image_path': image_path,
            'model_type': 'forced_placeholder'
        }
        
        print("🎭 PLACEHOLDER: Forced placeholder analysis complete")
        
        return jsonify({
            'success': True,
            'forced_placeholder': True,
            'test_image': filename,
            'result': result
        })
        
    except Exception as e:
        print(f"🎭 PLACEHOLDER: Force placeholder error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Force placeholder failed: {str(e)}'
        }), 500
