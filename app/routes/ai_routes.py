"""
AI Analysis Routes for Rebar Detection
Handles requests for AI model inference and analysis
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
                    'display': f"{result['dimensions']['length']}{result['dimensions']['unit']} × {result['dimensions']['width']}{result['dimensions']['unit']} × {result['dimensions']['height']}{result['dimensions']['unit']}"
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
                    'analyzed': f"/static/captured_images/{os.path.basename(result['analyzed_image_path'])}"
                },
                'metadata': {
                    'processing_time': '2.3s',  # Could be actual timing
                    'model_confidence': 'High',
                    'placeholder_mode': result.get('placeholder', False)
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
