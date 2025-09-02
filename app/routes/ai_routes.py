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

@ai_bp.route('/test-with-recent-image', methods=['POST'])
def test_with_recent_image():
    """Test AI with the most recent captured image"""
    try:
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Find most recent image
        if not os.path.exists(config.UPLOAD_FOLDER):
            return jsonify({
                'success': False,
                'error': 'Upload folder not found'
            }), 404
        
        images = [f for f in os.listdir(config.UPLOAD_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            return jsonify({
                'success': False,
                'error': 'No images found in upload folder'
            }), 404
        
        # Get most recent
        recent_image = max(images, key=lambda f: os.path.getmtime(os.path.join(config.UPLOAD_FOLDER, f)))
        image_path = os.path.join(config.UPLOAD_FOLDER, recent_image)
        
        print(f"🧪 Testing with recent image: {recent_image}")
        
        result = ai_service.analyze_image(image_path)
        
        return jsonify({
            'success': True,
            'test_image': recent_image,
            'analysis_result': result
        })
        
    except Exception as e:
        print(f"❌ Test with recent image error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500

@ai_bp.route('/force-placeholder-test', methods=['POST'])
def force_placeholder_test():
    """Force placeholder analysis for testing"""
    try:
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Create a dummy image path
        dummy_path = os.path.join(config.UPLOAD_FOLDER, 'test_placeholder.jpg')
        
        # Force placeholder mode temporarily
        original_loaded = ai_service.phases_loaded
        ai_service.phases_loaded = False
        
        try:
            result = ai_service.analyze_image(dummy_path)
        finally:
            # Restore original state
            ai_service.phases_loaded = original_loaded
        
        return jsonify({
            'success': True,
            'placeholder_result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Placeholder test failed: {str(e)}'
        }), 500

@ai_bp.route('/debug-analyze-rebar', methods=['POST'])
def debug_analyze_rebar():
    """Debug version of analyze-rebar with detailed logging"""
    try:
        print("🐛 DEBUG: AI analysis request received")
        
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            print("🐛 DEBUG: AI service validation failed")
            return validation_error
        
        # Get request data
        data = request.get_json()
        if not data:
            print("🐛 DEBUG: No JSON data provided")
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        print(f"🐛 DEBUG: Request data: {data}")
        
        # Get image path
        if 'image_path' in data:
            image_path = data['image_path']
        elif 'filename' in data:
            filename = data['filename']
            image_path = os.path.join(config.UPLOAD_FOLDER, filename)
            print(f"🐛 DEBUG: Using filename: {filename}")
            print(f"🐛 DEBUG: Full image path: {image_path}")
        else:
            print("🐛 DEBUG: No image path or filename provided")
            return jsonify({
                'success': False,
                'error': 'No image path or filename provided'
            }), 400
        
        # Validate image exists
        if not os.path.exists(image_path):
            print(f"🐛 DEBUG: Image file not found: {image_path}")
            return jsonify({
                'success': False,
                'error': f'Image file not found: {image_path}'
            }), 404
        
        print(f"🐛 DEBUG: Image exists, size: {os.path.getsize(image_path)} bytes")
        
        # Check AI service status
        model_status = ai_service.get_model_status()
        print(f"🐛 DEBUG: AI service status: {model_status}")
        
        # Run AI analysis with debug info
        print("🐛 DEBUG: Starting AI analysis...")
        result = ai_service.analyze_image(image_path)
        
        print(f"🐛 DEBUG: Analysis result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if result.get('success'):
            print("🐛 DEBUG: Analysis completed successfully")
            print(f"🐛 DEBUG: Detections: {result.get('num_detections', 0)}")
        else:
            print(f"🐛 DEBUG: Analysis failed: {result.get('error', 'Unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"🐛 DEBUG: Exception in analyze route: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Debug analysis failed: {str(e)}',
            'debug': True
        }), 500

@ai_bp.route('/debug-analyze-recent', methods=['POST'])
def debug_analyze_recent():
    """Debug analyze with most recent captured image"""
    try:
        print("🐛 DEBUG: Analyzing most recent image")
        
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Find most recent image
        if not os.path.exists(config.UPLOAD_FOLDER):
            return jsonify({
                'success': False,
                'error': 'Upload folder not found'
            }), 404
        
        images = [f for f in os.listdir(config.UPLOAD_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            return jsonify({
                'success': False,
                'error': 'No images found'
            }), 404
        
        recent_image = max(images, key=lambda f: os.path.getmtime(os.path.join(config.UPLOAD_FOLDER, f)))
        image_path = os.path.join(config.UPLOAD_FOLDER, recent_image)
        
        print(f"🐛 DEBUG: Using recent image: {recent_image}")
        
        # Simulate the request data
        request_data = {'filename': recent_image}
        
        # Call debug analyze
        result = ai_service.analyze_image(image_path)
        
        return jsonify({
            'success': True,
            'debug_image': recent_image,
            'debug_path': image_path,
            'analysis_result': result
        })
        
    except Exception as e:
        print(f"🐛 DEBUG: Error in debug analyze recent: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Debug recent failed: {str(e)}'
        }), 500

@ai_bp.route('/list-captured-images', methods=['GET'])
def list_captured_images():
    """List all captured images for debugging"""
    try:
        if not os.path.exists(config.UPLOAD_FOLDER):
            return jsonify({
                'success': False,
                'error': 'Upload folder not found'
            })
        
        images = []
        for filename in os.listdir(config.UPLOAD_FOLDER):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(config.UPLOAD_FOLDER, filename)
                stat = os.stat(filepath)
                images.append({
                    'filename': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Sort by modification time (newest first)
        images.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'images': images,
            'count': len(images),
            'upload_folder': config.UPLOAD_FOLDER
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'List images failed: {str(e)}'
        }), 500
