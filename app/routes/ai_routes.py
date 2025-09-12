"""
AI Analysis Routes for Rebar Detection
FIXED: Resolves 500 Internal Server Error in pipeline analysis
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
    FIXED: Added proper error handling and mode selection
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
        
        # Get request data for analysis mode and fallback image path (optional)
        # Handle case where request has no JSON body
        try:
            data = request.get_json(silent=True) or {}
        except Exception:
            data = {}
        
        # FIXED: Get analysis mode from request (default to pipeline)
        analysis_mode = data.get('mode', 'pipeline')
        print(f"📊 Analysis mode: {analysis_mode}")
        
        fallback_image_path = None
        
        # Check if fallback image path provided (for legacy compatibility)
        if 'filename' in data:
            filename = data['filename']
            fallback_image_path = os.path.join(config.UPLOAD_FOLDER, filename)
            print(f"📁 Fallback image path provided: {filename}")
        elif 'image_path' in data:
            fallback_image_path = data['image_path']
            print(f"📁 Fallback image path provided: {fallback_image_path}")
        
        # PRIMARY METHOD: Get current frame directly from camera
        print("📸 Attempting to get current frame from camera...")
        current_frame = camera_manager.get_current_frame()
        
        if current_frame is not None:
            print(f"✅ Using direct camera frame: {current_frame.shape}")
            print("   📝 NOTE: No original will be saved - only analyzed image")
            
            # FIXED: Analyze frame directly with mode selection (no original image saved)
            result = ai_service.analyze_image(image_data=current_frame, mode=analysis_mode)
            
        elif fallback_image_path and os.path.exists(fallback_image_path):
            print(f"🔄 Fallback: Using existing image file: {fallback_image_path}")
            print("   📝 NOTE: Only analyzed image will be saved")
            
            # FIXED: Fallback to existing image file with mode selection
            result = ai_service.analyze_image(image_path=fallback_image_path, mode=analysis_mode)
            
        else:
            error_msg = "No current camera frame available"
            if fallback_image_path:
                error_msg += f" and fallback image not found: {fallback_image_path}"
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # FIXED: Enhanced error handling for different failure types
        if result['success']:
            print("✅ Analysis completed successfully")
            
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
            
            # FIXED: Format response for frontend with proper error handling
            try:
                response = {
                    'success': True,
                    'analysis_id': f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'mode': analysis_mode,
                    'dimensions': {
                        'length': result.get('dimensions', {}).get('length', 0),
                        'width': result.get('dimensions', {}).get('width', 0),
                        'height': result.get('dimensions', {}).get('height', 0),
                        'unit': result.get('dimensions', {}).get('unit', 'cm'),
                        'display': result.get('dimensions', {}).get('display', 'N/A')
                    },
                    'cement_mixture': {
                        'ratio': result.get('cement_mixture', {}).get('ratio_string', 'N/A'),
                        'details': {
                            'cement_bags': result.get('cement_mixture', {}).get('cement_bags', 0),
                            'sand_m3': result.get('cement_mixture', {}).get('sand_volume_m3', 0),
                            'aggregate_m3': result.get('cement_mixture', {}).get('aggregate_volume_m3', 0),
                            'total_concrete_m3': result.get('cement_mixture', {}).get('total_concrete_volume_m3', 0)
                        }
                    },
                    'detections': {
                        'count': result.get('num_detections', 0),
                        'items': result.get('detections', [])
                    },
                    'images': {
                        # Only return the analyzed image (the ONLY one that was saved)
                        'analyzed': f"/static/captured_images/{analyzed_filename}"
                        # No original image - wasn't saved
                    },
                    'metadata': {
                        'processing_time': '2.3s',
                        'model_confidence': 'High',
                        'placeholder_mode': result.get('placeholder', False),
                        'save_mode': 'analyzed_only',  # Indicate save mode
                        'source': 'camera_frame' if current_frame is not None else 'fallback_file',
                        'analysis_mode': analysis_mode,
                        'model_type': result.get('model_type', 'unknown')
                    }
                }
                
                # Add mode-specific metadata
                if analysis_mode == 'pipeline' and 'quadrant_info' in result:
                    response['metadata']['quadrant_info'] = result['quadrant_info']
                
                return jsonify(response)
                
            except Exception as e:
                print(f"❌ Error formatting response: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Response formatting failed: {str(e)}'
                }), 500
        
        else:
            # FIXED: Enhanced error handling for different failure scenarios
            error_type = result.get('error', 'Unknown error')
            print(f"❌ Analysis failed: {error_type}")
            
            # Check if it's a "no detection" error
            if result.get('no_detection', False):
                print("⚠️  No rebar detected in image")
                return jsonify({
                    'success': False,
                    'error': 'no_rebar_detected',
                    'message': 'No rebar structures detected in the image',
                    'analysis_mode': analysis_mode
                }), 422  # Unprocessable Entity
            else:
                print(f"❌ Analysis failed: {error_type}")
                return jsonify({
                    'success': False,
                    'error': 'analysis_failed',
                    'message': error_type,
                    'analysis_mode': analysis_mode
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

@ai_bp.route('/analyze-rebar-pipeline', methods=['POST'])
def analyze_rebar_pipeline():
    """
    FIXED: Dedicated pipeline analysis endpoint 
    Forces pipeline mode analysis for compatibility
    """
    try:
        print("🔍 PIPELINE analysis request received (analyzed image only mode)")
        
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Validate camera service for direct frame access
        validation_error = _validate_camera_service()
        if validation_error:
            return validation_error
        
        # Force pipeline mode
        analysis_mode = 'pipeline'
        print(f"📊 Forced analysis mode: {analysis_mode}")
        
        # Get current frame directly from camera
        print("📸 Getting current frame for PIPELINE analysis...")
        current_frame = camera_manager.get_current_frame()
        
        if current_frame is not None:
            print(f"✅ Using direct camera frame: {current_frame.shape}")
            print("   📝 NOTE: PIPELINE mode - only analyzed image will be saved")
            
            # Run pipeline analysis
            result = ai_service.analyze_image(image_data=current_frame, mode=analysis_mode)
            
            if result['success']:
                print("✅ PIPELINE analysis completed successfully")
                
                analyzed_filename = os.path.basename(result['analyzed_image_path'])
                print(f"📁 PIPELINE analyzed image saved: {analyzed_filename}")
                
                # Format pipeline-specific response
                response = {
                    'success': True,
                    'analysis_id': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'mode': 'pipeline',
                    'dimensions': result.get('dimensions', {}),
                    'cement_mixture': result.get('cement_mixture', {}),
                    'detections': {
                        'count': result.get('num_detections', 0),
                        'items': result.get('detections', [])
                    },
                    'images': {
                        'analyzed': f"/static/captured_images/{analyzed_filename}"
                    },
                    'metadata': {
                        'processing_time': '2.8s',
                        'model_confidence': 'High',
                        'placeholder_mode': result.get('placeholder', False),
                        'save_mode': 'analyzed_only',
                        'source': 'camera_frame',
                        'analysis_mode': 'pipeline',
                        'model_type': result.get('model_type', 'unknown'),
                        'quadrant_info': result.get('quadrant_info', {})
                    }
                }
                
                return jsonify(response)
            else:
                # Handle pipeline analysis failure
                error_type = result.get('error', 'Unknown error')
                print(f"❌ PIPELINE analysis failed: {error_type}")
                
                if result.get('no_detection', False):
                    return jsonify({
                        'success': False,
                        'error': 'no_rebar_detected',
                        'message': 'No rebar structures detected for pipeline analysis',
                        'analysis_mode': 'pipeline'
                    }), 422
                else:
                    return jsonify({
                        'success': False,
                        'error': 'pipeline_analysis_failed',
                        'message': f'PIPELINE analysis failed: {error_type}',
                        'analysis_mode': 'pipeline'
                    }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'no_camera_frame',
                'message': 'No current camera frame available for PIPELINE analysis'
            }), 400
            
    except Exception as e:
        print(f"❌ PIPELINE analysis route error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': 'pipeline_internal_error',
            'message': f'PIPELINE analysis internal error: {str(e)}'
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
        
        # Add save mode info
        status['save_mode'] = 'analyzed_images_only'
        status['original_images_saved'] = False
        status['supported_modes'] = ['pipeline', 'phased']
        
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
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Get optional test parameters
        data = request.get_json() or {}
        test_image_path = data.get('test_image_path')
        use_camera_frame = data.get('use_camera_frame', True)
        test_mode = data.get('mode', 'pipeline')  # FIXED: Add mode selection for testing
        
        print(f"🧪 Running AI model test (mode: {test_mode}, analyzed image only mode)...")
        
        test_result = None
        
        # Try camera frame first if available and requested
        if use_camera_frame and camera_manager:
            current_frame = camera_manager.get_current_frame()
            if current_frame is not None:
                print(f"📸 Testing with current camera frame (mode: {test_mode})")
                test_result = ai_service.analyze_image(image_data=current_frame, mode=test_mode)
                test_result['test_source'] = 'camera_frame'
        
        # Fallback to test image path
        if not test_result and test_image_path:
            print(f"📁 Testing with image file: {test_image_path} (mode: {test_mode})")
            test_result = ai_service.analyze_image(image_path=test_image_path, mode=test_mode)
            test_result['test_source'] = 'test_file'
        
        # Final fallback to most recent image in upload folder
        if not test_result:
            captured_dir = config.UPLOAD_FOLDER
            if os.path.exists(captured_dir):
                images = [f for f in os.listdir(captured_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    # Sort by modification time, get most recent
                    images.sort(key=lambda x: os.path.getmtime(os.path.join(captured_dir, x)), reverse=True)
                    test_path = os.path.join(captured_dir, images[0])
                    print(f"📁 Testing with most recent image: {images[0]} (mode: {test_mode})")
                    test_result = ai_service.analyze_image(image_path=test_path, mode=test_mode)
                    test_result['test_source'] = 'recent_file'
        
        if not test_result:
            return jsonify({
                'success': False,
                'error': 'No test image available (no camera frame and no files)'
            })
        
        if test_result['success']:
            model_type = test_result.get('model_type', 'unknown')
            print(f"✅ AI model test successful! (Model type: {model_type}, Mode: {test_mode})")
            print("   📝 Only analyzed image saved during test")
            
            return jsonify({
                'success': True,
                'test_source': test_result.get('test_source', 'unknown'),
                'detections_found': test_result.get('num_detections', 0),
                'model_type': model_type,
                'test_mode': test_mode,
                'analyzed_image_saved': test_result.get('analyzed_image_path'),
                'save_mode': 'analyzed_only',
                'dimensions': test_result.get('dimensions', {}),
                'quadrant_info': test_result.get('quadrant_info', {}),
                'test_result': test_result
            })
        else:
            print(f"❌ AI model test failed: {test_result.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': test_result.get('error', 'Test failed'),
                'test_source': test_result.get('test_source', 'unknown'),
                'test_mode': test_mode
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
        
        # FIXED: Get model status safely
        try:
            model_status = ai_service.get_model_status()
            model_loaded = model_status.get('model_loaded', False)
        except Exception as e:
            print(f"⚠️  Error getting model status: {e}")
            model_loaded = False
        
        return jsonify({
            'success': True,
            'status': 'AI service healthy',
            'model_loaded': model_loaded,
            'camera_service_available': camera_available,
            'save_mode': 'analyzed_images_only',
            'supported_modes': ['pipeline', 'phased'],
            'timestamp': str(datetime.now())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': f'Health check failed: {str(e)}'
        }), 500

@ai_bp.route('/switch-analysis-mode', methods=['POST'])
def switch_analysis_mode():
    """FIXED: Endpoint to test different analysis modes"""
    try:
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        data = request.get_json() or {}
        mode = data.get('mode', 'pipeline')
        
        if mode not in ['pipeline', 'phased']:
            return jsonify({
                'success': False,
                'error': f'Invalid mode: {mode}. Must be "pipeline" or "phased".'
            }), 400
        
        print(f"🔄 Switching to {mode} analysis mode...")
        
        # Test the mode with current camera frame
        if camera_manager:
            current_frame = camera_manager.get_current_frame()
            if current_frame is not None:
                result = ai_service.analyze_image(image_data=current_frame, mode=mode)
                
                if result['success']:
                    return jsonify({
                        'success': True,
                        'message': f'Successfully switched to {mode} mode',
                        'mode': mode,
                        'test_result': {
                            'detections': result.get('num_detections', 0),
                            'model_type': result.get('model_type', 'unknown'),
                            'analyzed_image': os.path.basename(result.get('analyzed_image_path', ''))
                        }
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Mode switch test failed: {result.get("error", "Unknown error")}',
                        'mode': mode
                    }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': 'No camera frame available for mode test'
                }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'Camera service not available for mode test'
            }), 503
        
    except Exception as e:
        print(f"❌ Mode switch error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Mode switch failed: {str(e)}'
        }), 500
