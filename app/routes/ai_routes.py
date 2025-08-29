"""
AI Analysis Routes for Rebar Detection - Minimal Working Version
"""

from flask import Blueprint, jsonify, request
import os
from datetime import datetime
from app.utils.config import config

ai_bp = Blueprint('ai', __name__)
ai_service = None

def init_ai_routes(ai_svc):
    global ai_service
    ai_service = ai_svc
    print("AI routes initialized with service")

def _validate_ai_service():
    if not ai_service:
        return jsonify({'success': False, 'error': 'AI service not available'}), 503
    return None

@ai_bp.route('/analyze-rebar', methods=['POST'])
def analyze_rebar():
    """Analyze captured image with retry mechanism"""
    try:
        print("🔍 AI analysis request received with retry mechanism")
        
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        if 'image_path' in data:
            image_path = data['image_path']
        elif 'filename' in data:
            filename = data['filename']
            image_path = os.path.join(config.UPLOAD_FOLDER, filename)
        else:
            return jsonify({'success': False, 'error': 'No image path provided'}), 400
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        
        print(f"📸 Analyzing with retry: {os.path.basename(image_path)}")
        
        # Use retry mechanism
        result = ai_service.analyze_image_with_retries(image_path, max_retries=3)
        
        if result['success']:
            print("✅ Analysis completed with auto-save")
            
            response = {
                'success': True,
                'dimensions': {
                    'length': result['dimensions']['length'],
                    'width': result['dimensions']['width'],
                    'height': result['dimensions']['height'],
                    'unit': result['dimensions']['unit'],
                    'display': result['dimensions']['display']
                },
                'cement_mixture': {
                    'ratio': result['cement_mixture']['ratio_string']
                },
                'detections': {
                    'count': result.get('num_detections', 0)
                },
                'images': {
                    'original': f"/static/captured_images/{os.path.basename(image_path)}",
                    'analyzed': f"/static/captured_images/{os.path.basename(result['analyzed_image_path'])}"
                },
                'metadata': {
                    'auto_saved': True,
                    'gallery_ready': True
                }
            }
            return jsonify(response)
        else:
            if result.get('no_detection', False):
                return jsonify({
                    'success': False,
                    'error': 'no_rebar_detected',
                    'message': 'No rebar detected after multiple attempts'
                }), 422
            else:
                return jsonify({
                    'success': False,
                    'error': 'analysis_failed',
                    'message': result.get('error', 'Analysis failed')
                }), 500
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@ai_bp.route('/ai-model-status', methods=['GET'])
def ai_model_status():
    """Get AI model status"""
    try:
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        status = ai_service.get_model_status()
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ai_bp.route('/ai-health-check', methods=['GET'])
def ai_health_check():
    """Health check for AI service"""
    try:
        if not ai_service:
            return jsonify({'success': False, 'status': 'AI service not initialized'}), 503
        
        return jsonify({
            'success': True,
            'status': 'AI service healthy',
            'model_loaded': ai_service.model_loaded
        })
    except Exception as e:
        return jsonify({'success': False, 'status': f'Health check failed: {str(e)}'}), 500
