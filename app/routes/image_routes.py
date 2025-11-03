# Updated app/routes/image_routes.py with metadata support and debug routes

from flask import Blueprint, jsonify, request
import os
from datetime import datetime

# Create a Blueprint for image management routes
image_bp = Blueprint('images', __name__)

# This will be injected when the blueprint is registered
image_service = None

def init_image_routes(img_service):
    """Initialize the image routes with service dependencies"""
    global image_service
    image_service = img_service
    print("Image routes initialized with service")

def _validate_image_service():
    """Helper function to validate image service availability"""
    if not image_service:
        return jsonify({
            'success': False,
            'error': 'Image service not available'
        }), 503
    return None

@image_bp.route('/upload-image', methods=['POST'])
def upload_image():
    """Upload image from web interface with validation"""
    try:
        print("Upload image route accessed")
        
        # Validate service first
        validation_error = _validate_image_service()
        if validation_error:
            return validation_error
        
        # Validate request data
        data = request.get_json()
        print(f"Received data: {data is not None}")
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False, 
                'error': 'No image data provided'
            }), 400
        
        image_data = data['image']
        
        # Basic validation of image data
        if not image_data or not isinstance(image_data, str):
            return jsonify({
                'success': False,
                'error': 'Invalid image data format'
            }), 400
        
        # Check if it looks like base64 data
        if not image_data.startswith('data:image/') and ',' not in image_data:
            return jsonify({
                'success': False,
                'error': 'Image data does not appear to be valid base64'
            }), 400
        
        result = image_service.save_base64_image(image_data, 'web_capture')
        
        if result['success']:
            print(f"Image uploaded successfully: {result.get('filename', 'unknown')}")
        else:
            print(f"Image upload failed: {result.get('error', 'unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in upload_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@image_bp.route('/get-images', methods=['GET'])
def get_images():
    """UPDATED: Get list of all analyzed images with metadata (FIXED for simplified_analysis_)"""
    try:
        # Validate service
        validation_error = _validate_image_service()
        if validation_error:
            return validation_error
            
        result = image_service.get_all_images()
        
        if result['success']:
            analyzed_count = len([img for img in result.get('images', []) if img.get('is_analyzed', False)])
            print(f"Retrieved {len(result.get('images', []))} total images ({analyzed_count} analyzed)")
            print(f"FIXED: Now includes simplified_analysis_ files in gallery")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get_images: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to retrieve images: {str(e)}'
        }), 500

@image_bp.route('/get-image-metadata/<filename>', methods=['GET'])
def get_image_metadata(filename):
    """UPDATED: Get specific image metadata (for gallery modal)"""
    try:
        # Validate service
        validation_error = _validate_image_service()
        if validation_error:
            return validation_error
        
        # Basic filename validation
        if not filename or '..' in filename or '/' in filename:
            return jsonify({
                'success': False,
                'error': 'Invalid filename'
            }), 400
            
        result = image_service.get_image_metadata(filename)
        
        if result['success']:
            print(f"Retrieved metadata for: {filename}")
        else:
            print(f"Failed to get metadata for {filename}: {result.get('error', 'unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in get_image_metadata: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Metadata retrieval failed: {str(e)}'
        }), 500

@image_bp.route('/delete-image/<filename>', methods=['DELETE'])
def delete_image(filename):
    """UPDATED: Delete image and its metadata"""
    try:
        # Validate service
        validation_error = _validate_image_service()
        if validation_error:
            return validation_error
        
        # Basic filename validation
        if not filename or '..' in filename or '/' in filename:
            return jsonify({
                'success': False,
                'error': 'Invalid filename'
            }), 400
            
        result = image_service.delete_image(filename)
        
        if result['success']:
            print(f"Image and metadata deleted successfully: {filename}")
        else:
            print(f"Failed to delete image {filename}: {result.get('error', 'unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in delete_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Delete failed: {str(e)}'
        }), 500

@image_bp.route('/clear-all-images', methods=['DELETE'])
def clear_all_images():
    """UPDATED: Delete all images and metadata"""
    try:
        # Validate service
        validation_error = _validate_image_service()
        if validation_error:
            return validation_error
            
        result = image_service.clear_all_images()
        
        if result['success']:
            print("All images and metadata cleared successfully")
        else:
            print(f"Failed to clear images: {result.get('error', 'unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in clear_all_images: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Clear failed: {str(e)}'
        }), 500

@image_bp.route('/image-service-status', methods=['GET'])
def image_service_status():
    """Get image service status - useful for debugging"""
    try:
        if not image_service:
            return jsonify({
                'success': False,
                'error': 'Image service not available'
            }), 503
        
        # Get basic service info
        return jsonify({
            'success': True,
            'status': {
                'service_available': True,
                'upload_folder': image_service.upload_folder,
                'allowed_extensions': list(image_service.allowed_extensions),
                'storage_mode': 'analyzed_images_only',
                'supports_simplified_analysis': True
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# NEW DEBUG ROUTES
@image_bp.route('/debug-images', methods=['GET'])
def debug_images():
    """Debug route to see all images in upload folder"""
    try:
        if not image_service:
            return jsonify({'error': 'Image service not available'}), 503
        
        upload_folder = image_service.upload_folder
        debug_info = {
            'upload_folder': upload_folder,
            'folder_exists': os.path.exists(upload_folder),
            'all_files': [],
            'analyzed_files': [],
            'original_files': []
        }
        
        if os.path.exists(upload_folder):
            for filename in os.listdir(upload_folder):
                if image_service._is_allowed_file(filename):
                    filepath = os.path.join(upload_folder, filename)
                    file_stats = os.stat(filepath)
                    
                    file_info = {
                        'filename': filename,
                        'size': file_stats.st_size,
                        'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        'is_analyzed': image_service._is_analyzed_image(filename)
                    }
                    
                    debug_info['all_files'].append(file_info)
                    
                    if image_service._is_analyzed_image(filename):
                        debug_info['analyzed_files'].append(file_info)
                    else:
                        debug_info['original_files'].append(file_info)
        
        print(f"🐛 Debug Images:")
        print(f"   Total files: {len(debug_info['all_files'])}")
        print(f"   Analyzed: {len(debug_info['analyzed_files'])}")
        print(f"   Original: {len(debug_info['original_files'])}")
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@image_bp.route('/cleanup-originals', methods=['POST'])
def cleanup_originals():
    """Remove original images, keep only analyzed ones"""
    try:
        if not image_service:
            return jsonify({'error': 'Image service not available'}), 503
        
        result = image_service.cleanup_original_images()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@image_bp.route('/storage-stats', methods=['GET'])
def storage_stats():
    """Get detailed storage statistics"""
    try:
        if not image_service:
            return jsonify({'error': 'Image service not available'}), 503
        
        result = image_service.get_storage_stats()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@image_bp.route('/all-images-debug', methods=['GET'])
def all_images_debug():
    """Get ALL images including originals (for debugging)"""
    try:
        if not image_service:
            return jsonify({'error': 'Image service not available'}), 503
        
        result = image_service.get_all_images_including_originals()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
