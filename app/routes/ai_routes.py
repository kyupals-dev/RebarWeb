"""
AI Analysis Routes for Rebar Detection with Enhanced Pipeline Support
UPDATED: Added gallery integration and 4-step pipeline visualization
FIXED: Only 2 classes - front_horizontal, front_vertical
"""

from flask import Blueprint, jsonify, request
import os
import json
from datetime import datetime
from app.utils.config import config

# Create a Blueprint for AI analysis routes
ai_bp = Blueprint('ai', __name__)

# This will be injected when the blueprint is registered
ai_service = None
camera_manager = None

def init_ai_routes(ai_svc, cam_manager=None):
    """Initialize the AI routes with service dependencies"""
    global ai_service, camera_manager
    ai_service = ai_svc
    camera_manager = cam_manager
    print("✅ AI routes initialized with AI service and camera manager")

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
    Analyze current camera frame for rebar detection using pipeline mode
    UPDATED: Enhanced with 4-step pipeline visualization and gallery integration
    """
    try:
        print("🔍 AI pipeline analysis request received")
        
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Validate camera service for direct frame access
        validation_error = _validate_camera_service()
        if validation_error:
            return validation_error
        
        # Get request data for optional parameters
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
        
        # PRIMARY METHOD: Get current frame directly from camera
        print("📸 Attempting to get current frame from camera...")
        current_frame = camera_manager.get_current_frame()
        
        if current_frame is not None:
            print(f"✅ Using direct camera frame: {current_frame.shape}")
            print("   📝 NOTE: Pipeline analysis with 4-step visualization")
            
            # Analyze frame with pipeline method
            result = ai_service.analyze_image(image_data=current_frame)
            
        elif fallback_image_path and os.path.exists(fallback_image_path):
            print(f"🔄 Fallback: Using existing image file: {fallback_image_path}")
            print("   📝 NOTE: Pipeline analysis with 4-step visualization")
            
            # Fallback to existing image file
            result = ai_service.analyze_image(image_path=fallback_image_path)
            
        else:
            error_msg = "No current camera frame available"
            if fallback_image_path:
                error_msg += f" and fallback image not found: {fallback_image_path}"
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        if result['success']:
            print("✅ Pipeline analysis completed successfully")
            
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
                    'error': 'Analysis succeeded but analyzed image file not found'
                }), 500
            
            # Add capture timestamp
            result['capture_timestamp'] = data.get('capture_timestamp', datetime.now().isoformat())
            
            # Log successful analysis
            filename = os.path.basename(result['analyzed_image_path'])
            print(f"📊 Pipeline analysis results:")
            print(f"   📁 Analyzed image: {filename}")
            print(f"   📐 Dimensions: {result.get('dimensions', {}).get('display', 'N/A')}")
            print(f"   🧮 Mixture: {result.get('cement_mixture', {}).get('ratio_string', 'N/A')}")
            print(f"   🔍 Detections: {result.get('num_detections', 0)}")
            
            return jsonify(result)
            
        else:
            print(f"❌ Pipeline analysis failed: {result.get('error', 'Unknown error')}")
            return jsonify(result), 400
        
    except Exception as e:
        error_msg = f'Analysis request failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@ai_bp.route('/save-to-gallery', methods=['POST'])
def save_to_gallery():
    """
    Save analysis results to gallery with metadata
    NEW: Handles 4-step pipeline images and detailed metadata
    """
    try:
        print("💾 Gallery save request received")
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        analyzed_image_path = data.get('analyzed_image_path')
        metadata = data.get('metadata', {})
        step_images = data.get('step_images', {})
        
        if not analyzed_image_path:
            return jsonify({
                'success': False,
                'error': 'No analyzed image path provided'
            }), 400
        
        # Verify analyzed image exists
        if not os.path.exists(analyzed_image_path):
            return jsonify({
                'success': False,
                'error': 'Analyzed image file not found'
            }), 404
        
        # Create gallery metadata file
        try:
            # Generate gallery entry
            gallery_entry = {
                'analyzed_image_path': analyzed_image_path,
                'analyzed_image_filename': os.path.basename(analyzed_image_path),
                'save_timestamp': datetime.now().isoformat(),
                'analysis_metadata': metadata,
                'step_images': step_images,
                'pipeline_type': 'quadrant_intersection',
                'dimensions': metadata.get('dimensions', {}),
                'cement_mixture': metadata.get('cement_mixture', {}),
                'detection_count': metadata.get('detections', 0),
                'model_type': metadata.get('model_type', 'unknown')
            }
            
            # Save metadata to gallery folder
            gallery_metadata_dir = os.path.join(config.UPLOAD_FOLDER, 'gallery_metadata')
            os.makedirs(gallery_metadata_dir, exist_ok=True)
            
            # Create metadata filename based on analyzed image
            base_name = os.path.splitext(os.path.basename(analyzed_image_path))[0]
            metadata_filename = f"{base_name}_metadata.json"
            metadata_path = os.path.join(gallery_metadata_dir, metadata_filename)
            
            # Write metadata file
            with open(metadata_path, 'w') as f:
                json.dump(gallery_entry, f, indent=2)
            
            print(f"✅ Gallery entry saved:")
            print(f"   📁 Image: {gallery_entry['analyzed_image_filename']}")
            print(f"   📋 Metadata: {metadata_filename}")
            print(f"   📐 Dimensions: {metadata.get('dimensions', {}).get('display', 'N/A')}")
            print(f"   🧮 Mixture: {metadata.get('cement_mixture', {}).get('ratio_string', 'N/A')}")
            
            return jsonify({
                'success': True,
                'gallery_entry': gallery_entry,
                'metadata_path': metadata_path
            })
            
        except Exception as e:
            print(f"❌ Error saving gallery metadata: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to save gallery metadata: {str(e)}'
            }), 500
        
    except Exception as e:
        error_msg = f'Gallery save failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@ai_bp.route('/test-ai-model', methods=['POST'])
def test_ai_model():
    """
    Test AI model functionality for debugging
    UPDATED: Tests pipeline mode with 2 classes
    """
    try:
        print("🧪 AI model test request received")
        
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Get optional test parameters
        data = request.get_json() or {}
        test_image_path = data.get('test_image_path')
        use_camera_frame = data.get('use_camera_frame', True)
        
        print("🧪 Running AI model test (pipeline mode, 2 classes)...")
        
        test_result = None
        
        # Try camera frame first if available and requested
        if use_camera_frame and camera_manager:
            current_frame = camera_manager.get_current_frame()
            if current_frame is not None:
                print("📸 Testing with current camera frame")
                test_result = ai_service.analyze_image(image_data=current_frame)
                test_result['test_source'] = 'camera_frame'
        
        # Fallback to test image path
        if not test_result and test_image_path:
            print(f"📁 Testing with image file: {test_image_path}")
            test_result = ai_service.analyze_image(image_path=test_image_path)
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
            print(f"✅ AI model test successful! Model type: {model_type}")
            print(f"   📊 Detections: {test_result.get('num_detections', 0)}")
            print(f"   📐 Dimensions: {test_result.get('dimensions', {}).get('display', 'N/A')}")
            print(f"   🧮 Mixture: {test_result.get('cement_mixture', {}).get('ratio_string', 'N/A')}")
            
            # Add test metadata
            test_result['test_timestamp'] = datetime.now().isoformat()
            test_result['test_mode'] = 'pipeline'
            
            return jsonify(test_result)
        else:
            print(f"❌ AI model test failed: {test_result.get('error', 'Unknown error')}")
            return jsonify(test_result), 400
        
    except Exception as e:
        error_msg = f'AI model test failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@ai_bp.route('/get-model-status', methods=['GET'])
def get_model_status():
    """
    Get current AI model status and configuration
    UPDATED: Shows pipeline configuration details
    """
    try:
        # Validate AI service
        validation_error = _validate_ai_service()
        if validation_error:
            return validation_error
        
        # Get model status from AI service
        status = ai_service.get_model_status()
        
        # Add additional runtime information
        status.update({
            'camera_available': camera_manager is not None,
            'upload_folder': config.UPLOAD_FOLDER,
            'upload_folder_exists': os.path.exists(config.UPLOAD_FOLDER),
            'pipeline_mode': True,
            'expected_classes': ['front_horizontal', 'front_vertical'],
            'expected_detections': {
                'front_vertical': 2,
                'front_horizontal': 11
            },
            'pipeline_constants': {
                'PX_TO_CM': getattr(ai_service, 'PX_TO_CM', 1/3.54),
                'OFFSET_CM': getattr(ai_service, 'OFFSET_CM', 4.5),
                'MIX_RATIO': getattr(ai_service, 'MIX_RATIO', [1, 2, 4])
            }
        })
        
        print(f"📊 Model status requested:")
        print(f"   Model loaded: {'✅' if status['model_loaded'] else '❌'}")
        print(f"   Detectron2: {'✅' if status['detectron2_available'] else '❌'}")
        print(f"   Classes: {status['class_names']}")
        print(f"   Pipeline mode: {'✅' if status['pipeline_mode'] else '❌'}")
        
        return jsonify(status)
        
    except Exception as e:
        error_msg = f'Model status request failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@ai_bp.route('/get-gallery-entries', methods=['GET'])
def get_gallery_entries():
    """
    Get all gallery entries with metadata
    NEW: Returns pipeline analysis results for gallery display
    """
    try:
        print("🖼️ Gallery entries request received")
        
        gallery_entries = []
        gallery_metadata_dir = os.path.join(config.UPLOAD_FOLDER, 'gallery_metadata')
        
        # Check if gallery metadata directory exists
        if not os.path.exists(gallery_metadata_dir):
            print("📁 No gallery metadata directory found")
            return jsonify({
                'success': True,
                'entries': [],
                'count': 0
            })
        
        # Load all metadata files
        try:
            metadata_files = [f for f in os.listdir(gallery_metadata_dir) 
                            if f.endswith('_metadata.json')]
            
            for metadata_file in metadata_files:
                metadata_path = os.path.join(gallery_metadata_dir, metadata_file)
                
                try:
                    with open(metadata_path, 'r') as f:
                        entry = json.load(f)
                    
                    # Verify analyzed image still exists
                    if os.path.exists(entry.get('analyzed_image_path', '')):
                        gallery_entries.append(entry)
                    else:
                        print(f"⚠️ Analyzed image missing for {metadata_file}")
                        
                except Exception as e:
                    print(f"⚠️ Error loading metadata {metadata_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            gallery_entries.sort(key=lambda x: x.get('save_timestamp', ''), reverse=True)
            
            print(f"✅ Found {len(gallery_entries)} gallery entries")
            
            return jsonify({
                'success': True,
                'entries': gallery_entries,
                'count': len(gallery_entries)
            })
            
        except Exception as e:
            print(f"❌ Error loading gallery entries: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to load gallery entries: {str(e)}'
            }), 500
        
    except Exception as e:
        error_msg = f'Gallery entries request failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@ai_bp.route('/delete-gallery-entry', methods=['POST'])
def delete_gallery_entry():
    """
    Delete a gallery entry and its associated files
    NEW: Allows cleanup of old analysis results
    """
    try:
        print("🗑️ Gallery entry deletion request received")
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        analyzed_image_filename = data.get('analyzed_image_filename')
        if not analyzed_image_filename:
            return jsonify({
                'success': False,
                'error': 'No analyzed image filename provided'
            }), 400
        
        # Find and delete the analyzed image
        analyzed_image_path = os.path.join(config.UPLOAD_FOLDER, analyzed_image_filename)
        deleted_files = []
        
        if os.path.exists(analyzed_image_path):
            os.remove(analyzed_image_path)
            deleted_files.append(analyzed_image_filename)
            print(f"🗑️ Deleted analyzed image: {analyzed_image_filename}")
        
        # Find and delete metadata file
        gallery_metadata_dir = os.path.join(config.UPLOAD_FOLDER, 'gallery_metadata')
        base_name = os.path.splitext(analyzed_image_filename)[0]
        metadata_filename = f"{base_name}_metadata.json"
        metadata_path = os.path.join(gallery_metadata_dir, metadata_filename)
        
        if os.path.exists(metadata_path):
            # Load metadata to find step images
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Delete step images if they exist
                step_images = metadata.get('step_images', {})
                for step_name, step_path in step_images.items():
                    if os.path.exists(step_path):
                        step_filename = os.path.basename(step_path)
                        os.remove(step_path)
                        deleted_files.append(step_filename)
                        print(f"🗑️ Deleted step image: {step_filename}")
                
            except Exception as e:
                print(f"⚠️ Error processing metadata for deletion: {e}")
            
            # Delete metadata file
            os.remove(metadata_path)
            deleted_files.append(metadata_filename)
            print(f"🗑️ Deleted metadata: {metadata_filename}")
        
        print(f"✅ Gallery entry deletion complete: {len(deleted_files)} files removed")
        
        return jsonify({
            'success': True,
            'deleted_files': deleted_files,
            'count': len(deleted_files)
        })
        
    except Exception as e:
        error_msg = f'Gallery entry deletion failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@ai_bp.route('/cleanup-orphaned-files', methods=['POST'])
def cleanup_orphaned_files():
    """
    Clean up orphaned image files that don't have corresponding metadata
    NEW: Maintenance function for file cleanup
    """
    try:
        print("🧹 Orphaned files cleanup request received")
        
        upload_dir = config.UPLOAD_FOLDER
        gallery_metadata_dir = os.path.join(upload_dir, 'gallery_metadata')
        
        if not os.path.exists(gallery_metadata_dir):
            return jsonify({
                'success': True,
                'message': 'No gallery metadata directory found',
                'orphaned_files': [],
                'count': 0
            })
        
        # Get all metadata files
        metadata_files = [f for f in os.listdir(gallery_metadata_dir) 
                         if f.endswith('_metadata.json')]
        
        # Get all referenced image files from metadata
        referenced_files = set()
        for metadata_file in metadata_files:
            metadata_path = os.path.join(gallery_metadata_dir, metadata_file)
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Add analyzed image
                analyzed_path = metadata.get('analyzed_image_path', '')
                if analyzed_path:
                    referenced_files.add(os.path.basename(analyzed_path))
                
                # Add step images
                step_images = metadata.get('step_images', {})
                for step_path in step_images.values():
                    if step_path:
                        referenced_files.add(os.path.basename(step_path))
                        
            except Exception as e:
                print(f"⚠️ Error reading metadata {metadata_file}: {e}")
                continue
        
        # Find orphaned image files
        orphaned_files = []
        if os.path.exists(upload_dir):
            all_images = [f for f in os.listdir(upload_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png')) and 
                         (f.startswith('analyzed_') or f.startswith('step'))]
            
            for image_file in all_images:
                if image_file not in referenced_files:
                    orphaned_files.append(image_file)
        
        # Get dry_run parameter
        data = request.get_json() or {}
        dry_run = data.get('dry_run', True)
        
        if dry_run:
            print(f"🔍 Found {len(orphaned_files)} orphaned files (dry run)")
            return jsonify({
                'success': True,
                'message': f'Found {len(orphaned_files)} orphaned files (dry run mode)',
                'orphaned_files': orphaned_files,
                'count': len(orphaned_files),
                'dry_run': True
            })
        else:
            # Actually delete orphaned files
            deleted_count = 0
            for orphaned_file in orphaned_files:
                try:
                    file_path = os.path.join(upload_dir, orphaned_file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"🗑️ Deleted orphaned file: {orphaned_file}")
                except Exception as e:
                    print(f"⚠️ Error deleting {orphaned_file}: {e}")
            
            print(f"✅ Cleanup complete: {deleted_count} orphaned files deleted")
            return jsonify({
                'success': True,
                'message': f'Cleanup complete: {deleted_count} files deleted',
                'orphaned_files': orphaned_files,
                'count': deleted_count,
                'dry_run': False
            })
        
    except Exception as e:
        error_msg = f'Orphaned files cleanup failed: {str(e)}'
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500
