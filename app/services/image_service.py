import os
import base64
import cv2
import json
from datetime import datetime
from app.utils.config import config

class ImageService:
    """
    Enhanced Image Service with 4-Step Analysis Metadata Support
    FIXED: Gallery shows only analyzed images with complete pipeline metadata
    """
    
    def __init__(self):
        self.upload_folder = config.UPLOAD_FOLDER
        self.allowed_extensions = config.ALLOWED_EXTENSIONS
        print("📁 Enhanced Image Service initialized (4-step analysis metadata support)")
    
    def _generate_filename(self, prefix='capture'):
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        return f'{prefix}_{timestamp}.jpg'
    
    def _is_allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _is_analyzed_image(self, filename):
        """
        Check if filename indicates an analyzed image with AI overlays
        FIXED: Exclude step images from gallery - they're for modal details only
        """
        
        if self._is_step_image(filename):
            return False
            
        analyzed_prefixes = [
            'analyzed_rebar_',         # Real model results
            'analyzed_placeholder_',   # Placeholder results
            'step4_cement_',          # Step 4 final results
            'real_analysis_',          # Legacy real model naming
            'placeholder_analysis_'    # Legacy placeholder naming
        ]
        
        return any(filename.startswith(prefix) for prefix in analyzed_prefixes)
    
    def _is_step_image(self, filename):
        """Check if filename is a step analysis image"""
        step_prefixes = ['step1_detection_', 'step2_intersections_', 'step3_polygon_', 'step4_cement_']
        return any(filename.startswith(prefix) for prefix in step_prefixes)
    
    def _extract_step_number(self, filename):
        """Extract step number from step image filename"""
        if filename.startswith('step1_'):
            return 1
        elif filename.startswith('step2_'):
            return 2
        elif filename.startswith('step3_'):
            return 3
        elif filename.startswith('step4_'):
            return 4
        return None
    
    def _get_metadata_path(self, image_filename):
        """Get metadata file path for an image"""
        base_name = os.path.splitext(image_filename)[0]
        return os.path.join(self.upload_folder, f'{base_name}_metadata.json')
    
    def save_analysis_with_metadata(self, analysis_results):
        """
        Save analysis results with complete 4-step metadata
        Called by AI service after completing 4-step analysis
        """
        try:
            if not analysis_results or not analysis_results.get('success'):
                return {'success': False, 'error': 'Invalid analysis results'}
            
            # Get the final analyzed image path (step 4)
            final_image_path = analysis_results.get('analyzed_image_path')
            if not final_image_path or not os.path.exists(final_image_path):
                return {'success': False, 'error': 'Final analyzed image not found'}
            
            filename = os.path.basename(final_image_path)
            
            # Prepare comprehensive metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': '4_step_pipeline',
                'model_type': analysis_results.get('model_type', 'simplified_4step_pipeline'),
                'placeholder': analysis_results.get('placeholder', False),
                
                # Detection data
                'detections': {
                    'total_count': analysis_results.get('num_detections', 0),
                    'front_vertical_count': analysis_results.get('pipeline_data', {}).get('front_vertical_count', 2),
                    'front_horizontal_count': analysis_results.get('pipeline_data', {}).get('front_horizontal_count', 11),
                    'intersection_count': analysis_results.get('pipeline_data', {}).get('intersection_count', 22),
                    'items': analysis_results.get('detections', [])
                },
                
                # Dimensional data
                'dimensions': analysis_results.get('dimensions', {}),
                
                # Cement mixture data
                'cement_mixture': analysis_results.get('cement_mixture', {}),
                
                # Step images data
                'step_images': {},
                
                # Pipeline data
                'pipeline_data': analysis_results.get('pipeline_data', {})
            }
            
            # Process step images
            step_images = analysis_results.get('step_images', {})
            if step_images:
                for step, path in step_images.items():
                    if path and os.path.exists(path):
                        step_filename = os.path.basename(path)
                        metadata['step_images'][step] = {
                            'filename': step_filename,
                            'url': f'/static/captured_images/{step_filename}',
                            'exists': True
                        }
                    else:
                        metadata['step_images'][step] = {
                            'filename': None,
                            'url': None,
                            'exists': False
                        }
            
            # Save metadata
            metadata_path = self._get_metadata_path(filename)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"✅ Saved 4-step analysis metadata: {os.path.basename(metadata_path)}")
            print(f"   📊 Detections: {metadata['detections']['total_count']}")
            print(f"   📐 Dimensions: {metadata['dimensions'].get('display', 'N/A')}")
            print(f"   🧮 Cement: {metadata['cement_mixture'].get('ratio_string', 'N/A')}")
            print(f"   🖼️  Step Images: {len([s for s in metadata['step_images'].values() if s['exists']])}/4")
            
            return {
                'success': True,
                'metadata_saved': True,
                'metadata_path': metadata_path,
                'filename': filename
            }
            
        except Exception as e:
            print(f"❌ Error saving analysis metadata: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_image_metadata(self, filename):
        """Get metadata for a specific analyzed image"""
        try:
            if not self._is_analyzed_image(filename):
                return {
                    'success': False,
                    'error': 'Not an analyzed image'
                }
            
            metadata_path = self._get_metadata_path(filename)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Verify step images still exist
                step_images = metadata.get('step_images', {})
                for step, step_data in step_images.items():
                    if step_data.get('filename'):
                        step_path = os.path.join(self.upload_folder, step_data['filename'])
                        step_data['exists'] = os.path.exists(step_path)
                
                print(f"📖 Retrieved metadata for: {filename}")
                return {
                    'success': True,
                    'metadata': metadata
                }
            else:
                # Create basic metadata if none exists
                basic_metadata = self._create_basic_metadata(filename)
                return {
                    'success': True,
                    'metadata': basic_metadata,
                    'generated': True
                }
                
        except Exception as e:
            print(f"❌ Error getting metadata for {filename}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_basic_metadata(self, filename):
        """Create basic metadata for images without saved metadata"""
        try:
            filepath = os.path.join(self.upload_folder, filename)
            if not os.path.exists(filepath):
                return {}
            
            file_stats = os.stat(filepath)
            
            return {
                'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'analysis_type': '4_step_pipeline',
                'model_type': 'unknown',
                'placeholder': True,
                'detections': {
                    'total_count': 13,
                    'front_vertical_count': 2,
                    'front_horizontal_count': 11,
                    'intersection_count': 22,
                    'items': []
                },
                'dimensions': {
                    'display': '28cm x 28cm x 57cm = 45000cm³',
                    'length': 28,
                    'width': 28,
                    'height': 57,
                    'unit': 'cm'
                },
                'cement_mixture': {
                    'ratio_string': '1 Cement : 2 Sand : 4 Aggregate'
                },
                'step_images': {
                    'step1': {'filename': None, 'url': None, 'exists': False},
                    'step2': {'filename': None, 'url': None, 'exists': False},
                    'step3': {'filename': None, 'url': None, 'exists': False},
                    'step4': {'filename': None, 'url': None, 'exists': False}
                },
                'generated': True
            }
            
        except Exception as e:
            print(f"❌ Error creating basic metadata: {str(e)}")
            return {}
    
    def get_all_images(self):
        """
        Get list of ONLY analyzed images with 4-step metadata
        ENHANCED: Now includes complete pipeline metadata
        """
        try:
            all_images = []
            analyzed_images = []
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        file_stats = os.stat(filepath)
                        
                        # Get image dimensions
                        image_info = self._log_image_dimensions(filepath, "Gallery Scan")
                        
                        image_data = {
                            'filename': filename,
                            'url': f'/static/captured_images/{filename}',
                            'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            'size': file_stats.st_size
                        }
                        
                        # Add dimension info if available
                        if image_info:
                            image_data.update({
                                'width': image_info['width'],
                                'height': image_info['height'],
                                'dimensions_text': f"{image_info['width']}x{image_info['height']}"
                            })
                        
                        all_images.append(image_data)
                        
                        # Filter for analyzed images only and add metadata
                        if self._is_analyzed_image(filename):
                            image_data['type'] = 'analyzed'
                            image_data['is_analyzed'] = True
                            
                            # Get metadata
                            metadata_result = self.get_image_metadata(filename)
                            if metadata_result['success']:
                                metadata = metadata_result['metadata']
                                image_data['metadata'] = metadata
                                
                                # Add searchable fields from metadata
                                image_data['detections_count'] = metadata.get('detections', {}).get('total_count', 0)
                                image_data['model_type'] = metadata.get('model_type', 'unknown')
                                image_data['has_step_images'] = any(
                                    step.get('exists', False) 
                                    for step in metadata.get('step_images', {}).values()
                                )
                                
                            analyzed_images.append(image_data)
                        else:
                            image_data['type'] = 'original'
                            image_data['is_analyzed'] = False
            
            # Sort analyzed images by timestamp (newest first)
            analyzed_images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            total_count = len(all_images)
            analyzed_count = len(analyzed_images)
            original_count = total_count - analyzed_count
            
            print(f"📚 Enhanced Gallery Filter Results:")
            print(f"   📊 Total images found: {total_count}")
            print(f"   ✅ Analyzed images (shown): {analyzed_count}")
            print(f"   🚫 Original images (hidden): {original_count}")
            print(f"   📋 With 4-step metadata: {len([img for img in analyzed_images if img.get('metadata')])}")
            
            return {
                'success': True,
                'images': analyzed_images,  # Only return analyzed images
                'stats': {
                    'total_files': total_count,
                    'analyzed_shown': analyzed_count,
                    'originals_hidden': original_count,
                    'with_metadata': len([img for img in analyzed_images if img.get('metadata')])
                }
            }
            
        except Exception as e:
            print(f"💥 Error getting enhanced images: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_image(self, filename):
        """Delete image and its metadata"""
        try:
            if not self._is_allowed_file(filename):
                return {
                    'success': False,
                    'error': 'Invalid file type'
                }
            
            filepath = os.path.join(self.upload_folder, filename)
            metadata_path = self._get_metadata_path(filename)
            
            if os.path.exists(filepath):
                image_type = "Analyzed" if self._is_analyzed_image(filename) else "Original"
                print(f"🗑️  Deleting {image_type} image with metadata: {filename}")
                
                # Delete main image
                os.remove(filepath)
                
                # Delete metadata if exists
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    print(f"🗑️  Deleted metadata: {os.path.basename(metadata_path)}")
                
                # If this is a final analyzed image, also delete related step images
                if self._is_analyzed_image(filename):
                    self._delete_related_step_images(filename)
                
                print(f"✅ {image_type} image and metadata deleted: {filename}")
                return {
                    'success': True,
                    'message': f'{image_type} image and metadata deleted successfully!'
                }
            else:
                return {
                    'success': False,
                    'error': 'Image not found'
                }
                
        except Exception as e:
            print(f"💥 Error deleting image: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _delete_related_step_images(self, analyzed_filename):
        """Delete step images related to an analyzed image"""
        try:
            # Extract timestamp from analyzed filename
            if 'analyzed_rebar_' in analyzed_filename:
                timestamp = analyzed_filename.replace('analyzed_rebar_', '').replace('.jpg', '')
                
                # Delete related step images
                step_patterns = [
                    f'step1_detection_{timestamp}.jpg',
                    f'step2_intersections_{timestamp}.jpg',
                    f'step3_polygon_{timestamp}.jpg',
                    f'step4_cement_{timestamp}.jpg'
                ]
                
                deleted_steps = 0
                for step_filename in step_patterns:
                    step_path = os.path.join(self.upload_folder, step_filename)
                    if os.path.exists(step_path):
                        os.remove(step_path)
                        deleted_steps += 1
                        print(f"🗑️  Deleted step image: {step_filename}")
                
                print(f"✅ Deleted {deleted_steps} related step images")
                
        except Exception as e:
            print(f"⚠️  Error deleting related step images: {str(e)}")
    
    def clear_all_images(self):
        """Delete all images and metadata"""
        try:
            deleted_count = 0
            analyzed_deleted = 0
            original_deleted = 0
            metadata_deleted = 0
            total_size = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    filepath = os.path.join(self.upload_folder, filename)
                    
                    if self._is_allowed_file(filename):
                        file_size = os.path.getsize(filepath)
                        total_size += file_size
                        
                        if self._is_analyzed_image(filename):
                            analyzed_deleted += 1
                        else:
                            original_deleted += 1
                        
                        os.remove(filepath)
                        deleted_count += 1
                        
                    elif filename.endswith('_metadata.json'):
                        os.remove(filepath)
                        metadata_deleted += 1
            
            print(f"🗑️  Complete Image & Metadata Cleanup:")
            print(f"   📊 Total images deleted: {deleted_count}")
            print(f"   ✅ Analyzed deleted: {analyzed_deleted}")
            print(f"   📁 Originals deleted: {original_deleted}")
            print(f"   📋 Metadata files deleted: {metadata_deleted}")
            print(f"   💾 Space freed: {total_size / 1024:.1f} KB")
            
            return {
                'success': True,
                'message': f'Cleared {deleted_count} images and {metadata_deleted} metadata files!',
                'details': {
                    'total_deleted': deleted_count,
                    'analyzed_deleted': analyzed_deleted,
                    'original_deleted': original_deleted,
                    'metadata_deleted': metadata_deleted,
                    'space_freed_kb': round(total_size / 1024, 1)
                }
            }
            
        except Exception as e:
            print(f"💥 Error clearing images: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _log_image_dimensions(self, filepath, source="Unknown"):
        """Log dimensions of saved image file"""
        try:
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    height, width, channels = img.shape
                    file_size = os.path.getsize(filepath)
                    
                    return {
                        'width': width,
                        'height': height,
                        'channels': channels,
                        'file_size': file_size,
                        'file_size_kb': round(file_size / 1024, 1)
                    }
            return None
        except Exception as e:
            print(f"💥 Error analyzing image {filepath}: {e}")
            return None
    
    # Legacy methods for compatibility
    def save_frame(self, frame, prefix='frame_capture'):
        """Save a cv2 frame (legacy method)"""
        try:
            filename = self._generate_filename(prefix)
            filepath = os.path.join(self.upload_folder, filename)
            
            success = cv2.imwrite(filepath, frame)
            
            if success:
                image_info = self._log_image_dimensions(filepath, "Frame Saved")
                return {
                    'success': True,
                    'filename': filename,
                    'message': 'Frame captured successfully!',
                    'dimensions': image_info
                }
            else:
                raise Exception("Failed to save frame")
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_base64_image(self, image_data, prefix='web_capture'):
        """Save a base64 encoded image (legacy method)"""
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            filename = self._generate_filename(prefix)
            filepath = os.path.join(self.upload_folder, filename)
            
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            image_info = self._log_image_dimensions(filepath, "Web Captured")
            
            return {
                'success': True,
                'filename': filename,
                'message': 'Image saved successfully!',
                'dimensions': image_info
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def cleanup_original_images(self):
        """
        Delete only original images, keeping analyzed images
        Useful for cleaning up duplicate originals when only analyzed images are needed
        """
        try:
            deleted_count = 0
            total_size = 0
            kept_count = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        
                        if not self._is_analyzed_image(filename):
                            # This is an original image - delete it
                            file_size = os.path.getsize(filepath)
                            total_size += file_size
                            os.remove(filepath)
                            deleted_count += 1
                            print(f"🗑️  Deleted original: {filename}")
                        else:
                            # This is an analyzed image - keep it
                            kept_count += 1
            
            print(f"🧹 Original Image Cleanup:")
            print(f"   🗑️  Originals deleted: {deleted_count}")
            print(f"   ✅ Analyzed kept: {kept_count}")
            print(f"   💾 Space freed: {total_size / 1024:.1f} KB")
            
            return {
                'success': True,
                'message': f'Cleaned up {deleted_count} original images, kept {kept_count} analyzed images',
                'details': {
                    'originals_deleted': deleted_count,
                    'analyzed_kept': kept_count,
                    'space_freed_kb': round(total_size / 1024, 1)
                }
            }
            
        except Exception as e:
            print(f"💥 Error cleaning up original images: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_storage_stats(self):
        """Get detailed storage statistics"""
        try:
            all_files = 0
            analyzed_files = 0
            original_files = 0
            step_files = 0
            metadata_files = 0
            total_size = 0
            analyzed_size = 0
            original_size = 0
            step_size = 0
            metadata_size = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    filepath = os.path.join(self.upload_folder, filename)
                    file_size = os.path.getsize(filepath)
                    
                    if self._is_allowed_file(filename):
                        all_files += 1
                        total_size += file_size
                        
                        if self._is_analyzed_image(filename):
                            analyzed_files += 1
                            analyzed_size += file_size
                        elif self._is_step_image(filename):
                            step_files += 1
                            step_size += file_size
                        else:
                            original_files += 1
                            original_size += file_size
                    elif filename.endswith('_metadata.json'):
                        metadata_files += 1
                        metadata_size += file_size
                        total_size += file_size
            
            stats = {
                'total_files': all_files,
                'analyzed_files': analyzed_files,
                'original_files': original_files,
                'step_files': step_files,
                'metadata_files': metadata_files,
                'total_size_kb': round(total_size / 1024, 1),
                'analyzed_size_kb': round(analyzed_size / 1024, 1),
                'original_size_kb': round(original_size / 1024, 1),
                'step_size_kb': round(step_size / 1024, 1),
                'metadata_size_kb': round(metadata_size / 1024, 1),
                'gallery_shows': analyzed_files,
                'hidden_from_gallery': original_files + step_files
            }
            
            print(f"📊 Enhanced Storage Statistics:")
            print(f"   📁 Total image files: {stats['total_files']}")
            print(f"   ✅ Analyzed (gallery): {stats['analyzed_files']} ({stats['analyzed_size_kb']} KB)")
            print(f"   📄 Originals (hidden): {stats['original_files']} ({stats['original_size_kb']} KB)")
            print(f"   🎞️  Step images: {stats['step_files']} ({stats['step_size_kb']} KB)")
            print(f"   📋 Metadata files: {stats['metadata_files']} ({stats['metadata_size_kb']} KB)")
            print(f"   💾 Total storage: {stats['total_size_kb']} KB")
            
            return {
                'success': True,
                'stats': stats
            }
            
        except Exception as e:
            print(f"💥 Error getting storage stats: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
