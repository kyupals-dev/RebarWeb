import os
import base64
import cv2
import json
from datetime import datetime
from app.utils.config import config

class ImageService:
    """
    Handles image saving, loading, and management operations with enhanced metadata
    MODIFIED: Gallery shows only analyzed images with full analysis metadata
    """
    
    def __init__(self):
        self.upload_folder = config.UPLOAD_FOLDER
        self.allowed_extensions = config.ALLOWED_EXTENSIONS
        print("📁 Enhanced Image Service initialized (analyzed images + metadata mode)")
    
    def _generate_filename(self, prefix='capture'):
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        return f'{prefix}_{timestamp}.jpg'
    
    def _is_allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _is_analyzed_image(self, filename):
        """Check if filename indicates an analyzed image with AI overlays"""
        analyzed_prefixes = [
            'analyzed_rebar_',         # Real model results
            'analyzed_placeholder_',   # Placeholder results
            'analyzed_simplified_',    # Simplified model results
            'real_analysis_',          # Legacy real model naming
            'placeholder_analysis_'    # Legacy placeholder naming
        ]
        
        return any(filename.startswith(prefix) for prefix in analyzed_prefixes)
    
    def _get_metadata_path(self, image_filename):
        """Get the metadata JSON file path for an image"""
        base_name = os.path.splitext(image_filename)[0]
        return os.path.join(self.upload_folder, f'{base_name}_metadata.json')
    
    def _save_image_metadata(self, image_filename, metadata):
        """Save metadata for an analyzed image"""
        try:
            metadata_path = self._get_metadata_path(image_filename)
            
            # Add timestamp if not present
            if 'analysis_date' not in metadata:
                metadata['analysis_date'] = datetime.now().isoformat()
            
            # Add filename reference
            metadata['image_filename'] = image_filename
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"💾 Metadata saved: {os.path.basename(metadata_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
            return False
    
    def _load_image_metadata(self, image_filename):
        """Load metadata for an analyzed image"""
        try:
            metadata_path = self._get_metadata_path(image_filename)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"📖 Metadata loaded: {os.path.basename(metadata_path)}")
                return metadata
            else:
                print(f"⚠️  No metadata file found: {os.path.basename(metadata_path)}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None
    
    def save_analyzed_image_with_metadata(self, image_path, analysis_result):
        """
        Save analyzed image with its metadata
        Called by AI service after analysis is complete
        """
        try:
            if not os.path.exists(image_path):
                print(f"❌ Analyzed image not found: {image_path}")
                return False
            
            filename = os.path.basename(image_path)
            
            # Extract and save metadata from analysis result
            metadata = {
                'analysis_date': datetime.now().isoformat(),
                'image_filename': filename,
                'analysis_type': 'rebar_detection',
                'dimensions': analysis_result.get('dimensions', {}),
                'cement_mixture': analysis_result.get('cement_mixture', {}),
                'detections': {
                    'count': analysis_result.get('num_detections', 0),
                    'front_vertical_count': analysis_result.get('front_vertical_count', 0),
                    'front_horizontal_count': analysis_result.get('front_horizontal_count', 0),
                    'intersection_count': analysis_result.get('intersection_count', 0),
                    'target_achieved': analysis_result.get('target_achieved', {})
                },
                'model_info': {
                    'model_type': analysis_result.get('model_type', 'unknown'),
                    'placeholder_mode': analysis_result.get('placeholder', False)
                }
            }
            
            # Save metadata
            self._save_image_metadata(filename, metadata)
            
            print(f"✅ Analyzed image with metadata saved: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving analyzed image with metadata: {e}")
            return False
    
    def _log_image_dimensions(self, filepath, source="Unknown"):
        """Log dimensions of saved image file"""
        try:
            if os.path.exists(filepath):
                # Read image to get dimensions
                img = cv2.imread(filepath)
                if img is not None:
                    height, width, channels = img.shape
                    file_size = os.path.getsize(filepath)
                    
                    print(f"📊 {source} Image Analysis:")
                    print(f"   📁 File: {os.path.basename(filepath)}")
                    print(f"   📐 Dimensions: {width} x {height} pixels")
                    print(f"   🎨 Channels: {channels}")
                    print(f"   💾 File Size: {file_size / 1024:.1f} KB")
                    print(f"   📍 Path: {filepath}")
                    
                    return {
                        'width': width,
                        'height': height,
                        'channels': channels,
                        'file_size': file_size,
                        'file_size_kb': round(file_size / 1024, 1)
                    }
                else:
                    print(f"❌ Could not read image: {filepath}")
                    return None
            else:
                print(f"❌ Image file not found: {filepath}")
                return None
        except Exception as e:
            print(f"💥 Error analyzing image {filepath}: {e}")
            return None
    
    def save_frame(self, frame, prefix='frame_capture'):
        """
        Save a cv2 frame as an image file with dimension logging
        NOTE: This method is now rarely used since we only save analyzed images
        """
        try:
            filename = self._generate_filename(prefix)
            filepath = os.path.join(self.upload_folder, filename)
            
            # Log frame dimensions before saving
            if frame is not None:
                height, width = frame.shape[:2]
                print(f"💾 Saving Frame:")
                print(f"   🖼️  Frame Shape: {frame.shape}")
                print(f"   📐 Frame Size: {width} x {height} pixels")
                print("   📝 NOTE: Consider if this should be an analyzed image instead")
            
            # Save the frame using OpenCV
            success = cv2.imwrite(filepath, frame)
            
            if success:
                print(f"✅ Frame saved: {filename}")
                
                # Analyze saved image
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
            print(f"💥 Error saving frame: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_base64_image(self, image_data, prefix='web_capture'):
        """Save a base64 encoded image with dimension logging"""
        try:
            # Remove the data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            filename = self._generate_filename(prefix)
            filepath = os.path.join(self.upload_folder, filename)
            
            print(f"💾 Saving base64 image to: {filepath}")
            
            # Decode and save the image
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            print(f"✅ Base64 image saved: {filename}")
            
            # Analyze saved image
            image_info = self._log_image_dimensions(filepath, "Web Captured")
            
            return {
                'success': True,
                'filename': filename,
                'message': 'Image saved successfully!',
                'dimensions': image_info
            }
            
        except Exception as e:
            print(f"💥 Error saving base64 image: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_all_images(self):
        """
        Get list of ONLY analyzed images with AI overlays and their metadata
        ENHANCED: Now includes analysis metadata for gallery modal
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
                        
                        # Filter for analyzed images only
                        if self._is_analyzed_image(filename):
                            image_data['type'] = 'analyzed'
                            
                            # Load metadata if available
                            metadata = self._load_image_metadata(filename)
                            if metadata:
                                image_data['metadata'] = metadata
                                image_data['has_analysis'] = True
                            else:
                                image_data['has_analysis'] = False
                            
                            analyzed_images.append(image_data)
                        else:
                            image_data['type'] = 'original'
            
            # Sort analyzed images by timestamp (newest first)
            analyzed_images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            total_count = len(all_images)
            analyzed_count = len(analyzed_images)
            original_count = total_count - analyzed_count
            
            print(f"📚 Enhanced Gallery Filter Results:")
            print(f"   📊 Total images found: {total_count}")
            print(f"   ✅ Analyzed images (shown): {analyzed_count}")
            print(f"   🚫 Original images (hidden): {original_count}")
            print(f"   📖 Images with metadata: {len([img for img in analyzed_images if img.get('has_analysis')])}")
            
            if original_count > 0:
                print(f"   📝 NOTE: {original_count} original images exist but are hidden from gallery")
                print("   💡 Consider cleaning up original images if no longer needed")
            
            return {
                'success': True,
                'images': analyzed_images,  # Only return analyzed images with metadata
                'stats': {
                    'total_files': total_count,
                    'analyzed_shown': analyzed_count,
                    'originals_hidden': original_count,
                    'with_metadata': len([img for img in analyzed_images if img.get('has_analysis')])
                }
            }
            
        except Exception as e:
            print(f"💥 Error getting images: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_image_metadata(self, filename):
        """
        Get specific image metadata for gallery modal
        ENHANCED: Returns full analysis results for modal display
        """
        try:
            # Basic filename validation
            if not filename or '..' in filename or '/' in filename:
                return {
                    'success': False,
                    'error': 'Invalid filename'
                }
            
            if not self._is_allowed_file(filename):
                return {
                    'success': False,
                    'error': 'Invalid file type'
                }
            
            # Check if file exists
            filepath = os.path.join(self.upload_folder, filename)
            if not os.path.exists(filepath):
                return {
                    'success': False,
                    'error': 'Image file not found'
                }
            
            # Load metadata
            metadata = self._load_image_metadata(filename)
            
            if metadata:
                print(f"📖 Retrieved metadata for gallery modal: {filename}")
                return {
                    'success': True,
                    'metadata': metadata,
                    'has_analysis': True
                }
            else:
                # Return basic info if no metadata available
                file_stats = os.stat(filepath)
                basic_metadata = {
                    'analysis_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'image_filename': filename,
                    'dimensions': {
                        'display': 'Analysis data not available',
                        'length': 25.4,
                        'width': 25.4,
                        'height': 200.0
                    },
                    'cement_mixture': {
                        'ratio_string': '1 Cement : 2 Sand : 3 Aggregate'
                    },
                    'detections': {
                        'count': 0,
                        'front_vertical_count': 0,
                        'front_horizontal_count': 0
                    }
                }
                
                print(f"📝 No metadata found, returning basic info: {filename}")
                return {
                    'success': True,
                    'metadata': basic_metadata,
                    'has_analysis': False
                }
            
        except Exception as e:
            print(f"💥 Error getting image metadata: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_image(self, filename):
        """Delete a specific image file and its metadata"""
        try:
            # Security check - only allow files in upload folder
            if not self._is_allowed_file(filename):
                return {
                    'success': False,
                    'error': 'Invalid file type'
                }
            
            filepath = os.path.join(self.upload_folder, filename)
            metadata_path = self._get_metadata_path(filename)
            
            if os.path.exists(filepath):
                # Log image info before deletion
                image_type = "Analyzed" if self._is_analyzed_image(filename) else "Original"
                print(f"🗑️  Deleting {image_type} image: {filename}")
                self._log_image_dimensions(filepath, f"Deleting {image_type}")
                
                # Delete image file
                os.remove(filepath)
                
                # Delete metadata file if it exists
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    print(f"🗑️  Deleted metadata: {os.path.basename(metadata_path)}")
                
                print(f"✅ {image_type} image and metadata deleted: {filename}")
                return {
                    'success': True,
                    'message': f'{image_type} image deleted successfully!'
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
    
    def clear_all_images(self):
        """Delete all images and metadata in the upload folder"""
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
                        # Image file
                        file_size = os.path.getsize(filepath)
                        total_size += file_size
                        
                        # Track type before deletion
                        if self._is_analyzed_image(filename):
                            analyzed_deleted += 1
                        else:
                            original_deleted += 1
                        
                        os.remove(filepath)
                        deleted_count += 1
                        
                    elif filename.endswith('_metadata.json'):
                        # Metadata file
                        os.remove(filepath)
                        metadata_deleted += 1
            
            print(f"🗑️  Enhanced Image Cleanup Complete:")
            print(f"   📊 Total images deleted: {deleted_count}")
            print(f"   ✅ Analyzed deleted: {analyzed_deleted}")
            print(f"   📁 Originals deleted: {original_deleted}")
            print(f"   📖 Metadata files deleted: {metadata_deleted}")
            print(f"   💾 Space freed: {total_size / 1024:.1f} KB")
            
            return {
                'success': True,
                'message': f'Cleared {deleted_count} images and {metadata_deleted} metadata files successfully!',
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
    
    def cleanup_original_images(self):
        """
        Delete only original images, keeping analyzed images and metadata
        Useful for cleaning up duplicate originals when only analyzed images are needed
        """
        try:
            deleted_count = 0
            total_size = 0
            kept_count = 0
            metadata_kept = 0
            
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
                            
                            # Check if it has metadata
                            if os.path.exists(self._get_metadata_path(filename)):
                                metadata_kept += 1
            
            print(f"🧹 Enhanced Original Image Cleanup:")
            print(f"   🗑️  Originals deleted: {deleted_count}")
            print(f"   ✅ Analyzed kept: {kept_count}")
            print(f"   📖 Metadata files kept: {metadata_kept}")
            print(f"   💾 Space freed: {total_size / 1024:.1f} KB")
            
            return {
                'success': True,
                'message': f'Cleaned up {deleted_count} original images, kept {kept_count} analyzed images with {metadata_kept} metadata files',
                'details': {
                    'originals_deleted': deleted_count,
                    'analyzed_kept': kept_count,
                    'metadata_kept': metadata_kept,
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
        """Get detailed storage statistics including metadata"""
        try:
            all_files = 0
            analyzed_files = 0
            original_files = 0
            metadata_files = 0
            total_size = 0
            analyzed_size = 0
            original_size = 0
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
                'metadata_files': metadata_files,
                'total_size_kb': round(total_size / 1024, 1),
                'analyzed_size_kb': round(analyzed_size / 1024, 1),
                'original_size_kb': round(original_size / 1024, 1),
                'metadata_size_kb': round(metadata_size / 1024, 1),
                'gallery_shows': analyzed_files,
                'hidden_from_gallery': original_files
            }
            
            print(f"📊 Enhanced Storage Statistics:")
            print(f"   📁 Total image files: {stats['total_files']}")
            print(f"   ✅ Analyzed (shown): {stats['analyzed_files']} ({stats['analyzed_size_kb']} KB)")
            print(f"   📄 Originals (hidden): {stats['original_files']} ({stats['original_size_kb']} KB)")
            print(f"   📖 Metadata files: {stats['metadata_files']} ({stats['metadata_size_kb']} KB)")
            print(f"   💾 Total size: {stats['total_size_kb']} KB")
            
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
