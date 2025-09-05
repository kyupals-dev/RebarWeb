import os
import base64
import cv2
from datetime import datetime
from app.utils.config import config

class ImageService:
    """
    Handles image saving, loading, and management operations with dimension logging
    FIXED: Gallery shows analyzed images including simplified_analysis_ files
    """
    
    def __init__(self):
        self.upload_folder = config.UPLOAD_FOLDER
        self.allowed_extensions = config.ALLOWED_EXTENSIONS
        print("📁 Image Service initialized (analyzed images only mode)")
    
    def _generate_filename(self, prefix='capture'):
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        return f'{prefix}_{timestamp}.jpg'
    
    def _is_allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _is_analyzed_image(self, filename):
        """FIXED: Check if filename indicates an analyzed image with AI overlays"""
        analyzed_prefixes = [
            'analyzed_rebar_',         # Real model results
            'analyzed_placeholder_',   # Placeholder results
            'simplified_analysis_',    # NEW: Simplified model results
            'real_analysis_',          # Legacy real model naming
            'placeholder_analysis_'    # Legacy placeholder naming
        ]
        
        return any(filename.startswith(prefix) for prefix in analyzed_prefixes)
    
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
        FIXED: Get list of ONLY analyzed images including simplified_analysis_ files
        """
        try:
            all_images = []
            analyzed_images = []
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        file_stats = os.stat(filepath)
                        
                        # Get image dimensions (reduced logging for gallery scan)
                        image_info = None
                        try:
                            img = cv2.imread(filepath)
                            if img is not None:
                                height, width, channels = img.shape
                                image_info = {
                                    'width': width,
                                    'height': height,
                                    'channels': channels
                                }
                        except:
                            pass
                        
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
                        
                        # FIXED: Filter for analyzed images (now includes simplified_analysis_)
                        if self._is_analyzed_image(filename):
                            image_data['type'] = 'analyzed'
                            analyzed_images.append(image_data)
                            print(f"📸 Gallery: Including {filename} (analyzed image)")
                        else:
                            image_data['type'] = 'original'
            
            # Sort analyzed images by timestamp (newest first)
            analyzed_images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            total_count = len(all_images)
            analyzed_count = len(analyzed_images)
            original_count = total_count - analyzed_count
            
            print(f"📚 FIXED Gallery Filter Results:")
            print(f"   📊 Total images found: {total_count}")
            print(f"   ✅ Analyzed images (shown): {analyzed_count}")
            print(f"   🚫 Original images (hidden): {original_count}")
            
            if analyzed_count > 0:
                print(f"   📝 Analyzed images in gallery:")
                for img in analyzed_images[:3]:  # Show first 3
                    print(f"      - {img['filename']}")
            
            if original_count > 0:
                print(f"   💡 {original_count} original images exist but are hidden from gallery")
            
            return {
                'success': True,
                'images': analyzed_images,  # Only return analyzed images
                'stats': {
                    'total_files': total_count,
                    'analyzed_shown': analyzed_count,
                    'originals_hidden': original_count
                }
            }
            
        except Exception as e:
            print(f"💥 Error getting images: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_all_images_including_originals(self):
        """
        Get ALL images including originals (for debugging/admin purposes)
        This method can be used for cleanup or debugging
        """
        try:
            images = []
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        file_stats = os.stat(filepath)
                        
                        # Get image dimensions
                        image_info = self._log_image_dimensions(filepath, "Full Scan")
                        
                        image_data = {
                            'filename': filename,
                            'url': f'/static/captured_images/{filename}',
                            'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            'size': file_stats.st_size,
                            'type': 'analyzed' if self._is_analyzed_image(filename) else 'original'
                        }
                        
                        # Add dimension info if available
                        if image_info:
                            image_data.update({
                                'width': image_info['width'],
                                'height': image_info['height'],
                                'dimensions_text': f"{image_info['width']}x{image_info['height']}"
                            })
                        
                        images.append(image_data)
            
            # Sort by timestamp (newest first)
            images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            analyzed_count = len([img for img in images if img['type'] == 'analyzed'])
            original_count = len([img for img in images if img['type'] == 'original'])
            
            print(f"📚 Full Image List:")
            print(f"   📊 Total images: {len(images)}")
            print(f"   ✅ Analyzed: {analyzed_count}")
            print(f"   📁 Originals: {original_count}")
            
            return {
                'success': True,
                'images': images,
                'stats': {
                    'total': len(images),
                    'analyzed': analyzed_count,
                    'originals': original_count
                }
            }
            
        except Exception as e:
            print(f"💥 Error getting all images: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_image_metadata(self, filename):
        """FIXED: Get metadata for specific image (for gallery modal)"""
        try:
            if not self._is_allowed_file(filename):
                return {
                    'success': False,
                    'error': 'Invalid file type'
                }
            
            filepath = os.path.join(self.upload_folder, filename)
            
            if not os.path.exists(filepath):
                return {
                    'success': False,
                    'error': 'Image not found'
                }
            
            # Get file stats
            file_stats = os.stat(filepath)
            
            # Get image dimensions
            image_info = self._log_image_dimensions(filepath, "Gallery Modal")
            
            metadata = {
                'filename': filename,
                'url': f'/static/captured_images/{filename}',
                'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'size': file_stats.st_size,
                'type': 'analyzed' if self._is_analyzed_image(filename) else 'original',
                'is_analyzed': self._is_analyzed_image(filename)
            }
            
            if image_info:
                metadata.update({
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'dimensions_text': f"{image_info['width']}x{image_info['height']}",
                    'file_size_kb': image_info['file_size_kb']
                })
            
            # Add analysis info for analyzed images
            if self._is_analyzed_image(filename):
                if filename.startswith('simplified_analysis_'):
                    metadata['analysis_type'] = 'Simplified Front Detection'
                    metadata['model_info'] = '2 Verticals + 11 Horizontals Pattern'
                elif filename.startswith('analyzed_rebar_'):
                    metadata['analysis_type'] = 'Full AI Model Analysis'
                    metadata['model_info'] = 'Detectron2 Mask R-CNN'
                else:
                    metadata['analysis_type'] = 'AI Analysis Result'
                    metadata['model_info'] = 'Rebar Detection Model'
            
            print(f"📋 Metadata for {filename}: {metadata['type']} image")
            
            return {
                'success': True,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"💥 Error getting metadata for {filename}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_image(self, filename):
        """Delete a specific image file"""
        try:
            # Security check - only allow files in upload folder
            if not self._is_allowed_file(filename):
                return {
                    'success': False,
                    'error': 'Invalid file type'
                }
            
            filepath = os.path.join(self.upload_folder, filename)
            
            if os.path.exists(filepath):
                # Log image info before deletion
                image_type = "Analyzed" if self._is_analyzed_image(filename) else "Original"
                print(f"🗑️  Deleting {image_type} image: {filename}")
                self._log_image_dimensions(filepath, f"Deleting {image_type}")
                
                os.remove(filepath)
                print(f"✅ {image_type} image deleted: {filename}")
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
        """Delete all images in the upload folder"""
        try:
            deleted_count = 0
            analyzed_deleted = 0
            original_deleted = 0
            total_size = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        file_size = os.path.getsize(filepath)
                        total_size += file_size
                        
                        # Track type before deletion
                        if self._is_analyzed_image(filename):
                            analyzed_deleted += 1
                        else:
                            original_deleted += 1
                        
                        os.remove(filepath)
                        deleted_count += 1
            
            print(f"🗑️  Image Cleanup Complete:")
            print(f"   📊 Total deleted: {deleted_count} images")
            print(f"   ✅ Analyzed deleted: {analyzed_deleted}")
            print(f"   📁 Originals deleted: {original_deleted}")
            print(f"   💾 Space freed: {total_size / 1024:.1f} KB")
            
            return {
                'success': True,
                'message': f'Cleared {deleted_count} images successfully!',
                'details': {
                    'total_deleted': deleted_count,
                    'analyzed_deleted': analyzed_deleted,
                    'original_deleted': original_deleted,
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
            total_size = 0
            analyzed_size = 0
            original_size = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        file_size = os.path.getsize(filepath)
                        
                        all_files += 1
                        total_size += file_size
                        
                        if self._is_analyzed_image(filename):
                            analyzed_files += 1
                            analyzed_size += file_size
                        else:
                            original_files += 1
                            original_size += file_size
            
            stats = {
                'total_files': all_files,
                'analyzed_files': analyzed_files,
                'original_files': original_files,
                'total_size_kb': round(total_size / 1024, 1),
                'analyzed_size_kb': round(analyzed_size / 1024, 1),
                'original_size_kb': round(original_size / 1024, 1),
                'gallery_shows': analyzed_files,
                'hidden_from_gallery': original_files
            }
            
            print(f"📊 Storage Statistics:")
            print(f"   📁 Total files: {stats['total_files']}")
            print(f"   ✅ Analyzed (shown): {stats['analyzed_files']} ({stats['analyzed_size_kb']} KB)")
            print(f"   📄 Originals (hidden): {stats['original_files']} ({stats['original_size_kb']} KB)")
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
