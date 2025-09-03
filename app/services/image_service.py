import os
import base64
import cv2
from datetime import datetime
from app.utils.config import config

class ImageService:
    """
    FIXED: Handles image saving, loading, and management with improved filtering
    """
    
    def __init__(self):
        self.upload_folder = config.UPLOAD_FOLDER
        self.allowed_extensions = config.ALLOWED_EXTENSIONS
        print("📁 Image Service initialized (analyzed images only mode) - FIXED VERSION")
    
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
        # Expanded list of analyzed image prefixes
        analyzed_prefixes = [
            'analyzed_',               # Most common
            'analyzed_rebar_',         # Real model results
            'analyzed_placeholder_',   # Placeholder results  
            'analyzed_enhanced_',      # Enhanced detection
            'analyzed_guaranteed_',    # Guaranteed detection
            'analyzed_real_',          # Real model
            'analyzed_simple_',        # Simple visualization
            'analyzed_emergency_',     # Emergency fallback
            'real_analysis_',          # Legacy real model naming
            'placeholder_analysis_',   # Legacy placeholder naming
            'phase_analysis_',         # Phase-based analysis
            'ai_analysis_',            # AI analysis results
            'rebar_analysis_',         # Rebar-specific analysis
        ]
        
        # Check if filename starts with any analyzed prefix
        for prefix in analyzed_prefixes:
            if filename.startswith(prefix):
                return True
        
        # Additional check: look for analysis-related keywords in filename
        analysis_keywords = ['analysis', 'detected', 'overlay', 'result']
        filename_lower = filename.lower()
        for keyword in analysis_keywords:
            if keyword in filename_lower and not filename_lower.startswith('frame_'):
                return True
        
        return False
    
    def _log_image_dimensions(self, filepath, source="Unknown"):
        """Log dimensions of saved image file"""
        try:
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                if img is not None:
                    height, width, channels = img.shape
                    file_size = os.path.getsize(filepath)
                    
                    print(f"📊 {source} Image Analysis:")
                    print(f"   📁 File: {os.path.basename(filepath)}")
                    print(f"   📐 Dimensions: {width} x {height} pixels")
                    print(f"   🎨 Channels: {channels}")
                    print(f"   💾 File Size: {file_size / 1024:.1f} KB")
                    
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
        """Save a cv2 frame as an image file with dimension logging"""
        try:
            filename = self._generate_filename(prefix)
            filepath = os.path.join(self.upload_folder, filename)
            
            if frame is not None:
                height, width = frame.shape[:2]
                print(f"💾 Saving Frame:")
                print(f"   🖼️  Frame Shape: {frame.shape}")
                print(f"   📐 Frame Size: {width} x {height} pixels")
            
            success = cv2.imwrite(filepath, frame)
            
            if success:
                print(f"✅ Frame saved: {filename}")
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
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            filename = self._generate_filename(prefix)
            filepath = os.path.join(self.upload_folder, filename)
            
            print(f"💾 Saving base64 image to: {filepath}")
            
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            print(f"✅ Base64 image saved: {filename}")
            
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
        FIXED: Get list of ONLY analyzed images with AI overlays
        """
        try:
            all_images = []
            analyzed_images = []
            
            if os.path.exists(self.upload_folder):
                print(f"📁 Scanning upload folder: {self.upload_folder}")
                
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        
                        try:
                            file_stats = os.stat(filepath)
                            
                            # Basic image data
                            image_data = {
                                'filename': filename,
                                'url': f'/static/captured_images/{filename}',
                                'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                                'size': file_stats.st_size
                            }
                            
                            # Get image dimensions (but don't spam the logs)
                            try:
                                img = cv2.imread(filepath)
                                if img is not None:
                                    height, width = img.shape[:2]
                                    image_data.update({
                                        'width': width,
                                        'height': height,
                                        'dimensions_text': f"{width}x{height}"
                                    })
                            except Exception:
                                pass  # Skip dimension info if failed
                            
                            all_images.append(image_data)
                            
                            # FIXED: Check if this is an analyzed image
                            if self._is_analyzed_image(filename):
                                image_data['type'] = 'analyzed'
                                image_data['is_analyzed'] = True
                                analyzed_images.append(image_data)
                                print(f"✅ Analyzed image: {filename}")
                            else:
                                image_data['type'] = 'original'
                                image_data['is_analyzed'] = False
                                
                        except Exception as e:
                            print(f"⚠️  Error processing {filename}: {e}")
                            continue
            
            # Sort analyzed images by timestamp (newest first)
            analyzed_images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            total_count = len(all_images)
            analyzed_count = len(analyzed_images)
            original_count = total_count - analyzed_count
            
            print(f"📚 FIXED Gallery Filter Results:")
            print(f"   📊 Total images found: {total_count}")
            print(f"   ✅ Analyzed images (shown): {analyzed_count}")
            print(f"   🚫 Original images (hidden): {original_count}")
            
            if original_count > 0:
                print(f"   📝 NOTE: {original_count} original images exist but are hidden from gallery")
                print("   💡 Consider cleaning up original images if no longer needed")
            
            # FIXED: Return only analyzed images with proper metadata
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
    
    def get_image_metadata(self, filename):
        """FIXED: Get specific image metadata"""
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
            image_info = self._log_image_dimensions(filepath, "Metadata Request")
            
            # Check if it's an analyzed image
            is_analyzed = self._is_analyzed_image(filename)
            
            metadata = {
                'filename': filename,
                'url': f'/static/captured_images/{filename}',
                'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'size': file_stats.st_size,
                'type': 'analyzed' if is_analyzed else 'original',
                'is_analyzed': is_analyzed
            }
            
            if image_info:
                metadata.update({
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'channels': image_info['channels'],
                    'dimensions_text': f"{image_info['width']}x{image_info['height']}"
                })
            
            return {
                'success': True,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"💥 Error getting image metadata: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_image(self, filename):
        """Delete a specific image file"""
        try:
            if not self._is_allowed_file(filename):
                return {
                    'success': False,
                    'error': 'Invalid file type'
                }
            
            filepath = os.path.join(self.upload_folder, filename)
            
            if os.path.exists(filepath):
                image_type = "Analyzed" if self._is_analyzed_image(filename) else "Original"
                print(f"🗑️  Deleting {image_type} image: {filename}")
                
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
        """Delete only original images, keeping analyzed images"""
        try:
            deleted_count = 0
            total_size = 0
            kept_count = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        
                        if not self._is_analyzed_image(filename):
                            file_size = os.path.getsize(filepath)
                            total_size += file_size
                            os.remove(filepath)
                            deleted_count += 1
                            print(f"🗑️  Deleted original: {filename}")
                        else:
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
