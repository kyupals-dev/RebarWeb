import os
import base64
import cv2
from datetime import datetime
from app.utils.config import config

class ImageService:
    """Handles image saving, loading, and management operations with dimension logging"""
    
    def __init__(self):
        self.upload_folder = config.UPLOAD_FOLDER
        self.allowed_extensions = config.ALLOWED_EXTENSIONS
    
    def _generate_filename(self, prefix='capture'):
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        return f'{prefix}_{timestamp}.jpg'
    
    def _is_allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
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
        """Save a cv2 frame as an image file with dimension logging"""
        try:
            filename = self._generate_filename(prefix)
            filepath = os.path.join(self.upload_folder, filename)
            
            # Log frame dimensions before saving
            if frame is not None:
                height, width = frame.shape[:2]
                print(f"💾 Saving A4Tech Frame:")
                print(f"   🖼️  Frame Shape: {frame.shape}")
                print(f"   📐 Frame Size: {width} x {height} pixels")
            
            # Save the frame using OpenCV
            success = cv2.imwrite(filepath, frame)
            
            if success:
                print(f"✅ A4Tech frame saved: {filename}")
                
                # Analyze saved image
                image_info = self._log_image_dimensions(filepath, "A4Tech Captured")
                
                return {
                    'success': True,
                    'filename': filename,
                    'message': 'Frame captured successfully!',
                    'dimensions': image_info
                }
            else:
                raise Exception("Failed to save frame")
                
        except Exception as e:
            print(f"💥 Error saving A4Tech frame: {str(e)}")
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
        """Get list of all captured images with metadata and dimensions"""
        try:
            images = []
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        file_stats = os.stat(filepath)
                        
                        # Get image dimensions
                        image_info = self._log_image_dimensions(filepath, "Gallery")
                        
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
                        
                        images.append(image_data)
            
            # Sort by timestamp (newest first)
            images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            print(f"📚 Found {len(images)} images in gallery")
            
            return {
                'success': True,
                'images': images
            }
            
        except Exception as e:
            print(f"💥 Error getting images: {str(e)}")
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
                self._log_image_dimensions(filepath, "Deleting")
                
                os.remove(filepath)
                print(f"🗑️  Image deleted: {filename}")
                return {
                    'success': True,
                    'message': 'Image deleted successfully!'
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
            total_size = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    if self._is_allowed_file(filename):
                        filepath = os.path.join(self.upload_folder, filename)
                        file_size = os.path.getsize(filepath)
                        total_size += file_size
                        
                        os.remove(filepath)
                        deleted_count += 1
            
            print(f"🗑️  Cleared {deleted_count} images ({total_size / 1024:.1f} KB total)")
            
            return {
                'success': True,
                'message': f'Cleared {deleted_count} images successfully!'
            }
            
        except Exception as e:
            print(f"💥 Error clearing images: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
