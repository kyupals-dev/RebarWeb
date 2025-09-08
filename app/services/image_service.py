import os
import base64
import cv2
import json
from datetime import datetime
from app.utils.config import config

class ImageService:
    """
    Handles image saving, loading, and management operations with dimension logging
    MODIFIED: Gallery shows only analyzed images with AI overlays + metadata support
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
        """Check if filename indicates an analyzed image with AI overlays"""
        analyzed_prefixes = [
            'analyzed_rebar_',         # Real model results  
            'analyzed_resnet101_',     # ResNet-101 results
            'analyzed_placeholder_',   # Placeholder results
            'real_analysis_',          # Legacy real model naming
            'placeholder_analysis_'    # Legacy placeholder naming
        ]
        
        return any(filename.startswith(prefix) for prefix in analyzed_prefixes)
    
    def _create_metadata_file(self, image_filename, metadata):
        """Create metadata JSON file for analyzed image"""
        try:
            # Create metadata filename
            base_name = os.path.splitext(image_filename)[0]
            metadata_filename = f"{base_name}_metadata.json"
            metadata_path = os.path.join(self.upload_folder, metadata_filename)
            
            # Default metadata structure
            default_metadata = {
                'filename': image_filename,
                'analysis_date': datetime.now().isoformat(),
                'rebar_dimensions': '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters',
                'cement_mixture_ratio': '1 Cement: 2 Sand: 4 Aggregate',
                'detections_found': '2 Verticals + 11 Horizontals Pattern',
                'model_type': 'Simplified Front Detection',
                'is_analyzed': True,
                'type': 'analyzed'
            }
            
            # Merge with provided metadata
            if metadata:
                default_metadata.update(metadata)
            
            # Save metadata file
            with open(metadata_path, 'w') as f:
                json.dump(default_metadata, f, indent=2)
            
            print(f"✅ Metadata saved: {metadata_filename}")
            return metadata_path
            
        except Exception as e:
            print(f"⚠️  Error creating metadata file: {e}")
            return None
    
    def _get_image_metadata(self, image_filename):
        """Get metadata for an image file"""
        try:
            # Look for metadata file
            base_name = os.path.splitext(image_filename)[0]
            metadata_filename = f"{base_name}_metadata.json"
            metadata_path = os.path.join(self.upload_folder, metadata_filename)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
            else:
                # Return default metadata for analyzed images
                if self._is_analyzed_image(image_filename):
                    return {
                        'filename': image_filename,
                        'analysis_date': 'Unknown',
                        'rebar_dimensions': '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters',
                        'cement_mixture_ratio': '1 Cement: 2 Sand: 4 Aggregate',
                        'detections_found': '2 Verticals + 11 Horizontals Pattern',
                        'model_type': 'Simplified Front Detection',
                        'is_analyzed': True,
                        'type': 'analyzed'
                    }
                else:
                    return {
                        'filename': image_filename,
                        'is_analyzed': False,
                        'type': 'original'
                    }
                    
        except Exception as e:
            print(f"⚠️  Error reading metadata: {e}")
            return {'filename': image_filename, 'is_analyzed': False, 'type': 'unknown'}
    
    def save_analyzed_image_with_metadata(self, image_path, analysis_results=None):
        """Save metadata for an analyzed image that was just created"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Analyzed image not found: {image_path}")
                return False
            
            filename = os.path.basename(image_path)
            
            # Extract metadata from analysis results
            metadata = {
                'filename': filename,
                'analysis_date': datetime.now().isoformat(),
                'is_analyzed': True,
                'type': 'analyzed'
            }
            
            if analysis_results:
                # Extract dimensions
                dimensions = analysis_results.get('dimensions', {})
                metadata['rebar_dimensions'] = dimensions.get('display', '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters')
                
                # Extract cement mixture
                mixture = analysis_results.get('cement_mixture', {})
                metadata['cement_mixture_ratio'] = mixture.get('ratio_string', '1 Cement: 2 Sand: 4 Aggregate')
                
                # Extract detection info
                detections = analysis_results.get('detections', [])
                verticals = [d for d in detections if 'vertical' in d.get('class_name', '')]
                horizontals = [d for d in detections if 'horizontal' in d.get('class_name', '')]
                metadata['detections_found'] = f"{len(verticals)} Verticals + {len(horizontals)} Horizontals Pattern"
                
                # Model type
                model_type = analysis_results.get('model_type', 'unknown')
                if 'resnet101' in model_type:
                    metadata['model_type'] = 'ResNet-101 Front Detection'
                elif 'placeholder' in model_type:
                    metadata['model_type'] = 'Simplified Front Detection'
                else:
                    metadata['model_type'] = 'Simplified Front Detection'
            else:
                # Default values
                metadata['rebar_dimensions'] = '27.36cm x 27.36cm x 200cm = 149,874 cubic centimeters'
                metadata['cement_mixture_ratio'] = '1 Cement: 2 Sand: 4 Aggregate'
                metadata['detections_found'] = '2 Verticals + 11 Horizontals Pattern'
                metadata['model_type'] = 'Simplified Front Detection'
            
            # Create metadata file
            self._create_metadata_file(filename, metadata)
            print(f"✅ Analyzed image metadata created for: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving analyzed image metadata: {e}")
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
        Get list of ONLY analyzed images with AI overlays (no originals) + metadata
        MODIFIED: Filters to show only meaningful results with metadata support
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
                        
                        # Get metadata
                        metadata = self._get_image_metadata(filename)
                        
                        image_data = {
                            'filename': filename,
                            'url': f'/static/captured_images/{filename}',
                            'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                            'size': file_stats.st_size,
                            'metadata': metadata
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
                            image_data['is_analyzed'] = True
                            analyzed_images.append(image_data)
                        else:
                            image_data['type'] = 'original'
                            image_data['is_analyzed'] = False
            
            # Sort analyzed images by timestamp (newest first)
            analyzed_images.sort(key=lambda x: x['timestamp'], reverse=True)
            
            total_count = len(all_images)
            analyzed_count = len(analyzed_images)
            original_count = total_count - analyzed_count
            
            print(f"📚 Gallery Filter Results:")
            print(f"   📊 Total images found: {total_count}")
            print(f"   ✅ Analyzed images (shown): {analyzed_count}")
            print(f"   🚫 Original images (hidden): {original_count}")
            
            if original_count > 0:
                print(f"   📝 NOTE: {original_count} original images exist but are hidden from gallery")
                print("   💡 Consider cleaning up original images if no longer needed")
            
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
        """Get specific image metadata (for gallery modal)"""
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
            
            # Get metadata
            metadata = self._get_image_metadata(filename)
            
            # Get file stats
            file_stats = os.stat(filepath)
            
            # Get image dimensions
            image_info = self._log_image_dimensions(filepath, "Metadata Request")
            
            result = {
                'success': True,
                'filename': filename,
                'metadata': metadata,
                'file_size': file_stats.st_size,
                'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'is_analyzed': metadata.get('is_analyzed', False)
            }
            
            if image_info:
                result.update({
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'dimensions_text': f"{image_info['width']}x{image_info['height']}"
                })
            
            return result
            
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
            
            if os.path.exists(filepath):
                # Log image info before deletion
                image_type = "Analyzed" if self._is_analyzed_image(filename) else "Original"
                print(f"🗑️  Deleting {image_type} image: {filename}")
                self._log_image_dimensions(filepath, f"Deleting {image_type}")
                
                # Delete image file
                os.remove(filepath)
                
                # Delete metadata file if it exists
                base_name = os.path.splitext(filename)[0]
                metadata_filename = f"{base_name}_metadata.json"
                metadata_path = os.path.join(self.upload_folder, metadata_filename)
                
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    print(f"🗑️  Deleted metadata: {metadata_filename}")
                
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
        """Delete all images and metadata files in the upload folder"""
        try:
            deleted_count = 0
            analyzed_deleted = 0
            original_deleted = 0
            metadata_deleted = 0
            total_size = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    filepath = os.path.join(self.upload_folder, filename)
                    file_size = os.path.getsize(filepath)
                    total_size += file_size
                    
                    if self._is_allowed_file(filename):
                        # Track type before deletion
                        if self._is_analyzed_image(filename):
                            analyzed_deleted += 1
                        else:
                            original_deleted += 1
                        
                        os.remove(filepath)
                        deleted_count += 1
                    elif filename.endswith('_metadata.json'):
                        # Delete metadata files
                        os.remove(filepath)
                        metadata_deleted += 1
            
            print(f"🗑️  Image Cleanup Complete:")
            print(f"   📊 Total images deleted: {deleted_count}")
            print(f"   ✅ Analyzed deleted: {analyzed_deleted}")
            print(f"   📁 Originals deleted: {original_deleted}")
            print(f"   📄 Metadata files deleted: {metadata_deleted}")
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
    
    def get_storage_stats(self):
        """Get detailed storage statistics"""
        try:
            all_files = 0
            analyzed_files = 0
            original_files = 0
            metadata_files = 0
            total_size = 0
            analyzed_size = 0
            original_size = 0
            
            if os.path.exists(self.upload_folder):
                for filename in os.listdir(self.upload_folder):
                    filepath = os.path.join(self.upload_folder, filename)
                    file_size = os.path.getsize(filepath)
                    total_size += file_size
                    
                    if self._is_allowed_file(filename):
                        all_files += 1
                        
                        if self._is_analyzed_image(filename):
                            analyzed_files += 1
                            analyzed_size += file_size
                        else:
                            original_files += 1
                            original_size += file_size
                    elif filename.endswith('_metadata.json'):
                        metadata_files += 1
            
            stats = {
                'total_files': all_files,
                'analyzed_files': analyzed_files,
                'original_files': original_files,
                'metadata_files': metadata_files,
                'total_size_kb': round(total_size / 1024, 1),
                'analyzed_size_kb': round(analyzed_size / 1024, 1),
                'original_size_kb': round(original_size / 1024, 1),
                'gallery_shows': analyzed_files,
                'hidden_from_gallery': original_files
            }
            
            print(f"📊 Storage Statistics:")
            print(f"   📁 Total images: {stats['total_files']}")
            print(f"   ✅ Analyzed (shown): {stats['analyzed_files']} ({stats['analyzed_size_kb']} KB)")
            print(f"   📄 Originals (hidden): {stats['original_files']} ({stats['original_size_kb']} KB)")
            print(f"   📋 Metadata files: {stats['metadata_files']}")
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
