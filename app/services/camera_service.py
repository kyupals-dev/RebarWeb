# Updated camera_service.py with optimized brightness and exposure settings
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
from app.utils.config import config

class CameraManager:
    """A4Tech camera with enhanced brightness, exposure, and 90° clockwise rotation"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_error = None
        self.camera_device = 0  # A4Tech is on /dev/video0
        self.frame_count = 0
        self.rotation_enabled = True  # Enable 90° clockwise rotation
        self.cropping_enabled = False  # Disable cropping in camera service
        self.brightness_enhancement = True  # Enable brightness enhancement
        self.last_captured_dimensions = None
        self.crop_log_count = 0
        
        # Enhanced camera settings for better brightness
        self.brightness_boost = 30  # Brightness adjustment (-100 to 100)
        self.contrast_boost = 20    # Contrast adjustment (-100 to 100)
        self.saturation_boost = 10  # Saturation adjustment (-100 to 100)
        self.exposure_boost = -3    # Exposure compensation
        self.gain_boost = 50        # ISO/Gain boost for low light
        
        print("🎥 Initializing Enhanced Camera Manager with Brightness Optimization...")
        print(f"   Brightness boost: +{self.brightness_boost}")
        print(f"   Contrast boost: +{self.contrast_boost}")
        print(f"   Saturation boost: +{self.saturation_boost}")
        print(f"   Exposure compensation: {self.exposure_boost}")
        print(f"   Gain boost: +{self.gain_boost}")
        
    def crop_black_borders(self, frame):
        """Remove black borders using manual or automatic detection"""
        if frame is None or not self.cropping_enabled:
            return frame
        
        try:
            height, width = frame.shape[:2]
            
            # Use manual cropping if enabled (more reliable)
            if hasattr(self, 'use_manual_crop') and self.use_manual_crop:
                left = getattr(self, 'manual_crop', {}).get('left', 0)
                right = width - getattr(self, 'manual_crop', {}).get('right', 0)
                top = getattr(self, 'manual_crop', {}).get('top', 0)
                bottom = height - getattr(self, 'manual_crop', {}).get('bottom', 0)
                
                # Ensure valid crop boundaries
                if left < right and top < bottom and left >= 0 and top >= 0:
                    cropped_frame = frame[top:bottom, left:right]
                    
                    # Log manual cropping info
                    self.crop_log_count += 1
                    if self.crop_log_count % 150 == 1:
                        crop_width = right - left
                        crop_height = bottom - top
                        original_area = height * width
                        cropped_area = crop_width * crop_height
                        
                        print(f"✂️  A4Tech Manual Cropping:")
                        print(f"   📐 Original: {width}x{height}")
                        print(f"   🎯 Manual Crop: {crop_width}x{crop_height}")
                        print(f"   📊 Content: {(cropped_area/original_area)*100:.1f}% of original")
                        print(f"   🔧 Removed - L:{left}, R:{getattr(self, 'manual_crop', {}).get('right', 0)}, T:{top}, B:{getattr(self, 'manual_crop', {}).get('bottom', 0)}")
                    
                    return cropped_frame
                else:
                    print("⚠️  Invalid manual crop settings, using original frame")
                    return frame
            
            # Fallback to automatic detection
            else:
                # Convert to grayscale for border detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Use row and column analysis for automatic detection
                row_sums = np.sum(gray > 20, axis=1)  # Count non-black pixels per row
                non_zero_rows = np.where(row_sums > width * 0.1)[0]  # Rows with >10% non-black pixels
                
                col_sums = np.sum(gray > 20, axis=0)  # Count non-black pixels per column
                non_zero_cols = np.where(col_sums > height * 0.1)[0]  # Columns with >10% non-black pixels
                
                if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
                    y_start = max(0, non_zero_rows[0] - 2)
                    y_end = min(height, non_zero_rows[-1] + 2)
                    x_start = max(0, non_zero_cols[0] - 2)
                    x_end = min(width, non_zero_cols[-1] + 2)
                    
                    crop_width = x_end - x_start
                    crop_height = y_end - y_start
                    original_area = height * width
                    cropped_area = crop_width * crop_height
                    
                    if cropped_area >= 0.5 * original_area:
                        cropped_frame = frame[y_start:y_end, x_start:x_end]
                        
                        self.crop_log_count += 1
                        if self.crop_log_count % 150 == 1:
                            print(f"🤖 A4Tech Auto Cropping:")
                            print(f"   📐 Original: {width}x{height}")
                            print(f"   ✂️  Auto Crop: {crop_width}x{crop_height}")
                            print(f"   📊 Content: {(cropped_area/original_area)*100:.1f}% of original")
                        
                        return cropped_frame
                
                return frame
                
        except Exception as e:
            print(f"⚠️  Error in cropping: {e}")
            return frame
    
    def enhance_brightness(self, frame):
        """Apply brightness and contrast enhancement to frame"""
        if frame is None or not self.brightness_enhancement:
            return frame
        
        try:
            # Convert to float for processing
            enhanced_frame = frame.astype(np.float32)
            
            # Apply brightness boost (additive)
            enhanced_frame = enhanced_frame + (self.brightness_boost * 2.55)  # Convert to 0-255 scale
            
            # Apply contrast boost (multiplicative around midpoint)
            contrast_factor = (100 + self.contrast_boost) / 100.0
            enhanced_frame = ((enhanced_frame - 127.5) * contrast_factor) + 127.5
            
            # Apply saturation boost in HSV space
            if self.saturation_boost != 0:
                hsv_frame = cv2.cvtColor(np.clip(enhanced_frame, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
                hsv_frame = hsv_frame.astype(np.float32)
                
                # Adjust saturation
                saturation_factor = (100 + self.saturation_boost) / 100.0
                hsv_frame[:, :, 1] = hsv_frame[:, :, 1] * saturation_factor
                
                # Clip saturation values
                hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1], 0, 255)
                
                # Convert back to BGR
                enhanced_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
            
            # Clamp values to 0-255 range and convert back to uint8
            enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)
            
            return enhanced_frame
            
        except Exception as e:
            print(f"⚠️  Error in brightness enhancement: {e}")
            return frame
    
    def rotate_frame_90_clockwise(self, frame):
        """Rotate frame 90 degrees clockwise to get 480x640 portrait format"""
        if frame is None:
            return None
        
        try:
            # cv2.ROTATE_90_CLOCKWISE rotates 90° clockwise
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # After rotation, should be 480x640 (height x width)
            height, width = rotated_frame.shape[:2]
            
            # Ensure output is exactly 480x640
            if height != 640 or width != 480:
                # Resize to exact 480x640 if needed
                rotated_frame = cv2.resize(rotated_frame, (480, 640))
                print(f"Resized rotated frame from {width}x{height} to 480x640")
            
            return rotated_frame
                
        except Exception as e:
            print(f"Rotation/resize error: {e}")
            # Fallback: resize original frame to 480x640
            try:
                return cv2.resize(frame, (480, 640))
            except:
                return frame
    
    def configure_camera_settings(self):
        """Configure camera settings for optimal brightness and exposure"""
        if not self.cap or not self.cap.isOpened():
            return False
        
        try:
            print("⚙️  Configuring enhanced camera settings for brightness...")
            
            # Basic resolution and format settings
            width_set = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            height_set = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            fps_set = self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Format settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for lower latency
            fourcc_set = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # ENHANCED BRIGHTNESS AND EXPOSURE SETTINGS
            
            # Set manual exposure mode for more control
            try:
                # Try to set manual exposure (may not work on all cameras)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure mode
                
                # Set exposure compensation
                exposure_set = self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_boost)
                print(f"   📸 Exposure set to: {self.exposure_boost} (success: {exposure_set})")
            except Exception as e:
                print(f"   ⚠️  Manual exposure setting failed, trying auto with bias: {e}")
                # Fallback to auto exposure with bias
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto exposure
                
            # Brightness adjustment
            try:
                brightness_set = self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_boost / 100.0)
                print(f"   💡 Brightness set to: {self.brightness_boost} (success: {brightness_set})")
            except Exception as e:
                print(f"   ⚠️  Brightness setting failed: {e}")
            
            # Contrast adjustment
            try:
                contrast_set = self.cap.set(cv2.CAP_PROP_CONTRAST, (50 + self.contrast_boost) / 100.0)
                print(f"   🎨 Contrast set to: {50 + self.contrast_boost} (success: {contrast_set})")
            except Exception as e:
                print(f"   ⚠️  Contrast setting failed: {e}")
            
            # Saturation adjustment
            try:
                saturation_set = self.cap.set(cv2.CAP_PROP_SATURATION, (50 + self.saturation_boost) / 100.0)
                print(f"   🌈 Saturation set to: {50 + self.saturation_boost} (success: {saturation_set})")
            except Exception as e:
                print(f"   ⚠️  Saturation setting failed: {e}")
            
            # Gain/ISO boost for low light
            try:
                gain_set = self.cap.set(cv2.CAP_PROP_GAIN, self.gain_boost)
                print(f"   📈 Gain set to: {self.gain_boost} (success: {gain_set})")
            except Exception as e:
                print(f"   ⚠️  Gain setting failed: {e}")
            
            # Additional low-light optimizations
            try:
                # Disable auto white balance for consistent colors
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Auto white balance on
                
                # Set gamma if supported
                self.cap.set(cv2.CAP_PROP_GAMMA, 120)  # Slightly higher gamma for visibility
                
                # Reduce noise reduction to preserve detail in low light
                # (Note: these may not be supported by all cameras)
                
            except Exception as e:
                print(f"   ⚠️  Additional settings failed: {e}")
            
            # Verify settings by reading them back
            actual_brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            actual_contrast = self.cap.get(cv2.CAP_PROP_CONTRAST) 
            actual_saturation = self.cap.get(cv2.CAP_PROP_SATURATION)
            actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            actual_gain = self.cap.get(cv2.CAP_PROP_GAIN)
            
            print("✅ Enhanced camera settings applied:")
            print(f"   💡 Actual brightness: {actual_brightness:.3f}")
            print(f"   🎨 Actual contrast: {actual_contrast:.3f}")
            print(f"   🌈 Actual saturation: {actual_saturation:.3f}")
            print(f"   📸 Actual exposure: {actual_exposure:.3f}")
            print(f"   📈 Actual gain: {actual_gain:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error configuring enhanced camera settings: {e}")
            return False
    
    def start_camera(self):
        """Start A4Tech camera with enhanced brightness settings and rotation"""
        if not self.is_running:
            try:
                print(f"🎥 Starting A4Tech camera with enhanced brightness and stable settings...")
                
                # Try different camera indices if default fails
                for camera_index in [0, 1, 2]:
                    self.cap = cv2.VideoCapture(camera_index)
                    if self.cap.isOpened():
                        self.camera_device = camera_index
                        break
                    else:
                        if self.cap:
                            self.cap.release()
                
                if not self.cap or not self.cap.isOpened():
                    self.last_error = "Cannot open any camera device"
                    print(f"❌ {self.last_error}")
                    return False
                
                print(f"✅ Camera found at index {self.camera_device}")
                
                # Apply enhanced camera settings
                settings_success = self.configure_camera_settings()
                if not settings_success:
                    print("⚠️  Some enhanced settings may not have been applied")
                
                # Test multiple frames to ensure stability with enhanced settings
                stable_frames = 0
                test_frames = []
                for i in range(10):
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        # Apply brightness enhancement
                        if self.brightness_enhancement:
                            test_frame = self.enhance_brightness(test_frame)
                        
                        test_frames.append(test_frame)
                        stable_frames += 1
                    else:
                        print(f"⚠️  Frame {i} failed")
                
                if stable_frames < 5:
                    self.cap.release()
                    self.last_error = "Camera produces unstable frames"
                    print(f"❌ {self.last_error} (only {stable_frames}/10 good frames)")
                    return False
                
                # Analyze brightness of test frames
                if test_frames:
                    avg_brightness = np.mean([np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in test_frames])
                    print(f"📊 Average frame brightness: {avg_brightness:.1f}/255")
                    
                    if avg_brightness < 80:
                        print("⚠️  Low brightness detected - brightness enhancement will be applied")
                        self.brightness_enhancement = True
                    elif avg_brightness > 200:
                        print("✅ Good brightness detected - minimal enhancement needed")
                        self.brightness_boost = 10  # Reduce boost for bright conditions
                
                # Get a good test frame for analysis
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.cap.release()
                    self.last_error = "Cannot capture test frame"
                    print(f"❌ {self.last_error}")
                    return False
                
                # Apply enhancements to test frame
                if self.brightness_enhancement:
                    test_frame = self.enhance_brightness(test_frame)
                
                # Test rotation and validate frame
                print(f"📐 A4Tech Original Frame: {test_frame.shape} (H x W x C)")
                
                if len(test_frame.shape) != 3 or test_frame.shape[2] != 3:
                    print(f"⚠️  Unusual frame format: {test_frame.shape}")
                
                rotated_test = self.rotate_frame_90_clockwise(test_frame)
                if rotated_test is not None:
                    print(f"🔄 A4Tech Rotated Frame: {rotated_test.shape} (H x W x C)")
                    print(f"✅ Processing: {test_frame.shape[1]}x{test_frame.shape[0]} → {rotated_test.shape[1]}x{rotated_test.shape[0]}")
                
                # Get actual settings for logging
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✅ A4Tech camera started successfully with brightness enhancement!")
                print(f"   📐 Resolution: {actual_width}x{actual_height}")
                print(f"   🔄 Rotation: {'ON' if self.rotation_enabled else 'OFF'}")
                print(f"   💡 Brightness enhancement: {'ON' if self.brightness_enhancement else 'OFF'}")
                print(f"   🎬 Frame rate: {actual_fps} fps")
                print(f"   📊 Stable frames: {stable_frames}/10")
                print(f"   🎨 Enhancement settings: B+{self.brightness_boost}, C+{self.contrast_boost}, S+{self.saturation_boost}")
                
                self.is_running = True
                self.last_error = None
                return True
                
            except Exception as e:
                self.last_error = f"A4Tech camera error: {str(e)}"
                print(f"❌ {self.last_error}")
                if self.cap:
                    self.cap.release()
                return False
        else:
            print("A4Tech camera already running")
            return True
    
    def get_frame(self):
        """Get enhanced and rotated frame from A4Tech camera with validation"""
        if not self.cap or not self.is_running:
            return None
            
        try:
            # Clear buffer to get latest frame
            ret = False
            frame = None
            
            # Try to get the most recent frame
            for _ in range(2):  # Clear buffer
                ret, frame = self.cap.read()
            
            if ret and frame is not None and frame.size > 0:
                self.frame_count += 1
                
                # Validate frame dimensions
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"⚠️  Invalid frame shape: {frame.shape}")
                    return None
                
                # Apply brightness enhancement first
                if self.brightness_enhancement:
                    frame = self.enhance_brightness(frame)
                
                # Apply 90° clockwise rotation (no cropping for capture)
                if self.rotation_enabled:
                    rotated_frame = self.rotate_frame_90_clockwise(frame)
                    if rotated_frame is not None:
                        frame = rotated_frame
                
                # Log every 600 frames (20 seconds at 30fps) with brightness info
                if self.frame_count % 600 == 0:
                    brightness_status = "Enhanced" if self.brightness_enhancement else "Standard"
                    print(f"📹 A4Tech camera: {self.frame_count} frames ({frame.shape[1]}x{frame.shape[0]}) - {brightness_status}")
                
                return frame
            else:
                print("⚠️  Failed to read frame or empty frame")
                return None
                
        except Exception as e:
            if self.frame_count % 100 == 0:  # Don't spam errors
                print(f"⚠️  A4Tech camera frame error: {e}")
            return None
    
    def get_current_frame(self):
        """Get current processed frame (thread-safe)"""
        try:
            with self.frame_lock:
                return self.current_frame.copy() if self.current_frame is not None else None
        except Exception:
            return None
    
    def update_current_frame(self, frame):
        """Update current frame (thread-safe)"""
        try:
            with self.frame_lock:
                self.current_frame = frame.copy() if frame is not None else None
        except Exception:
            pass
    
    def capture_and_save_image(self, filepath):
        """Capture current frame and save with dimension logging and brightness info"""
        try:
            frame = self.get_current_frame()
            if frame is not None:
                # Log frame dimensions before saving
                height, width = frame.shape[:2]
                brightness_info = "Enhanced" if self.brightness_enhancement else "Standard"
                
                print(f"📸 A4Tech Capture Dimensions ({brightness_info}):")
                print(f"   🖼️  Frame Shape: {frame.shape} (H x W x C)")
                print(f"   📐 Resolution: {width} x {height} pixels")
                print(f"   🔄 Rotation: {'Applied' if self.rotation_enabled else 'None'}")
                print(f"   💡 Brightness: {brightness_info} (B+{self.brightness_boost}, C+{self.contrast_boost})")
                print(f"   ✂️  Cropping: {'Applied' if self.cropping_enabled else 'None'}")
                print(f"   💾 Saving to: {filepath}")
                
                # Save the image
                success = cv2.imwrite(filepath, frame)
                
                if success:
                    # Verify saved image dimensions
                    saved_img = cv2.imread(filepath)
                    if saved_img is not None:
                        saved_height, saved_width = saved_img.shape[:2]
                        file_size = os.path.getsize(filepath)
                        
                        print(f"✅ A4Tech Enhanced Image Saved Successfully:")
                        print(f"   📁 File: {os.path.basename(filepath)}")
                        print(f"   📐 Saved Size: {saved_width} x {saved_height} pixels")
                        print(f"   💾 File Size: {file_size / 1024:.1f} KB")
                        print(f"   🎨 Enhancement: {brightness_info}")
                        
                        self.last_captured_dimensions = {
                            'width': saved_width,
                            'height': saved_height,
                            'file_size': file_size,
                            'filepath': filepath,
                            'brightness_enhanced': self.brightness_enhancement,
                            'enhancement_settings': f"B+{self.brightness_boost}, C+{self.contrast_boost}, S+{self.saturation_boost}"
                        }
                        
                        return True
                    else:
                        print(f"❌ Could not verify saved image: {filepath}")
                        return False
                else:
                    print(f"❌ Failed to save image: {filepath}")
                    return False
            else:
                print("❌ No current frame available for capture")
                return False
                
        except Exception as e:
            print(f"💥 Error capturing image: {e}")
            return False
    
    def adjust_brightness_settings(self, brightness=None, contrast=None, saturation=None):
        """Dynamically adjust brightness settings"""
        try:
            settings_changed = False
            
            if brightness is not None:
                self.brightness_boost = max(-50, min(100, brightness))
                settings_changed = True
                print(f"💡 Brightness adjusted to: +{self.brightness_boost}")
            
            if contrast is not None:
                self.contrast_boost = max(-50, min(100, contrast))
                settings_changed = True
                print(f"🎨 Contrast adjusted to: +{self.contrast_boost}")
                
            if saturation is not None:
                self.saturation_boost = max(-50, min(100, saturation))
                settings_changed = True
                print(f"🌈 Saturation adjusted to: +{self.saturation_boost}")
            
            if settings_changed and self.cap and self.cap.isOpened():
                # Try to apply settings to camera hardware if possible
                try:
                    if brightness is not None:
                        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness_boost / 100.0)
                    if contrast is not None:
                        self.cap.set(cv2.CAP_PROP_CONTRAST, (50 + self.contrast_boost) / 100.0)
                    if saturation is not None:
                        self.cap.set(cv2.CAP_PROP_SATURATION, (50 + self.saturation_boost) / 100.0)
                except Exception as e:
                    print(f"⚠️  Hardware setting adjustment failed, using software enhancement: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error adjusting brightness settings: {e}")
            return False
    
    def toggle_brightness_enhancement(self):
        """Toggle brightness enhancement on/off"""
        self.brightness_enhancement = not self.brightness_enhancement
        status = "enabled" if self.brightness_enhancement else "disabled"
        print(f"💡 A4Tech brightness enhancement {status}")
        return self.brightness_enhancement
    
    def toggle_cropping(self):
        """Toggle black border cropping on/off"""
        self.cropping_enabled = not self.cropping_enabled
        status = "enabled" if self.cropping_enabled else "disabled"
        print(f"✂️  A4Tech black border cropping {status}")
        return self.cropping_enabled
    
    def toggle_rotation(self):
        """Toggle 90° rotation on/off"""
        self.rotation_enabled = not self.rotation_enabled
        status = "enabled" if self.rotation_enabled else "disabled"
        print(f"🔄 A4Tech camera rotation {status}")
        return self.rotation_enabled
    
    def stop_camera(self):
        """Stop A4Tech camera"""
        try:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            print("🛑 A4Tech camera stopped")
        except Exception as e:
            print(f"Error stopping camera: {e}")
        finally:
            self.is_running = False
    
    def get_status(self):
        """Get A4Tech camera status with processing and brightness info"""
        return {
            'is_running': self.is_running,
            'has_frame': self.current_frame is not None,
            'last_error': self.last_error,
            'camera_device': f'/dev/video{self.camera_device}',
            'camera_type': 'A4Tech FHD 1080P PC Camera',
            'frames_captured': self.frame_count,
            'rotation_enabled': self.rotation_enabled,
            'cropping_enabled': self.cropping_enabled,
            'brightness_enhancement': self.brightness_enhancement,
            'brightness_settings': {
                'brightness_boost': self.brightness_boost,
                'contrast_boost': self.contrast_boost,
                'saturation_boost': self.saturation_boost,
                'exposure_boost': self.exposure_boost,
                'gain_boost': self.gain_boost
            },
            'processing': f"{'Rotation' if self.rotation_enabled else ''}{' + ' if self.rotation_enabled and self.cropping_enabled else ''}{'Cropping' if self.cropping_enabled else ''}{' + ' if (self.rotation_enabled or self.cropping_enabled) and self.brightness_enhancement else ''}{'Brightness Enhancement' if self.brightness_enhancement else ''}",
            'last_capture': self.last_captured_dimensions
        }

def camera_thread_worker(camera_manager):
    """Enhanced worker for A4Tech camera with brightness processing"""
    print("🚀 Starting A4Tech camera worker thread with brightness enhancement...")
    
    # Start camera
    if not camera_manager.start_camera():
        print("❌ Failed to start A4Tech camera")
        return
    
    print("🎬 A4Tech camera thread running with rotation, cropping, and brightness enhancement...")
    consecutive_failures = 0
    max_failures = 10
    
    while True:
        try:
            # Get processed frame (rotated, cropped, and brightness enhanced)
            frame = camera_manager.get_frame()
            if frame is not None:
                camera_manager.update_current_frame(frame)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"⚠️  Too many A4Tech camera failures, restarting...")
                    camera_manager.stop_camera()
                    threading.Event().wait(2)
                    
                    if camera_manager.start_camera():
                        consecutive_failures = 0
                        print("✅ A4Tech camera restarted with brightness enhancement")
                    else:
                        print("❌ Failed to restart A4Tech camera")
                        break
            
            # Higher FPS timing for lower latency
            threading.Event().wait(1.0 / 30.0)  # 30 FPS for real-time feel
            
        except KeyboardInterrupt:
            print("🛑 A4Tech camera thread interrupted")
            break
        except Exception as e:
            print(f"💥 A4Tech camera thread error: {e}")
            consecutive_failures += 1
            threading.Event().wait(0.5)
    
    camera_manager.stop_camera()
    print("🏁 A4Tech enhanced camera thread finished")
