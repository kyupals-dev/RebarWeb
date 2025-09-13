# Updated camera_service.py with optimizations for reduced latency and better performance
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import numpy as np
import time
from datetime import datetime
from app.utils.config import config

class CameraManager:
    """A4Tech camera with 90° clockwise rotation and optimization for reduced latency"""
    
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
        self.last_captured_dimensions = None
        self.crop_log_count = 0
        
        # OPTIMIZATION: New parameters for reduced latency
        self.frame_buffer_size = 1  # Minimize buffering for lower latency
        self.jpeg_quality = 75      # Reduce quality slightly for faster transmission
        self.target_fps = 15        # Reduce from 30 to 15 FPS for less lag
        self.frame_skip_threshold = 2  # Skip frames if processing is slow
        
        print("📹 Camera Manager initialized with low-latency optimizations")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Buffer Size: {self.frame_buffer_size}")
        print(f"   JPEG Quality: {self.jpeg_quality}")
    
    def crop_black_borders(self, frame):
        """Remove black borders using manual or automatic detection"""
        if frame is None or not self.cropping_enabled:
            return frame
        
        try:
            height, width = frame.shape[:2]
            
            # Use manual cropping if enabled (more reliable)
            if hasattr(self, 'use_manual_crop') and self.use_manual_crop:
                left = self.manual_crop['left']
                right = width - self.manual_crop['right']
                top = self.manual_crop['top']
                bottom = height - self.manual_crop['bottom']
                
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
                        print(f"   🔧 Removed - L:{left}, R:{self.manual_crop['right']}, T:{top}, B:{self.manual_crop['bottom']}")
                    
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
    
    def start_camera(self):
        """Start A4Tech camera with optimized settings for reduced latency"""
        if not self.is_running:
            try:
                print(f"🎥 Starting optimized A4Tech camera (target: {self.target_fps} FPS)...")
                
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
                
                # Configure camera with optimized settings
                print("⚙️  Configuring A4Tech camera for low latency...")
                
                # Set resolution to capture landscape then rotate to 480x640 portrait
                width_set = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                height_set = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # OPTIMIZATION: Set target FPS
                fps_set = self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                
                # OPTIMIZATION: Reduced buffer size for lower latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.frame_buffer_size)
                
                # Use MJPG for better compatibility
                fourcc_set = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                # OPTIMIZATION: Disable auto exposure for consistent frame timing
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual exposure
                
                # Test multiple frames to ensure stability
                stable_frames = 0
                for i in range(10):
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        stable_frames += 1
                    else:
                        print(f"⚠️  Frame {i} failed")
                
                if stable_frames < 5:
                    self.cap.release()
                    self.last_error = "Camera produces unstable frames"
                    print(f"❌ {self.last_error} (only {stable_frames}/10 good frames)")
                    return False
                
                # Get a good test frame for analysis
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.cap.release()
                    self.last_error = "Cannot capture test frame"
                    print(f"❌ {self.last_error}")
                    return False
                
                # Test rotation and validate frame
                print(f"📐 A4Tech Original Frame: {test_frame.shape} (H x W x C)")
                
                if len(test_frame.shape) != 3 or test_frame.shape[2] != 3:
                    print(f"⚠️  Unusual frame format: {test_frame.shape}")
                
                rotated_test = self.rotate_frame_90_clockwise(test_frame)
                if rotated_test is not None:
                    print(f"🔄 A4Tech Rotated Frame: {rotated_test.shape} (H x W x C)")
                    print(f"✅ Processing: {test_frame.shape[1]}x{test_frame.shape[0]} → {rotated_test.shape[1]}x{rotated_test.shape[0]}")
                
                # Get actual settings
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✅ Optimized A4Tech camera started!")
                print(f"   📐 Resolution: {actual_width}x{actual_height}")
                print(f"   🔄 Rotation: {'ON' if self.rotation_enabled else 'OFF'}")
                print(f"   🎬 Frame rate: {actual_fps} fps (target: {self.target_fps})")
                print(f"   📊 Stable frames: {stable_frames}/10")
                print(f"   🚀 Buffer size: {self.frame_buffer_size} (optimized)")
                print(f"   📷 JPEG quality: {self.jpeg_quality}% (optimized)")
                
                self.is_running = True
                self.last_error = None
                return True
                
            except Exception as e:
                self.last_error = f"Optimized A4Tech camera error: {str(e)}"
                print(f"❌ {self.last_error}")
                if self.cap:
                    self.cap.release()
                return False
        else:
            print("Optimized A4Tech camera already running")
            return True
    
    def get_frame(self):
        """Get frame with optimization for reduced latency"""
        if not self.cap or not self.is_running:
            return None
            
        try:
            # OPTIMIZATION: Clear buffer to get most recent frame
            ret = False
            frame = None
            
            # Clear buffer to get latest frame - reduced iterations for speed
            for _ in range(self.frame_buffer_size + 1):
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            if ret and frame is not None and frame.size > 0:
                self.frame_count += 1
                
                # OPTIMIZATION: Skip frame processing if we're behind
                if self.frame_count % self.frame_skip_threshold != 0:
                    return self.current_frame if hasattr(self, 'current_frame') else frame
                
                # Validate frame dimensions
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"⚠️  Invalid frame shape: {frame.shape}")
                    return None
                
                # Apply 90° clockwise rotation only (no cropping for capture)
                if self.rotation_enabled:
                    rotated_frame = self.rotate_frame_90_clockwise(frame)
                    if rotated_frame is not None:
                        frame = rotated_frame
                
                # Log every 900 frames (60 seconds at 15fps)
                if self.frame_count % 900 == 0:
                    print(f"📹 Optimized A4Tech camera: {self.frame_count} frames ({frame.shape[1]}x{frame.shape[0]})")
                
                return frame
            else:
                print("⚠️  Failed to read frame or empty frame")
                return None
                
        except Exception as e:
            if self.frame_count % 100 == 0:  # Don't spam errors
                print(f"⚠️  Optimized A4Tech camera frame error: {e}")
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
        """Capture current frame and save with dimension logging"""
        try:
            frame = self.get_current_frame()
            if frame is not None:
                # Log frame dimensions before saving
                height, width = frame.shape[:2]
                print(f"📸 Optimized A4Tech Capture Dimensions:")
                print(f"   🖼️  Frame Shape: {frame.shape} (H x W x C)")
                print(f"   📐 Resolution: {width} x {height} pixels")
                print(f"   🔄 Rotation: {'Applied' if self.rotation_enabled else 'None'}")
                print(f"   ✂️  Cropping: {'Applied' if self.cropping_enabled else 'None'}")
                print(f"   💾 Saving to: {filepath}")
                
                # Save the image with optimized quality
                success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                
                if success:
                    # Verify saved image dimensions
                    saved_img = cv2.imread(filepath)
                    if saved_img is not None:
                        saved_height, saved_width = saved_img.shape[:2]
                        file_size = os.path.getsize(filepath)
                        
                        print(f"✅ Optimized A4Tech Image Saved:")
                        print(f"   📁 File: {os.path.basename(filepath)}")
                        print(f"   📐 Saved Size: {saved_width} x {saved_height} pixels")
                        print(f"   💾 File Size: {file_size / 1024:.1f} KB")
                        print(f"   📷 Quality: {self.jpeg_quality}% (optimized)")
                        
                        self.last_captured_dimensions = {
                            'width': saved_width,
                            'height': saved_height,
                            'file_size': file_size,
                            'filepath': filepath,
                            'quality': self.jpeg_quality
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
            print(f"💥 Error capturing optimized image: {e}")
            return False
    
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
    
    def set_fps(self, fps):
        """Set target FPS (optimization feature)"""
        if 5 <= fps <= 30:
            self.target_fps = fps
            if self.cap and self.is_running:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            print(f"🎬 Target FPS updated to: {fps}")
            return True
        else:
            print(f"⚠️  Invalid FPS value: {fps} (must be 5-30)")
            return False
    
    def set_jpeg_quality(self, quality):
        """Set JPEG quality for captures (optimization feature)"""
        if 50 <= quality <= 100:
            self.jpeg_quality = quality
            print(f"📷 JPEG quality updated to: {quality}%")
            return True
        else:
            print(f"⚠️  Invalid JPEG quality: {quality} (must be 50-100)")
            return False
    
    def stop_camera(self):
        """Stop optimized A4Tech camera"""
        try:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            print("🛑 Optimized A4Tech camera stopped")
        except Exception as e:
            print(f"Error stopping optimized camera: {e}")
        finally:
            self.is_running = False
    
    def get_status(self):
        """Get optimized A4Tech camera status with processing info"""
        return {
            'is_running': self.is_running,
            'has_frame': self.current_frame is not None,
            'last_error': self.last_error,
            'camera_device': f'/dev/video{self.camera_device}',
            'camera_type': 'A4Tech FHD 1080P PC Camera (Optimized)',
            'frames_captured': self.frame_count,
            'rotation_enabled': self.rotation_enabled,
            'cropping_enabled': self.cropping_enabled,
            'processing': f"{'Rotation' if self.rotation_enabled else ''}{' + ' if self.rotation_enabled and self.cropping_enabled else ''}{'Cropping' if self.cropping_enabled else ''}",
            'last_capture': self.last_captured_dimensions,
            'optimization': {
                'target_fps': self.target_fps,
                'buffer_size': self.frame_buffer_size,
                'jpeg_quality': self.jpeg_quality,
                'frame_skip_threshold': self.frame_skip_threshold
            }
        }

def camera_thread_worker(camera_manager):
    """Optimized worker thread for reduced latency"""
    print("🚀 Starting optimized A4Tech camera worker thread...")
    
    # Start camera
    if not camera_manager.start_camera():
        print("❌ Failed to start optimized A4Tech camera")
        return
    
    print("🎬 Optimized A4Tech camera thread running...")
    consecutive_failures = 0
    max_failures = 10
    target_fps = camera_manager.target_fps  # Use camera's target FPS
    frame_time = 1.0 / target_fps
    
    while True:
        try:
            start_time = time.time()
            
            # Get processed frame (rotated and optimized)
            frame = camera_manager.get_frame()
            if frame is not None:
                camera_manager.update_current_frame(frame)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"⚠️  Restarting optimized A4Tech camera after {consecutive_failures} failures...")
                    camera_manager.stop_camera()
                    time.sleep(2)
                    
                    if camera_manager.start_camera():
                        consecutive_failures = 0
                        print("✅ Optimized A4Tech camera restarted")
                    else:
                        print("❌ Failed to restart optimized A4Tech camera")
                        break
            
            # OPTIMIZATION: Consistent frame timing
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            print("🛑 Optimized A4Tech camera thread interrupted")
            break
        except Exception as e:
            print(f"💥 Optimized A4Tech camera thread error: {e}")
            consecutive_failures += 1
            time.sleep(0.5)
    
    camera_manager.stop_camera()
    print("🏁 Optimized A4Tech camera thread finished")
