# Updated camera_service.py with proper black border cropping
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
    """A4Tech camera with 90° clockwise rotation and black border cropping"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_error = None
        self.camera_device = 0  # A4Tech is on /dev/video0
        self.frame_count = 0
        self.rotation_enabled = True  # Enable 90° clockwise rotation
        self.cropping_enabled = True  # Enable black border cropping
        self.last_captured_dimensions = None
        self.crop_log_count = 0
        
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
        
    def crop_black_borders(self, frame):
        """Remove black borders using manual or automatic detection"""
        if frame is None or not self.cropping_enabled:
            return frame
        
        try:
            height, width = frame.shape[:2]
            
            # Use manual cropping if enabled (more reliable)
            if self.use_manual_crop:
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
        """Start A4Tech camera with optimal settings and rotation"""
        if not self.is_running:
            try:
                print(f"🎥 Starting A4Tech camera with rotation and stable settings...")
                
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
                
                # Configure camera with stable settings
                print("⚙️  Configuring A4Tech camera for stability...")
                
                # Set resolution to capture landscape then rotate to 480x640 portrait
                width_set = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                height_set = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Use stable frame rate
                fps_set = self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Minimal buffer for lower latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Use MJPG for better compatibility
                fourcc_set = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                # Stable exposure settings
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto exposure
                
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
                
                print(f"✅ A4Tech camera started successfully!")
                print(f"   📐 Resolution: {actual_width}x{actual_height}")
                print(f"   🔄 Rotation: {'ON' if self.rotation_enabled else 'OFF'}")
                print(f"   🎬 Frame rate: {actual_fps} fps")
                print(f"   📊 Stable frames: {stable_frames}/10")
                
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
        """Get rotated frame from A4Tech camera with validation"""
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
                
                # Apply 90° clockwise rotation only (no cropping for capture)
                if self.rotation_enabled:
                    rotated_frame = self.rotate_frame_90_clockwise(frame)
                    if rotated_frame is not None:
                        frame = rotated_frame
                
                # Log every 600 frames (20 seconds at 30fps)
                if self.frame_count % 600 == 0:
                    print(f"📹 A4Tech camera: {self.frame_count} frames ({frame.shape[1]}x{frame.shape[0]})")
                
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
        """Capture current frame and save with dimension logging"""
        try:
            frame = self.get_current_frame()
            if frame is not None:
                # Log frame dimensions before saving
                height, width = frame.shape[:2]
                print(f"📸 A4Tech Capture Dimensions:")
                print(f"   🖼️  Frame Shape: {frame.shape} (H x W x C)")
                print(f"   📐 Resolution: {width} x {height} pixels")
                print(f"   🔄 Rotation: {'Applied' if self.rotation_enabled else 'None'}")
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
                        
                        print(f"✅ A4Tech Image Saved Successfully:")
                        print(f"   📁 File: {os.path.basename(filepath)}")
                        print(f"   📐 Saved Size: {saved_width} x {saved_height} pixels")
                        print(f"   💾 File Size: {file_size / 1024:.1f} KB")
                        
                        self.last_captured_dimensions = {
                            'width': saved_width,
                            'height': saved_height,
                            'file_size': file_size,
                            'filepath': filepath
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
        """Get A4Tech camera status with processing info"""
        return {
            'is_running': self.is_running,
            'has_frame': self.current_frame is not None,
            'last_error': self.last_error,
            'camera_device': f'/dev/video{self.camera_device}',
            'camera_type': 'A4Tech FHD 1080P PC Camera',
            'frames_captured': self.frame_count,
            'rotation_enabled': self.rotation_enabled,
            'cropping_enabled': self.cropping_enabled,
            'processing': f"{'Rotation' if self.rotation_enabled else ''}{' + ' if self.rotation_enabled and self.cropping_enabled else ''}{'Cropping' if self.cropping_enabled else ''}",
            'last_capture': self.last_captured_dimensions
        }

def camera_thread_worker(camera_manager):
    """Optimized worker for A4Tech camera with processing"""
    print("🚀 Starting A4Tech camera worker thread with processing...")
    
    # Start camera
    if not camera_manager.start_camera():
        print("❌ Failed to start A4Tech camera")
        return
    
    print("🎬 A4Tech camera thread running with rotation and cropping...")
    consecutive_failures = 0
    max_failures = 10
    
    while True:
        try:
            # Get processed frame (rotated and cropped)
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
                        print("✅ A4Tech camera restarted with processing")
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
    print("🏁 A4Tech camera thread finished")
