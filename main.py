# main.py - Complete 480x640 format implementation (Fixed)
from app import create_app
from app.services.camera_service import CameraManager, camera_thread_worker
from app.services.image_service import ImageService
from app.routes.camera_routes import init_camera_routes
from app.routes.image_routes import init_image_routes
from app.utils.config import config
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
from datetime import datetime

class TkinterCameraFrame:
    """Tkinter camera display window for 480x640 portrait format"""
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.root = tk.Tk()
        self.root.title("A4Tech Camera Feed - 480x640")
        self.root.geometry("520x720")  # Slightly larger for UI elements
        self.root.configure(bg='#2c3e50')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label
        title_label = tk.Label(main_frame, text="Live A4Tech Camera Feed (480x640)", 
                              font=('Arial', 16, 'bold'), 
                              bg='#2c3e50', fg='white')
        title_label.pack(pady=(0, 10))
        
        # Camera display label - PORTRAIT 480x640
        self.camera_label = tk.Label(main_frame, bg='black', 
                                   width=480, height=640)
        self.camera_label.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Initializing A4Tech camera (480x640)...", 
                                   font=('Arial', 10), 
                                   bg='#2c3e50', fg='#ecf0f1')
        self.status_label.pack(pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Capture button
        self.capture_btn = tk.Button(button_frame, text="Capture 480x640 Image", 
                                   command=self.capture_image,
                                   bg='#3498db', fg='white', 
                                   font=('Arial', 12, 'bold'),
                                   padx=20, pady=5)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Toggle camera button
        self.toggle_btn = tk.Button(button_frame, text="Stop Camera", 
                                  command=self.toggle_camera,
                                  bg='#e74c3c', fg='white', 
                                  font=('Arial', 12, 'bold'),
                                  padx=20, pady=5)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Info display
        self.info_label = tk.Label(main_frame, text="Format: 480x640 Portrait | Status: Ready", 
                                 font=('Arial', 9), 
                                 bg='#2c3e50', fg='#bdc3c7')
        self.info_label.pack(pady=(5, 0))
        
        self.is_running = True
        self.frame_count = 0
        self.update_frame()
        
    def update_frame(self):
        if self.is_running and self.camera_manager:
            current_frame = self.camera_manager.get_current_frame()
            
            if current_frame is not None:
                self.frame_count += 1
                
                # Validate and ensure frame is 480x640
                height, width = current_frame.shape[:2]
                
                if width != 480 or height != 640:
                    # Resize to 480x640 if not already
                    current_frame = cv2.resize(current_frame, (480, 640))
                    if self.frame_count % 100 == 1:  # Log occasionally
                        print(f"Tkinter: Resized frame from {width}x{height} to 480x640")
                
                try:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Convert to PhotoImage (already 480x640)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update the label
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo
                    
                    # Update status
                    self.status_label.configure(text="A4Tech Active - 480x640 Portrait Feed")
                    self.info_label.configure(text=f"Format: 480x640 Portrait | Frames: {self.frame_count}")
                    
                except Exception as e:
                    print(f"Tkinter display error: {e}")
                    self.status_label.configure(text="Display error - check camera")
            else:
                self.status_label.configure(text="No A4Tech camera feed available")
                self.info_label.configure(text="Format: 480x640 Portrait | Status: No Feed")
        
        if self.is_running:
            # Schedule next update - 30ms for smooth display
            self.root.after(30, self.update_frame)
    
    def capture_image(self):
        if self.camera_manager:
            current_frame = self.camera_manager.get_current_frame()
            
            if current_frame is not None:
                try:
                    # Ensure frame is exactly 480x640
                    height, width = current_frame.shape[:2]
                    if width != 480 or height != 640:
                        current_frame = cv2.resize(current_frame, (480, 640))
                        print(f"Capture: Resized from {width}x{height} to 480x640")
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    filename = f'tkinter_480x640_{timestamp}.jpg'
                    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
                    
                    # Save the 480x640 image
                    success = cv2.imwrite(filepath, current_frame)
                    
                    if success:
                        # Verify saved dimensions
                        saved_img = cv2.imread(filepath)
                        if saved_img is not None:
                            saved_height, saved_width = saved_img.shape[:2]
                            file_size = os.path.getsize(filepath)
                            
                            self.status_label.configure(text=f"480x640 image saved: {filename}")
                            self.info_label.configure(text=f"Saved: {saved_width}x{saved_height} | Size: {file_size/1024:.1f}KB")
                            
                            print("Tkinter captured 480x640 image:")
                            print(f"   File: {filename}")
                            print(f"   Dimensions: {saved_width}x{saved_height}")
                            print(f"   File size: {file_size / 1024:.1f} KB")
                        else:
                            self.status_label.configure(text="Could not verify saved image")
                    else:
                        self.status_label.configure(text="Failed to save 480x640 image")
                        
                except Exception as e:
                    print(f"Capture error: {e}")
                    self.status_label.configure(text=f"Capture failed: {str(e)}")
            else:
                self.status_label.configure(text="No frame available to capture")
    
    def toggle_camera(self):
        if self.camera_manager.is_running:
            self.camera_manager.stop_camera()
            self.toggle_btn.configure(text="Start Camera", bg='#27ae60')
            self.status_label.configure(text="A4Tech camera stopped")
            self.info_label.configure(text="Format: 480x640 Portrait | Status: Stopped")
        else:
            if self.camera_manager.start_camera():
                self.toggle_btn.configure(text="Stop Camera", bg='#e74c3c')
                self.status_label.configure(text="A4Tech camera started (480x640)")
                self.info_label.configure(text="Format: 480x640 Portrait | Status: Starting...")
            else:
                self.status_label.configure(text="Failed to start A4Tech camera")
                self.info_label.configure(text="Format: 480x640 Portrait | Status: Failed")
    
    def on_closing(self):
        print("Closing Tkinter camera window...")
        self.is_running = False
        if self.camera_manager:
            # Don't stop camera manager here - let main process handle it
            pass
        self.root.destroy()
    
    def start(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("Starting Tkinter 480x640 camera window...")
        self.root.mainloop()

def start_tkinter_window(camera_manager):
    """Start the tkinter camera window for 480x640 display"""
    try:
        print("Initializing Tkinter 480x640 camera interface...")
        tk_camera = TkinterCameraFrame(camera_manager)
        tk_camera.start()
    except Exception as e:
        print(f"Error starting Tkinter window: {e}")

def main():
    try:
        print("Starting Rebar Vista with 480x640 format...")
        print("=" * 60)
        
        # Initialize services first
        print("Initializing camera and image services...")
        camera_manager = CameraManager()
        image_service = ImageService()
        
        # Create Flask app with services
        print("Creating Flask web application...")
        app = create_app()
        
        # Initialize routes with services AFTER app creation
        with app.app_context():
            init_camera_routes(camera_manager, image_service)
            init_image_routes(image_service)
            print("Flask routes initialized")
        
        # Ensure upload folder exists
        config.ensure_upload_folder()
        print(f"Upload folder ready: {config.UPLOAD_FOLDER}")
        
        print("Starting camera thread for 480x640 capture...")
        # Start camera thread
        camera_thread = threading.Thread(
            target=camera_thread_worker, 
            args=(camera_manager,), 
            daemon=True
        )
        camera_thread.start()
        
        print("Starting Tkinter 480x640 display window...")
        # Start tkinter window in separate thread
        tkinter_thread = threading.Thread(
            target=start_tkinter_window, 
            args=(camera_manager,), 
            daemon=True
        )
        tkinter_thread.start()
        
        print("Starting Flask web server...")
        print("Available routes:")
        for rule in app.url_map.iter_rules():
            methods = ', '.join(rule.methods - {'HEAD', 'OPTIONS'})
            print(f"  {rule.endpoint}: {rule.rule} [{methods}]")
        
        print("=" * 60)
        print("REBAR VISTA READY - 480x640 FORMAT")
        print("Web interface: Camera display at 480x640")
        print("Tkinter window: Live 480x640 camera feed")
        print("Image capture: All images saved as 480x640")
        print("=" * 60)
        
        # Check SSL certificates and start server
        if not os.path.exists(config.SSL_CERT_PATH):
            print(f"SSL certificate not found at {config.SSL_CERT_PATH}")
            print("Running HTTP server (no SSL)...")
            app.run(
                host=config.HOST,
                port=config.PORT,
                use_reloader=False,
                threaded=True,
                debug=config.DEBUG
            )
        else:
            print(f"Using SSL certificates from {config.SSL_CERT_PATH}")
            print("Running HTTPS server...")
            app.run(
                host=config.HOST,
                port=config.PORT,
                ssl_context=config.ssl_context,
                use_reloader=False,
                threaded=True,
                debug=config.DEBUG
            )
            
    except KeyboardInterrupt:
        print("\nShutting down Rebar Vista...")
        print("Goodbye!")
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("REBAR VISTA - 480x640 FORMAT")
    print("AI-powered rebar detection system")
    print("Portrait 480x640 image processing")
    print("")
    main()
