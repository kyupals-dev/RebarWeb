# main.py - Complete implementation with AI service and Distance Sensor integration
# MODIFIED: AI routes now receive camera manager for direct frame access

from app import create_app
from app.services.camera_service import CameraManager, camera_thread_worker
from app.services.image_service import ImageService
from app.services.ai_service import AIService
from app.services.distance_service import DistanceService  # Import Distance Service
from app.routes.camera_routes import init_camera_routes
from app.routes.image_routes import init_image_routes
from app.routes.ai_routes import init_ai_routes
from app.routes.sensor_routes import init_sensor_routes  # Import Sensor Routes
from app.utils.config import config
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
from datetime import datetime

class TkinterCameraFrame:
    """Tkinter camera display window for 480 x 640 portrait format with distance sensor"""
    def __init__(self, camera_manager, distance_service):
        self.camera_manager = camera_manager
        self.distance_service = distance_service
        self.root = tk.Tk()
        self.root.title("Rebar Vista Camera Feed - 480x640 (Analyzed Images Only)")
        self.root.geometry("520x780")  # Slightly taller for distance display
        self.root.configure(bg='#2c3e50')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label with save mode indicator
        title_label = tk.Label(main_frame, 
                              text="Rebar Vista Camera (480x640) + Distance\nSave Mode: Analyzed Images Only", 
                              font=('Arial', 14, 'bold'), 
                              bg='#2c3e50', fg='white')
        title_label.pack(pady=(0, 5))
        
        # Distance display frame
        distance_frame = tk.Frame(main_frame, bg='#2c3e50')
        distance_frame.pack(pady=5)
        
        # Distance labels
        self.distance_label = tk.Label(distance_frame, text="Distance: --cm", 
                                     font=('Arial', 12, 'bold'), 
                                     bg='#2c3e50', fg='white')
        self.distance_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.distance_status_label = tk.Label(distance_frame, text="CHECKING", 
                                            font=('Arial', 11, 'bold'), 
                                            bg='#95a5a6', fg='white',
                                            padx=10, pady=2)
        self.distance_status_label.pack(side=tk.LEFT)
        
        # Camera display label - PORTRAIT 480x640
        self.camera_label = tk.Label(main_frame, bg='black', 
                                   width=480, height=640)
        self.camera_label.pack(pady=5)
        
        # Status label with save mode info
        self.status_label = tk.Label(main_frame, 
                                   text="Initializing camera and distance sensor (analyzed images only)...", 
                                   font=('Arial', 10), 
                                   bg='#2c3e50', fg='#ecf0f1')
        self.status_label.pack(pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Note: Capture button removed since we don't save originals via Tkinter
        # The web interface handles capture + AI analysis
        
        # Toggle camera button
        self.toggle_btn = tk.Button(button_frame, text="Stop Camera", 
                                  command=self.toggle_camera,
                                  bg='#e74c3c', fg='white', 
                                  font=('Arial', 12, 'bold'),
                                  padx=20, pady=5)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Distance test button
        self.distance_btn = tk.Button(button_frame, text="Test Distance", 
                                    command=self.test_distance,
                                    bg='#f39c12', fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    padx=20, pady=5)
        self.distance_btn.pack(side=tk.LEFT, padx=5)
        
        # AI test button
        self.ai_test_btn = tk.Button(button_frame, text="Test AI", 
                                   command=self.test_ai,
                                   bg='#9b59b6', fg='white', 
                                   font=('Arial', 12, 'bold'),
                                   padx=20, pady=5)
        self.ai_test_btn.pack(side=tk.LEFT, padx=5)
        
        # Info display with save mode
        self.info_label = tk.Label(main_frame, 
                                 text="Format: 480x640 Portrait | Distance: Checking | Save: Analyzed Only", 
                                 font=('Arial', 9), 
                                 bg='#2c3e50', fg='#bdc3c7')
        self.info_label.pack(pady=(5, 0))
        
        self.is_running = True
        self.frame_count = 0
        self.last_distance_update = 0
        self.update_frame()
        self.update_distance()
        
    def update_distance(self):
        """Update distance display"""
        if self.is_running and self.distance_service:
            try:
                reading = self.distance_service.get_current_reading()
                
                if reading['success']:
                    # Update distance text
                    distance_text = f"Distance: {reading['distance_text']}"
                    self.distance_label.configure(text=distance_text)
                    
                    # Update status with color coding
                    status_text = reading['status_text']
                    status_color = reading['status_color']
                    
                    self.distance_status_label.configure(text=status_text)
                    
                    # Set background color based on status
                    if status_color == 'green':
                        self.distance_status_label.configure(bg='#2ecc71')  # Optimal
                    elif status_color == 'red':
                        self.distance_status_label.configure(bg='#e74c3c')  # Too close/Error
                    elif status_color == 'yellow':
                        self.distance_status_label.configure(bg='#f1c40f', fg='#2c3e50')  # Too far
                    else:
                        self.distance_status_label.configure(bg='#95a5a6', fg='white')  # Unknown
                    
                    # Update info label
                    optimal_range = reading.get('optimal_range', '160-200cm')
                    self.info_label.configure(
                        text=f"Format: 480x640 | Distance: {reading['distance_text']} | Range: {optimal_range} | Save: Analyzed Only"
                    )
                    
                else:
                    # Error case
                    self.distance_label.configure(text="Distance: ERROR")
                    self.distance_status_label.configure(text="ERROR", bg='#e74c3c', fg='white')
                    self.info_label.configure(
                        text="Format: 480x640 | Distance: ERROR | Save: Analyzed Only"
                    )
                    
            except Exception as e:
                print(f"Tkinter distance update error: {e}")
                self.distance_label.configure(text="Distance: ERROR")
                self.distance_status_label.configure(text="ERROR", bg='#e74c3c', fg='white')
        
        if self.is_running:
            # Update every 500ms (matching the distance service update rate)
            self.root.after(500, self.update_distance)
    
    def test_distance(self):
        """Test distance sensor functionality"""
        if self.distance_service:
            try:
                test_result = self.distance_service.test_sensor()
                
                if test_result['success']:
                    avg_distance = test_result.get('average_distance', 0)
                    readings_count = test_result.get('readings_count', 0)
                    
                    self.status_label.configure(
                        text=f"Distance test passed: {avg_distance:.1f}cm average from {readings_count} readings"
                    )
                else:
                    error = test_result.get('error', 'Unknown error')
                    self.status_label.configure(text=f"Distance test failed: {error}")
                
            except Exception as e:
                self.status_label.configure(text=f"Distance test error: {str(e)}")
        else:
            self.status_label.configure(text="Distance service not available")
                    
    def test_ai(self):
        """Test AI service with current camera frame"""
        if self.camera_manager:
            current_frame = self.camera_manager.get_current_frame()
            
            if current_frame is not None:
                self.status_label.configure(text="Testing AI with current frame (analyzed image only)...")
                print("🧪 Testing AI service with current camera frame")
                print("📝 NOTE: AI test will save only analyzed image if successful")
            else:
                self.status_label.configure(text="No camera frame available for AI test")
        else:
            self.status_label.configure(text="Camera manager not available for AI test")
    
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
                    self.status_label.configure(text="Rebar Vista Active - 480x640 (Analyzed Images Only Mode)")
                    
                except Exception as e:
                    print(f"Tkinter display error: {e}")
                    self.status_label.configure(text="Display error - check camera")
            else:
                self.status_label.configure(text="No Rebar Vista camera feed available")
        
        if self.is_running:
            # Schedule next update - 30ms for smooth display
            self.root.after(30, self.update_frame)
    
    def toggle_camera(self):
        if self.camera_manager.is_running:
            self.camera_manager.stop_camera()
            self.toggle_btn.configure(text="Start Camera", bg='#27ae60')
            self.status_label.configure(text="Rebar Vista camera stopped")
        else:
            if self.camera_manager.start_camera():
                self.toggle_btn.configure(text="Stop Camera", bg='#e74c3c')
                self.status_label.configure(text="Rebar Vista camera started (480x640, analyzed images only)")
            else:
                self.status_label.configure(text="Failed to start Rebar Vista camera")
    
    def on_closing(self):
        print("Closing Tkinter camera window...")
        self.is_running = False
        if self.camera_manager:
            # Don't stop camera manager here - let main process handle it
            pass
        self.root.destroy()
    
    def start(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("Starting Tkinter 480x640 camera window (analyzed images only mode)...")
        self.root.mainloop()

def start_tkinter_window(camera_manager, distance_service):
    """Start the tkinter camera window for 480x640 display with distance sensor"""
    try:
        print("Initializing Tkinter 480x640 camera interface (analyzed images only)...")
        tk_camera = TkinterCameraFrame(camera_manager, distance_service)
        tk_camera.start()
    except Exception as e:
        print(f"Error starting Tkinter window: {e}")

def main():
    try:
        print("Starting Rebar Vista with AI and Distance Sensor Integration...")
        print("📝 SAVE MODE: Only analyzed images with AI overlays will be saved")
        print("=" * 70)
        
        # Initialize services
        print("Initializing camera, image, AI, and distance services...")
        camera_manager = CameraManager()
        image_service = ImageService()
        ai_service = AIService()
        distance_service = DistanceService()  # Initialize Distance Service
        
        # Create Flask app with services
        print("Creating Flask web application...")
        app = create_app()
        
        # Initialize routes with services AFTER app creation
        with app.app_context():
            init_camera_routes(camera_manager, image_service)
            init_image_routes(image_service)
            init_ai_routes(ai_service, camera_manager)  # MODIFIED: Pass camera_manager to AI routes
            init_sensor_routes(distance_service)  # Initialize Sensor Routes
            print("Flask routes initialized (AI routes have camera access)")
        
        # Ensure upload folder exists
        config.ensure_upload_folder()
        print(f"Upload folder ready: {config.UPLOAD_FOLDER}")
        
        # Ensure model folder exists
        model_folder = "/home/team10/RebarWeb/app/model"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print(f"Created model folder: {model_folder}")
        else:
            print(f"Model folder ready: {model_folder}")
        
        print("Starting camera thread for 480x640 capture...")
        # Start camera thread
        camera_thread = threading.Thread(
            target=camera_thread_worker, 
            args=(camera_manager,), 
            daemon=True
        )
        camera_thread.start()
        
        print("Starting distance sensor monitoring...")
        # Start distance monitoring
        distance_service.start_monitoring()
        
        print("Starting Tkinter 480x640 display window (analyzed images only)...")
        # Start tkinter window in separate thread
        tkinter_thread = threading.Thread(
            target=start_tkinter_window, 
            args=(camera_manager, distance_service), 
            daemon=True
        )
        tkinter_thread.start()
        
        print("Starting Flask web server...")
        print("Available routes:")
        for rule in app.url_map.iter_rules():
            methods = ', '.join(rule.methods - {'HEAD', 'OPTIONS'})
            print(f"  {rule.endpoint}: {rule.rule} [{methods}]")
        
        print("=" * 70)
        print("REBAR VISTA READY - AI-POWERED ANALYSIS (ANALYZED IMAGES ONLY)")
        print("Web interface: Camera display with AI analysis (saves analyzed images only)")
        print("Tkinter window: Live 480x640 camera feed with distance overlay")
        print("AI Analysis: Detectron2 rebar detection and measurement")
        print("Distance Sensor: HC-SR04 optimal positioning (160-200cm)")
        print("Save Mode: ONLY analyzed images with AI overlays are saved")
        print("Gallery: Shows only analyzed images (no duplicates)")
        print("=" * 70)
        
        # Print AI service status
        ai_status = ai_service.get_model_status()
        print("\n=== AI Service Status ===")
        print(f"Detectron2 Available: {'✅' if ai_status['detectron2_available'] else '❌'}")
        print(f"Model Loaded: {'✅' if ai_status['model_loaded'] else '❌'}")
        print(f"Model Path: {ai_status['model_path']}")
        print(f"Model Exists: {'✅' if ai_status['model_exists'] else '❌'}")
        print(f"Classes: {ai_status['class_names']}")
        print(f"Detection Threshold: {ai_status['threshold']}")
        print(f"Save Mode: {ai_status.get('save_mode', 'analyzed_images_only')}")
        if not ai_status['model_loaded']:
            print("⚠️  AI will use placeholder results until model is available")
        print("========================")
        
        # Print distance sensor status
        distance_status = distance_service.get_sensor_status()
        print("\n=== Distance Sensor Status ===")
        print(f"GPIO Available: {'✅' if distance_status['gpio_available'] else '❌'}")
        print(f"Sensor Available: {'✅' if distance_status['sensor_available'] else '❌'}")
        print(f"Monitoring Running: {'✅' if distance_status['is_running'] else '❌'}")
        print(f"GPIO Pins: TRIG={distance_status['gpio_pins']['trigger']}, ECHO={distance_status['gpio_pins']['echo']}")
        print(f"Optimal Range: {distance_status['optimal_range']['min']}-{distance_status['optimal_range']['max']}{distance_status['optimal_range']['unit']}")
        if distance_status['last_error']:
            print(f"⚠️  Last Error: {distance_status['last_error']}")
        
        # Print image service info
        image_stats = image_service.get_storage_stats()
        if image_stats['success']:
            stats = image_stats['stats']
            print("\n=== Image Storage Status ===")
            print(f"Total Files: {stats['total_files']}")
            print(f"Analyzed Images (Gallery): {stats['analyzed_files']} ({stats['analyzed_size_kb']} KB)")
            print(f"Original Images (Hidden): {stats['original_files']} ({stats['original_size_kb']} KB)")
            print(f"Total Storage: {stats['total_size_kb']} KB")
            if stats['original_files'] > 0:
                print(f"💡 Tip: Use /cleanup-originals endpoint to remove {stats['original_files']} hidden original images")
            print("============================\n")
        
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
        
        # Clean shutdown of services
        try:
            if 'distance_service' in locals():
                distance_service.stop_monitoring()
                print("Distance sensor monitoring stopped")
        except Exception as e:
            print(f"Error stopping distance service: {e}")
        
        try:
            if 'camera_manager' in locals():
                camera_manager.stop_camera()
                print("Camera stopped")
        except Exception as e:
            print(f"Error stopping camera: {e}")
        
        print("Goodbye!")
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("REBAR VISTA - AI-POWERED REBAR DETECTION (ANALYZED IMAGES ONLY)")
    print("AI-powered rebar detection with optimal positioning system")
    print("Portrait 480x640 image processing with Detectron2 and HC-SR04")
    print("Optimal distance range: 160-200cm for best analysis results")
    print("Save Mode: Only analyzed images with AI overlays are saved")
    print("")
    main()
