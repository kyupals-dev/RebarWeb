# main.py - Complete implementation with AI service and Distance Sensor integration
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
        self.root.title("Rebar Vista Camera Feed - 480x640 with Distance Sensor")
        self.root.geometry("520x780")  # Slightly taller for distance display
        self.root.configure(bg='#2c3e50')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label
        title_label = tk.Label(main_frame, text="Live Rebar Vista Camera Feed (480x640) + Distance", 
                              font=('Arial', 16, 'bold'), 
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
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Initializing Rebar Vista camera and distance sensor...", 
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
        
        # Distance test button
        self.distance_btn = tk.Button(button_frame, text="Test Distance", 
                                    command=self.test_distance,
                                    bg='#f39c12', fg='white', 
                                    font=('Arial', 12, 'bold'),
                                    padx=20, pady=5)
        self.distance_btn.pack(side=tk.LEFT, padx=5)
        
        # Info display
        self.info_label = tk.Label(main_frame, text="Format: 480x640 Portrait | Distance: Checking | Status: Ready", 
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
                        text=f"Format: 480x640 Portrait | Distance: {reading['distance_text']} | Range: {optimal_range} | Status: {status_text}"
                    )
                    
                else:
                    # Error case
                    self.distance_label.configure(text="Distance: ERROR")
                    self.distance_status_label.configure(text="ERROR", bg='#e74c3c', fg='white')
                    self.info_label.configure(
                        text="Format: 480x640 Portrait | Distance: ERROR | Status: Sensor unavailable"
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
                    simulation_mode = test_result.get('simulation_mode', False)
                    
                    mode_text = " (SIMULATION)" if simulation_mode else ""
                    self.status_label.configure(
                        text=f"Distance test passed: {avg_distance:.1f}cm average from {readings_count} readings{mode_text}"
                    )
                else:
                    error = test_result.get('error', 'Unknown error')
                    self.status_label.configure(text=f"Distance test failed: {error}")
                    
            except Exception as e:
                self.status_label.configure(text=f"Distance test error: {str(e)}")
        else:
            self.status_label.configure(text="Distance service not available")
    
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
                    self.status_label.configure(text="Rebar Vista Active - 480x640 Portrait Feed with Distance Sensor")
                    
                except Exception as e:
                    print(f"Tkinter display error: {e}")
                    self.status_label.configure(text="Display error - check camera")
            else:
                self.status_label.configure(text="No Rebar Vista camera feed available")
        
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
                    
                    # Get current distance reading for metadata
                    distance_info = ""
                    if self.distance_service:
                        reading = self.distance_service.get_current_reading()
                        if reading['success']:
                            distance_info = f"_{reading['distance']:.0f}cm_{reading['status']}"
                    
                    # Generate filename with timestamp and distance info
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    filename = f'tkinter_480x640{distance_info}_{timestamp}.jpg'
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
                            
                            print("Tkinter captured 480x640 image with distance:")
                            print(f"   File: {filename}")
                            print(f"   Dimensions: {saved_width}x{saved_height}")
                            print(f"   File size: {file_size / 1024:.1f} KB")
                            if distance_info:
                                print(f"   Distance info: {distance_info}")
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
            self.status_label.configure(text="Rebar Vista camera stopped")
        else:
            if self.camera_manager.start_camera():
                self.toggle_btn.configure(text="Stop Camera", bg='#e74c3c')
                self.status_label.configure(text="Rebar Vista camera started (480x640)")
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
        print("Starting Tkinter 480x640 camera window with distance sensor...")
        self.root.mainloop()

def start_tkinter_window(camera_manager, distance_service):
    """Start the tkinter camera window for 480x640 display with distance sensor"""
    try:
        print("Initializing Tkinter 480x640 camera interface with distance sensor...")
        tk_camera = TkinterCameraFrame(camera_manager, distance_service)
        tk_camera.start()
    except Exception as e:
        print(f"Error starting Tkinter window: {e}")

def main():
    try:
        print("Starting Rebar Vista with AI and Distance Sensor Integration...")
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
            init_ai_routes(ai_service)
            init_sensor_routes(distance_service)  # Initialize Sensor Routes
            print("Flask routes initialized (including AI and sensor routes)")
        
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
        
        print("Starting Tkinter 480x640 display window with distance sensor...")
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
        print("REBAR VISTA READY - AI-POWERED ANALYSIS WITH DISTANCE SENSOR")
        print("Web interface: Camera display at 480x640 with AI analysis and distance")
        print("Tkinter window: Live 480x640 camera feed with distance overlay")
        print("AI Analysis: Detectron2 rebar detection and measurement")
        print("Distance Sensor: HC-SR04 optimal positioning (160-200cm)")
        print("Image format: All images saved as 480x640")
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
        if not ai_status['model_loaded']:
            print("⚠️  AI will use placeholder results until model is available")
        print("========================")
        
        # Print distance sensor status
        distance_status = distance_service.get_sensor_status()
        print("\n=== Distance Sensor Status ===")
        print(f"GPIO Available: {'✅' if distance_status['gpio_available'] else '❌'}")
        print(f"Sensor Available: {'✅' if distance_status['sensor_available'] else '❌'}")
        print(f"Monitoring Running: {'✅' if distance_status['is_running'] else '❌'}")
        print(f"Simulation Mode: {'✅' if distance_status['simulation_mode'] else '❌'}")
        print(f"GPIO Pins: TRIG={distance_status['gpio_pins']['trigger']}, ECHO={distance_status['gpio_pins']['echo']}")
        print(f"Optimal Range: {distance_status['optimal_range']['min']}-{distance_status['optimal_range']['max']}{distance_status['optimal_range']['unit']}")
        if distance_status['last_error']:
            print(f"⚠️  Last Error: {distance_status['last_error']}")
        if distance_status['simulation_mode']:
            print("⚠️  Distance sensor using simulation mode")
        print("==============================\n")
        
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
    print("REBAR VISTA - AI-POWERED REBAR DETECTION WITH DISTANCE SENSOR")
    print("AI-powered rebar detection and optimal positioning system")
    print("Portrait 480x640 image processing with Detectron2 and HC-SR04")
    print("Optimal distance range: 160-200cm for best analysis results")
    print("")
    main()
