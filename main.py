# main.py - Complete implementation with 4-Step AI Pipeline and Distance Sensor integration
# FIXED: Complete integration of ImageService, AIService with metadata support

from app import create_app
from app.services.camera_service import CameraManager, camera_thread_worker
from app.services.image_service import ImageService
from app.services.ai_service import AIService
from app.services.distance_service import DistanceService
from app.routes.camera_routes import init_camera_routes
from app.routes.image_routes import init_image_routes
from app.routes.ai_routes import init_ai_routes
from app.routes.sensor_routes import init_sensor_routes
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
        self.root.title("Rebar Vista 4-Step Analysis - 480x640 (Analyzed Images Only)")
        self.root.geometry("520x780")  # Slightly taller for distance display
        self.root.configure(bg='#2c3e50')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label with 4-step pipeline indicator
        title_label = tk.Label(main_frame, 
                              text="Rebar Vista 4-Step AI Pipeline (480x640)\nDetection → Intersections → Polygon → Cement\nSave Mode: Analyzed Images Only", 
                              font=('Arial', 12, 'bold'), 
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
        
        # Status label with 4-step pipeline info
        self.status_label = tk.Label(main_frame, 
                                   text="Initializing 4-step AI pipeline and distance sensor...", 
                                   font=('Arial', 10), 
                                   bg='#2c3e50', fg='#ecf0f1')
        self.status_label.pack(pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
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
        self.ai_test_btn = tk.Button(button_frame, text="Test 4-Step AI", 
                                   command=self.test_ai,
                                   bg='#9b59b6', fg='white', 
                                   font=('Arial', 12, 'bold'),
                                   padx=15, pady=5)
        self.ai_test_btn.pack(side=tk.LEFT, padx=5)
        
        # Info display with 4-step pipeline
        self.info_label = tk.Label(main_frame, 
                                 text="Format: 480x640 | Distance: Checking | Pipeline: 4-Step | Save: Analyzed Only", 
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
                        text=f"480x640 | Distance: {reading['distance_text']} | Range: {optimal_range} | Pipeline: 4-Step | Save: Analyzed Only"
                    )
                    
                else:
                    # Error case
                    self.distance_label.configure(text="Distance: ERROR")
                    self.distance_status_label.configure(text="ERROR", bg='#e74c3c', fg='white')
                    
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
    
    def test_ai(self):
        """Test 4-step AI pipeline with current camera frame"""
        if self.camera_manager:
            current_frame = self.camera_manager.get_current_frame()
            
            if current_frame is not None:
                self.status_label.configure(text="Testing 4-step AI pipeline with current frame...")
                print("🧪 Testing 4-step AI pipeline with current camera frame")
                print("📝 NOTE: 4-step AI test will save visualization images if successful")
            else:
                self.status_label.configure(text="No camera frame available for 4-step AI test")
        else:
            self.status_label.configure(text="Camera manager not available for 4-step AI test")
    
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
                    self.status_label.configure(text="4-Step AI Pipeline Ready - 480x640 (Analyzed Images Only)")
                    
                except Exception as e:
                    print(f"Tkinter display error: {e}")
                    self.status_label.configure(text="Display error - check camera")
            else:
                self.status_label.configure(text="No camera feed available for 4-step analysis")
        
        if self.is_running:
            # Schedule next update - 30ms for smooth display
            self.root.after(30, self.update_frame)
    
    def toggle_camera(self):
        if self.camera_manager.is_running:
            self.camera_manager.stop_camera()
            self.toggle_btn.configure(text="Start Camera", bg='#27ae60')
            self.status_label.configure(text="Camera stopped - 4-step AI pipeline unavailable")
        else:
            if self.camera_manager.start_camera():
                self.toggle_btn.configure(text="Stop Camera", bg='#e74c3c')
                self.status_label.configure(text="Camera started - 4-step AI pipeline ready (480x640)")
            else:
                self.status_label.configure(text="Failed to start camera - 4-step AI pipeline unavailable")
    
    def on_closing(self):
        print("Closing Tkinter camera window...")
        self.is_running = False
        if self.camera_manager:
            # Don't stop camera manager here - let main process handle it
            pass
        self.root.destroy()
    
    def start(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("Starting Tkinter 480x640 camera window with 4-step AI pipeline...")
        self.root.mainloop()

def start_tkinter_window(camera_manager, distance_service):
    """Start the tkinter camera window for 480x640 display with distance sensor"""
    try:
        print("Initializing Tkinter 480x640 camera interface with 4-step pipeline...")
        tk_camera = TkinterCameraFrame(camera_manager, distance_service)
        tk_camera.start()
    except Exception as e:
        print(f"Error starting Tkinter window: {e}")

def integrate_services(camera_manager, image_service, ai_service, distance_service):
    """Integrate all services with proper dependencies for 4-step pipeline"""
    try:
        print("🔗 Integrating services for 4-step AI pipeline...")
        
        # CRITICAL: Integrate AI service with image service for metadata
        ai_service.image_service = image_service
        print("✅ AI service integrated with image service for metadata")
        
        # Test service integrations
        ai_status = ai_service.get_model_status()
        print(f"✅ AI service status: {ai_status['model_type']}")
        
        try:
            image_stats = image_service.get_storage_stats()
            if image_stats['success']:
                print(f"✅ Image service ready: {image_stats['stats']['total_files']} files")
            else:
                print(f"⚠️  Image service warning: {image_stats['error']}")
        except Exception as e:
            print(f"⚠️  Image service integration warning: {e}")
        
        distance_status = distance_service.get_sensor_status()
        print(f"✅ Distance service ready: {'active' if distance_status['is_running'] else 'inactive'}")
        
        print("✅ All services integrated successfully for 4-step pipeline")
        return True
        
    except Exception as e:
        print(f"❌ Error integrating services: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Continuing with limited service integration...")
        return False

def main():
    try:
        print("STARTING REBAR VISTA 4-STEP AI ANALYSIS PIPELINE")
        print("=" * 70)
        print("🤖 Pipeline: Detection → Intersections → Polygon → Cement")
        print("📝 Save Mode: Only analyzed images with 4-step visualizations")
        print("🎯 Expected: 2 vertical + 11 horizontal rebars = 13 detections")
        print("=" * 70)
        
        # Initialize services
        print("Initializing camera, image, AI, and distance services...")
        camera_manager = CameraManager()
        image_service = ImageService()
        ai_service = AIService()
        distance_service = DistanceService()
        
        # CRITICAL: Integrate services with dependencies
        print("Integrating services for 4-step pipeline...")
        if not integrate_services(camera_manager, image_service, ai_service, distance_service):
            print("⚠️  Service integration had issues, continuing with reduced functionality")
        
        # Create Flask app with services
        print("Creating Flask web application...")
        app = create_app()
        
        # Initialize routes with services AFTER app creation
        with app.app_context():
            init_camera_routes(camera_manager, image_service)
            init_image_routes(image_service)
            init_ai_routes(ai_service, camera_manager)  # AI routes have camera access
            init_sensor_routes(distance_service)
            print("Flask routes initialized (4-step AI pipeline ready)")
        
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
        
        print("Starting Tkinter 480x640 display window...")
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
        print("REBAR VISTA 4-STEP AI PIPELINE READY")
        print("=" * 70)
        print("🌐 Web Interface: Camera with 4-step AI analysis display")
        print("🖥️  Tkinter Window: Live 480x640 feed with distance overlay")
        print("🤖 AI Analysis: Simplified 4-step pipeline with step visualization")
        print("📏 Distance Sensor: HC-SR04 optimal positioning (160-200cm)")
        print("💾 Save Mode: ONLY analyzed images with 4-step visualizations")
        print("📚 Gallery: Shows analyzed images with complete metadata")
        print("=" * 70)
        
        # Print AI service status
        ai_status = ai_service.get_model_status()
        print("\n=== 4-Step AI Pipeline Status ===")
        print(f"Detectron2 Available: {'✅' if ai_status['detectron2_available'] else '❌'}")
        print(f"Model Loaded: {'✅' if ai_status['model_loaded'] else '❌'}")
        print(f"Model Path: {ai_status['model_path']}")
        print(f"Model Exists: {'✅' if ai_status['model_exists'] else '❌'}")
        print(f"Classes (Fixed): {ai_status['class_names']}")
        print(f"Expected Detections:")
        print(f"  - Front Vertical: {ai_status['expected_detections']['front_vertical']}")
        print(f"  - Front Horizontal: {ai_status['expected_detections']['front_horizontal']}")
        print(f"  - Total Expected: {ai_status['expected_detections']['total']}")
        print(f"Detection Threshold: {ai_status['threshold']}")
        print(f"Pipeline Type: {ai_status['model_type']}")
        if not ai_status['model_loaded']:
            print("⚠️  AI will use placeholder 4-step results until real model is available")
            print("   Placeholder will show expected pipeline visualization")
        print("==================================")
        
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
        print("==============================")
        
        # Print enhanced image service info
        try:
            image_stats = image_service.get_storage_stats()
            if image_stats['success']:
                stats = image_stats['stats']
                print("\n=== Enhanced Image Storage Status ===")
                print(f"Total Image Files: {stats['total_files']}")
                print(f"Analyzed Images (Gallery): {stats['analyzed_files']} ({stats['analyzed_size_kb']} KB)")
                print(f"Original Images (Hidden): {stats['original_files']} ({stats['original_size_kb']} KB)")
                print(f"Step Images (4-Step): {stats['step_files']} ({stats['step_size_kb']} KB)")
                print(f"Metadata Files: {stats['metadata_files']} ({stats['metadata_size_kb']} KB)")
                print(f"Total Storage: {stats['total_size_kb']} KB")
                print(f"Gallery Shows: {stats['gallery_shows']} analyzed images")
                print(f"Hidden from Gallery: {stats['hidden_from_gallery']} files")
                if stats['original_files'] > 0:
                    print(f"💡 Cleanup available: {stats['original_files']} hidden original images")
                print("=====================================\n")
        except Exception as e:
            print(f"⚠️  Could not get enhanced image storage stats: {e}\n")
        
        # Check SSL certificates and start server
        if not os.path.exists(config.SSL_CERT_PATH):
            print(f"SSL certificate not found at {config.SSL_CERT_PATH}")
            print("Running HTTP server (no SSL)...")
            print(f"🌐 Access via: http://{config.current_ip}:{config.PORT}")
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
            print(f"🌐 Access via: https://{config.current_ip}:{config.PORT}")
            print(f"📱 Mobile access: https://{config.current_ip}:{config.PORT}")
            print(f"🏠 Local access: https://localhost:{config.PORT}")
            app.run(
                host=config.HOST,
                port=config.PORT,
                ssl_context=config.ssl_context,
                use_reloader=False,
                threaded=True,
                debug=config.DEBUG
            )
            
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("SHUTTING DOWN REBAR VISTA 4-STEP PIPELINE")
        print("=" * 50)
        
        # Clean shutdown of services
        try:
            if 'distance_service' in locals():
                distance_service.stop_monitoring()
                print("✅ Distance sensor monitoring stopped")
        except Exception as e:
            print(f"⚠️  Error stopping distance service: {e}")
        
        try:
            if 'camera_manager' in locals():
                camera_manager.stop_camera()
                print("✅ Camera stopped")
        except Exception as e:
            print(f"⚠️  Error stopping camera: {e}")
        
        print("=" * 50)
        print("REBAR VISTA 4-STEP PIPELINE SHUTDOWN COMPLETE")
        print("Thank you for using Rebar Vista!")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error starting 4-step pipeline application: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 50)
        print("TROUBLESHOOTING TIPS:")
        print("1. Check model file exists: /home/team10/RebarWeb/app/model/model_final.pth")
        print("2. Verify 2 classes: front_horizontal, front_vertical")
        print("3. Ensure all required files are updated")
        print("4. Check SSL certificates if HTTPS fails")
        print("=" * 50)

if __name__ == '__main__':
    print("=" * 70)
    print("REBAR VISTA - 4-STEP AI ANALYSIS PIPELINE")
    print("=" * 70)
    print("🤖 AI-powered rebar detection with simplified 4-step visualization")
    print("📐 Portrait 480x640 image processing with Detectron2 and HC-SR04")
    print("🎯 Expected detections: 2 vertical + 11 horizontal rebars = 13 total")
    print("🔄 Pipeline: Detection → Intersections → Polygon → Cement Mixture")
    print("💾 Save Mode: Only analyzed images with 4-step visualizations")
    print("📚 Gallery: Enhanced metadata with complete analysis details")
    print("🌐 HTTPS Support: SSL certificates for secure mobile access")
    print("=" * 70)
    print("")
    main()
