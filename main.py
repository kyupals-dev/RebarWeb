# main.py - Fixed Pipeline Implementation for Raspberry Pi 5
# FIXED: Corrected method names and error handling

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
    """Tkinter camera display window for 480 x 640 portrait format with pipeline status"""
    def __init__(self, camera_manager, distance_service):
        self.camera_manager = camera_manager
        self.distance_service = distance_service
        self.root = tk.Tk()
        self.root.title("Rebar Vista Camera Feed - Pipeline Mode (480x640)")
        self.root.geometry("520x820")  # Taller for pipeline status
        self.root.configure(bg='#2c3e50')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label with pipeline mode indicator
        title_label = tk.Label(main_frame, 
                              text="Rebar Vista Camera (480x640) - Pipeline Mode\nSave Mode: Analyzed Images with 4-Step Visualization", 
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
                                            padx=10, pady=5, relief='raised')
        self.distance_status_label.pack(side=tk.LEFT)
        
        # Pipeline status frame
        pipeline_frame = tk.LabelFrame(main_frame, text="Pipeline Status", 
                                     font=('Arial', 10, 'bold'),
                                     bg='#2c3e50', fg='white')
        pipeline_frame.pack(pady=5, fill=tk.X)
        
        # Pipeline status labels
        self.pipeline_status_label = tk.Label(pipeline_frame, text="Ready for Analysis", 
                                            font=('Arial', 10), 
                                            bg='#2c3e50', fg='white')
        self.pipeline_status_label.pack(pady=2)
        
        self.detection_count_label = tk.Label(pipeline_frame, text="Expected: 2 verticals + 11 horizontals", 
                                            font=('Arial', 9), 
                                            bg='#2c3e50', fg='#bdc3c7')
        self.detection_count_label.pack(pady=1)
        
        # Camera display frame
        camera_frame = tk.Frame(main_frame, bg='black', relief='sunken', bd=2)
        camera_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Camera label (480x640 aspect ratio)
        self.camera_label = tk.Label(camera_frame, bg='black', text="Initializing Camera...", 
                                   fg='white', font=('Arial', 16))
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons frame
        controls_frame = tk.Frame(main_frame, bg='#2c3e50')
        controls_frame.pack(pady=10, fill=tk.X)
        
        # Pipeline analysis button
        self.analyze_button = tk.Button(controls_frame, text="Run Pipeline Analysis", 
                                      font=('Arial', 12, 'bold'),
                                      bg='#27ae60', fg='white', 
                                      activebackground='#2ecc71',
                                      command=self.trigger_pipeline_analysis,
                                      relief='raised', bd=3, padx=20, pady=8)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        # Status display button
        self.status_button = tk.Button(controls_frame, text="Show Status", 
                                     font=('Arial', 10),
                                     bg='#3498db', fg='white',
                                     activebackground='#5dade2',
                                     command=self.show_status,
                                     relief='raised', bd=2, padx=15, pady=5)
        self.status_button.pack(side=tk.LEFT, padx=5)
        
        # Gallery button
        self.gallery_button = tk.Button(controls_frame, text="Open Gallery", 
                                      font=('Arial', 10),
                                      bg='#9b59b6', fg='white',
                                      activebackground='#bb8fce',
                                      command=self.open_gallery,
                                      relief='raised', bd=2, padx=15, pady=5)
        self.gallery_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize variables
        self.current_frame = None
        self.is_running = False
        self.pipeline_analyzing = False
        
    def start(self):
        """Start the camera display loop"""
        print("🖥️ Starting Tkinter camera display with pipeline status...")
        self.is_running = True
        self.update_display()
        self.update_distance()
        self.update_pipeline_status()
        self.root.mainloop()
    
    def update_display(self):
        """Update camera display with current frame"""
        if not self.is_running:
            return
        
        try:
            # FIXED: Check if camera_manager has is_running attribute
            if self.camera_manager and hasattr(self.camera_manager, 'is_running') and self.camera_manager.is_running:
                # Get current frame
                frame = self.camera_manager.get_current_frame()
                
                if frame is not None:
                    self.current_frame = frame
                    
                    # Convert frame for Tkinter display
                    # Resize for display (maintain aspect ratio)
                    display_height = 480  # Fixed display height
                    aspect_ratio = frame.shape[1] / frame.shape[0]  # width/height
                    display_width = int(display_height * aspect_ratio)
                    
                    # Resize frame
                    display_frame = cv2.resize(frame, (display_width, display_height))
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update label
                    self.camera_label.configure(image=photo, text="")
                    self.camera_label.image = photo  # Keep a reference
                else:
                    self.camera_label.configure(image="", text="No Camera Frame", 
                                              fg='red', font=('Arial', 14))
            else:
                self.camera_label.configure(image="", text="Camera Not Running", 
                                          fg='orange', font=('Arial', 14))
        
        except Exception as e:
            print(f"⚠️ Tkinter display error: {e}")
            self.camera_label.configure(image="", text="Display Error", 
                                      fg='red', font=('Arial', 12))
        
        # Schedule next update
        if self.is_running:
            self.root.after(100, self.update_display)  # 10 FPS
    
    def update_distance(self):
        """Update distance sensor display"""
        if not self.is_running:
            return
        
        try:
            if self.distance_service:
                # FIXED: Use correct method name
                if hasattr(self.distance_service, 'get_current_reading'):
                    reading = self.distance_service.get_current_reading()
                elif hasattr(self.distance_service, 'get_latest_reading'):
                    reading = self.distance_service.get_latest_reading()
                else:
                    # Fallback: try to get status
                    reading = None
                    print("⚠️ Distance service method not found")
                
                if reading and reading.get('success'):
                    distance_text = reading.get('distance_text', '--cm')
                    status_text = reading.get('status_text', 'UNKNOWN')
                    status = reading.get('status', 'unknown')
                    
                    # Update distance label
                    self.distance_label.config(text=f"Distance: {distance_text}")
                    
                    # Update status label with color coding
                    self.distance_status_label.config(text=status_text)
                    
                    if status == 'optimal':
                        self.distance_status_label.config(bg='#27ae60')  # Green
                    elif status in ['too_close', 'too_far']:
                        self.distance_status_label.config(bg='#e74c3c')  # Red
                    else:
                        self.distance_status_label.config(bg='#f39c12')  # Orange
                else:
                    self.distance_label.config(text="Distance: --cm")
                    self.distance_status_label.config(text="NO SIGNAL", bg='#95a5a6')
            else:
                self.distance_label.config(text="Distance: UNAVAILABLE")
                self.distance_status_label.config(text="DISABLED", bg='#7f8c8d')
        
        except Exception as e:
            print(f"⚠️ Distance update error: {e}")
            self.distance_label.config(text="Distance: ERROR")
            self.distance_status_label.config(text="ERROR", bg='#e74c3c')
        
        # Schedule next update
        if self.is_running:
            self.root.after(500, self.update_distance)  # 2 Hz
    
    def update_pipeline_status(self):
        """Update pipeline analysis status"""
        if not self.is_running:
            return
        
        try:
            # Update pipeline status based on current state
            if self.pipeline_analyzing:
                self.pipeline_status_label.config(text="Running Pipeline Analysis...", fg='#f39c12')
                self.detection_count_label.config(text="Processing: Detection → Quadrants → Polygon → Cement")
            else:
                self.pipeline_status_label.config(text="Ready for Pipeline Analysis", fg='#27ae60')
                self.detection_count_label.config(text="Expected: 2 verticals + 11 horizontals")
        
        except Exception as e:
            print(f"⚠️ Pipeline status update error: {e}")
        
        # Schedule next update
        if self.is_running:
            self.root.after(1000, self.update_pipeline_status)  # 1 Hz
    
    def trigger_pipeline_analysis(self):
        """Trigger pipeline analysis via web API"""
        try:
            print("🔄 Triggering pipeline analysis from Tkinter interface...")
            self.pipeline_analyzing = True
            self.analyze_button.config(state='disabled', text="Analyzing...", bg='#f39c12')
            
            # Use threading to avoid blocking UI
            import requests
            import threading
            
            def run_analysis():
                try:
                    # Get the server URL from config
                    server_url = config.get_server_url() if hasattr(config, 'get_server_url') else f"https://localhost:{config.PORT}"
                    
                    # Call the web API for pipeline analysis
                    response = requests.post(f'{server_url}/analyze-rebar', 
                                           json={'source': 'tkinter_interface'},
                                           verify=False, timeout=30)
                    
                    if response.ok:
                        result = response.json()
                        if result.get('success'):
                            self.show_analysis_result(result)
                        else:
                            self.show_analysis_error(result.get('error', 'Analysis failed'))
                    else:
                        self.show_analysis_error(f"Request failed: {response.status_code}")
                
                except Exception as e:
                    self.show_analysis_error(f"Analysis request error: {str(e)}")
                
                finally:
                    # Re-enable button
                    self.root.after(0, lambda: (
                        setattr(self, 'pipeline_analyzing', False),
                        self.analyze_button.config(state='normal', text="Run Pipeline Analysis", bg='#27ae60')
                    ))
            
            # Start analysis in background thread
            analysis_thread = threading.Thread(target=run_analysis, daemon=True)
            analysis_thread.start()
            
        except Exception as e:
            print(f"❌ Pipeline analysis trigger error: {e}")
            self.pipeline_analyzing = False
            self.analyze_button.config(state='normal', text="Run Pipeline Analysis", bg='#27ae60')
    
    def show_analysis_result(self, result):
        """Show analysis result in popup"""
        def show_popup():
            import tkinter.messagebox as msgbox
            
            dimensions = result.get('dimensions', {})
            mixture = result.get('cement_mixture', {})
            
            message = f"""Pipeline Analysis Complete!
            
Dimensions: {dimensions.get('display', 'N/A')}
Cement Mixture: {mixture.get('ratio_string', 'N/A')}
Detections: {result.get('num_detections', 0)}
Model: {result.get('model_type', 'Unknown')}

Results saved to gallery with 4-step visualization."""
            
            msgbox.showinfo("Pipeline Analysis Results", message)
        
        self.root.after(0, show_popup)
    
    def show_analysis_error(self, error_msg):
        """Show analysis error in popup"""
        def show_popup():
            import tkinter.messagebox as msgbox
            msgbox.showerror("Pipeline Analysis Error", f"Analysis failed:\n\n{error_msg}")
        
        self.root.after(0, show_popup)
    
    def show_status(self):
        """Show system status"""
        try:
            import tkinter.messagebox as msgbox
            
            # FIXED: Check camera status properly
            camera_status = "Unknown"
            if self.camera_manager:
                if hasattr(self.camera_manager, 'is_running'):
                    camera_status = "Running" if self.camera_manager.is_running else "Stopped"
                else:
                    camera_status = "Available"
            else:
                camera_status = "Not Available"
            
            distance_status = "Available" if self.distance_service else "Unavailable"
            
            status_msg = f"""System Status:

Camera: {camera_status}
Distance Sensor: {distance_status}
Pipeline Mode: Active
Save Mode: Analyzed Images Only
Expected Classes: front_horizontal, front_vertical
Expected Detections: 2 verticals + 11 horizontals

Current Frame: {'Available' if self.current_frame is not None else 'None'}
Display: {'Running' if self.is_running else 'Stopped'}"""
            
            msgbox.showinfo("System Status", status_msg)
            
        except Exception as e:
            print(f"❌ Status display error: {e}")
    
    def open_gallery(self):
        """Open gallery in web browser"""
        try:
            import webbrowser
            server_url = config.get_server_url() if hasattr(config, 'get_server_url') else f"https://localhost:{config.PORT}"
            webbrowser.open(f'{server_url}/result.html')
        except Exception as e:
            print(f"❌ Gallery open error: {e}")
    
    def stop(self):
        """Stop the display"""
        self.is_running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

def start_tkinter_window(camera_manager, distance_service):
    """Start the tkinter camera window for 480x640 display with enhanced pipeline status"""
    try:
        print("Initializing Tkinter 480x640 camera interface with pipeline support...")
        tk_camera = TkinterCameraFrame(camera_manager, distance_service)
        tk_camera.start()
    except Exception as e:
        print(f"Error starting Tkinter window: {e}")

def main():
    distance_service = None
    camera_manager = None
    
    try:
        print("Starting Rebar Vista with Enhanced Pipeline Integration...")
        print("📝 PIPELINE MODE: 4-step visualization with exact formulas")
        print("🔧 SAVE MODE: Only analyzed images with overlays will be saved")
        print("=" * 70)
        
        # Initialize services
        print("Initializing camera, image, AI, and distance services...")
        camera_manager = CameraManager()
        image_service = ImageService()
        ai_service = AIService()
        distance_service = DistanceService()
        
        # Create Flask app with services
        print("Creating Flask web application...")
        app = create_app()
        
        # Initialize routes with services AFTER app creation
        with app.app_context():
            init_camera_routes(camera_manager, image_service)
            init_image_routes(image_service)
            init_ai_routes(ai_service, camera_manager)
            init_sensor_routes(distance_service)
            print("✅ Flask routes initialized with enhanced pipeline support")
        
        # Ensure upload folder exists
        config.ensure_upload_folder()
        print(f"📁 Upload folder ready: {config.UPLOAD_FOLDER}")
        
        # Ensure gallery metadata folder exists
        gallery_metadata_dir = os.path.join(config.UPLOAD_FOLDER, 'gallery_metadata')
        os.makedirs(gallery_metadata_dir, exist_ok=True)
        print(f"📁 Gallery metadata folder ready: {gallery_metadata_dir}")
        
        # Ensure model folder exists
        model_folder = "/home/team10/RebarWeb/app/model"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
            print(f"📁 Created model folder: {model_folder}")
        else:
            print(f"📁 Model folder ready: {model_folder}")
        
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
        
        print("Starting Tkinter 480x640 display window with pipeline status...")
        # Start tkinter window in separate thread
        tkinter_thread = threading.Thread(
            target=start_tkinter_window, 
            args=(camera_manager, distance_service), 
            daemon=True
        )
        tkinter_thread.start()
        
        print("Starting Flask web server with HTTPS...")
        print("Available routes:")
        for rule in app.url_map.iter_rules():
            methods = ', '.join(rule.methods - {'HEAD', 'OPTIONS'})
            print(f"  {rule.endpoint}: {rule.rule} [{methods}]")
        
        print("=" * 70)
        print("REBAR VISTA READY - ENHANCED PIPELINE MODE")
        print("Web interface: HTTPS camera display with 4-step pipeline analysis")
        print("Tkinter window: Live 480x640 camera feed with pipeline controls")
        print("AI Analysis: Detectron2 with exact pipeline formulas")
        print("Distance Sensor: HC-SR04 optimal positioning (160-200cm)")
        print("Pipeline Steps: Detection → Quadrants → Polygon → Cement")
        print("Save Mode: ONLY analyzed images with 4-step visualization")
        print("Gallery: Enhanced with pipeline metadata and step images")
        print("=" * 70)
        
        # Print AI service status
        ai_status = ai_service.get_model_status()
        print("\n=== AI Service Status ===")
        print(f"Detectron2 Available: {'✅' if ai_status['detectron2_available'] else '❌'}")
        print(f"Model Loaded: {'✅' if ai_status['model_loaded'] else '❌'}")
        print(f"Model Path: {ai_status['model_path']}")
        print(f"Model Exists: {'✅' if ai_status['model_exists'] else '❌'}")
        print(f"Classes: {ai_status['class_names']} (2 classes)")
        print(f"Detection Threshold: {ai_status['threshold']}")
        print(f"Pipeline Constants: PX_TO_CM={getattr(ai_service, 'PX_TO_CM', 'N/A')}, OFFSET_CM={getattr(ai_service, 'OFFSET_CM', 'N/A')}")
        print(f"Mix Ratio: {getattr(ai_service, 'MIX_RATIO', 'N/A')}")
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
        print(f"Simulation Mode: {'✅' if distance_status['simulation_mode'] else '❌'}")
        print(f"GPIO Pins: TRIG={distance_status['gpio_pins']['trigger']}, ECHO={distance_status['gpio_pins']['echo']}")
        print(f"Optimal Range: {distance_status['optimal_range']['min']}-{distance_status['optimal_range']['max']}{distance_status['optimal_range']['unit']}")
        print("===============================")
        
        # Print SSL/HTTPS status
        print("\n=== HTTPS Configuration ===")
        print(f"Host: {config.HOST}")
        print(f"Port: {config.PORT}")
        if hasattr(config, 'SSL_CERT_PATH'):
            print(f"SSL Cert: {config.SSL_CERT_PATH}")
            print(f"SSL Key: {config.SSL_KEY_PATH}")
        if hasattr(config, 'get_local_ip'):
            print(f"Local IP: {config.get_local_ip()}")
            print(f"HTTPS URL: https://{config.get_local_ip()}:{config.PORT}")
        print("============================")
        
        # Start Flask app with SSL (HTTPS) if available
        print(f"\n🚀 Starting server on {config.HOST}:{config.PORT}")
        
        # Check if SSL is configured
        ssl_context = None
        if hasattr(config, 'get_ssl_context'):
            ssl_context = config.get_ssl_context()
        
        if ssl_context:
            print(f"🔒 HTTPS enabled - Access via: https://{config.get_local_ip()}:{config.PORT}")
            app.run(
                host=config.HOST,
                port=config.PORT,
                debug=config.DEBUG,
                ssl_context=ssl_context,
                threaded=True
            )
        else:
            print(f"🌐 HTTP mode - Access via: http://{config.HOST}:{config.PORT}")
            app.run(
                host=config.HOST,
                port=config.PORT,
                debug=config.DEBUG,
                threaded=True
            )
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔄 Cleaning up services...")
        
        # Stop distance monitoring
        try:
            if distance_service:
                if hasattr(distance_service, 'stop_monitoring'):
                    distance_service.stop_monitoring()
                    print("✅ Distance service stopped")
                else:
                    print("⚠️ Distance service stop method not found")
        except Exception as e:
            print(f"⚠️ Error stopping distance service: {e}")
        
        # Stop camera
        try:
            if camera_manager:
                # FIXED: Check for available stop methods
                if hasattr(camera_manager, 'stop'):
                    camera_manager.stop()
                    print("✅ Camera service stopped")
                elif hasattr(camera_manager, 'cleanup'):
                    camera_manager.cleanup()
                    print("✅ Camera service cleaned up")
                else:
                    print("⚠️ Camera service stop method not found")
        except Exception as e:
            print(f"⚠️ Error stopping camera service: {e}")
        
        print("👋 Rebar Vista shutdown complete")

if __name__ == "__main__":
    main()
