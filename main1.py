from flask import Flask, render_template, Response, jsonify, request
import os
import base64
from datetime import datetime
from flask import send_from_directory
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

app = Flask(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'static/captured_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables for camera sharing
camera = None
current_frame = None
frame_lock = threading.Lock()

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_running = False
        
    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)  # Use default camera
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_running = True
                return True
            else:
                print("Failed to open camera")
                return False
        return True
    
    def get_frame(self):
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.is_running = False

# Global camera manager
camera_manager = CameraManager()

class TkinterCameraFrame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Camera Feed - Tkinter")
        self.root.geometry("660x520")
        self.root.configure(bg='#2c3e50')
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label
        title_label = tk.Label(main_frame, text="Live Camera Feed", 
                              font=('Arial', 16, 'bold'), 
                              bg='#2c3e50', fg='white')
        title_label.pack(pady=(0, 10))
        
        # Camera display label
        self.camera_label = tk.Label(main_frame, bg='black', 
                                   width=640, height=480)
        self.camera_label.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Initializing camera...", 
                                   font=('Arial', 10), 
                                   bg='#2c3e50', fg='#ecf0f1')
        self.status_label.pack(pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Capture button
        self.capture_btn = tk.Button(button_frame, text="Capture Image", 
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
        
        self.is_running = True
        self.update_frame()
        
    def update_frame(self):
        global current_frame, frame_lock
        
        if self.is_running:
            with frame_lock:
                if current_frame is not None:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Resize to fit the label
                    pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update the label
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo
                    
                    self.status_label.configure(text="Camera Active - Live Feed")
                else:
                    self.status_label.configure(text="No camera feed available")
            
            # Schedule next update
            self.root.after(30, self.update_frame)  # ~33 FPS
    
    def capture_image(self):
        global current_frame, frame_lock
        
        with frame_lock:
            if current_frame is not None:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = f'tkinter_capture_{timestamp}.jpg'
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # Save the image
                cv2.imwrite(filepath, current_frame)
                
                self.status_label.configure(text=f"Image saved: {filename}")
                print(f"Image captured and saved: {filename}")
            else:
                self.status_label.configure(text="No frame to capture")
    
    def toggle_camera(self):
        if camera_manager.is_running:
            camera_manager.stop_camera()
            self.toggle_btn.configure(text="Start Camera", bg='#27ae60')
            self.status_label.configure(text="Camera stopped")
        else:
            if camera_manager.start_camera():
                self.toggle_btn.configure(text="Stop Camera", bg='#e74c3c')
                self.status_label.configure(text="Camera started")
            else:
                self.status_label.configure(text="Failed to start camera")
    
    def on_closing(self):
        self.is_running = False
        camera_manager.stop_camera()
        self.root.destroy()
    
    def start(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def camera_thread():
    """Thread function to continuously capture frames"""
    global current_frame, frame_lock
    
    camera_manager.start_camera()
    
    while True:
        frame = camera_manager.get_frame()
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
        
        # Small delay to prevent excessive CPU usage
        threading.Event().wait(0.033)  # ~30 FPS

def start_tkinter_window():
    """Start the tkinter camera window in a separate thread"""
    tk_camera = TkinterCameraFrame()
    tk_camera.start()

@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def service_worker():
    return send_from_directory('static', 'sw.js')

#Route for splash page
@app.route('/')
def splash():
    return render_template('splash.html')

# Route for welcome page
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

# Route for mainpage.html
@app.route('/mainpage.html')
def mainpage():
    return render_template('mainpage.html')

# Route for result.html
@app.route('/result.html')
def result():
    return render_template('result.html')

# Route to get camera feed for web interface
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global current_frame, frame_lock
        
        while True:
            with frame_lock:
                if current_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', current_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            threading.Event().wait(0.033)  # ~30 FPS
    
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to upload/save captured images
@app.route('/upload-image', methods=['POST'])
def upload_image():
    try:
        print("Upload image route accessed")
        
        # Get the image data from the request
        data = request.get_json()
        print(f"Received data: {data is not None}")
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
            
        image_data = data['image']
        
        # Remove the data URL prefix (data:image/jpeg;base64,)
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f'web_capture_{timestamp}.jpg'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        print(f"Saving to: {filepath}")
        
        # Decode base64 and save the image
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(image_data))
        
        print(f"Image saved successfully: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Image saved successfully!'
        })
        
    except Exception as e:
        print(f"Error in upload_image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to capture image from current frame (for web interface)
@app.route('/capture-current-frame', methods=['POST'])
def capture_current_frame():
    try:
        global current_frame, frame_lock
        
        with frame_lock:
            if current_frame is not None:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = f'frame_capture_{timestamp}.jpg'
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # Save the current frame
                cv2.imwrite(filepath, current_frame)
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'message': 'Current frame captured successfully!'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No current frame available'
                }), 400
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to get all captured images
@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        images = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file_stats = os.stat(filepath)
                    
                    images.append({
                        'filename': filename,
                        'url': f'/static/captured_images/{filename}',
                        'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        'size': file_stats.st_size
                    })
        
        # Sort by timestamp (newest first)
        images.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'images': images
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to delete an image
@app.route('/delete-image/<filename>', methods=['DELETE'])
def delete_image(filename):
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({
                'success': True,
                'message': 'Image deleted successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route to clear all images
@app.route('/clear-all-images', methods=['DELETE'])
def clear_all_images():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    os.remove(os.path.join(UPLOAD_FOLDER, filename))
        
        return jsonify({
            'success': True,
            'message': 'All images cleared successfully!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Flask app with Tkinter camera frame...")
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")

    # Start camera thread
    camera_thread_obj = threading.Thread(target=camera_thread, daemon=True)
    camera_thread_obj.start()
    
    # Start tkinter window in a separate thread
    tkinter_thread = threading.Thread(target=start_tkinter_window, daemon=True)
    tkinter_thread.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=8000, ssl_context=('D:/Web App PD/certificates/192.168.100.61.pem', 'D:/Web App PD/certificates/192.168.100.61-key.pem'), use_reloader=False, threaded=True, debug=True)


    
