"""
Enhanced Configuration for Pipeline Mode Rebar Vista
UPDATED: Complete pipeline constants and HTTPS SSL support
FIXED: Matches training config with exact 2 classes
"""

import os
import socket
import subprocess
import shutil

def get_local_ip():
    """Get the local IP address of the device"""
    try:
        # Method 1: Connect to remote address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and ip != "127.0.0.1":
                return ip
    except Exception:
        pass
    
    try:
        # Method 2: Use hostname command
        result = subprocess.run(['hostname', '-I'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            ip = result.stdout.strip().split()[0]
            if ip and ip != "127.0.0.1":
                return ip
    except Exception:
        pass
    
    # Fallback to localhost
    print("Warning: Could not determine local IP, using localhost")
    return "127.0.0.1"

class Config:
    """
    Enhanced Application Configuration for Pipeline Mode
    UPDATED: Complete SSL, pipeline constants, and 2-class model support
    """
    
    def __init__(self):
        # Server settings
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', self._find_available_port([8000, 8001, 8080, 5000, 5001])))
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Upload settings
        self.UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/captured_images')
        self.ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
        
        # Camera settings - EXACT MATCH TO TRAINING
        self.CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', 480))  # Training width
        self.CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', 640))  # Training height
        self.CAMERA_FPS = float(os.getenv('CAMERA_FPS', 30.0))
        
        # PIPELINE CONSTANTS - EXACT FROM PROJECT KNOWLEDGE
        self.PIPELINE_MODE = True
        self.PX_TO_CM = 1 / 3.54  # conversion factor (3.54 px = 1 cm)
        self.OFFSET_CM = 4.5      # allowance for formworks per side
        
        # Cement mixture constants - EXACT FROM PIPELINE
        self.CEMENT_BAG_WEIGHT = 40      # kg
        self.MIX_RATIO = (1, 2, 4)      # cement : sand : gravel (aggregate)
        self.WATER_CEMENT_RATIO = 0.53
        self.DRY_VOLUME_FACTOR = 1.54
        
        # Material Densities (kg/m³) - EXACT FROM PIPELINE
        self.CEMENT_DENSITY = 1440
        self.SAND_DENSITY = 1600
        self.GRAVEL_DENSITY = 1500
        
        # Model configuration - EXACT MATCH TO TRAINING CONFIG (2 CLASSES ONLY)
        self.MODEL_CONFIG = {
            'config_file': "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            'num_classes': 2,  # FIXED: Only front_horizontal, front_vertical
            'class_names': ["front_horizontal", "front_vertical"],
            'score_thresh_test': 0.3,
            'device': 'cpu',  # Raspberry Pi 5
            'input_format': 'BGR',
            'expected_detections': {
                'front_vertical': 2,
                'front_horizontal': 11
            }
        }
        
        # Distance sensor optimal range
        self.DISTANCE_OPTIMAL_MIN = 160  # cm
        self.DISTANCE_OPTIMAL_MAX = 200  # cm
        
        # Get current IP for SSL certificate generation
        self.current_ip = get_local_ip()
        
        print("🔧 PIPELINE Config initialized:")
        print(f"   Camera: {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT}")
        print(f"   Classes: {self.MODEL_CONFIG['class_names']} (2 classes)")
        print(f"   PX_TO_CM: {self.PX_TO_CM}")
        print(f"   Mix ratio: {self.MIX_RATIO}")
        print(f"   Expected detections: {self.MODEL_CONFIG['expected_detections']}")
        print(f"   Local IP: {self.current_ip}")
        
        # Setup SSL paths and generate certificates
        self._setup_ssl_configuration()
    
    def get_local_ip(self):
        """Return the current local IP"""
        return self.current_ip
    
    def _find_available_port(self, ports):
        """Find an available port from the list"""
        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    print(f"✅ Port {port} is available")
                    return port
            except OSError:
                print(f"⚠️  Port {port} is busy")
                continue
        
        # If all ports busy, default to 8000
        print("❌ All preferred ports busy, using 8000")
        return 8000
    
    def _setup_ssl_configuration(self):
        """Setup SSL certificate paths and generate certificates if needed"""
        # SSL certificate directory
        self.SSL_DIR = "/home/team10/RebarWeb/ssl"
        self.SSL_CERT_PATH = os.path.join(self.SSL_DIR, f"cert_{self.current_ip}.pem")
        self.SSL_KEY_PATH = os.path.join(self.SSL_DIR, f"key_{self.current_ip}.pem")
        
        # Ensure SSL directory exists
        os.makedirs(self.SSL_DIR, exist_ok=True)
        
        # Check if certificates exist for current IP
        if not os.path.exists(self.SSL_CERT_PATH) or not os.path.exists(self.SSL_KEY_PATH):
            print(f"🔐 Generating SSL certificates for IP: {self.current_ip}")
            self._generate_ssl_certificates()
        else:
            print(f"🔐 Using existing SSL certificates for IP: {self.current_ip}")
        
        # Verify certificate files
        if os.path.exists(self.SSL_CERT_PATH) and os.path.exists(self.SSL_KEY_PATH):
            print(f"✅ SSL certificates ready:")
            print(f"   Cert: {self.SSL_CERT_PATH}")
            print(f"   Key: {self.SSL_KEY_PATH}")
        else:
            print("❌ SSL certificate generation failed, using HTTP mode")
            self.SSL_CERT_PATH = None
            self.SSL_KEY_PATH = None
    
    def _generate_ssl_certificates(self):
        """Generate self-signed SSL certificates for HTTPS"""
        try:
            # Check if openssl is available
            result = subprocess.run(['which', 'openssl'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ OpenSSL not found, cannot generate SSL certificates")
                return False
            
            # Generate private key
            key_cmd = [
                'openssl', 'genrsa', 
                '-out', self.SSL_KEY_PATH, 
                '2048'
            ]
            
            result = subprocess.run(key_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"❌ Failed to generate private key: {result.stderr}")
                return False
            
            # Generate certificate
            cert_cmd = [
                'openssl', 'req', 
                '-new', '-x509', 
                '-key', self.SSL_KEY_PATH,
                '-out', self.SSL_CERT_PATH,
                '-days', '365',
                '-subj', f'/C=PH/ST=Metro Manila/L=Pasay/O=Rebar Vista/OU=AI Team/CN={self.current_ip}'
            ]
            
            result = subprocess.run(cert_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"❌ Failed to generate certificate: {result.stderr}")
                return False
            
            print(f"✅ SSL certificates generated successfully for {self.current_ip}")
            return True
            
        except Exception as e:
            print(f"❌ SSL certificate generation error: {e}")
            return False
    
    def ensure_upload_folder(self):
        """Ensure upload folder exists with proper permissions"""
        try:
            if not os.path.exists(self.UPLOAD_FOLDER):
                os.makedirs(self.UPLOAD_FOLDER, mode=0o755)
                print(f"📁 Created upload folder: {self.UPLOAD_FOLDER}")
            
            # Ensure gallery metadata subfolder exists
            gallery_metadata_dir = os.path.join(self.UPLOAD_FOLDER, 'gallery_metadata')
            if not os.path.exists(gallery_metadata_dir):
                os.makedirs(gallery_metadata_dir, mode=0o755)
                print(f"📁 Created gallery metadata folder: {gallery_metadata_dir}")
            
            # Test write permissions
            test_file = os.path.join(self.UPLOAD_FOLDER, 'test_write.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"✅ Upload folder writable: {self.UPLOAD_FOLDER}")
            except Exception as e:
                print(f"❌ Upload folder not writable: {e}")
                
        except Exception as e:
            print(f"❌ Error setting up upload folder: {e}")
    
    def get_ssl_context(self):
        """Get SSL context for Flask app"""
        if self.SSL_CERT_PATH and self.SSL_KEY_PATH:
            if os.path.exists(self.SSL_CERT_PATH) and os.path.exists(self.SSL_KEY_PATH):
                return (self.SSL_CERT_PATH, self.SSL_KEY_PATH)
        return None
    
    def get_server_url(self):
        """Get the complete server URL with protocol"""
        protocol = "https" if self.get_ssl_context() else "http"
        return f"{protocol}://{self.current_ip}:{self.PORT}"
    
    def get_pipeline_constants(self):
        """Get all pipeline constants for reference"""
        return {
            'PX_TO_CM': self.PX_TO_CM,
            'OFFSET_CM': self.OFFSET_CM,
            'CEMENT_BAG_WEIGHT': self.CEMENT_BAG_WEIGHT,
            'MIX_RATIO': self.MIX_RATIO,
            'WATER_CEMENT_RATIO': self.WATER_CEMENT_RATIO,
            'DRY_VOLUME_FACTOR': self.DRY_VOLUME_FACTOR,
            'CEMENT_DENSITY': self.CEMENT_DENSITY,
            'SAND_DENSITY': self.SAND_DENSITY,
            'GRAVEL_DENSITY': self.GRAVEL_DENSITY,
            'DISTANCE_OPTIMAL_MIN': self.DISTANCE_OPTIMAL_MIN,
            'DISTANCE_OPTIMAL_MAX': self.DISTANCE_OPTIMAL_MAX
        }
    
    def validate_model_config(self):
        """Validate model configuration matches training setup"""
        required_classes = ["front_horizontal", "front_vertical"]
        
        if self.MODEL_CONFIG['class_names'] != required_classes:
            print(f"⚠️  Model class mismatch!")
            print(f"   Expected: {required_classes}")
            print(f"   Configured: {self.MODEL_CONFIG['class_names']}")
            return False
        
        if self.MODEL_CONFIG['num_classes'] != 2:
            print(f"⚠️  Model class count mismatch! Expected: 2, Got: {self.MODEL_CONFIG['num_classes']}")
            return False
        
        print("✅ Model configuration matches training setup")
        return True
    
    def print_configuration_summary(self):
        """Print complete configuration summary"""
        print("\n" + "=" * 60)
        print("REBAR VISTA PIPELINE CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Server: {self.get_server_url()}")
        print(f"Camera: {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT} @ {self.CAMERA_FPS}fps")
        print(f"Upload: {self.UPLOAD_FOLDER}")
        print(f"SSL: {'✅' if self.get_ssl_context() else '❌'}")
        print("\nPipeline Constants:")
        constants = self.get_pipeline_constants()
        for key, value in constants.items():
            print(f"  {key}: {value}")
        print("\nModel Configuration:")
        print(f"  Classes: {self.MODEL_CONFIG['class_names']}")
        print(f"  Count: {self.MODEL_CONFIG['num_classes']}")
        print(f"  Threshold: {self.MODEL_CONFIG['score_thresh_test']}")
        print(f"  Expected Detections: {self.MODEL_CONFIG['expected_detections']}")
        print("=" * 60)

# Create global config instance
config = Config()

# Validate configuration on import
if __name__ == "__main__":
    config.print_configuration_summary()
    config.validate_model_config()
