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
    Application configuration settings for PIPELINE MODE
    UPDATED: Enhanced SSL certificate management for IP changes + Pipeline Constants
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
        
        # PIPELINE CONSTANTS - EXACT MATCH TO YOUR TRAINING CODE
        self.PIPELINE_MODE = True
        self.PX_TO_CM = 1 / 3.54  # conversion factor (3.54 px = 1 cm)
        self.OFFSET_CM = 4.5      # allowance for formworks per side
        
        # Cement mixture constants - EXACT FROM PIPELINE
        self.CEMENT_BAG_WEIGHT = 40      # kg
        self.MIX_RATIO = (1, 2, 4)       # cement : sand : gravel
        self.WATER_CEMENT_RATIO = 0.53
        self.DRY_VOLUME_FACTOR = 1.54
        
        # Material Densities (kg/m³) - EXACT FROM PIPELINE
        self.CEMENT_DENSITY = 1440
        self.SAND_DENSITY = 1600
        self.GRAVEL_DENSITY = 1500
        
        # Model configuration - EXACT MATCH TO YOUR TRAINING CONFIG
        self.MODEL_CONFIG = {
            'config_file': "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            'num_classes': 2,  # front_horizontal, front_vertical
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
        
        print("🔧 PIPELINE Config initialized:")
        print(f"   Camera: {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT}")
        print(f"   Classes: {self.MODEL_CONFIG['class_names']}")
        print(f"   PX_TO_CM: {self.PX_TO_CM}")
        print(f"   Mix ratio: {self.MIX_RATIO}")
        print(f"   Expected detections: {self.MODEL_CONFIG['expected_detections']}")
        
        # Setup SSL paths
        self._setup_ssl_paths()
    
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
        
        # If all ports busy, default to 8000 and let it fail with clear message
        print("❌ All preferred ports busy, using 8000")
        return 8000
    
    def _setup_ssl_paths(self):
        """Setup SSL certificate paths dynamically based on current IP"""
        # Get current IP address
        self.current_ip = get_local_ip()
        
        # Define certificate directory (relative to project root)
        # Go up from app/utils to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cert_dir = os.path.join(project_root, 'certificates')
        
        # Set SSL paths
        self.SSL_CERT_PATH = os.getenv('SSL_CERT_PATH', 
                                      os.path.join(cert_dir, f'{self.current_ip}.pem'))
        self.SSL_KEY_PATH = os.getenv('SSL_KEY_PATH', 
                                     os.path.join(cert_dir, f'{self.current_ip}-key.pem'))
        
        print(f"Device IP: {self.current_ip}")
        print(f"SSL Certificate Path: {self.SSL_CERT_PATH}")
        print(f"SSL Key Path: {self.SSL_KEY_PATH}")
    
    @property
    def ssl_context(self):
        """Get SSL context tuple"""
        return (self.SSL_CERT_PATH, self.SSL_KEY_PATH)
    
    def ensure_upload_folder(self):
        """Create upload folder if it doesn't exist"""
        if not os.path.exists(self.UPLOAD_FOLDER):
            os.makedirs(self.UPLOAD_FOLDER)
            print(f"Created upload folder: {self.UPLOAD_FOLDER}")
    
    def ensure_certificate_folder(self):
        """Create certificate folder if it doesn't exist"""
        cert_dir = os.path.dirname(self.SSL_CERT_PATH)
        if not os.path.exists(cert_dir):
            os.makedirs(cert_dir)
            print(f"Created certificate folder: {cert_dir}")
    
    def _check_openssl_available(self):
        """Check if OpenSSL is available on the system"""
        return shutil.which('openssl') is not None
    
    def force_ssl_regeneration(self):
        """Force regeneration of SSL certificates for current IP"""
        print(f"🔄 Force regenerating SSL certificates for IP: {self.current_ip}")
        
        # Remove existing certificates
        try:
            if os.path.exists(self.SSL_CERT_PATH):
                os.remove(self.SSL_CERT_PATH)
                print(f"   Removed old certificate: {self.SSL_CERT_PATH}")
            if os.path.exists(self.SSL_KEY_PATH):
                os.remove(self.SSL_KEY_PATH)
                print(f"   Removed old key: {self.SSL_KEY_PATH}")
        except Exception as e:
            print(f"   Warning: Could not remove old certificates: {e}")
        
        # Generate new certificates
        return self.generate_ssl_certificates()
    
    def get_ssl_status_for_ip(self):
        """Get detailed SSL status for current IP"""
        return {
            'current_ip': self.current_ip,
            'cert_path': self.SSL_CERT_PATH,
            'key_path': self.SSL_KEY_PATH,
            'cert_exists': os.path.exists(self.SSL_CERT_PATH),
            'key_exists': os.path.exists(self.SSL_KEY_PATH),
            'cert_dir_exists': os.path.exists(os.path.dirname(self.SSL_CERT_PATH)),
            'openssl_available': self._check_openssl_available(),
            'expected_cert_name': f'{self.current_ip}.pem',
            'expected_key_name': f'{self.current_ip}-key.pem'
        }
    
    def clean_old_certificates(self):
        """Remove certificates for old IP addresses"""
        cert_dir = os.path.dirname(self.SSL_CERT_PATH)
        
        if not os.path.exists(cert_dir):
            return
        
        print("🧹 Cleaning old certificates...")
        
        try:
            # List all certificate files
            for filename in os.listdir(cert_dir):
                if filename.endswith(('.pem', '.crt', '.key', '.csr')):
                    filepath = os.path.join(cert_dir, filename)
                    
                    # Skip current IP certificates
                    if self.current_ip in filename:
                        continue
                    
                    # Remove old certificates
                    try:
                        os.remove(filepath)
                        print(f"   🗑️ Removed old certificate: {filename}")
                    except Exception as e:
                        print(f"   ⚠️ Could not remove {filename}: {e}")
                        
        except Exception as e:
            print(f"⚠️ Error cleaning old certificates: {e}")
    
    def generate_ssl_certificates(self):
        """Generate mobile-compatible SSL certificates for the current IP address"""
        try:
            if not self._check_openssl_available():
                print("Error: OpenSSL not found. Please install it with:")
                print("sudo apt install openssl")
                return False
            
            current_ip = self.current_ip
            cert_dir = os.path.dirname(self.SSL_CERT_PATH)
            
            self.ensure_certificate_folder()
            
            print(f"Generating mobile-compatible SSL certificates for IP: {current_ip}")
            
            # Enhanced certificate configuration with proper key usage for mobile devices
            cert_config = f"""[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
CN = {current_ip}
O = Rebar Vista Pipeline
OU = Development
C = PH
L = Quezon City
ST = Metro Manila

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names
extendedKeyUsage = serverAuth, clientAuth

[alt_names]
DNS.1 = localhost
DNS.2 = raspberrypi
DNS.3 = raspberrypi.local
DNS.4 = *.local
DNS.5 = rebar-vista.local
IP.1 = 127.0.0.1
IP.2 = {current_ip}
IP.3 = ::1
"""
            
            config_file = os.path.join(cert_dir, 'ssl.conf')
            
            with open(config_file, 'w') as f:
                f.write(cert_config)
            
            print("Generating private key...")
            result = subprocess.run([
                'openssl', 'genrsa', '-out', self.SSL_KEY_PATH, '2048'
            ], capture_output=True, text=True, cwd=cert_dir)
            
            if result.returncode != 0:
                print(f"Error generating private key: {result.stderr}")
                return False
            
            print("Generating certificate signing request...")
            csr_path = os.path.join(cert_dir, f'{current_ip}.csr')
            result = subprocess.run([
                'openssl', 'req', '-new', 
                '-key', self.SSL_KEY_PATH,
                '-out', csr_path,
                '-config', config_file
            ], capture_output=True, text=True, cwd=cert_dir)
            
            if result.returncode != 0:
                print(f"Error generating CSR: {result.stderr}")
                return False
            
            print("Generating mobile-compatible certificate...")
            result = subprocess.run([
                'openssl', 'x509', '-req',
                '-in', csr_path,
                '-signkey', self.SSL_KEY_PATH,
                '-out', self.SSL_CERT_PATH,
                '-days', '365',
                '-extensions', 'v3_req',
                '-extfile', config_file
            ], capture_output=True, text=True, cwd=cert_dir)
            
            if result.returncode != 0:
                print(f"Error generating certificate: {result.stderr}")
                return False
            
            # Set proper permissions
            os.chmod(self.SSL_KEY_PATH, 0o600)
            os.chmod(self.SSL_CERT_PATH, 0o644)
            
            # Clean up temporary files
            if os.path.exists(csr_path):
                os.remove(csr_path)
            if os.path.exists(config_file):
                os.remove(config_file)
            
            print(f"✅ Mobile-compatible SSL certificates generated!")
            print(f"Certificate: {self.SSL_CERT_PATH}")
            print(f"Private Key: {self.SSL_KEY_PATH}")
            print(f"Valid for 365 days")
            
            # Verify certificate extensions
            print("Verifying certificate compatibility...")
            try:
                result = subprocess.run([
                    'openssl', 'x509', '-in', self.SSL_CERT_PATH, '-text', '-noout'
                ], capture_output=True, text=True)
                
                if 'serverAuth' in result.stdout:
                    print("✅ Certificate includes serverAuth extension (required for HTTPS)")
                if 'digitalSignature' in result.stdout:
                    print("✅ Certificate includes digitalSignature (required for mobile)")
                if 'keyEncipherment' in result.stdout:
                    print("✅ Certificate includes keyEncipherment (required for SSL)")
                
                # Check for subject alternative names
                if f'IP Address:{current_ip}' in result.stdout:
                    print(f"✅ Certificate includes IP SAN: {current_ip}")
                
            except Exception as e:
                print(f"⚠️  Could not verify certificate: {e}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error running OpenSSL command: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error generating SSL certificates: {e}")
            return False
    
    def validate_ssl_certificates(self):
        """Check if SSL certificates exist and are valid for current IP - ENHANCED VERSION"""
        cert_exists = os.path.exists(self.SSL_CERT_PATH)
        key_exists = os.path.exists(self.SSL_KEY_PATH)
        
        print(f"🔍 Validating SSL certificates for IP: {self.current_ip}")
        print(f"   Certificate exists: {cert_exists}")
        print(f"   Key exists: {key_exists}")
        
        # Check if certificates exist
        if not (cert_exists and key_exists):
            print(f"❌ SSL certificates missing for IP {self.current_ip}")
            print("🔄 Generating new certificates...")
            return self.generate_ssl_certificates()
        
        # Verify certificate is for current IP
        try:
            print("🔍 Verifying certificate matches current IP...")
            result = subprocess.run([
                'openssl', 'x509', '-in', self.SSL_CERT_PATH, '-text', '-noout'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                cert_content = result.stdout
                
                # Check if certificate contains current IP
                ip_in_cert = f'IP Address:{self.current_ip}' in cert_content
                
                if ip_in_cert:
                    print(f"✅ SSL certificates valid for IP {self.current_ip}")
                    
                    # Additional check: verify certificate is not expired
                    expiry_result = subprocess.run([
                        'openssl', 'x509', '-in', self.SSL_CERT_PATH, '-checkend', '86400'  # Check if expires in 24h
                    ], capture_output=True, text=True, timeout=5)
                    
                    if expiry_result.returncode == 0:
                        print("✅ Certificate is not expired")
                        return True
                    else:
                        print("⚠️ Certificate is expired or expiring soon, regenerating...")
                        return self.force_ssl_regeneration()
                else:
                    print(f"⚠️ Certificate is for different IP, regenerating for {self.current_ip}")
                    
                    # Show what IPs are in the certificate
                    ip_lines = [line.strip() for line in cert_content.split('\n') if 'IP Address:' in line]
                    if ip_lines:
                        print(f"   Certificate currently contains: {ip_lines}")
                    
                    return self.force_ssl_regeneration()
            else:
                print(f"⚠️ Could not read certificate: {result.stderr}")
                return self.force_ssl_regeneration()
                
        except subprocess.TimeoutExpired:
            print("⚠️ Certificate verification timed out, regenerating...")
            return self.force_ssl_regeneration()
        except Exception as e:
            print(f"⚠️ Error verifying certificate: {e}")
            return self.force_ssl_regeneration()
        
        # Fallback: regenerate certificates
        return self.generate_ssl_certificates()
    
    def get_pipeline_config(self):
        """Get pipeline-specific configuration"""
        return {
            'pipeline_mode': self.PIPELINE_MODE,
            'px_to_cm': self.PX_TO_CM,
            'offset_cm': self.OFFSET_CM,
            'mix_ratio': self.MIX_RATIO,
            'cement_bag_weight': self.CEMENT_BAG_WEIGHT,
            'water_cement_ratio': self.WATER_CEMENT_RATIO,
            'dry_volume_factor': self.DRY_VOLUME_FACTOR,
            'material_densities': {
                'cement': self.CEMENT_DENSITY,
                'sand': self.SAND_DENSITY,
                'gravel': self.GRAVEL_DENSITY
            },
            'model_config': self.MODEL_CONFIG,
            'distance_range': {
                'min': self.DISTANCE_OPTIMAL_MIN,
                'max': self.DISTANCE_OPTIMAL_MAX,
                'unit': 'cm'
            }
        }
    
    def get_status(self):
        """Get current configuration status"""
        return {
            'ip_address': self.current_ip,
            'ssl_cert_exists': os.path.exists(self.SSL_CERT_PATH),
            'ssl_key_exists': os.path.exists(self.SSL_KEY_PATH),
            'upload_folder_exists': os.path.exists(self.UPLOAD_FOLDER),
            'openssl_available': self._check_openssl_available(),
            'ssl_cert_path': self.SSL_CERT_PATH,
            'ssl_key_path': self.SSL_KEY_PATH,
            'pipeline_mode': self.PIPELINE_MODE,
            'camera_resolution': f"{self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT}",
            'expected_classes': self.MODEL_CONFIG['class_names'],
            'mix_ratio': f"{self.MIX_RATIO[0]}:{self.MIX_RATIO[1]}:{self.MIX_RATIO[2]}"
        }
    
    def print_status(self):
        """Print current configuration status"""
        status = self.get_status()
        print("\n=== Rebar Vista PIPELINE Configuration Status ===")
        print(f"Mode: PIPELINE ANALYSIS (Quadrant Intersections)")
        print(f"IP Address: {status['ip_address']}")
        print(f"Server Host: {self.HOST}")
        print(f"Server Port: {self.PORT}")
        print(f"Camera Resolution: {status['camera_resolution']} (Training Match)")
        print(f"Expected Classes: {status['expected_classes']}")
        print(f"Mix Ratio: {status['mix_ratio']}")
        print(f"PX_TO_CM Factor: {self.PX_TO_CM}")
        print(f"Offset per side: {self.OFFSET_CM}cm")
        print(f"Distance Range: {self.DISTANCE_OPTIMAL_MIN}-{self.DISTANCE_OPTIMAL_MAX}cm")
        print(f"OpenSSL Available: {'✅' if status['openssl_available'] else '❌'}")
        print(f"SSL Certificate: {'✅' if status['ssl_cert_exists'] else '❌'}")
        print(f"  Path: {status['ssl_cert_path']}")
        print(f"SSL Private Key: {'✅' if status['ssl_key_exists'] else '❌'}")
        print(f"  Path: {status['ssl_key_path']}")
        print(f"Upload Folder: {'✅' if status['upload_folder_exists'] else '❌'}")
        print(f"  Path: {self.UPLOAD_FOLDER}")
        print("\n=== Network Access URLs ===")
        print(f"🏠 Local access: https://localhost:{self.PORT}")
        print(f"🌐 Network access: https://{self.current_ip}:{self.PORT}")
        print(f"📱 Mobile/Tablet: https://{self.current_ip}:{self.PORT}")
        print(f"🦊 Firefox: https://{self.current_ip}:{self.PORT}")
        print("========================================\n")
    
    def regenerate_certificates(self):
        """Force regeneration of SSL certificates"""
        print("🔄 Force regenerating SSL certificates...")
        
        # Delete existing certificates
        try:
            if os.path.exists(self.SSL_CERT_PATH):
                os.remove(self.SSL_CERT_PATH)
                print(f"Deleted old certificate: {self.SSL_CERT_PATH}")
            if os.path.exists(self.SSL_KEY_PATH):
                os.remove(self.SSL_KEY_PATH)
                print(f"Deleted old key: {self.SSL_KEY_PATH}")
        except Exception as e:
            print(f"Warning: Could not delete old certificates: {e}")
        
        # Generate new certificates
        return self.generate_ssl_certificates()

# Create a global config instance
config = Config()

# Print status when module is imported (helpful for debugging)
if __name__ == "__main__":
    config.print_status()
else:
    # Only print brief status when imported
    print(f"PIPELINE Config loaded - IP: {config.current_ip}")
    print(f"Camera: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}, Mix: {config.MIX_RATIO}")
