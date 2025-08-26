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
    """Application configuration settings with mobile-compatible SSL certificate generation"""
    
    def __init__(self):
        # Server settings
        self.HOST = os.getenv('HOST', '0.0.0.0')
        self.PORT = int(os.getenv('PORT', 8000))
        self.DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Upload settings
        self.UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/captured_images')
        self.ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
        
        # Camera settings
        self.CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', 640))
        self.CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', 480))
        self.CAMERA_FPS = float(os.getenv('CAMERA_FPS', 30.0))
        
        # Setup SSL paths
        self._setup_ssl_paths()
    
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
O = Rebar Vista
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
        """Check if SSL certificates exist and generate them if they don't"""
        cert_exists = os.path.exists(self.SSL_CERT_PATH)
        key_exists = os.path.exists(self.SSL_KEY_PATH)
        
        if cert_exists and key_exists:
            print(f"✅ SSL certificates found:")
            print(f"  Certificate: {self.SSL_CERT_PATH}")
            print(f"  Private Key: {self.SSL_KEY_PATH}")
            
            # Verify certificate is not expired and has correct extensions
            try:
                result = subprocess.run([
                    'openssl', 'x509', '-in', self.SSL_CERT_PATH, '-checkend', '86400'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Certificate is valid and not expiring soon")
                else:
                    print("⚠️  Certificate is expired or expiring soon, regenerating...")
                    return self.generate_ssl_certificates()
                    
            except Exception as e:
                print(f"⚠️  Could not verify certificate validity: {e}")
            
            return True
        
        if not cert_exists:
            print(f"❌ SSL certificate not found: {self.SSL_CERT_PATH}")
        if not key_exists:
            print(f"❌ SSL key not found: {self.SSL_KEY_PATH}")
        
        # If certificates don't exist, try to generate them
        print("🔄 Attempting to generate mobile-compatible SSL certificates...")
        if self.generate_ssl_certificates():
            return True
        else:
            print("❌ Failed to generate SSL certificates. Will run without SSL.")
            return False
    
    def get_status(self):
        """Get current configuration status"""
        return {
            'ip_address': self.current_ip,
            'ssl_cert_exists': os.path.exists(self.SSL_CERT_PATH),
            'ssl_key_exists': os.path.exists(self.SSL_KEY_PATH),
            'upload_folder_exists': os.path.exists(self.UPLOAD_FOLDER),
            'openssl_available': self._check_openssl_available(),
            'ssl_cert_path': self.SSL_CERT_PATH,
            'ssl_key_path': self.SSL_KEY_PATH
        }
    
    def print_status(self):
        """Print current configuration status"""
        status = self.get_status()
        print("\n=== Rebar Vista Configuration Status ===")
        print(f"IP Address: {status['ip_address']}")
        print(f"Server Host: {self.HOST}")
        print(f"Server Port: {self.PORT}")
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
    print(f"Config loaded - IP: {config.current_ip}")
