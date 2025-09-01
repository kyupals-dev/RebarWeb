






















T

#!/bin/bash

echo "🔒 SSL CERTIFICATE FIX FOR IP CHANGE"
echo "===================================="
echo "Old IP: 192.168.100.80 → New IP: 10.77.15.178"
echo ""

# Get current directory (should be /home/team10/RebarWeb)
PROJECT_ROOT="/home/team10/RebarWeb"
CERT_DIR="$PROJECT_ROOT/certificates"
NEW_IP="10.77.15.178"

echo "1. Checking current certificate status..."
echo "   Certificate directory: $CERT_DIR"

if [ -d "$CERT_DIR" ]; then
    echo "   📁 Certificate directory exists"
    echo "   Current certificates:"
    ls -la "$CERT_DIR/"
else
    echo "   📁 Certificate directory does not exist"
fi

echo ""
echo "2. Removing old certificates..."
if [ -d "$CERT_DIR" ]; then
    # Remove old certificates for previous IPs
    rm -f "$CERT_DIR"/192.168.100.80.*
    rm -f "$CERT_DIR"/*.pem
    rm -f "$CERT_DIR"/*.csr
    echo "   ✅ Old certificates removed"
else
    mkdir -p "$CERT_DIR"
    echo "   📁 Created certificate directory"
fi

echo ""
echo "3. Checking OpenSSL availability..."
if command -v openssl >/dev/null 2>&1; then
    echo "   ✅ OpenSSL is available"
    openssl version
else
    echo "   ❌ OpenSSL not found. Installing..."
    sudo apt update
    sudo apt install openssl -y
fi

echo ""
echo "4. Generating new SSL certificates for IP: $NEW_IP"

# Create certificate configuration
CERT_CONFIG="$CERT_DIR/ssl_$NEW_IP.conf"
cat > "$CERT_CONFIG" << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
CN = $NEW_IP
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
IP.2 = $NEW_IP
IP.3 = ::1
EOF

echo "   📝 Certificate configuration created"

# Generate private key
PRIVATE_KEY="$CERT_DIR/$NEW_IP-key.pem"
echo "   🔑 Generating private key..."
openssl genrsa -out "$PRIVATE_KEY" 2048

# Generate certificate signing request
CSR_FILE="$CERT_DIR/$NEW_IP.csr"
echo "   📋 Generating certificate signing request..."
openssl req -new -key "$PRIVATE_KEY" -out "$CSR_FILE" -config "$CERT_CONFIG"

# Generate self-signed certificate
CERTIFICATE="$CERT_DIR/$NEW_IP.pem"
echo "   🏆 Generating self-signed certificate..."
openssl x509 -req -in "$CSR_FILE" -signkey "$PRIVATE_KEY" -out "$CERTIFICATE" -days 365 -extensions v3_req -extfile "$CERT_CONFIG"

# Set proper permissions
chmod 600 "$PRIVATE_KEY"
chmod 644 "$CERTIFICATE"

echo ""
echo "5. Verifying new certificates..."
if [ -f "$CERTIFICATE" ] && [ -f "$PRIVATE_KEY" ]; then
    echo "   ✅ Certificate files created successfully"
    echo "   📄 Certificate: $CERTIFICATE"
    echo "   🔑 Private key: $PRIVATE_KEY"
    
    # Test certificate validity
    echo "   🧪 Testing certificate..."
    openssl x509 -in "$CERTIFICATE" -text -noout | grep -E "(Subject:|DNS:|IP Address:)" | head -10
    
    # Check certificate dates
    echo "   📅 Certificate validity:"
    openssl x509 -in "$CERTIFICATE" -noout -dates
else
    echo "   ❌ Certificate generation failed"
    exit 1
fi

# Clean up temporary files
rm -f "$CSR_FILE" "$CERT_CONFIG"

echo ""
echo "6. Testing Python SSL configuration..."
python3 << EOF
import os
import sys
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from app.utils.config import config
    print(f"   Current IP detected: {config.current_ip}")
    print(f"   SSL Certificate Path: {config.SSL_CERT_PATH}")
    print(f"   SSL Key Path: {config.SSL_KEY_PATH}")
    print(f"   Certificate exists: {os.path.exists(config.SSL_CERT_PATH)}")
    print(f"   Private key exists: {os.path.exists(config.SSL_KEY_PATH)}")
    print("   ✅ Python SSL configuration working")
except Exception as e:
    print(f"   ❌ Python SSL configuration error: {e}")
EOF

echo ""
echo "🎉 SSL CERTIFICATE SETUP COMPLETE!"
echo "=================================="
echo "✅ New certificates generated for IP: $NEW_IP"
echo "✅ Certificates are valid for 365 days"
echo "✅ Mobile-compatible extensions included"
echo ""
echo "🚀 Next steps:"
echo "1. Start Rebar Vista: python3 main.py"
echo "2. Access via HTTPS: https://$NEW_IP:8000"
echo "3. Accept security warning on first visit"
echo ""
echo "📱 For mobile access:"
echo "- Visit: https://$NEW_IP:8000"
echo "- Accept the self-signed certificate warning"
echo "- Add to home screen for app-like experience"
