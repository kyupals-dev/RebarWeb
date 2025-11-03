#!/usr/bin/env python3
"""
Force SSL Certificate Regeneration for IP Change
Run this script when your IP address changes to fix SSL certificates

Usage: python3 force_ssl_fix.py
"""

import os
import sys
import subprocess

# Add project root to Python path
project_root = "/home/team10/RebarWeb"
sys.path.insert(0, project_root)

def main():
    print("🔒 FORCE SSL CERTIFICATE FIX FOR REBAR VISTA")
    print("============================================")
    
    try:
        from app.utils.config import config
        
        print(f"Current IP: {config.current_ip}")
        print(f"Project root: {project_root}")
        print(f"SSL Certificate Path: {config.SSL_CERT_PATH}")
        print(f"SSL Key Path: {config.SSL_KEY_PATH}")
        
        # Check current status
        ssl_status = config.get_ssl_status_for_ip()
        print(f"\n📊 Current SSL Status:")
        print(f"   Certificate exists: {'✅' if ssl_status['cert_exists'] else '❌'}")
        print(f"   Private key exists: {'✅' if ssl_status['key_exists'] else '❌'}")
        print(f"   OpenSSL available: {'✅' if ssl_status['openssl_available'] else '❌'}")
        
        # Clean old certificates first
        print(f"\n🧹 Cleaning old certificates...")
        config.clean_old_certificates()
        
        # Force regeneration
        print(f"\n🔄 Force regenerating SSL certificates for IP {config.current_ip}...")
        success = config.force_ssl_regeneration()
        
        if success:
            print("\n✅ SSL CERTIFICATES REGENERATED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📄 Certificate: {config.SSL_CERT_PATH}")
            print(f"🔑 Private key: {config.SSL_KEY_PATH}")
            print(f"🌐 HTTPS URL: https://{config.current_ip}:{config.PORT}")
            print(f"🏠 Local HTTPS: https://localhost:{config.PORT}")
            print("📱 Accept security warning on first mobile visit")
            
            # Verify files exist
            if os.path.exists(config.SSL_CERT_PATH) and os.path.exists(config.SSL_KEY_PATH):
                print("\n🔍 Certificate verification:")
                
                # Get file sizes
                cert_size = os.path.getsize(config.SSL_CERT_PATH)
                key_size = os.path.getsize(config.SSL_KEY_PATH)
                print(f"   Certificate size: {cert_size} bytes")
                print(f"   Private key size: {key_size} bytes")
                
                # Test certificate
                try:
                    result = subprocess.run([
                        'openssl', 'x509', '-in', config.SSL_CERT_PATH, '-noout', '-subject'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        print(f"   Subject: {result.stdout.strip()}")
                        print("   ✅ Certificate is valid")
                    else:
                        print("   ⚠️  Certificate validation failed")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not verify certificate: {e}")
            else:
                print("\n❌ Certificate files not found after generation!")
                
        else:
            print("\n❌ SSL CERTIFICATE REGENERATION FAILED")
            print("=" * 45)
            print("Possible issues:")
            print("1. OpenSSL is not installed:")
            print("   sudo apt update && sudo apt install openssl")
            print("2. Permission issues with certificate directory")
            print("3. Disk space issues")
            print("")
            print(f"🌐 Use HTTP instead: http://{config.current_ip}:{config.PORT}")
            
        print(f"\n🚀 NEXT STEPS:")
        print("=============")
        print("1. Start Rebar Vista: python3 main.py")
        if success:
            print(f"2. Visit: https://{config.current_ip}:{config.PORT}")
            print("3. Accept the security warning (self-signed certificate)")
            print("4. Bookmark for future use")
        else:
            print(f"2. Visit: http://{config.current_ip}:{config.PORT}")
            print("3. Fix SSL issues and run this script again")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the RebarWeb directory:")
        print("   cd /home/team10/RebarWeb")
        print("   python3 force_ssl_fix.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("app/utils/config.py"):
        print("❌ Error: This script must be run from the RebarWeb directory")
        print("Current directory:", os.getcwd())
        print("Expected files not found. Please run:")
        print("   cd /home/team10/RebarWeb")
        print("   python3 force_ssl_fix.py")
        sys.exit(1)
    
    main()
