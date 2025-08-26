#!/bin/bash
cd ~/RebarWeb
mkdir -p certificates
cd certificates

IP=$(hostname -I | awk '{print $1}')
echo "Generating SSL certificates for IP: $IP"

cat > ssl.conf << EOFSSL
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
CN = $IP
O = Rebar Vista
OU = Development
C = PH

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = localhost
DNS.2 = raspberrypi
DNS.3 = raspberrypi.local
DNS.4 = *.local
IP.1 = 127.0.0.1
IP.2 = $IP
IP.3 = ::1
EOFSSL

echo "Generating private key..."
openssl genrsa -out ${IP}-key.pem 2048

echo "Generating certificate signing request..."
openssl req -new -key ${IP}-key.pem -out ${IP}.csr -config ssl.conf

echo "Generating self-signed certificate..."
openssl x509 -req -in ${IP}.csr -signkey ${IP}-key.pem -out ${IP}.pem -days 365 -extensions v3_req -extfile ssl.conf

chmod 600 ${IP}-key.pem
chmod 644 ${IP}.pem

rm ${IP}.csr ssl.conf

echo "✅ SSL certificates generated successfully!"
echo "Certificate: ${IP}.pem"
echo "Private Key: ${IP}-key.pem"

openssl x509 -in ${IP}.pem -text -noout | grep -A 10 "X509v3"
