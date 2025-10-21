#!/bin/bash
# Setup Private Instance 1 with Node Exporter
# Usage: ./setup_private_instance_1.sh

set -e

NODE_EXPORTER_VERSION="1.8.2"

echo "🚀 Setting up Private Instance 1 with Node Exporter..."

# Download and install Node Exporter
echo "Downloading Node Exporter v${NODE_EXPORTER_VERSION}..."
cd /tmp
wget -q https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz

echo "Installing Node Exporter..."
tar xzf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
sudo cp node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
sudo chmod +x /usr/local/bin/node_exporter

# Clean up
rm -rf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64*

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/node_exporter.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=Node Exporter
Documentation=https://github.com/prometheus/node_exporter
After=network.target

[Service]
Type=simple
User=ubuntu
ExecStart=/usr/local/bin/node_exporter
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Start and enable Node Exporter
echo "Starting Node Exporter..."
sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl restart node_exporter

# Wait for service to start
sleep 2

# Check status
echo ""
echo "✅ Node Exporter installed and running!"
sudo systemctl status node_exporter --no-pager | head -10

echo ""
echo "📊 Node Exporter endpoint: http://$(hostname -I | awk '{print $1}'):9100/metrics"

# Test endpoint
echo ""
echo "Testing metrics endpoint..."
curl -s http://localhost:9100/metrics | head -5
