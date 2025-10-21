#!/bin/bash
# Setup Private Instance 1 with Node Exporter
# Usage: ./setup_private_instance_1.sh

set -e

NODE_EXPORTER_VERSION="1.8.2"

echo "🚀 Setting up Private Instance 1 with Node Exporter..."

# Check if Node Exporter is already installed and running correctly
if systemctl is-active --quiet node_exporter 2>/dev/null; then
    echo "✅ Node Exporter service is already running"
    
    # Check if it's responding with metrics
    if curl -sf http://localhost:9100/metrics >/dev/null; then
        echo "✅ Node Exporter is responding correctly"
        echo "📊 Node Exporter metrics available at: http://$(hostname -I | awk '{print $1}'):9100/metrics"
        exit 0
    else
        echo "⚠️  Service running but not responding, will reinstall..."
    fi
else
    echo "📦 Node Exporter not running, proceeding with installation..."
fi

# Download and install Node Exporter
echo "Downloading Node Exporter v${NODE_EXPORTER_VERSION}..."
cd /tmp

if [ ! -f "node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz" ]; then
    wget -q https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
fi

echo "Installing Node Exporter..."
tar xzf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz

# Stop existing service if running
sudo systemctl stop node_exporter 2>/dev/null || true

# Install binary
sudo cp node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
sudo chmod +x /usr/local/bin/node_exporter

# Clean up
rm -rf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64*

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/node_exporter.service > /dev/null << 'EOF'
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
EOF

# Start and enable service
echo "Starting Node Exporter..."
sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter

# Wait for service to be ready
sleep 3

# Verify installation
echo ""
if systemctl is-active --quiet node_exporter && curl -sf http://localhost:9100/metrics >/dev/null; then
    echo "✅ Node Exporter successfully installed and running!"
    echo "📊 Metrics: http://$(hostname -I | awk '{print $1}'):9100/metrics"
    curl -s http://localhost:9100/metrics | head -5
else
    echo "❌ Installation completed but service verification failed"
    sudo systemctl status node_exporter --no-pager
    exit 1
fi