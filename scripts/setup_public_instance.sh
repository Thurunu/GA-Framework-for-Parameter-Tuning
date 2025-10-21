#!/bin/bash
# Setup Public Instance with Docker, Prometheus, and Grafana
# Usage: ./setup_public_instance.sh <private_1_ip> <private_2_ip>

set -e

PRIVATE_1_IP="${1}"
PRIVATE_2_IP="${2}"
APP_DIR="${3:-/home/ubuntu/ga-framework}"
# DOCKER_NETWORK="monitoring_network"

echo "🚀 Setting up Public Instance with Docker and Prometheus..."

# Install Docker
cd "$APP_DIR/scripts"
chmod +x install_docker.sh
./install_docker.sh

# Stop all running Docker containers
echo "Stopping all running Docker containers..."
sudo docker stop $(sudo docker ps -q) 2>/dev/null || echo "✅ Stopped all running containers"

# sudo docker network create "$DOCKER_NETWORK" 2>/dev/null || echo "✓ Network $DOCKER_NETWORK already exists"

# Create Prometheus configuration directory
mkdir -p "/home/ubuntu/prometheus"

# Create Grafana data directory
mkdir -p "/home/ubuntu/grafana"

# Create Prometheus configuration file
cat > "/home/ubuntu/prometheus/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter on Private Instance 1
  - job_name: 'node-exporter-private-1'
    static_configs:
      - targets: ['${PRIVATE_1_IP}:9100']
        labels:
          instance: 'private-instance-1'

  # Node Exporter on Private Instance 2
  - job_name: 'node-exporter-private-2'
    static_configs:
      - targets: ['${PRIVATE_2_IP}:9100']
        labels:
          instance: 'private-instance-2'

  # MySQL Exporter on Private Instance 2
  - job_name: 'mysql-exporter'
    static_configs:
      - targets: ['${PRIVATE_2_IP}:9104']
        labels:
          instance: 'private-instance-2-mysql'
EOF

echo "✅ Prometheus configuration created"

# Start Docker Compose services (Prometheus & Grafana)
echo "Starting Prometheus and Grafana with Docker Compose..."
cd "$APP_DIR/docker"

# Stop and remove existing containers if they exist
sudo docker compose down -v 2>/dev/null || true

# Start services
sudo docker compose up -d

# Wait for containers to start
echo "Waiting for services to start..."
sleep 10

# Check status
echo ""
echo "✅ Public Instance setup complete!"
echo "📊 Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
echo "📈 Grafana: http://$(hostname -I | awk '{print $1}'):3000"
echo ""
echo "Docker containers:"
sudo docker ps --filter "name=prometheus" --filter "name=grafana"
