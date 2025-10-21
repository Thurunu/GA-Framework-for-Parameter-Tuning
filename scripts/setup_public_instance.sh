#!/bin/bash
# Setup Public Instance with Docker, Prometheus, and Grafana
# Usage: ./setup_public_instance.sh <private_1_ip> <private_2_ip>

set -e

PRIVATE_1_IP="${1}"
PRIVATE_2_IP="${2}"
APP_DIR="${3:-/home/ubuntu/ga-framework}"
DOCKER_NETWORK="monitoring_network"

echo "🚀 Setting up Public Instance with Docker and Prometheus..."

# Install Docker
cd "$APP_DIR/scripts"
chmod +x install_docker.sh
./install_docker.sh

# Create Docker network for monitoring
sudo docker network create "$DOCKER_NETWORK" 2>/dev/null || echo "✓ Network $DOCKER_NETWORK already exists"

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

# Stop and remove existing containers if they exist
sudo docker compose down -v 2>/dev/null || true
sudo docker compose up -d

# Start Prometheus
# echo "Starting Prometheus..."
# sudo docker run -d \
#   --name prometheus \
#   --network "$DOCKER_NETWORK" \
#   --restart unless-stopped \
#   -p 9090:9090 \
#   -v "$APP_DIR/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml" \
#   prom/prometheus:latest \
#   --config.file=/etc/prometheus/prometheus.yml

# Start Grafana
# echo "Starting Grafana..."
# sudo docker run -d \
#   --name grafana \
#   --network "$DOCKER_NETWORK" \
#   --restart unless-stopped \
#   -p 3000:3000 \
#   -e "GF_SECURITY_ADMIN_PASSWORD=${{ secrets.GRAFANA_PASSWORD:-admin }}" \
#   grafana/grafana:latest

# Wait for containers to start
sleep 10

# Check status
echo ""
echo "✅ Public Instance setup complete!"
echo "📊 Prometheus: http://prometheus:9090"
echo "📈 Grafana: http://grafana:3000"
echo ""
echo "Docker containers:"
sudo docker ps --filter "name=prometheus" --filter "name=grafana"
