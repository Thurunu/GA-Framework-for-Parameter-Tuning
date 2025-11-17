#!/bin/bash
# Setup Public Instance with Docker, Prometheus, and Grafana
# Usage: ./setup_public_instance.sh <private_1_ip> <private_2_ip> [app_dir]

set -e

# Validate arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <private_1_ip> <private_2_ip> [app_dir]"
    exit 1
fi

PRIVATE_1_IP="${1}"
PRIVATE_2_IP="${2}"
PRIVATE_3_IP="${3}"
APP_DIR="${3:-/home/ubuntu/ga-framework}"

echo "🚀 Setting up Public Instance with Docker and Monitoring..."

# ============================================
# CHECK IF CONTAINERS ARE ALREADY RUNNING
# ============================================
PROMETHEUS_RUNNING=$(sudo docker ps --filter "name=prometheus" --filter "status=running" -q 2>/dev/null)
GRAFANA_RUNNING=$(sudo docker ps --filter "name=grafana" --filter "status=running" -q 2>/dev/null)

if [ -n "$PROMETHEUS_RUNNING" ] && [ -n "$GRAFANA_RUNNING" ]; then
    echo "✅ Prometheus and Grafana containers are already running"
    
    # Verify Prometheus is responding
    if curl -sf http://localhost:9090/-/healthy >/dev/null 2>&1; then
        echo "✅ Prometheus is healthy and responding"
    else
        echo "⚠️  Prometheus container running but not responding"
    fi
    
    # Verify Grafana is responding
    if curl -sf http://localhost:3000/api/health >/dev/null 2>&1; then
        echo "✅ Grafana is healthy and responding"
    else
        echo "⚠️  Grafana container running but not responding"
    fi
    
    echo ""
    echo "Current containers:"
    sudo docker ps --filter "name=prometheus" --filter "name=grafana" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo "📊 Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
    echo "📈 Grafana: http://$(hostname -I | awk '{print $1}'):3000"
    echo ""
    echo "⏭️  Skipping setup - everything is already running"
    exit 0
fi

echo "📦 Containers not running, proceeding with setup..."

# ============================================
# INSTALL DOCKER
# ============================================
echo ""
echo "Checking Docker installation..."

if command -v docker >/dev/null 2>&1 && sudo docker ps >/dev/null 2>&1; then
    echo "✅ Docker is already installed and running"
else
    echo "Installing Docker..."
    cd "$APP_DIR/scripts"
    chmod +x install_docker.sh
    ./install_docker.sh
fi

# ============================================
# CLEANUP OLD CONTAINERS
# ============================================
echo ""
echo "Cleaning up old containers..."

# Stop Prometheus and Grafana if they exist
sudo docker stop prometheus grafana 2>/dev/null || true

# Remove old containers
sudo docker rm prometheus grafana 2>/dev/null || true

echo "✅ Cleanup complete"

# ============================================
# CREATE DIRECTORIES
# ============================================
echo ""
echo "Setting up directories..."

mkdir -p "/home/ubuntu/prometheus"
mkdir -p "/home/ubuntu/grafana"
sudo chown -R 472:472 /home/ubuntu/grafana

echo "✅ Directories created"

# ============================================
# CREATE PROMETHEUS CONFIGURATION
# ============================================
echo ""
echo "Creating Prometheus configuration..."

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
  # Node Exporter on Private Instance 3
  - job_name: 'node-exporter-private-3'
    static_configs:
      - targets: ['${PRIVATE_3_IP}:9100']
        labels:
          instance: 'private-instance-3'

  # MySQL Exporter on Private Instance 2
  - job_name: 'mysql-exporter'
    static_configs:
      - targets: ['${PRIVATE_2_IP}:9104']
        labels:
          instance: 'private-instance-2-mysql'
EOF

echo "✅ Prometheus configuration created"

# ============================================
# START SERVICES WITH DOCKER COMPOSE
# ============================================
echo ""
echo "Starting Prometheus and Grafana with Docker Compose..."

cd "$APP_DIR/docker"

# Stop any existing compose services
sudo docker compose down -v 2>/dev/null || true

# Start services
sudo docker compose up -d

# Wait for containers to start
echo "Waiting for services to start..."
sleep 10

# ============================================
# VERIFY DEPLOYMENT
# ============================================
echo ""
echo "Verifying deployment..."

# Check if containers are running
if ! sudo docker ps --filter "name=prometheus" --filter "status=running" -q | grep -q .; then
    echo "❌ Prometheus container failed to start"
    sudo docker logs prometheus 2>/dev/null || true
    exit 1
fi

if ! sudo docker ps --filter "name=grafana" --filter "status=running" -q | grep -q .; then
    echo "❌ Grafana container failed to start"
    sudo docker logs grafana 2>/dev/null || true
    exit 1
fi

echo "✅ Containers are running"

# Check if services are responding
echo ""
echo "Checking service health..."

# Wait for Prometheus to be ready
for i in {1..30}; do
    if curl -sf http://localhost:9090/-/healthy >/dev/null 2>&1; then
        echo "✅ Prometheus is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Prometheus not responding after 30 seconds"
    fi
    sleep 1
done

# Wait for Grafana to be ready
for i in {1..30}; do
    if curl -sf http://localhost:3000/api/health >/dev/null 2>&1; then
        echo "✅ Grafana is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Grafana not responding after 30 seconds"
    fi
    sleep 1
done

# ============================================
# DISPLAY STATUS
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Public Instance setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Services:"
echo "  📊 Prometheus: http://$(hostname -I | awk '{print $1}'):9090"
echo "  📈 Grafana:    http://$(hostname -I | awk '{print $1}'):3000"
echo ""
echo "Default Grafana credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "Running containers:"
sudo docker ps --filter "name=prometheus" --filter "name=grafana" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""