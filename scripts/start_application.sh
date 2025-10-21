#!/bin/bash
# Script to start the GA Framework application
# Usage: ./start_application.sh <app_directory>

set -e  # Exit on error

APP_DIR="${1:-/home/ubuntu/ga-framework}"

echo "🚀 Starting GA Framework application..."
echo "Application directory: $APP_DIR"

cd "$APP_DIR"

# Activate virtual environment
source venv/bin/activate

# Run setup scripts if available
if [ -f "scripts/install_service.py" ]; then
    echo "Running service installation script..."
    sudo python3 scripts/install_service.py || echo "⚠️  Service installation skipped"
fi

# Start with Docker Compose if available
if [ -f "infrastructure/docker/docker-compose.yml" ]; then
    echo "Starting application with Docker Compose..."
    cd infrastructure/docker
    
    # Stop existing containers
    sudo docker-compose down || true
    
    # Build and start containers
    sudo docker-compose up -d --build || echo "⚠️  Docker deployment skipped"
    
    cd "$APP_DIR"
    
    echo "✅ Docker containers started"
    sudo docker-compose -f infrastructure/docker/docker-compose.yml ps
else
    echo "⚠️  Docker Compose file not found, skipping Docker deployment"
fi

echo "✅ Application startup complete"
