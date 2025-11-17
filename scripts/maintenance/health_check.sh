#!/bin/bash
# Script to perform health checks on deployed application
# Usage: ./health_check.sh <app_directory>

set -e  # Exit on error

APP_DIR="${1:-/home/ubuntu/ga-framework}"

echo "🏥 Running health checks..."
echo "Application directory: $APP_DIR"

cd "$APP_DIR"

# Activate virtual environment
source venv/bin/activate

# Check Python version
echo ""
echo "Python Environment:"
python3 -c "import sys; print(f'✅ Python version: {sys.version}')"

# Check if key dependencies are installed
echo ""
echo "Checking Python packages..."
python3 -c "import numpy, scipy; print('✅ numpy and scipy installed')" || echo "⚠️  Some packages missing"

# Try importing application modules
echo ""
echo "Checking application modules..."
python3 -c "import sys; sys.path.insert(0, 'src'); import GeneticAlgorithm; print('✅ GeneticAlgorithm module OK')" || echo "⚠️  GeneticAlgorithm import failed"

# Check Docker if available
if command -v docker &> /dev/null; then
    echo ""
    echo "Docker Status:"
    sudo docker ps 2>/dev/null | head -5 || echo "⚠️  Docker not running"
fi

# Check MySQL connection if client is available
if command -v mysql &> /dev/null; then
    echo ""
    echo "MySQL client available for database connections"
fi

# Check disk space
echo ""
echo "Disk Space:"
df -h "$APP_DIR" | tail -1

# Check memory
echo ""
echo "Memory Usage:"
free -h | grep -E "^Mem:"

echo ""
echo "✅ Health check complete"
