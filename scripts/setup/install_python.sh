#!/bin/bash
# Script to install Python dependencies on EC2 instances
# Usage: ./install_dependencies.sh <app_directory>

set -e  # Exit on error

# APP_DIR="${1:-/home/ubuntu/ga-framework}"

# echo "📦 Installing dependencies for GA Framework..."
# echo "Application directory: $APP_DIR"

# cd "$APP_DIR"

# Install Python and pip if not installed
if ! command -v python3 &> /dev/null; then
    echo "Installing Python3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

# # Create virtual environment
# echo "Creating virtual environment..."
# python3 -m venv venv

# # Activate virtual environment
# source venv/bin/activate

# # Install requirements
# echo "Installing Python packages..."
# pip install --upgrade pip
# pip install -r requirements.txt

# echo "✅ Dependencies installed successfully"
# echo "Python version: $(python3 --version)"
# echo "Pip version: $(pip --version)"
