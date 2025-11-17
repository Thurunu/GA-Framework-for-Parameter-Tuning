#!/bin/bash
# Setup Private Instance 2 with MySQL, MySQL Exporter, Node Exporter, and Optimization App
# Usage: ./setup_private_instance_2.sh <public_ip> <private_1_ip> <private_2_ip> <mysql_password> [app_dir]

set -e

# Validate arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <public_ip> <private_1_ip> <private_2_ip> <mysql_password> [app_dir]"
    exit 1
fi

PUBLIC_IP="${1}"
PRIVATE_1_IP="${2}"
PRIVATE_2_IP="${3}"
MYSQL_PASSWORD="${4}"
APP_DIR="${5:-/home/ubuntu/ga-framework}"
NODE_EXPORTER_SERVICE="/etc/systemd/system/node_exporter.service"
MYSQL_EXPORTER_SERVICE="/etc/systemd/system/mysql_exporter.service"
MYSQL_EXPORTER_CNF="/etc/.mysqld_exporter.cnf"
NODE_EXPORTER_VERSION="1.8.2"
MYSQL_EXPORTER_VERSION="0.15.1"

echo "🚀 Setting up Private Instance 2 with MySQL, Exporters, and Application..."

# ============================================
# 1. CHECK AND INSTALL MYSQL
# ============================================
echo ""
echo "Checking MySQL Server..."

# Check if MySQL is installed, running, and properly configured
MYSQL_NEEDS_INSTALL=false

if ! systemctl is-active mysql 2>/dev/null; then
    echo "📦 MySQL is not running"
    MYSQL_NEEDS_INSTALL=true
elif sudo mysql -e "USE ga_optimization_db;" 2>/dev/null; then
    echo "⚠️  MySQL running but ga_optimization_db doesn't exist"
    MYSQL_NEEDS_INSTALL=false
elif ! sudo mysql -e "SELECT User FROM mysql.user WHERE User='ga_app_user';" 2>/dev/null | grep -q "ga_app_user"; then
    echo "⚠️  MySQL running but ga_app_user doesn't exist"
    MYSQL_NEEDS_INSTALL=true
else
    echo "✅ MySQL Server already installed and configured correctly"
    echo "   - Database: ga_optimization_db ✓"
    echo "   - User: ga_app_user ✓"
fi

if [ "$MYSQL_NEEDS_INSTALL" = true ]; then
    echo "Installing/Configuring MySQL Server..."
    cd "$APP_DIR/scripts"
    chmod +x install_mysql.sh
    ./install_mysql.sh "$PUBLIC_IP" "$PRIVATE_1_IP" "$PRIVATE_2_IP" "$MYSQL_PASSWORD"
    echo "✅ MySQL Server installed and configured"
fi

# ============================================
# 2. CHECK AND INSTALL PYTHON DEPENDENCIES
# ============================================
echo ""
echo "Checking Python environment..."

PYTHON_NEEDS_INSTALL=false

if ! command -v python3 &>/dev/null; then
    echo "📦 Python3 not found"
    PYTHON_NEEDS_INSTALL=true
elif ! python3 -m venv --help &>/dev/null 2>&1; then
    echo "📦 Python venv module not found"
    PYTHON_NEEDS_INSTALL=true
elif [ ! -d "$APP_DIR/venv" ]; then
    echo "📦 Virtual environment doesn't exist at $APP_DIR/venv"
    PYTHON_NEEDS_INSTALL=false
else
    echo "✅ Python3 and virtual environment already configured"
fi

if [ "$PYTHON_NEEDS_INSTALL" = true ]; then
    echo "Installing Python dependencies..."
    cd "$APP_DIR/scripts"
    chmod +x install_dependencies.sh
    ./install_dependencies.sh "$APP_DIR"
    echo "✅ Python dependencies installed"
fi

# ============================================
# 3. CHECK AND INSTALL NODE EXPORTER
# ============================================
echo ""
echo "Checking Node Exporter..."

if systemctl is-active node_exporter 2>/dev/null && \
   curl -sf http://localhost:9100/metrics >/dev/null 2>&1; then
    echo "✅ Node Exporter already running and responding correctly"
else
    echo "Installing Node Exporter v${NODE_EXPORTER_VERSION}..."
    cd /tmp
    
    # Download if not exists
    if [ ! -f "node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz" ]; then
        wget -q https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    fi
    
    tar xzf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    
    # Stop old service
    sudo systemctl stop node_exporter 2>/dev/null || true
    
    sudo cp node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
    sudo chmod +x /usr/local/bin/node_exporter
    rm -rf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64*

    # Create Node Exporter service
if [ ! -s "$NODE_EXPORTER_SERVICE" ]; then
    echo "📄 Creating Node Exporter systemd service..."
    sudo tee "$NODE_EXPORTER_SERVICE" >/dev/null <<'EOF'

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

else
    echo "✅ Node Exporter service file already exists and is not empty — skipping creation"
fi
    sudo systemctl daemon-reload
    sudo systemctl enable node_exporter
    sudo systemctl start node_exporter
    
    sleep 2
    
    if systemctl is-active --quiet node_exporter; then
        echo "✅ Node Exporter installed and running"
    else
        echo "❌ Node Exporter failed to start"
        sudo systemctl status node_exporter --no-pager
    fi
fi

# ============================================
# 4. CHECK AND INSTALL MYSQL EXPORTER
# ============================================
echo ""
echo "Checking MySQL Exporter..."

if systemctl is-active mysql_exporter 2>/dev/null && \
   curl -sf http://localhost:9104/metrics >/dev/null 2>&1; then
    echo "✅ MySQL Exporter already running and responding correctly"
else
    echo "Installing MySQL Exporter v${MYSQL_EXPORTER_VERSION}..."
    cd /tmp
    
    # Download if not exists
    if [ ! -f "mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64.tar.gz" ]; then
        wget -q https://github.com/prometheus/mysqld_exporter/releases/download/v${MYSQL_EXPORTER_VERSION}/mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64.tar.gz
    fi
    
    tar xzf mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64.tar.gz
    
    # Stop old service
    sudo systemctl stop mysql_exporter 2>/dev/null || true
    
    sudo cp mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64/mysqld_exporter /usr/local/bin/
    sudo chmod +x /usr/local/bin/mysqld_exporter
    rm -rf mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64*

    # Create MySQL Exporter configuration
    if [ ! -s "$MYSQL_EXPORTER_CNF" ]; then
    echo "📄 Creating MySQL Exporter systemd service..."
    sudo tee "$MYSQL_EXPORTER_CNF" >/dev/null <<EOF
[client]
user=ga_app_user
password=${MYSQL_PASSWORD}
host=localhost
port=3306
EOF
    sudo chown ubuntu:ubuntu "$MYSQL_EXPORTER_CNF"
    sudo chmod 600 "$MYSQL_EXPORTER_CNF"

    fi

    # Create MySQL Exporter service
    if [ ! -s "$MYSQL_EXPORTER_SERVICE" ]; then
    echo "📄 Creating MySQL Exporter systemd service..."
    sudo tee "$MYSQL_EXPORTER_SERVICE" >/dev/null <<'EOF'
[Unit]
Description=MySQL Exporter
Documentation=https://github.com/prometheus/mysqld_exporter
After=network.target mysql.service

[Service]
Type=simple
User=ubuntu
ExecStart=/usr/local/bin/mysqld_exporter --config.my-cnf=/etc/.mysqld_exporter.cnf
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
else 
    echo "✅ MySQL Exporter service file already exists and is not empty — skipping creation"
fi

    sudo systemctl daemon-reload
    sudo systemctl enable mysql_exporter
    sudo systemctl start mysql_exporter
    
    sleep 2
    
    if systemctl is-active mysql_exporter; then
        echo "✅ MySQL Exporter installed and running"
    else
        echo "❌ MySQL Exporter failed to start"
        sudo systemctl status mysql_exporter --no-pager
    fi
fi

# ============================================
# 5. ENSURE PYTHON PACKAGES ARE INSTALLED
# ============================================
echo ""
echo "Ensuring Python packages are up to date..."
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip >/dev/null 2>&1
echo "✅ System Python packages updated"

# ============================================
# FINAL STATUS CHECK
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Private Instance 2 setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Services Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# MySQL Status
if systemctl is-active mysql; then
    echo "✅ MySQL Server: Running"
    sudo systemctl status mysql --no-pager | grep -E "Active:|Main PID:" | sed 's/^/   /'
else
    echo "❌ MySQL Server: Not Running"
fi

echo ""

# Node Exporter Status
if systemctl is-active node_exporter; then
    echo "✅ Node Exporter: Running"
    sudo systemctl status node_exporter --no-pager | grep -E "Active:|Main PID:" | sed 's/^/   /'
else
    echo "❌ Node Exporter: Not Running"
fi

echo ""

# MySQL Exporter Status
if systemctl is-active mysql_exporter; then
    echo "✅ MySQL Exporter: Running"
    sudo systemctl status mysql_exporter --no-pager | grep -E "Active:|Main PID:" | sed 's/^/   /'
else
    echo "❌ MySQL Exporter: Not Running"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Endpoints:"
echo "  📊 Node Exporter:  http://$(hostname -I | awk '{print $1}'):9100/metrics"
echo "  🗄️  MySQL Exporter: http://$(hostname -I | awk '{print $1}'):9104/metrics"
echo "  🗄️  MySQL Server:   $(hostname -I | awk '{print $1}'):3306"
echo ""

# Verify endpoints are responding
echo "Verifying endpoints..."
if curl -sf http://localhost:9100/metrics >/dev/null; then
    echo "  ✅ Node Exporter responding"
else
    echo "  ⚠️  Node Exporter not responding"
fi

if curl -sf http://localhost:9104/metrics >/dev/null; then
    echo "  ✅ MySQL Exporter responding"
else
    echo "  ⚠️  MySQL Exporter not responding"
fi

echo ""
echo "Setup completed successfully!"