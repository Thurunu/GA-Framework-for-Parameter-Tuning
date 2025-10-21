#!/bin/bash
# Setup Private Instance 2 with MySQL, MySQL Exporter, Node Exporter, and Optimization App
# Usage: ./setup_private_instance_2.sh <public_ip> <private_1_ip> <private_2_ip> <mysql_password>

set -e

PUBLIC_IP="${1}"
PRIVATE_1_IP="${2}"
PRIVATE_2_IP="${3}"
MYSQL_PASSWORD="${4}"
APP_DIR="${5:-/home/ubuntu/ga-framework}"

NODE_EXPORTER_VERSION="1.8.2"
MYSQL_EXPORTER_VERSION="0.15.1"

echo "🚀 Setting up Private Instance 2 with MySQL, Exporters, and Application..."

# 1. Install MySQL
# Check if MySQL is already installed and the database exists
if systemctl is-active --quiet mysql && sudo mysql -e "USE ga_optimization_db;" 2>/dev/null; then
    echo "✅ MySQL Server already installed and ga_optimization_db exists"
    
    # Check if ga_app_user exists
    if sudo mysql -e "SELECT User FROM mysql.user WHERE User='ga_app_user';" 2>/dev/null | grep -q "ga_app_user"; then
        echo "✅ MySQL user ga_app_user already exists"
        echo "⏭️  Skipping MySQL installation and configuration"
    else
        echo "⚠️  MySQL installed but user missing, reconfiguring..."
        cd "$APP_DIR/scripts"
        chmod +x install_mysql.sh
        ./install_mysql.sh "$PUBLIC_IP" "$PRIVATE_1_IP" "$PRIVATE_2_IP" "$MYSQL_PASSWORD"
    fi
else
    echo "Installing MySQL Server..."
    cd "$APP_DIR/scripts"
    chmod +x install_mysql.sh
    ./install_mysql.sh "$PUBLIC_IP" "$PRIVATE_1_IP" "$PRIVATE_2_IP" "$MYSQL_PASSWORD"
fi

# 2. Install Python dependencies
# Check if Python and venv are already installed
if command -v python3 &> /dev/null && python3 -m venv --help &> /dev/null 2>&1; then
    echo "✅ Python3 and venv already installed"
    
    # Check if virtual environment exists
    if [ -d "$APP_DIR/venv" ]; then
        echo "✅ Virtual environment already exists at $APP_DIR/venv"
        echo "⏭️  Skipping Python setup (will update dependencies if needed)"
    else
        echo "📦 Installing Python dependencies..."
        chmod +x install_dependencies.sh
        ./install_dependencies.sh "$APP_DIR"
    fi
else
    echo "📦 Installing Python dependencies..."
    chmod +x install_dependencies.sh
    ./install_dependencies.sh "$APP_DIR"
fi

# 3. Install Node Exporter
if systemctl is-active --quiet node_exporter && curl -s http://localhost:9100/metrics > /dev/null 2>&1; then
    echo "✅ Node Exporter already running and responding correctly, skipping installation"
else
    echo "Installing Node Exporter v${NODE_EXPORTER_VERSION}..."
    cd /tmp
    wget -q https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    tar xzf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz
    
    # Stop service before copying binary
    sudo systemctl stop node_exporter 2>/dev/null || true
    
    sudo cp node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
    sudo chmod +x /usr/local/bin/node_exporter
    rm -rf node_exporter-${NODE_EXPORTER_VERSION}.linux-amd64*

# Create Node Exporter service
sudo tee /etc/systemd/system/node_exporter.service > /dev/null << 'NODEEOF'
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
NODEEOF

    sudo systemctl daemon-reload
    sudo systemctl enable node_exporter
    sudo systemctl restart node_exporter
    echo "✅ Node Exporter installed"
fi

# 4. Install MySQL Exporter
if systemctl is-active --quiet mysql_exporter && curl -s http://localhost:9104/metrics > /dev/null 2>&1; then
    echo "✅ MySQL Exporter already running and responding correctly, skipping installation"
else
    echo "Installing MySQL Exporter v${MYSQL_EXPORTER_VERSION}..."
    cd /tmp
    wget -q https://github.com/prometheus/mysqld_exporter/releases/download/v${MYSQL_EXPORTER_VERSION}/mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64.tar.gz
    tar xzf mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64.tar.gz
    
    # Stop service before copying binary
    sudo systemctl stop mysql_exporter 2>/dev/null || true
    
    sudo cp mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64/mysqld_exporter /usr/local/bin/
    sudo chmod +x /usr/local/bin/mysqld_exporter
    rm -rf mysqld_exporter-${MYSQL_EXPORTER_VERSION}.linux-amd64*

# Create MySQL Exporter configuration
sudo tee /etc/.mysqld_exporter.cnf > /dev/null << MYSQLEOF
[client]
user=ga_app_user
password=${MYSQL_PASSWORD}
host=localhost
port=3306
MYSQLEOF

sudo chmod 600 /etc/.mysqld_exporter.cnf

# Create MySQL Exporter service
sudo tee /etc/systemd/system/mysql_exporter.service > /dev/null << 'MYSQLSERVEOF'
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
MYSQLSERVEOF

    sudo systemctl daemon-reload
    sudo systemctl enable mysql_exporter
    sudo systemctl restart mysql_exporter
    echo "✅ MySQL Exporter installed"
fi

# Setup python
sudo apt-get -y update
sudo apt-get -y install python3-venv python3-pip

# Wait for services to start
sleep 3

# Display status
echo ""
echo "✅ Private Instance 2 setup complete!"
echo ""
echo "Services status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MySQL Server:"
sudo systemctl status mysql --no-pager | grep -E "Active:|Main PID:"
echo ""
echo "Node Exporter:"
sudo systemctl status node_exporter --no-pager | grep -E "Active:|Main PID:"
echo ""
echo "MySQL Exporter:"
sudo systemctl status mysql_exporter --no-pager | grep -E "Active:|Main PID:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Node Exporter: http://$(hostname -I | awk '{print $1}'):9100/metrics"
echo "🗄️  MySQL Exporter: http://$(hostname -I | awk '{print $1}'):9104/metrics"
echo "🗄️  MySQL Server: $(hostname -I | awk '{print $1}'):3306"
