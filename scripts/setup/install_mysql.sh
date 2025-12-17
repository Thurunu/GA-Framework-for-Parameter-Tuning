#!/bin/bash
# Script to install and configure MySQL server
# Usage: ./install_mysql.sh <public_ip> <private_1_ip> <private_2_ip> <mysql_password>

set -e  # Exit on error

PUBLIC_IP="65.2.180.161"
PRIVATE_1_IP="13.127.66.112"
PRIVATE_2_IP="65.2.131.133"
MYSQL_PASSWORD="12345678"

echo "🗄️  Installing and configuring MySQL Server..."
echo "Allowed IPs: $PUBLIC_IP, $PRIVATE_1_IP, $PRIVATE_2_IP"

# Update and install MySQL
echo "📦 Installing MySQL Server..."
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server

# Start and enable MySQL
echo "🔧 Starting MySQL service..."
sudo systemctl start mysql
sudo systemctl enable mysql

# Wait for MySQL to be ready
echo "⏳ Waiting for MySQL to be ready..."
sleep 5

# Configure MySQL to listen on all interfaces
echo "🔧 Configuring MySQL to accept remote connections..."
sudo sed -i 's/bind-address.*/bind-address = 0.0.0.0/' /etc/mysql/mysql.conf.d/mysqld.cnf

# Restart MySQL to apply changes
sudo systemctl restart mysql
sleep 3

echo "✅ MySQL installed and configured"

# Create database and users
echo "👤 Creating MySQL database and users..."

sudo mysql << EOSQL
-- Create database
CREATE DATABASE IF NOT EXISTS ga_framework;

-- Create application user with password
CREATE USER IF NOT EXISTS 'ga_app_user'@'%' IDENTIFIED BY '${MYSQL_PASSWORD}';

-- Grant privileges on the database
GRANT ALL PRIVILEGES ON ga_framework.* TO 'ga_app_user'@'%';

-- Grant access from specific IPs for better security
CREATE USER IF NOT EXISTS 'ga_app_user'@'${PUBLIC_IP}' IDENTIFIED BY '${MYSQL_PASSWORD}';
CREATE USER IF NOT EXISTS 'ga_app_user'@'${PRIVATE_1_IP}' IDENTIFIED BY '${MYSQL_PASSWORD}';
CREATE USER IF NOT EXISTS 'ga_app_user'@'${PRIVATE_2_IP}' IDENTIFIED BY '${MYSQL_PASSWORD}';

GRANT ALL PRIVILEGES ON ga_framework.* TO 'ga_app_user'@'${PUBLIC_IP}';
GRANT ALL PRIVILEGES ON ga_framework.* TO 'ga_app_user'@'${PRIVATE_1_IP}';
GRANT ALL PRIVILEGES ON ga_framework.* TO 'ga_app_user'@'${PRIVATE_2_IP}';

-- Flush privileges
FLUSH PRIVILEGES;

-- Show created users
SELECT User, Host FROM mysql.user WHERE User = 'ga_app_user';
EOSQL

echo "✅ MySQL user 'ga_app_user' created with access from all instances"
echo "📊 Database 'ga_framework' created"

# Verify MySQL is running
echo ""
echo "MySQL Status:"
sudo systemctl status mysql --no-pager | head -10

echo ""
echo "✅ MySQL setup complete"
echo "📍 MySQL Server listening on port 3306"
echo "👤 Username: ga_app_user"
echo "🗄️  Database: ga_framework"
