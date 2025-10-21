# Deployment Scripts

This directory contains reusable bash scripts for deploying and managing the GA Framework application on EC2 instances.

## Scripts Overview

### 1. `install_dependencies.sh`
**Purpose:** Install Python and all project dependencies

**Usage:**
```bash
./install_dependencies.sh [app_directory]
```

**What it does:**
- Installs Python3, pip, and venv if not present
- Creates a virtual environment
- Installs all packages from `requirements.txt`
- Displays installed versions

**Example:**
```bash
cd /home/ubuntu/ga-framework/scripts
chmod +x install_dependencies.sh
./install_dependencies.sh /home/ubuntu/ga-framework
```

---

### 2. `install_docker.sh`
**Purpose:** Install Docker and Docker Compose

**Usage:**
```bash
./install_docker.sh
```

**What it does:**
- Checks if Docker is already installed
- Installs Docker CE if needed
- Installs Docker Compose if needed
- Adds current user to docker group
- Displays installed versions

**Example:**
```bash
cd /home/ubuntu/ga-framework/scripts
chmod +x install_docker.sh
./install_docker.sh
```

---

### 3. `install_mysql.sh`
**Purpose:** Install and configure MySQL server with remote access

**Usage:**
```bash
./install_mysql.sh <public_ip> <private_1_ip> <private_2_ip> <mysql_password>
```

**Parameters:**
- `public_ip`: IP address of public instance
- `private_1_ip`: IP address of private instance 1
- `private_2_ip`: IP address of private instance 2
- `mysql_password`: Password for the MySQL user

**What it does:**
- Installs MySQL Server
- Configures MySQL to accept remote connections
- Creates `ga_framework` database
- Creates `ga_app_user` with access from specified IPs
- Grants necessary privileges

**Example:**
```bash
cd /home/ubuntu/ga-framework/scripts
chmod +x install_mysql.sh
./install_mysql.sh "13.127.45.123" "13.127.45.124" "13.127.45.125" "mySecurePassword123"
```

**Database Connection Info:**
- Host: Private Instance 2 IP
- Port: 3306
- Username: `ga_app_user`
- Database: `ga_framework`

---

### 4. `start_application.sh`
**Purpose:** Start the GA Framework application

**Usage:**
```bash
./start_application.sh [app_directory]
```

**What it does:**
- Activates Python virtual environment
- Runs `install_service.py` if available
- Starts Docker containers if `docker-compose.yml` exists
- Displays container status

**Example:**
```bash
cd /home/ubuntu/ga-framework/scripts
chmod +x start_application.sh
./start_application.sh /home/ubuntu/ga-framework
```

---

### 5. `health_check.sh`
**Purpose:** Verify application health and configuration

**Usage:**
```bash
./health_check.sh [app_directory]
```

**What it does:**
- Checks Python version
- Verifies key dependencies (numpy, scipy)
- Tests application module imports
- Displays Docker container status
- Shows disk space and memory usage

**Example:**
```bash
cd /home/ubuntu/ga-framework/scripts
chmod +x health_check.sh
./health_check.sh /home/ubuntu/ga-framework
```

---

## GitHub Actions Integration

These scripts are automatically executed by the GitHub Actions workflow (`.github/workflows/aws.yml`) during deployment:

```yaml
# Python dependencies installation
- name: Install Python dependencies on all instances
  run: |
    ssh -i ~/.ssh/deploy_key.pem ubuntu@$INSTANCE_IP bash << 'EOF'
      cd /home/ubuntu/ga-framework/scripts
      chmod +x install_dependencies.sh
      ./install_dependencies.sh /home/ubuntu/ga-framework
    EOF

# Docker installation
- name: Setup Docker on all instances
  run: |
    ssh -i ~/.ssh/deploy_key.pem ubuntu@$INSTANCE_IP bash << 'EOF'
      cd /home/ubuntu/ga-framework/scripts
      chmod +x install_docker.sh
      ./install_docker.sh
    EOF

# MySQL installation (Private Instance 2 only)
- name: Install MySQL and create user for all instances
  run: |
    ssh -i ~/.ssh/deploy_key.pem ubuntu@$INSTANCE_IP bash << ENDSSH
      cd /home/ubuntu/ga-framework/scripts
      chmod +x install_mysql.sh
      ./install_mysql.sh "$PUBLIC_IP" "$PRIVATE_1_IP" "$PRIVATE_2_IP" "$MYSQL_PASSWORD"
    ENDSSH

# Application startup
- name: Start application on all instances
  run: |
    ssh -i ~/.ssh/deploy_key.pem ubuntu@$INSTANCE_IP bash << 'EOF'
      cd /home/ubuntu/ga-framework/scripts
      chmod +x start_application.sh
      ./start_application.sh /home/ubuntu/ga-framework
    EOF

# Health checks
- name: Health check all instances
  run: |
    ssh -i ~/.ssh/deploy_key.pem ubuntu@$INSTANCE_IP bash << 'EOF'
      cd /home/ubuntu/ga-framework/scripts
      chmod +x health_check.sh
      ./health_check.sh /home/ubuntu/ga-framework
    EOF
```

## Benefits of This Approach

### 1. **Cleaner Workflow File**
- GitHub Actions workflow is now much shorter and readable
- Easy to understand deployment flow
- Less chance of YAML syntax errors

### 2. **Reusable Scripts**
- Scripts can be run manually for debugging
- Can be used in other CI/CD tools
- Easy to test locally before deploying

### 3. **Maintainability**
- Changes to installation logic only require updating one script
- Easier to version control
- Better separation of concerns

### 4. **Debugging**
- Can SSH into instances and run scripts manually
- Scripts have better error messages
- Easier to test individual components

### 5. **Flexibility**
- Scripts can be called from anywhere
- Easy to add new functionality
- Can be integrated with other tools

## Manual Deployment

You can also use these scripts for manual deployment:

```bash
# 1. SSH into instance
ssh -i my-key.pem ubuntu@<instance-ip>

# 2. Navigate to application directory
cd /home/ubuntu/ga-framework/scripts

# 3. Make scripts executable (one time)
chmod +x *.sh

# 4. Run scripts in order
./install_dependencies.sh /home/ubuntu/ga-framework
./install_docker.sh
./start_application.sh /home/ubuntu/ga-framework
./health_check.sh /home/ubuntu/ga-framework
```

## Troubleshooting

### Script Permission Denied
```bash
chmod +x /home/ubuntu/ga-framework/scripts/*.sh
```

### MySQL Installation Fails
```bash
# Check if MySQL is already installed
systemctl status mysql

# View installation logs
sudo journalctl -u mysql
```

### Docker Installation Fails
```bash
# Check Docker status
systemctl status docker

# Verify Docker group
groups $USER
```

### Health Check Fails
```bash
# Check virtual environment
source /home/ubuntu/ga-framework/venv/bin/activate
pip list

# Check application files
ls -la /home/ubuntu/ga-framework/
```

## Adding New Scripts

To add new deployment scripts:

1. Create script in `scripts/` directory
2. Make it executable: `chmod +x script_name.sh`
3. Add documentation to this README
4. Update GitHub Actions workflow to use the new script
5. Test manually before committing

---

**Last Updated:** October 21, 2025
**Maintained by:** GA Framework Team
