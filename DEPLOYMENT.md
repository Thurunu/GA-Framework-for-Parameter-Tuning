# AWS EC2 Deployment Guide - GA Framework

## 🏗️ Architecture Overview

This deployment uses **3 EC2 instances** with the following roles:

```
┌─────────────────────────────────────────────┐
│         PUBLIC INSTANCE                     │
│  ┌──────────────────────────────────────┐  │
│  │ Docker                                │  │
│  │  ├─ Prometheus (Port 9090)           │  │
│  │  └─ Grafana (Port 3000)              │  │
│  └──────────────────────────────────────┘  │
└───────────────┬─────────────────────────────┘
                │ Scrapes metrics
    ┌───────────┴───────────┐
    │                       │
┌───▼───────────┐     ┌─────▼─────────────────┐
│ PRIVATE       │     │ PRIVATE               │
│ INSTANCE 1    │     │ INSTANCE 2            │
│ ┌───────────┐ │     │ ┌─────────────────┐  │
│ │Node       │ │     │ │MySQL Server     │  │
│ │Exporter   │ │     │ │(Port 3306)      │  │
│ │:9100      │ │     │ └─────────────────┘  │
│ └───────────┘ │     │ ┌─────────────────┐  │
└───────────────┘     │ │MySQL Exporter   │  │
                      │ │:9104            │  │
                      │ └─────────────────┘  │
                      │ ┌─────────────────┐  │
                      │ │Node Exporter    │  │
                      │ │:9100            │  │
                      │ └─────────────────┘  │
                      │ ┌─────────────────┐  │
                      │ │GA Framework App │  │
                      │ │(Python)         │  │
                      │ └─────────────────┘  │
                      └─────────────────────┘
```

## 📋 Prerequisites

### 1. Terraform Infrastructure (Managed Locally)
You manage infrastructure with Terraform locally connected to HashiCorp Cloud.

After running `terraform apply`, get the instance IPs:
```bash
cd infrastructure
terraform output
```

### 2. GitHub Secrets Configuration

Go to **Repository Settings → Secrets and variables → Actions** and add:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `PUBLIC_INSTANCE_IP` | Public EC2 instance IP | `13.127.45.123` |
| `PRIVATE_INSTANCE_1_IP` | Private instance 1 IP | `13.127.45.124` |
| `PRIVATE_INSTANCE_2_IP` | Private instance 2 IP | `13.127.45.125` |
| `EC2_SSH_KEY` | Complete SSH private key | `-----BEGIN RSA PRIVATE KEY-----\n...` |
| `MYSQL_PASSWORD` | MySQL password | `SecurePassword123!` |
| `GRAFANA_PASSWORD` | Grafana admin password (optional) | `admin` |

## 🚀 Deployment Workflow

### Automatic Deployment

The GitHub Actions workflow (`deploy-aws.yml`) automatically runs when you:
- Push to `main` or `development` branches
- Manually trigger via Actions tab

### Deployment Steps

1. **SSH Setup** - Configures SSH keys for all instances
2. **File Copy** - Copies application files to all instances
3. **Public Instance Setup**:
   - Installs Docker
   - Configures Prometheus with targets
   - Starts Prometheus container
   - Starts Grafana container
4. **Private Instance 1 Setup**:
   - Installs Node Exporter
   - Configures systemd service
5. **Private Instance 2 Setup**:
   - Installs MySQL Server
   - Creates database and users
   - Installs Python dependencies
   - Installs Node Exporter
   - Installs MySQL Exporter
   - Starts GA Framework application
6. **Health Checks** - Verifies all services

## 📊 Accessing Services

### Prometheus (Metrics Aggregation)
```
URL: http://<PUBLIC_INSTANCE_IP>:9090
```

Available targets:
- Prometheus itself
- Node Exporter (Private Instance 1)
- Node Exporter (Private Instance 2)
- MySQL Exporter (Private Instance 2)

### Grafana (Dashboards)
```
URL: http://<PUBLIC_INSTANCE_IP>:3000
Username: admin
Password: <GRAFANA_PASSWORD from secrets>
```

**First Time Setup:**
1. Add Prometheus as data source: `http://prometheus:9090`
2. Import dashboards:
   - Node Exporter Full: Dashboard ID `1860`
   - MySQL Overview: Dashboard ID `7362`

### MySQL Database
```
Host: <PRIVATE_INSTANCE_2_IP>
Port: 3306
Username: ga_app_user
Password: <MYSQL_PASSWORD>
Database: ga_framework
```

**Connect from any instance:**
```bash
mysql -h <PRIVATE_INSTANCE_2_IP> -u ga_app_user -p ga_framework
```

### Metrics Endpoints

**Private Instance 1:**
```
http://<PRIVATE_INSTANCE_1_IP>:9100/metrics
```

**Private Instance 2:**
```
Node Exporter: http://<PRIVATE_INSTANCE_2_IP>:9100/metrics
MySQL Exporter: http://<PRIVATE_INSTANCE_2_IP>:9104/metrics
```

## 🛠️ Manual Deployment

If you need to deploy manually:

```bash
# 1. Get instance IPs from Terraform
cd infrastructure
terraform output

# 2. SSH into Public Instance
ssh -i my-key.pem ubuntu@<PUBLIC_INSTANCE_IP>

# 3. Clone or copy your code
git clone <repo-url> /home/ubuntu/ga-framework
cd /home/ubuntu/ga-framework/scripts

# 4. Run setup script
chmod +x setup_public_instance.sh
./setup_public_instance.sh <PRIVATE_1_IP> <PRIVATE_2_IP>

# Repeat for other instances with their respective scripts
```

## 🔍 Monitoring & Debugging

### Check Service Status

**Public Instance:**
```bash
ssh ubuntu@<PUBLIC_IP>
sudo docker ps
sudo docker logs prometheus
sudo docker logs grafana
```

**Private Instance 1:**
```bash
ssh ubuntu@<PRIVATE_1_IP>
systemctl status node_exporter
curl localhost:9100/metrics | head
```

**Private Instance 2:**
```bash
ssh ubuntu@<PRIVATE_2_IP>
systemctl status mysql
systemctl status node_exporter
systemctl status mysql_exporter
curl localhost:9100/metrics | head
curl localhost:9104/metrics | head
```

### View Application Logs

```bash
ssh ubuntu@<PRIVATE_2_IP>
cd /home/ubuntu/ga-framework
source venv/bin/activate
# Check running processes, logs, etc.
```

### Restart Services

**Prometheus/Grafana:**
```bash
ssh ubuntu@<PUBLIC_IP>
sudo docker restart prometheus grafana
```

**Exporters:**
```bash
ssh ubuntu@<INSTANCE_IP>
sudo systemctl restart node_exporter
sudo systemctl restart mysql_exporter  # Only on Private Instance 2
```

## 🔐 Security Considerations

### Security Groups (Terraform)

Ensure your security groups allow:

**Public Instance:**
- Port 22 (SSH) - Your IP only
- Port 9090 (Prometheus) - Your IP only
- Port 3000 (Grafana) - Your IP only

**Private Instance 1:**
- Port 22 (SSH) - Your IP only
- Port 9100 (Node Exporter) - From Public Instance

**Private Instance 2:**
- Port 22 (SSH) - Your IP only
- Port 3306 (MySQL) - From all instances within VPC
- Port 9100 (Node Exporter) - From Public Instance
- Port 9104 (MySQL Exporter) - From Public Instance

### Best Practices

1. ✅ Use strong passwords for MySQL and Grafana
2. ✅ Restrict SSH access to your IP only
3. ✅ Keep security group rules minimal
4. ✅ Regularly update packages on instances
5. ✅ Use private subnets for Private Instances (if possible)
6. ✅ Enable CloudWatch logging
7. ✅ Rotate secrets regularly

## 🔄 Updating the Application

### Via GitHub Actions (Recommended)
```bash
git add .
git commit -m "Update application"
git push origin main
```

The workflow will automatically:
1. Copy updated files
2. Restart services
3. Run health checks

### Manual Update
```bash
ssh ubuntu@<PRIVATE_2_IP>
cd /home/ubuntu/ga-framework
git pull
source venv/bin/activate
pip install -r requirements.txt
cd scripts
./start_application.sh
```

## 📦 Backup & Recovery

### MySQL Backup
```bash
ssh ubuntu@<PRIVATE_2_IP>
mysqldump -u ga_app_user -p ga_framework > backup_$(date +%Y%m%d).sql
```

### Restore MySQL
```bash
mysql -u ga_app_user -p ga_framework < backup_20250421.sql
```

### Grafana Dashboards
Dashboards are stored in Docker volumes. To backup:
```bash
ssh ubuntu@<PUBLIC_IP>
sudo docker exec grafana cat /var/lib/grafana/grafana.db > grafana_backup.db
```

## 🚨 Troubleshooting

### Prometheus Not Scraping Targets

1. Check if exporters are running:
```bash
curl http://<INSTANCE_IP>:9100/metrics
```

2. Check Prometheus logs:
```bash
sudo docker logs prometheus
```

3. Verify network connectivity:
```bash
telnet <INSTANCE_IP> 9100
```

### MySQL Connection Issues

1. Check MySQL is running:
```bash
systemctl status mysql
```

2. Verify user permissions:
```bash
mysql -u root
SELECT User, Host FROM mysql.user WHERE User = 'ga_app_user';
```

3. Check MySQL logs:
```bash
sudo tail -f /var/log/mysql/error.log
```

### Application Not Starting

1. Check Python environment:
```bash
source /home/ubuntu/ga-framework/venv/bin/activate
python --version
pip list
```

2. Check for errors:
```bash
cd /home/ubuntu/ga-framework
python src/MainIntegration.py
```

3. Review logs if service-based:
```bash
journalctl -u ga-framework -f
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks

**Weekly:**
- Check Prometheus/Grafana dashboards for anomalies
- Review application logs
- Verify all exporters are reporting

**Monthly:**
- Update system packages: `sudo apt update && sudo apt upgrade`
- Backup MySQL database
- Review and rotate secrets if needed

**Quarterly:**
- Update exporter versions
- Review security group rules
- Audit access logs

---

**Repository:** GA-Framework-for-Parameter-Tuning  
**Last Updated:** October 21, 2025
