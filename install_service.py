#!/usr/bin/env python3
"""
Install Continuous Kernel Optimizer as a System Service
"""

import os
import sys
import shutil
from pathlib import Path

def install_service():
    """Install the continuous optimizer as a system service"""
    
    # Check if running as root
    if os.geteuid() != 0:
        print("Error: This script must be run as root (use sudo)")
        sys.exit(1)
    
    print("Installing Continuous Kernel Optimizer Service...")
    
    # Installation directory
    install_dir = Path("/opt/kernel-optimizer")
    
    # Create installation directory
    print(f"Creating installation directory: {install_dir}")
    install_dir.mkdir(parents=True, exist_ok=True)
    
    # Current directory (where the scripts are)
    current_dir = Path.cwd()
    
    # Copy Python files
    python_files = [
        "BayesianOptimzation.py",
        "GeneticAlgorithm.py", 
        "HybridOptimizationEngine.py",
        "PerformanceMonitor.py",
        "KernelParameterInterface.py",
        "ProcessWorkloadDetector.py",
        "ContinuousOptimizer.py",
        "MainIntegration.py",
        "requirements.txt"
    ]
    
    print("Copying framework files...")
    for file_name in python_files:
        src_file = current_dir / file_name
        if src_file.exists():
            dst_file = install_dir / file_name
            shutil.copy2(src_file, dst_file)
            print(f"  Copied: {file_name}")
        else:
            print(f"  Warning: {file_name} not found, skipping...")
    
    # Set permissions
    print("Setting permissions...")
    os.chown(install_dir, 0, 0)  # root:root
    for file_path in install_dir.glob("*.py"):
        os.chmod(file_path, 0o755)
        os.chown(file_path, 0, 0)
    
    # Create log directory
    log_dir = Path("/var/log/kernel-optimizer")
    print(f"Creating log directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    os.chown(log_dir, 0, 0)
    
    # Create log file
    log_file = log_dir / "continuous_optimizer.log"
    log_file.touch()
    os.chmod(log_file, 0o644)
    os.chown(log_file, 0, 0)
    
    # Create systemd service file
    service_content = """[Unit]
Description=Continuous Kernel Parameter Optimizer
Documentation=https://github.com/your-repo/kernel-optimizer
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=root
Group=root
ExecStart=/usr/bin/python3 /opt/kernel-optimizer/ContinuousOptimizer.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Set working directory
WorkingDirectory=/opt/kernel-optimizer

# Environment variables
Environment=PYTHONPATH=/opt/kernel-optimizer
Environment=PYTHONUNBUFFERED=1

# Resource limits to prevent excessive resource usage
MemoryLimit=1G
CPUQuota=20%

# Security settings
NoNewPrivileges=false
PrivateTmp=true
ProtectSystem=false
ProtectHome=true

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path("/etc/systemd/system/kernel-optimizer.service")
    print(f"Creating systemd service file: {service_file}")
    
    with open(service_file, 'w', encoding='utf-8') as f:
        f.write(service_content)
    
    os.chmod(service_file, 0o644)
    os.chown(service_file, 0, 0)
    
    # Install Python dependencies
    print("Installing Python dependencies...")
    os.system(f"pip3 install -r {install_dir}/requirements.txt")
    
    # Reload systemd
    print("Reloading systemd daemon...")
    os.system("systemctl daemon-reload")
    
    # Enable service
    print("Enabling kernel-optimizer service...")
    os.system("systemctl enable kernel-optimizer.service")
    
    print("\n" + "="*60)
    print("✅ Installation Complete!")
    print("="*60)
    print("The Continuous Kernel Optimizer has been installed as a system service.")
    print()
    print("Usage Commands:")
    print("  Start service:    sudo systemctl start kernel-optimizer")
    print("  Stop service:     sudo systemctl stop kernel-optimizer")
    print("  Service status:   sudo systemctl status kernel-optimizer")
    print("  View live logs:   sudo journalctl -u kernel-optimizer -f")
    print("  View log file:    sudo tail -f /var/log/kernel-optimizer/continuous_optimizer.log")
    print()
    print("Files installed to: /opt/kernel-optimizer/")
    print("Logs directory:     /var/log/kernel-optimizer/")
    print("Service file:       /etc/systemd/system/kernel-optimizer.service")
    print()
    print("⚠️  IMPORTANT:")
    print("  - This service modifies kernel parameters that affect system performance")
    print("  - It automatically creates backups before making changes")
    print("  - Monitor the service logs, especially during initial operation")
    print("  - Test on non-production systems first")
    print()
    print("To start the service now:")
    print("  sudo systemctl start kernel-optimizer")

def uninstall_service():
    """Uninstall the continuous optimizer service"""
    
    if os.geteuid() != 0:
        print("Error: This script must be run as root (use sudo)")
        sys.exit(1)
    
    print("Uninstalling Continuous Kernel Optimizer Service...")
    
    # Stop and disable service
    print("Stopping and disabling service...")
    os.system("systemctl stop kernel-optimizer.service")
    os.system("systemctl disable kernel-optimizer.service")
    
    # Remove service file
    service_file = Path("/etc/systemd/system/kernel-optimizer.service")
    if service_file.exists():
        service_file.unlink()
        print("Removed service file")
    
    # Reload systemd
    os.system("systemctl daemon-reload")
    
    # Ask about removing files
    response = input("Remove installation directory /opt/kernel-optimizer? [y/N]: ")
    if response.lower() in ['y', 'yes']:
        install_dir = Path("/opt/kernel-optimizer")
        if install_dir.exists():
            shutil.rmtree(install_dir)
            print("Removed installation directory")
    
    response = input("Remove log directory /var/log/kernel-optimizer? [y/N]: ")
    if response.lower() in ['y', 'yes']:
        log_dir = Path("/var/log/kernel-optimizer")
        if log_dir.exists():
            shutil.rmtree(log_dir)
            print("Removed log directory")
    
    print("Uninstallation complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_service()
    else:
        install_service()
