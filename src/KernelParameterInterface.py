#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Kernel Parameter Interface
This module handles kernel parameter reading, writing, and validation
"""

import os
import subprocess
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class KernelParameter:
    """Data class representing a kernel parameter"""
    name: str
    current_value: Any
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    subsystem: str = ""
    writable: bool = True
    requires_reboot: bool = False

class KernelParameterInterface:
    """Interface for managing Linux kernel parameters"""
    
    def __init__(self, backup_dir: str = "/tmp/kernel_optimizer_backup"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Define key parameters for optimization based on your paper
        self.optimization_parameters = {
            # Memory Management Parameters
            'vm.swappiness': KernelParameter(
                name='vm.swappiness',
                current_value=None,
                default_value=60,
                min_value=0,
                max_value=100,
                description='Controls swap usage tendency',
                subsystem='memory'
            ),
            'vm.dirty_ratio': KernelParameter(
                name='vm.dirty_ratio',
                current_value=None,
                default_value=20,
                min_value=1,
                max_value=90,
                description='Percentage of memory that can be dirty before writeback',
                subsystem='memory'
            ),
            'vm.dirty_background_ratio': KernelParameter(
                name='vm.dirty_background_ratio',
                current_value=None,
                default_value=10,
                min_value=1,
                max_value=50,
                description='Percentage for background writeback',
                subsystem='memory'
            ),
            'vm.vfs_cache_pressure': KernelParameter(
                name='vm.vfs_cache_pressure',
                current_value=None,
                default_value=100,
                min_value=1,
                max_value=1000,
                description='Controls VFS cache reclaim pressure',
                subsystem='memory'
            ),
            
            # CPU Scheduling Parameters (EEVDF-compatible for kernel 6.6+)
            'kernel.sched_cfs_bandwidth_slice_us': KernelParameter(
                name='kernel.sched_cfs_bandwidth_slice_us',
                current_value=None,
                default_value=5000,      # 5ms
                min_value=1000,          # 1ms
                max_value=20000,         # 20ms
                description='CFS bandwidth time slice in microseconds (EEVDF scheduler)',
                subsystem='cpu'
            ),
            'kernel.sched_latency_ns': KernelParameter(
                name='kernel.sched_latency_ns',
                current_value=None,
                default_value=6000000,   # 6ms
                min_value=1000000,       # 1ms
                max_value=50000000,      # 50ms
                description='Target preemption latency for CPU-bound tasks',
                subsystem='cpu'
            ),
            'kernel.sched_rt_period_us': KernelParameter(
                name='kernel.sched_rt_period_us',
                current_value=None,
                default_value=1000000,   # 1 second
                min_value=1,
                max_value=10000000,      # 10 seconds
                description='Period over which RT task bandwidth is measured',
                subsystem='cpu'
            ),
            'kernel.sched_rt_runtime_us': KernelParameter(
                name='kernel.sched_rt_runtime_us',
                current_value=None,
                default_value=950000,    # 950ms (95% of period)
                min_value=0,
                max_value=1000000,       # 1 second
                description='Portion of period that RT tasks can use',
                subsystem='cpu'
            ),
            
            # Network Parameters
            'net.core.rmem_max': KernelParameter(
                name='net.core.rmem_max',
                current_value=None,
                default_value=212992,
                min_value=8192,
                max_value=134217728,     # 128MB
                description='Maximum receive buffer size',
                subsystem='network'
            ),
            'net.core.wmem_max': KernelParameter(
                name='net.core.wmem_max',
                current_value=None,
                default_value=212992,
                min_value=8192,
                max_value=134217728,     # 128MB
                description='Maximum send buffer size',
                subsystem='network'
            ),
            'net.core.netdev_max_backlog': KernelParameter(
                name='net.core.netdev_max_backlog',
                current_value=None,
                default_value=1000,
                min_value=100,
                max_value=10000,
                description='Maximum packets in network device queue',
                subsystem='network'
            ),
            'net.ipv4.tcp_rmem': KernelParameter(
                name='net.ipv4.tcp_rmem',
                current_value=None,
                default_value='4096 87380 6291456',
                min_value=None,
                max_value=None,
                description='TCP receive buffer sizes (min default max)',
                subsystem='network'
            ),
            'net.ipv4.tcp_wmem': KernelParameter(
                name='net.ipv4.tcp_wmem',
                current_value=None,
                default_value='4096 16384 4194304',
                min_value=None,
                max_value=None,
                description='TCP send buffer sizes (min default max)',
                subsystem='network'
            ),
            'net.ipv4.tcp_congestion_control': KernelParameter(
                name='net.ipv4.tcp_congestion_control',
                current_value=None,
                default_value='cubic',
                min_value=None,
                max_value=None,
                description='TCP congestion control algorithm',
                subsystem='network'
            ),
            
            # File System Parameters
            'fs.file-max': KernelParameter(
                name='fs.file-max',
                current_value=None,
                default_value=2097152,
                min_value=1024,
                max_value=10485760,
                description='Maximum number of file handles',
                subsystem='filesystem'
            ),
            'fs.nr_open': KernelParameter(
                name='fs.nr_open',
                current_value=None,
                default_value=1048576,
                min_value=1024,
                max_value=10485760,
                description='Maximum file descriptors per process',
                subsystem='filesystem'
            )
        }
        
        # Setup logging first (before loading current values)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load current values
        self._load_current_values()
    
    def _load_current_values(self):
        """Load current kernel parameter values"""
        for param_name, param in self.optimization_parameters.items():
            try:
                current_val = self._read_parameter(param_name)
                param.current_value = current_val
            except Exception as e:
                self.logger.warning("Could not read parameter %s: %s", param_name, e)
                param.current_value = param.default_value
    
    def _read_parameter(self, param_name: str) -> Any:
        """Read a single kernel parameter value"""
        try:
            # Check if running on Linux/Unix system
            if os.name == 'posix':
                # Use sysctl to read parameter
                result = subprocess.run(
                    ['sysctl', '-n', param_name],
                    capture_output=True,
                    text=True,
                    check=True
                )
                value = result.stdout.strip()
                
                # Try to convert to appropriate type
                if value.isdigit():
                    return int(value)
                elif value.replace('.', '').isdigit():
                    return float(value)
                else:
                    return value
            else:
                # Windows fallback - return default value with warning
                print(f"Warning: Cannot read kernel parameter {param_name} on Windows. Using default value.")
                param_info = self.optimization_parameters.get(param_name)
                return param_info.default_value if param_info else 0
                    
        except subprocess.CalledProcessError:
            # Fallback: try reading from /proc/sys (Linux only)
            if os.name == 'posix':
                proc_path = f"/proc/sys/{param_name.replace('.', '/')}"
                try:
                    with open(proc_path, 'r', encoding='utf-8') as f:
                        value = f.read().strip()
                        if value.isdigit():
                            return int(value)
                        elif value.replace('.', '').isdigit():
                            return float(value)
                        else:
                            return value
                except (FileNotFoundError, PermissionError):
                    # Parameter doesn't exist on this system, return default
                    self.logger.info("Parameter %s not available on this system, using default", param_name)
                    param_info = self.optimization_parameters.get(param_name)
                    return param_info.default_value if param_info else 0
            else:
                # Windows fallback
                param_info = self.optimization_parameters.get(param_name)
                return param_info.default_value if param_info else 0
    
    def _write_parameter(self, param_name: str, value: Any) -> bool:
        """Write a kernel parameter value"""
        try:
            # Check if running on Linux/Unix system
            if os.name == 'posix':
                # Convert value to string
                str_value = str(int(value))
                print("------------------")
                # Use sysctl to write parameter
                result = subprocess.run(
                    ['sysctl', '-w', f'{param_name}={str_value}'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                self.logger.info("Set %s = %s", param_name, str_value)
                return True
            else:
                # Windows - simulate successful write for testing
                print(f"Simulated setting {param_name} = {value} (Windows)")
                return True
                
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to set %s = %s: %s", param_name, value, e)
            return False
    
    def backup_current_parameters(self) -> str:
        """Create backup of current kernel parameters"""
        timestamp = int(time.time())
        backup_file = self.backup_dir / f"kernel_params_backup_{timestamp}.json"
        
        backup_data = {
            'timestamp': timestamp,
            'parameters': {}
        }
        
        for param_name, param in self.optimization_parameters.items():
            backup_data['parameters'][param_name] = {
                'value': param.current_value,
                'subsystem': param.subsystem
            }
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2)
        
        self.logger.info("Created backup: %s", backup_file)
        return str(backup_file)
    
    def restore_from_backup(self, backup_file: str) -> bool:
        """Restore parameters from backup file"""
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            success_count = 0
            for param_name, param_data in backup_data['parameters'].items():
                if self._write_parameter(param_name, param_data['value']):
                    success_count += 1
            
            self.logger.info("Restored %s/%s parameters", success_count, len(backup_data['parameters']))
            self._load_current_values()  # Reload current values
            return success_count == len(backup_data['parameters'])
            
        except Exception as e:
            self.logger.error("Failed to restore from backup: %s", e)
            return False
    
    def apply_parameter_set(self, parameters: Dict[str, Any]) -> Dict[str, bool]:
        """Apply a set of kernel parameters"""
        results = {}
        
        # Create backup before applying changes
        self.backup_current_parameters()
        
        for param_name, value in parameters.items():
            if param_name in self.optimization_parameters:
                # Validate value is within bounds
                param = self.optimization_parameters[param_name]
                if self._validate_parameter_value(param, value):
                    results[param_name] = self._write_parameter(param_name, value)
                    if results[param_name]:
                        param.current_value = value
                else:
                    results[param_name] = False
                    self.logger.warning("Invalid value %s for parameter %s", value, param_name)
            else:
                results[param_name] = False
                self.logger.warning("Unknown parameter: %s", param_name)
        
        return results
    
    def _validate_parameter_value(self, param: KernelParameter, value: Any) -> bool:
        """Validate parameter value against constraints"""
        if param.min_value is not None:
            try:
                if float(value) < float(param.min_value):
                    return False
            except (ValueError, TypeError):
                pass
        
        if param.max_value is not None:
            try:
                if float(value) > float(param.max_value):
                    return False
            except (ValueError, TypeError):
                pass
        
        return True
    
    def get_parameter_info(self, param_name: str) -> Optional[KernelParameter]:
        """Get information about a specific parameter"""
        return self.optimization_parameters.get(param_name)
    
    def get_parameters_by_subsystem(self, subsystem: str) -> Dict[str, KernelParameter]:
        """Get all parameters belonging to a specific subsystem"""
        return {
            name: param for name, param in self.optimization_parameters.items()
            if param.subsystem == subsystem
        }
    
    def get_all_parameters(self) -> Dict[str, KernelParameter]:
        """Get all optimization parameters"""
        return self.optimization_parameters.copy()
    
    def get_current_configuration(self) -> Dict[str, Any]:
        """Get current parameter configuration as a simple dict"""
        return {
            name: param.current_value 
            for name, param in self.optimization_parameters.items()
        }
    
    def reset_to_defaults(self) -> Dict[str, bool]:
        """Reset all parameters to their default values"""
        default_config = {
            name: param.default_value 
            for name, param in self.optimization_parameters.items()
        }
        return self.apply_parameter_set(default_config)
    
    def export_current_config(self, filename: str):
        """Export current configuration to file"""
        config_data = {
            'timestamp': time.time(),
            'parameters': {}
        }
        
        for name, param in self.optimization_parameters.items():
            config_data['parameters'][name] = {
                'current_value': param.current_value,
                'default_value': param.default_value,
                'subsystem': param.subsystem,
                'description': param.description
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info("Exported configuration to %s", filename)

# Example usage and testing
if __name__ == "__main__":
    # Note: This requires root privileges to modify kernel parameters
    kernel_interface = KernelParameterInterface()
    
    print("Current Kernel Parameters:")
    print("=" * 50)
    
    current_config = kernel_interface.get_current_configuration()
    for param_name, value in current_config.items():
        param_info = kernel_interface.get_parameter_info(param_name)
        print(f"{param_name}: {value} (default: {param_info.default_value}) [{param_info.subsystem}]")
    
    print("\nParameters by Subsystem:")
    print("=" * 50)
    
    subsystems = set(param.subsystem for param in kernel_interface.optimization_parameters.values())
    for subsystem in subsystems:
        params = kernel_interface.get_parameters_by_subsystem(subsystem)
        print(f"\n{subsystem.upper()}:")
        for name, param in params.items():
            print(f"  {name}: {param.current_value}")
    
    # Export current configuration
    kernel_interface.export_current_config("current_kernel_config.json")
    
    print(f"\nConfiguration exported to current_kernel_config.json")
    print(f"Backup directory: {kernel_interface.backup_dir}")