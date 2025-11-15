#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Kernel Parameter Interface
This module handles kernel parameter reading, writing, and validation
"""

import os
import subprocess
import json
import time
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
from pathlib import Path
import logging

# Import data classes and config loader
from DataClasses import KernelParameter
from LoadConfigs import LoadConfigs

class KernelParameterInterface:
    """Interface for managing Linux kernel parameters"""
    
    def __init__(self, backup_dir: str = "/tmp/kernel_optimizer_backup", config_file: str = None):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Load kernel parameters from YAML configuration
        config_loader = LoadConfigs()
        self.optimization_parameters = config_loader.load_kernel_parameters(config_file)
        
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
                    self.logger.warning(
                        "Invalid value %s for parameter %s (valid range: %s to %s)",
                        value, param_name, param.min_value, param.max_value
                    )
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