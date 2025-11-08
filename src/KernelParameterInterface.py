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
    
    def __init__(self, backup_dir: str = "/tmp/kernel_optimizer_backup", config_file: str = None, dry_run: bool = False):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Setup logging first (before loading parameters)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Dry-run mode: simulate writes, still perform real reads when possible
        self.dry_run = dry_run
        
        # Load kernel parameters from YAML configuration
        self.optimization_parameters = self._load_parameters(config_file)
        
        # Load current values
        self._load_current_values()
        
        # Check parameter availability and warn about unavailable ones
        self._check_system_compatibility()
    
    def _load_parameters(self, config_file: str = None) -> Dict[str, KernelParameter]:
        """Load kernel parameter definitions from YAML configuration file"""
        if config_file is None:
            # Default to config/kernel_parameters.yml relative to project root
            current_dir = Path(__file__).parent
            config_file = current_dir.parent / "config" / "kernel_parameters.yml"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            parameters = {}
            for param_name, param_data in config['parameters'].items():
                parameters[param_name] = KernelParameter(
                    name=param_name,
                    current_value=None,  # Will be loaded later
                    default_value=param_data['default_value'],
                    min_value=param_data.get('min_value'),
                    max_value=param_data.get('max_value'),
                    description=param_data.get('description', ''),
                    subsystem=param_data.get('subsystem', ''),
                    writable=param_data.get('writable', True),
                    requires_reboot=param_data.get('requires_reboot', False)
                )
            
            print(f"Loaded {len(parameters)} kernel parameters from configuration")
            return parameters
            
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found. Using default parameters.")
            return self._get_default_parameters()
        except Exception as e:
            print(f"Error loading kernel parameters: {e}")
            print("Falling back to default parameters.")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, KernelParameter]:
        """Get default kernel parameters as fallback"""
        # Minimal fallback set of critical parameters
        return {
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
            )
        }
    
    def _check_system_compatibility(self):
        """Check which parameters are available on this system"""
        if os.name != 'posix':
            self.logger.info("Running on Windows - parameter checks skipped")
            return
        
        unavailable_params = []
        for param_name in list(self.optimization_parameters.keys()):
            if not self.check_parameter_availability(param_name):
                unavailable_params.append(param_name)
        
        if unavailable_params:
            self.logger.warning(
                "The following parameters are not available on this system and will be skipped: %s",
                ", ".join(unavailable_params)
            )
            self.logger.info(
                "This may be due to: kernel version, missing kernel modules, or system configuration. "
                "Optimization will continue with available parameters only."
            )
    
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
            # In dry-run mode, simulate success for known/available params after validation
            if getattr(self, 'dry_run', False):
                # Only simulate for available parameters
                if os.name != 'posix' or self.check_parameter_availability(param_name):
                    self.logger.info("[dry-run] Would set %s = %s", param_name, value)
                    return True
                self.logger.warning("[dry-run] Parameter %s not available, skipping", param_name)
                return False
            
            # Check if running on Linux/Unix system
            if os.name == 'posix':
                # Convert value to string while preserving multi-value tokens
                if isinstance(value, (int, float)):
                    str_value = str(int(value)) if isinstance(value, int) or float(value).is_integer() else str(float(value))
                else:
                    # Accept strings like "4 4 1 7" or "4096 87380 6291456"
                    str_value = str(value).strip()
                
                # Try sysctl method first
                result = subprocess.run(
                    ['sysctl', '-w', f'{param_name}={str_value}'],
                    capture_output=True,
                    text=True,
                    check=False  # Don't raise exception, check manually
                )
                
                if result.returncode == 0:
                    self.logger.info("Set %s = %s", param_name, str_value)
                    return True
                else:
                    # Log the actual error from sysctl
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    self.logger.warning("sysctl failed for %s: %s", param_name, error_msg)
                    
                    # Try alternative method: write directly to /proc/sys
                    proc_path = f"/proc/sys/{param_name.replace('.', '/')}"
                    try:
                        with open(proc_path, 'w', encoding='utf-8') as f:
                            f.write(str_value)
                        self.logger.info("Set %s = %s (via /proc/sys)", param_name, str_value)
                        return True
                    except (FileNotFoundError, PermissionError, OSError) as proc_error:
                        self.logger.error(
                            "Failed to set %s = %s: Parameter may not exist or is read-only. "
                            "sysctl error: %s, /proc/sys error: %s",
                            param_name, str_value, error_msg, str(proc_error)
                        )
                        return False
            else:
                # Windows - simulate successful write for testing
                print(f"Simulated setting {param_name} = {value} (Windows)")
                return True
                
        except Exception as e:
            self.logger.error("Unexpected error setting %s = %s: %s", param_name, value, e)
            return False

    # Public helper APIs expected by integration tests
    def set_parameter(self, param_name: str, value: Any) -> bool:
        """Set a single parameter value with validation and dry-run support"""
        # If we know the parameter, validate range and track current_value
        if param_name in self.optimization_parameters:
            param = self.optimization_parameters[param_name]
            if not self.check_parameter_availability(param_name) and os.name == 'posix':
                self.logger.warning("Parameter %s not available on this system", param_name)
                return False
            if not self._validate_parameter_value(param, value):
                self.logger.warning("Rejected %s=%s due to bounds [%s, %s]", param_name, value, param.min_value, param.max_value)
                return False
            success = self._write_parameter(param_name, value)
            if success:
                # Update in-memory current value on success (or dry-run)
                param.current_value = value
            return success
        else:
            # Unknown parameter: attempt write only if not in dry-run and on posix; otherwise skip
            if getattr(self, 'dry_run', False):
                self.logger.info("[dry-run] Unknown parameter %s, skipping", param_name)
                return False
            return self._write_parameter(param_name, value)

    def get_parameter(self, param_name: str) -> Optional[Any]:
        """Read a parameter value; prefer live read, fall back to cached/default"""
        try:
            value = self._read_parameter(param_name)
            return value
        except Exception:
            # Fallback to known parameter cache
            param = self.optimization_parameters.get(param_name)
            return param.current_value if param else None

    def validate_parameter(self, param_name: str, value: Any) -> bool:
        """Validate a value against known parameter bounds; True if unknown"""
        param = self.optimization_parameters.get(param_name)
        if not param:
            return True
        return self._validate_parameter_value(param, value)
    
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
    
    def check_parameter_availability(self, param_name: str) -> bool:
        """Check if a kernel parameter exists and is writable on the system"""
        if os.name != 'posix':
            return True  # On Windows, simulate availability for testing
        
        # Try to read the parameter
        try:
            result = subprocess.run(
                ['sysctl', '-n', param_name],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return True
            
            # Try /proc/sys path
            proc_path = f"/proc/sys/{param_name.replace('.', '/')}"
            return os.path.exists(proc_path) and os.access(proc_path, os.R_OK)
        except Exception:
            return False
    
    def get_available_parameters(self) -> Dict[str, KernelParameter]:
        """Get only the parameters that are available on this system"""
        available = {}
        for param_name, param in self.optimization_parameters.items():
            if self.check_parameter_availability(param_name):
                available[param_name] = param
            else:
                self.logger.info("Parameter %s is not available on this system, excluding from optimization", param_name)
        return available
    
    def apply_parameter_set(self, parameters: Dict[str, Any]) -> Dict[str, bool]:
        """Apply a set of kernel parameters"""
        results = {}
        
        # Create backup before applying changes
        self.backup_current_parameters()
        
        for param_name, value in parameters.items():
            if param_name in self.optimization_parameters:
                # Check if parameter is available on this system
                if not self.check_parameter_availability(param_name):
                    results[param_name] = False
                    self.logger.warning("Parameter %s is not available on this system, skipping", param_name)
                    continue
                
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