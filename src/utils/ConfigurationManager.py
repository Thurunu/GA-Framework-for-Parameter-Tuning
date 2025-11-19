#!/usr/bin/env python3
"""
Dynamic Configuration Manager
Handles real-time configuration updates from central management server
"""

import yaml
import json
import os
import sys
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

# Add project paths
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'utils'))
sys.path.insert(0, os.path.join(project_root, 'system'))
sys.path.insert(0, os.path.join(project_root, 'workload'))
sys.path.insert(0, os.path.join(project_root, 'data'))

from LoadConfigs import LoadConfigs
from KernelParameterInterface import KernelParameterInterface
from CentralDataStore import get_data_store

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Manages dynamic configuration updates from central server.
    Handles parameter additions, workload additions, and real-time updates.
    """

    def __init__(self, config_dir: str = None):
        """
        Initialize Configuration Manager

        Args:
            config_dir: Path to configuration directory (auto-detected if None)
        """
        if config_dir is None:
            config_dir = os.path.join(project_root, '..', 'config')
        
        self.config_dir = Path(config_dir)
        self.backup_dir = self.config_dir / 'backups'
        self.backup_dir.mkdir(exist_ok=True)

        # Configuration file paths
        self.kernel_params_file = self.config_dir / 'kernel_parameters.yml'
        self.workload_patterns_file = self.config_dir / 'workload_patterns.yml'
        self.optimization_profiles_file = self.config_dir / 'optimization_profiles.yml'
        self.process_priorities_file = self.config_dir / 'process_priorities.yml'

        # Config loader instance
        self.config_loader = LoadConfigs()

        # Change log for tracking updates
        self.change_log = []

        logger.info(f"Configuration Manager initialized: {self.config_dir}")

    # =========================================================================
    # BACKUP & RESTORE
    # =========================================================================

    def create_backup(self, config_type: str) -> str:
        """
        Create backup of configuration file before updating

        Args:
            config_type: Type of config ('kernel_params', 'workloads', etc.)

        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        config_files = {
            'kernel_params': self.kernel_params_file,
            'workloads': self.workload_patterns_file,
            'optimization_profiles': self.optimization_profiles_file,
            'process_priorities': self.process_priorities_file
        }

        source_file = config_files.get(config_type)
        if not source_file or not source_file.exists():
            logger.warning(f"Config file not found: {source_file}")
            return None

        backup_file = self.backup_dir / f"{config_type}_{timestamp}.yml"
        shutil.copy2(source_file, backup_file)
        
        logger.info(f"✅ Backup created: {backup_file}")
        return str(backup_file)

    def restore_from_backup(self, backup_file: str) -> bool:
        """
        Restore configuration from backup file

        Args:
            backup_file: Path to backup file

        Returns:
            True if restoration successful
        """
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False

            # Determine config type from filename
            filename = backup_path.stem
            for config_type in ['kernel_params', 'workloads', 'optimization_profiles', 'process_priorities']:
                if config_type in filename:
                    config_files = {
                        'kernel_params': self.kernel_params_file,
                        'workloads': self.workload_patterns_file,
                        'optimization_profiles': self.optimization_profiles_file,
                        'process_priorities': self.process_priorities_file
                    }
                    target_file = config_files[config_type]
                    shutil.copy2(backup_path, target_file)
                    logger.info(f"✅ Restored from backup: {backup_file}")
                    return True

            logger.error(f"Could not determine config type from: {backup_file}")
            return False

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    # =========================================================================
    # KERNEL PARAMETERS - ADD/UPDATE
    # =========================================================================

    def add_kernel_parameter(self, param_name: str, param_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Add a new kernel parameter to configuration

        Args:
            param_name: Name of parameter (e.g., 'vm.swappiness')
            param_config: Parameter configuration dict with keys:
                - default_value
                - min_value
                - max_value
                - description
                - subsystem
                - writable (default: True)
                - requires_reboot (default: False)

        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('kernel_params')

            # Load current configuration
            with open(self.kernel_params_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {'parameters': {}}

            # Check if parameter already exists
            if param_name in config['parameters']:
                return False, f"Parameter '{param_name}' already exists. Use update_kernel_parameter instead."

            # Validate required fields
            required_fields = ['default_value', 'min_value', 'max_value', 'description', 'subsystem']
            for field in required_fields:
                if field not in param_config:
                    return False, f"Missing required field: {field}"

            # Add parameter with defaults
            config['parameters'][param_name] = {
                'default_value': int(param_config['default_value']),
                'min_value': int(param_config['min_value']),
                'max_value': int(param_config['max_value']),
                'description': param_config['description'],
                'subsystem': param_config['subsystem'],
                'writable': param_config.get('writable', True),
                'requires_reboot': param_config.get('requires_reboot', False)
            }

            # Write updated configuration
            with open(self.kernel_params_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'add_parameter',
                'parameter': param_name,
                'config': param_config
            })

            logger.info(f"✅ Added kernel parameter: {param_name}")
            return True, f"Successfully added parameter: {param_name}"

        except Exception as e:
            logger.error(f"Failed to add kernel parameter: {e}")
            return False, f"Error: {str(e)}"

    def update_kernel_parameter(self, param_name: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update an existing kernel parameter configuration

        Args:
            param_name: Name of parameter to update
            updates: Dictionary of fields to update

        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('kernel_params')

            # Load current configuration
            with open(self.kernel_params_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Check if parameter exists
            if param_name not in config['parameters']:
                return False, f"Parameter '{param_name}' not found. Use add_kernel_parameter to create it."

            # Update fields
            for field, value in updates.items():
                if field in ['default_value', 'min_value', 'max_value']:
                    config['parameters'][param_name][field] = int(value)
                else:
                    config['parameters'][param_name][field] = value

            # Write updated configuration
            with open(self.kernel_params_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'update_parameter',
                'parameter': param_name,
                'updates': updates
            })

            logger.info(f"✅ Updated kernel parameter: {param_name}")
            return True, f"Successfully updated parameter: {param_name}"

        except Exception as e:
            logger.error(f"Failed to update kernel parameter: {e}")
            return False, f"Error: {str(e)}"

    def delete_kernel_parameter(self, param_name: str) -> Tuple[bool, str]:
        """
        Delete a kernel parameter from configuration

        Args:
            param_name: Name of parameter to delete

        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('kernel_params')

            # Load current configuration
            with open(self.kernel_params_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Check if parameter exists
            if param_name not in config['parameters']:
                return False, f"Parameter '{param_name}' not found."

            # Delete parameter
            deleted_config = config['parameters'].pop(param_name)

            # Write updated configuration
            with open(self.kernel_params_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'delete_parameter',
                'parameter': param_name,
                'deleted_config': deleted_config
            })

            logger.info(f"✅ Deleted kernel parameter: {param_name}")
            return True, f"Successfully deleted parameter: {param_name}"

        except Exception as e:
            logger.error(f"Failed to delete kernel parameter: {e}")
            return False, f"Error: {str(e)}"

    # =========================================================================
    # WORKLOAD PATTERNS - ADD/UPDATE
    # =========================================================================

    def add_workload_pattern(self, workload_name: str, workload_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Add a new workload pattern to configuration

        Args:
            workload_name: Name of workload (e.g., 'database', 'web_server')
            workload_config: Workload configuration dict with keys:
                - description
                - process_patterns (list of regex patterns)

        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('workloads')

            # Load current configuration
            with open(self.workload_patterns_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {'patterns': {}}

            # Check if workload already exists
            if workload_name in config['patterns']:
                return False, f"Workload '{workload_name}' already exists. Use update_workload_pattern instead."

            # Validate required fields
            if 'description' not in workload_config or 'process_patterns' not in workload_config:
                return False, "Missing required fields: 'description' and 'process_patterns'"

            # Add workload pattern
            config['patterns'][workload_name] = {
                'description': workload_config['description'],
                'process_patterns': workload_config['process_patterns']
            }

            # Write updated configuration
            with open(self.workload_patterns_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'add_workload',
                'workload': workload_name,
                'config': workload_config
            })

            logger.info(f"✅ Added workload pattern: {workload_name}")
            return True, f"Successfully added workload: {workload_name}"

        except Exception as e:
            logger.error(f"Failed to add workload pattern: {e}")
            return False, f"Error: {str(e)}"

    def update_workload_pattern(self, workload_name: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update an existing workload pattern

        Args:
            workload_name: Name of workload to update
            updates: Dictionary of fields to update

        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('workloads')

            # Load current configuration
            with open(self.workload_patterns_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Check if workload exists
            if workload_name not in config['patterns']:
                return False, f"Workload '{workload_name}' not found. Use add_workload_pattern to create it."

            # Update fields
            for field, value in updates.items():
                config['patterns'][workload_name][field] = value

            # Write updated configuration
            with open(self.workload_patterns_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'update_workload',
                'workload': workload_name,
                'updates': updates
            })

            logger.info(f"✅ Updated workload pattern: {workload_name}")
            return True, f"Successfully updated workload: {workload_name}"

        except Exception as e:
            logger.error(f"Failed to update workload pattern: {e}")
            return False, f"Error: {str(e)}"

    def delete_workload_pattern(self, workload_name: str) -> Tuple[bool, str]:
        """
        Delete a workload pattern from configuration

        Args:
            workload_name: Name of workload to delete

        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('workloads')

            # Load current configuration
            with open(self.workload_patterns_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Check if workload exists
            if workload_name not in config['patterns']:
                return False, f"Workload '{workload_name}' not found."

            # Delete workload
            deleted_config = config['patterns'].pop(workload_name)

            # Write updated configuration
            with open(self.workload_patterns_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'delete_workload',
                'workload': workload_name,
                'deleted_config': deleted_config
            })

            logger.info(f"✅ Deleted workload pattern: {workload_name}")
            return True, f"Successfully deleted workload: {workload_name}"

        except Exception as e:
            logger.error(f"Failed to delete workload pattern: {e}")
            return False, f"Error: {str(e)}"

    # =========================================================================
    # APPLY CONFIGURATION CHANGES TO SYSTEM
    # =========================================================================

    def apply_parameter_changes_to_system(self, param_name: str, value: int) -> Tuple[bool, str]:
        """
        Apply kernel parameter change to running system

        Args:
            param_name: Parameter name
            value: New value to apply

        Returns:
            (success, message)
        """
        try:
            kernel_interface = KernelParameterInterface()
            
            # Apply the parameter
            success = kernel_interface.set_parameter(param_name, value)
            
            if success:
                logger.info(f"✅ Applied {param_name} = {value} to system")
                return True, f"Successfully applied {param_name} = {value}"
            else:
                return False, f"Failed to apply {param_name}"

        except Exception as e:
            logger.error(f"Failed to apply parameter to system: {e}")
            return False, f"Error: {str(e)}"

    def reload_configurations(self) -> Dict[str, bool]:
        """
        Reload all configurations from disk and update CentralDataStore

        Returns:
            Dictionary of reload statuses for each config type
        """
        reload_status = {}

        try:
            # Reload kernel parameters
            self.config_loader.load_kernel_parameters()
            reload_status['kernel_params'] = True
        except Exception as e:
            logger.error(f"Failed to reload kernel parameters: {e}")
            reload_status['kernel_params'] = False

        try:
            # Reload optimization profiles and update CentralDataStore
            profiles = self.config_loader.load_optimization_profiles()
            
            # Update CentralDataStore with new optimization profiles
            data_store = get_data_store()
            data_store.set_optimization_profiles(profiles)
            
            reload_status['optimization_profiles'] = True
            logger.info(f"✅ Reloaded optimization profiles into CentralDataStore: {list(profiles.keys())}")
        except Exception as e:
            logger.error(f"Failed to reload optimization profiles: {e}")
            reload_status['optimization_profiles'] = False

        try:
            # Reload workload patterns
            self.config_loader.load_workload_patterns()
            reload_status['workload_patterns'] = True
        except Exception as e:
            logger.error(f"Failed to reload workload patterns: {e}")
            reload_status['workload_patterns'] = False

        logger.info(f"Configuration reload status: {reload_status}")
        return reload_status

    # =========================================================================
    # BATCH UPDATES
    # =========================================================================

    def update_optimization_profile(self, profile_name: str, updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update an existing optimization profile
        
        Args:
            profile_name: Name of profile to update
            updates: Dictionary of fields to update
            
        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('optimization_profiles')
            
            # Load configuration file
            with open(self.optimization_profiles_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {'profiles': {}}
            
            # Check if profile exists
            if profile_name not in config['profiles']:
                return False, f"Profile '{profile_name}' not found. Use add_new_optimization_profile to create it."
            
            # Update fields
            for field, value in updates.items():
                config['profiles'][profile_name][field] = value
            
            # Write updated configuration
            with open(self.optimization_profiles_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'update_optimization_profile',
                'profile': profile_name,
                'updates': updates
            })
            
            logger.info(f"✅ Updated optimization profile: {profile_name}")
            print(f"✅ Updated optimization profile: {profile_name}")
            return True, f"Successfully updated profile: {profile_name}"

        except Exception as e:
            logger.error(f"Failed to update optimization profile: {e}")
            return False, f"Error: {str(e)}"
    def delete_optimization_profile(self, profile_name: str) -> Tuple[bool, str]:
        """
        Delete an optimization profile
        
        Args:
            profile_name: Name of profile to delete
            
        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('optimization_profiles')
            
            # Load configuration file
            with open(self.optimization_profiles_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {'profiles': {}}
            
            # Check if profile exists
            if profile_name not in config['profiles']:
                return False, f"Profile '{profile_name}' not found."
            
            # Delete the profile
            deleted_config = config['profiles'].pop(profile_name)
            
            # Write updated configuration
            with open(self.optimization_profiles_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'delete_optimization_profile',
                'profile': profile_name,
                'deleted_config': deleted_config
            })
            
            logger.info(f"✅ Deleted optimization profile: {profile_name}")
            print(f"✅ Deleted optimization profile: {profile_name}")
            return True, f"Successfully deleted profile: {profile_name}"

        except Exception as e:
            logger.error(f"Failed to delete optimization profile: {e}")
            return False, f"Error: {str(e)}"
    def add_new_optimization_profile(self, profile_name: str, profile_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Add a new optimization profile
        
        Args:
            profile_name: Name of the new profile
            profile_config: Profile configuration dict with keys:
                - workload_type
                - strategy
                - evaluation_budget
                - time_budget
                - parameter_bounds
                - performance_weights
                
        Returns:
            (success, message)
        """
        try:
            # Create backup first
            self.create_backup('optimization_profiles')
            
            # Load configuration file
            with open(self.optimization_profiles_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {'profiles': {}}
            
            # Check if profile already exists
            if profile_name in config['profiles']:
                return False, f"Profile '{profile_name}' already exists. Use update_optimization_profile to modify it."
            
            # Validate required fields
            required_fields = ['workload_type', 'strategy', 'evaluation_budget', 'time_budget', 'parameter_bounds', 'performance_weights']
            for field in required_fields:
                if field not in profile_config:
                    return False, f"Missing required field: {field}"
            
            # Add new optimization profile
            config['profiles'][profile_name] = {
                'workload_type': profile_config['workload_type'],
                'strategy': profile_config['strategy'],
                'evaluation_budget': profile_config['evaluation_budget'],
                'time_budget': profile_config['time_budget'],
                'parameter_bounds': profile_config['parameter_bounds'],
                'performance_weights': profile_config['performance_weights']
            }
            
            # Write updated configuration
            with open(self.optimization_profiles_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            # Log the change
            self.change_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'add_optimization_profile',
                'profile': profile_name,
                'config': profile_config
            })
            
            logger.info(f"✅ Added optimization profile: {profile_name}")
            print(f"✅ Added optimization profile: {profile_name}")
            return True, f"Successfully added profile: {profile_name}"

        except Exception as e:
            logger.error(f"Failed to add new optimization profile: {e}")
            return False, f"Error: {str(e)}"

    # =========================================================================
    # BATCH UPDATES
    # =========================================================================

    def apply_batch_update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply multiple configuration updates in a batch

        Args:
            updates: Dictionary containing multiple update operations:
                {
                    'add_parameters': [{'name': '...', 'config': {...}}, ...],
                    'update_parameters': [{'name': '...', 'updates': {...}}, ...],
                    'add_workloads': [{'name': '...', 'config': {...}}, ...],
                    'update_workloads': [{'name': '...', 'updates': {...}}, ...],
                    'add_optimization_profiles': [{'name': '...', 'config': {...}}, ...],
                    'update_optimization_profiles': [{'name': '...', 'updates': {...}}, ...],
                    'delete_optimization_profiles': [{'name': '...'}, ...],
                    'apply_to_system': [{'param': '...', 'value': ...}, ...]
                }

        Returns:
            Results dictionary with success/failure for each operation
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_operations': 0,
            'successful': 0,
            'failed': 0,
            'details': []
        }

        # Add new parameters
        for param_update in updates.get('add_parameters', []):
            results['total_operations'] += 1
            success, message = self.add_kernel_parameter(param_update['name'], param_update['config'])
            results['details'].append({
                'operation': 'add_parameter',
                'name': param_update['name'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Update existing parameters
        for param_update in updates.get('update_parameters', []):
            results['total_operations'] += 1
            success, message = self.update_kernel_parameter(param_update['name'], param_update['updates'])
            results['details'].append({
                'operation': 'update_parameter',
                'name': param_update['name'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Add new workloads
        for workload_update in updates.get('add_workloads', []):
            results['total_operations'] += 1
            success, message = self.add_workload_pattern(workload_update['name'], workload_update['config'])
            results['details'].append({
                'operation': 'add_workload',
                'name': workload_update['name'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Update existing workloads
        for workload_update in updates.get('update_workloads', []):
            results['total_operations'] += 1
            success, message = self.update_workload_pattern(workload_update['name'], workload_update['updates'])
            results['details'].append({
                'operation': 'update_workload',
                'name': workload_update['name'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Add new optimization profiles
        for profile_update in updates.get('add_optimization_profiles', []):
            results['total_operations'] += 1
            success, message = self.add_new_optimization_profile(profile_update['name'], profile_update['config'])
            results['details'].append({
                'operation': 'add_optimization_profile',
                'name': profile_update['name'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Update existing optimization profiles
        for profile_update in updates.get('update_optimization_profiles', []):
            results['total_operations'] += 1
            success, message = self.update_optimization_profile(profile_update['name'], profile_update['updates'])
            results['details'].append({
                'operation': 'update_optimization_profile',
                'name': profile_update['name'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Delete optimization profiles
        for profile_delete in updates.get('delete_optimization_profiles', []):
            results['total_operations'] += 1
            success, message = self.delete_optimization_profile(profile_delete['name'])
            results['details'].append({
                'operation': 'delete_optimization_profile',
                'name': profile_delete['name'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Apply changes to system
        for system_update in updates.get('apply_to_system', []):
            results['total_operations'] += 1
            success, message = self.apply_parameter_changes_to_system(
                system_update['param'], system_update['value']
            )
            results['details'].append({
                'operation': 'apply_to_system',
                'parameter': system_update['param'],
                'value': system_update['value'],
                'success': success,
                'message': message
            })
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1

        # Reload configurations after batch update
        reload_status = self.reload_configurations()
        results['reload_status'] = reload_status

        logger.info(f"Batch update completed: {results['successful']}/{results['total_operations']} successful")
        return results

    # =========================================================================
    # EXPORT & REPORTING
    # =========================================================================

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Get list of all configuration changes"""
        return self.change_log

    def export_change_log(self, filename: str):
        """Export change log to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.change_log, f, indent=2)
        logger.info(f"Change log exported to: {filename}")

    def get_current_configurations(self) -> Dict[str, Any]:
        """Get current state of all configurations"""
        configs = {}

        try:
            with open(self.kernel_params_file, 'r', encoding='utf-8') as f:
                configs['kernel_parameters'] = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load kernel parameters: {e}")
            configs['kernel_parameters'] = {}

        try:
            with open(self.workload_patterns_file, 'r', encoding='utf-8') as f:
                configs['workload_patterns'] = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load workload patterns: {e}")
            configs['workload_patterns'] = {}

        return configs


# Singleton instance
_config_manager_instance = None


def get_config_manager() -> ConfigurationManager:
    """Get singleton instance of ConfigurationManager"""
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigurationManager()
    return _config_manager_instance


