#!/usr/bin/env python3
"""
Centralized Configuration Loader
Loads all YAML configuration files for the optimization system
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import os, sys

# Import required data classes from centralized location
from DataClasses import KernelParameter, OptimizationProfile, ProcessInfo, WorkloadInfo
from WorkloadCharacterizer import OptimizationStrategy
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'workload'))
sys.path.insert(0, os.path.join(project_root, 'data'))


class LoadConfigs:
    """Centralized configuration loader for all YAML configs"""

    def __init__(self):
        """Initialize the configuration loader"""
        # Get the config directory path
        configuration_dir = os.path.join(project_root, '..', 'config')
        
        # Config file paths
        self.kernel_params_file = os.path.join(configuration_dir, "kernel_parameters.yml")
        self.optimization_profiles_file = os.path.join(configuration_dir, "optimization_profiles.yml")
        self.process_priorities_file = os.path.join(configuration_dir, "process_priorities.yml")
        self.workload_patterns_file = os.path.join(configuration_dir, "workload_patterns.yml")

    # =====================================================
    # KERNEL PARAMETERS
    # =====================================================
    
    def load_kernel_parameters(self, config_file: Optional[str] = None) -> Dict[str, KernelParameter]:
        """Load kernel parameter definitions from YAML configuration file"""
        if config_file is None:
            config_file = self.kernel_params_file
        
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
            
            print(f"✅ Loaded {len(parameters)} kernel parameters from configuration")
            print(f"✅✅✅ Done")
            return parameters
            
        except FileNotFoundError:
            print(f"⚠️ Warning: Configuration file {config_file} not found. Using default parameters.")
            return self._get_default_kernel_parameters()
        except Exception as e:
            print(f"❌ Error loading kernel parameters: {e}")
            print("Falling back to default parameters.")
            return self._get_default_kernel_parameters()
    
    def _get_default_kernel_parameters(self) -> Dict[str, KernelParameter]:
        """Get default kernel parameters as fallback"""
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

    # =====================================================
    # OPTIMIZATION PROFILES
    # =====================================================
    
    def load_optimization_profiles(self, config_file: Optional[str] = None) -> Dict[str, OptimizationProfile]:
        """Load optimization profiles from YAML configuration file"""
        if config_file is None:
            config_file = self.optimization_profiles_file
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            profiles = {}
            for profile_name, profile_data in config['profiles'].items():
                # Convert parameter bounds from lists to tuples
                parameter_bounds = {
                    param: tuple(bounds) 
                    for param, bounds in profile_data['parameter_bounds'].items()
                }
                
                # Convert strategy string to enum
                strategy = OptimizationStrategy[profile_data['strategy']]
                
                profiles[profile_name] = OptimizationProfile(
                    workload_type=profile_data['workload_type'],
                    parameter_bounds=parameter_bounds,
                    strategy=strategy,
                    evaluation_budget=profile_data['evaluation_budget'],
                    time_budget=profile_data['time_budget'],
                    performance_weights=profile_data['performance_weights']
                )
            
            print(f"✅ Loaded {len(profiles)} optimization profiles from configuration")
            return profiles
            
        except FileNotFoundError:
            print(f"⚠️ Warning: Configuration file {config_file} not found. Using default profiles.")
            return self._get_default_optimization_profiles()
        except Exception as e:
            print(f"❌ Error loading optimization profiles: {e}")
            print("Falling back to default profiles.")
            return self._get_default_optimization_profiles()
    
    def _get_default_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Get default optimization profiles as fallback"""
        return {
            'general': OptimizationProfile(
                workload_type='general',
                parameter_bounds={
                    'vm.swappiness': (10, 80),
                    'vm.dirty_ratio': (10, 30),
                },
                strategy=OptimizationStrategy.ADAPTIVE,
                evaluation_budget=10,
                time_budget=120.0,
                performance_weights={
                    'cpu_efficiency': 0.25,
                    'memory_efficiency': 0.25,
                    'io_throughput': 0.25,
                    'network_throughput': 0.25
                }
            )
        }

    # =====================================================
    # PROCESS PRIORITIES
    # =====================================================
    
    def load_process_priorities(self, config_file: Optional[str] = None) -> Dict[str, ProcessInfo]:
        """Load process priority configurations from YAML"""
        if config_file is None:
            config_file = self.process_priorities_file
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Loaded process priorities configuration")
            return config
            
        except FileNotFoundError:
            print(f"⚠️ Warning: Configuration file {config_file} not found.")
            return {}
        except Exception as e:
            print(f"❌ Error loading process priorities: {e}")
            return {}

    # =====================================================
    # WORKLOAD PATTERNS
    # =====================================================
    
    def load_workload_patterns(self, config_file: Optional[str] = None) -> Dict[str, WorkloadInfo]:
        """Load workload pattern configurations from YAML"""
        if config_file is None:
            config_file = self.workload_patterns_file
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Loaded workload patterns configuration")
            return config
            
        except FileNotFoundError:
            print(f"⚠️ Warning: Configuration file {config_file} not found.")
            return {}
        except Exception as e:
            print(f"❌ Error loading workload patterns: {e}")
            return {}

    # =====================================================
    # VALIDATION LAYER (Phase 2)
    # =====================================================
    
    def validate_optimization_profile(self, profile_name: str, profile_data: Dict, 
                                     kernel_params: Dict, workload_patterns: Dict) -> tuple[bool, list]:
        """
        Validate an optimization profile for referential integrity
        
        Args:
            profile_name: Name of the profile
            profile_data: Profile configuration data
            kernel_params: Available kernel parameters
            workload_patterns: Available workload patterns
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check 1: Workload type must exist in workload_patterns
        workload_type = profile_data.get('workload_type')
        if workload_type not in workload_patterns.get('patterns', {}):
            errors.append(f"Profile '{profile_name}': workload_type '{workload_type}' not found in workload_patterns.yml")
        
        # Check 2: All parameters must exist in kernel_parameters
        param_bounds = profile_data.get('parameter_bounds', {})
        for param_name in param_bounds.keys():
            if param_name not in kernel_params.get('parameters', {}):
                errors.append(f"Profile '{profile_name}': parameter '{param_name}' not found in kernel_parameters.yml")
        
        return (len(errors) == 0, errors)
    
    def validate_process_priority(self, priority_name: str, priority_data: Dict,
                                  workload_patterns: Dict) -> tuple[bool, list]:
        """
        Validate a process priority mapping for referential integrity
        
        Args:
            priority_name: Name of the priority mapping
            priority_data: Priority configuration data
            workload_patterns: Available workload patterns
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check: If priority mapping references a workload type (not system_critical or background_tasks),
        # it must exist in workload_patterns
        # Exception: system_critical and background_tasks have their own patterns
        special_priorities = ['system_critical', 'background_tasks']
        
        if priority_name not in special_priorities:
            if priority_name not in workload_patterns.get('patterns', {}):
                # Only add patterns field if this is a special priority
                if 'patterns' not in priority_data:
                    errors.append(f"Priority '{priority_name}': must reference a workload type in workload_patterns.yml or define its own patterns")
        
        return (len(errors) == 0, errors)
    
    def validate_all_configs(self) -> Dict[str, Any]:
        """
        Load and validate all configurations for referential integrity
        
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': list,
                'warnings': list,
                'configs': dict (if valid)
            }
        """
        errors = []
        warnings = []
        
        # Load all configs
        kernel_params = None
        workload_patterns = None
        optimization_profiles = None
        process_priorities = None
        
        try:
            # Load base configs first
            with open(self.kernel_params_file, 'r', encoding='utf-8') as f:
                kernel_params = yaml.safe_load(f)
            
            with open(self.workload_patterns_file, 'r', encoding='utf-8') as f:
                workload_patterns = yaml.safe_load(f)
            
            with open(self.optimization_profiles_file, 'r', encoding='utf-8') as f:
                optimization_profiles = yaml.safe_load(f)
            
            with open(self.process_priorities_file, 'r', encoding='utf-8') as f:
                process_priorities = yaml.safe_load(f)
        
        except Exception as e:
            errors.append(f"Failed to load configuration files: {e}")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Validate optimization profiles
        for profile_name, profile_data in optimization_profiles.get('profiles', {}).items():
            is_valid, profile_errors = self.validate_optimization_profile(
                profile_name, profile_data, kernel_params, workload_patterns
            )
            errors.extend(profile_errors)
        
        # Validate process priorities
        for priority_name, priority_data in process_priorities.get('priority_mappings', {}).items():
            is_valid, priority_errors = self.validate_process_priority(
                priority_name, priority_data, workload_patterns
            )
            errors.extend(priority_errors)
        
        # Check for orphaned workload patterns (patterns defined but not used)
        used_workload_types = set()
        for profile_data in optimization_profiles.get('profiles', {}).values():
            used_workload_types.add(profile_data.get('workload_type'))
        
        defined_workload_types = set(workload_patterns.get('patterns', {}).keys())
        orphaned = defined_workload_types - used_workload_types
        
        if orphaned:
            warnings.append(f"Workload patterns defined but not used in optimization profiles: {orphaned}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'configs': {
                'kernel_parameters': kernel_params,
                'workload_patterns': workload_patterns,
                'optimization_profiles': optimization_profiles,
                'process_priorities': process_priorities
            } if len(errors) == 0 else None
        }
    
    # =====================================================
    # LOAD ALL CONFIGS
    # =====================================================
    
    def load_all_configs(self) -> Dict[str, Any]:
        """Load all configuration files at once"""
        return {
            'kernel_parameters': self.load_kernel_parameters(),
            'optimization_profiles': self.load_optimization_profiles(),
            'process_priorities': self.load_process_priorities(),
            'workload_patterns': self.load_workload_patterns()
        }