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