#!/usr/bin/env python3
"""
Workload Classifier
Classifies processes into workload types based on patterns and resource usage
"""

import re
import yaml
from typing import Dict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessInfo:
    """Information about a running process"""
    pid: int
    name: str
    cmdline: str
    cpu_percent: float
    memory_percent: float
    io_read_bytes: int
    io_write_bytes: int
    network_connections: int
    start_time: float
    workload_type: str = "unknown"


class WorkloadClassifier:
    """Classifies processes into workload types based on patterns and resource metrics"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize workload classifier with patterns from YAML configuration
        
        Args:
            config_file: Path to workload patterns YAML file (optional)
        """
        self.workload_patterns = {}
        self.fallback_thresholds = {}
        self._load_patterns(config_file)
    
    def _load_patterns(self, config_file: str = None):
        """
        Load workload patterns from YAML configuration file
        
        Args:
            config_file: Path to YAML configuration file
        """
        if config_file is None:
            # Default to config/workload_patterns.yml relative to project root
            current_dir = Path(__file__).parent
            config_file = current_dir.parent / "config" / "workload_patterns.yml"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Load patterns
            for workload_name, workload_data in config['patterns'].items():
                self.workload_patterns[workload_name] = workload_data['process_patterns']
            
            # Load fallback thresholds for resource-based classification
            self.fallback_thresholds = config.get('fallback_thresholds', {})
            
            print(f"✓ Loaded {len(self.workload_patterns)} workload patterns")
            
        except FileNotFoundError:
            print(f"⚠ Warning: Configuration file {config_file} not found. Using default patterns.")
            self._load_default_patterns()
        except yaml.YAMLError as e:
            print(f"⚠ Error parsing YAML configuration: {e}")
            print("Falling back to default patterns.")
            self._load_default_patterns()
        except (OSError, IOError) as e:
            print(f"⚠ Error loading workload patterns: {e}")
            print("Falling back to default patterns.")
            self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default patterns as fallback when config file is unavailable"""
        self.workload_patterns = {
            'database': [r'mysql.*', r'postgres.*', r'mongodb.*', r'redis.*'],
            'web_server': [r'nginx.*', r'apache.*', r'httpd.*', r'node.*'],
            'general': [r'.*']
        }
        self.fallback_thresholds = {
            'cpu_intensive': {'cpu_percent': 80},
            'memory_intensive': {'memory_percent': 50},
            'io_intensive': {'io_bytes_per_second': 1000000},
            'network_intensive': {'connection_count': 10}
        }
        print(f"✓ Loaded {len(self.workload_patterns)} default workload patterns")
    
    def classify_process(self, proc_info: ProcessInfo) -> str:
        """
        Classify process based on name, command line, and resource usage
        
        Classification Strategy:
        1. First, try pattern matching against process name and command line
        2. If no pattern matches, use resource-based classification (CPU, memory, I/O, network)
        3. Default to 'general' if no classification matches
        
        Args:
            proc_info: ProcessInfo object containing process details
            
        Returns:
            Workload type string (e.g., 'database', 'web_server', 'cpu_intensive', etc.)
        """
        full_cmd = f"{proc_info.name} {proc_info.cmdline}".lower()
        
        # Strategy 1: Pattern matching (most specific)
        for workload_type, patterns in self.workload_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, full_cmd):
                        return workload_type
                except re.error:
                    # Skip invalid regex patterns
                    continue
        
        # Strategy 2: Resource-based classification (fallback)
        return self._classify_by_resources(proc_info)
    
    def _classify_by_resources(self, proc_info: ProcessInfo) -> str:
        """
        Classify process based on resource usage when pattern matching fails
        
        Args:
            proc_info: ProcessInfo object containing process metrics
            
        Returns:
            Resource-based workload type or 'general'
        """
        # CPU-intensive check
        cpu_threshold = self.fallback_thresholds.get(
            'cpu_intensive', {}
        ).get('cpu_percent', 80)
        
        if proc_info.cpu_percent > cpu_threshold:
            return 'cpu_intensive'
        
        # Memory-intensive check
        memory_threshold = self.fallback_thresholds.get(
            'memory_intensive', {}
        ).get('memory_percent', 50)
        
        if proc_info.memory_percent > memory_threshold:
            return 'memory_intensive'
        
        # I/O-intensive check
        io_threshold = self.fallback_thresholds.get(
            'io_intensive', {}
        ).get('io_bytes_per_second', 1000000)
        
        total_io = proc_info.io_read_bytes + proc_info.io_write_bytes
        if total_io > io_threshold:
            return 'io_intensive'
        
        # Network-intensive check
        network_threshold = self.fallback_thresholds.get(
            'network_intensive', {}
        ).get('connection_count', 10)
        
        if proc_info.network_connections > network_threshold:
            return 'network_intensive'
        
        # Default classification
        return 'general'
    
    def get_workload_patterns(self) -> Dict[str, list]:
        """
        Get all loaded workload patterns
        
        Returns:
            Dictionary mapping workload types to their patterns
        """
        return self.workload_patterns.copy()
    
    def get_fallback_thresholds(self) -> Dict:
        """
        Get resource-based classification thresholds
        
        Returns:
            Dictionary of threshold configurations
        """
        return self.fallback_thresholds.copy()
    
    def add_pattern(self, workload_type: str, pattern: str):
        """
        Add a new pattern for workload classification
        
        Args:
            workload_type: Type of workload (e.g., 'database')
            pattern: Regex pattern to match process names/commands
        """
        if workload_type not in self.workload_patterns:
            self.workload_patterns[workload_type] = []
        
        self.workload_patterns[workload_type].append(pattern)
        print(f"✓ Added pattern '{pattern}' for workload type '{workload_type}'")
    
    def set_threshold(self, workload_type: str, metric: str, value: float):
        """
        Set or update a resource-based classification threshold
        
        Args:
            workload_type: Type of workload (e.g., 'cpu_intensive')
            metric: Metric name (e.g., 'cpu_percent')
            value: Threshold value
        """
        if workload_type not in self.fallback_thresholds:
            self.fallback_thresholds[workload_type] = {}
        
        self.fallback_thresholds[workload_type][metric] = value
        print(f"✓ Set threshold for '{workload_type}.{metric}' to {value}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing WorkloadClassifier")
    print("=" * 60)
    
    # Initialize classifier
    classifier = WorkloadClassifier()
    
    print(f"\nLoaded patterns: {list(classifier.get_workload_patterns().keys())}")
    print(f"Fallback thresholds: {list(classifier.get_fallback_thresholds().keys())}")
    
    # Test classification with sample processes
    test_processes = [
        ProcessInfo(
            pid=1234, name="mysqld", cmdline="mysqld --defaults-file=/etc/my.cnf",
            cpu_percent=45.0, memory_percent=30.0,
            io_read_bytes=1000000, io_write_bytes=500000,
            network_connections=50, start_time=123456.0
        ),
        ProcessInfo(
            pid=5678, name="nginx", cmdline="nginx: master process",
            cpu_percent=15.0, memory_percent=5.0,
            io_read_bytes=100000, io_write_bytes=200000,
            network_connections=100, start_time=123457.0
        ),
        ProcessInfo(
            pid=9012, name="python3", cmdline="python3 heavy_computation.py",
            cpu_percent=95.0, memory_percent=20.0,
            io_read_bytes=1000, io_write_bytes=1000,
            network_connections=0, start_time=123458.0
        ),
        ProcessInfo(
            pid=3456, name="chrome", cmdline="chrome --type=renderer",
            cpu_percent=10.0, memory_percent=65.0,
            io_read_bytes=50000, io_write_bytes=30000,
            network_connections=5, start_time=123459.0
        ),
    ]
    
    print("\nClassifying test processes:")
    print("-" * 60)
    
    for proc in test_processes:
        workload = classifier.classify_process(proc)
        print(f"{proc.name:15} → {workload:20} (CPU: {proc.cpu_percent:.1f}%, MEM: {proc.memory_percent:.1f}%)")
    
    # Test adding custom patterns
    print("\nAdding custom pattern...")
    classifier.add_pattern('custom_app', r'myapp.*')
    
    # Test setting custom thresholds
    print("Setting custom threshold...")
    classifier.set_threshold('cpu_intensive', 'cpu_percent', 70)
    
    print("\n" + "=" * 60)
    print("WorkloadClassifier testing completed!")
