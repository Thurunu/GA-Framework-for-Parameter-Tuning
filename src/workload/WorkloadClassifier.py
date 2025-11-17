#!/usr/bin/env python3
"""
Workload Classifier
Classifies processes into workload types based on patterns and resource usage
"""

import re
import os, sys
from typing import Dict

from DataClasses import WorkloadInfo
from LoadConfigs import LoadConfigs
from CentralDataStore import get_data_store
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'utils'))
sys.path.insert(0, os.path.join(project_root, 'data'))

class WorkloadClassifier:
    """Classifies processes into workload types based on patterns and resource metrics"""

    def __init__(self, config_file: str = None):
        """
        Initialize workload classifier with patterns from YAML configuration

        Args:
            config_file: Path to workload patterns YAML file (optional)
        """
        self.data_store = get_data_store()
        self.workload_patterns = {}
        self.fallback_thresholds = {}
        self.config_loader = LoadConfigs()
        self._load_patterns(config_file)

    def _load_patterns(self, config_file: str = None):
        """
        Load workload patterns from YAML configuration file

        Args:
            config_file: Path to YAML configuration file
        """

        try:
            if config_file is None:
                self.config = self.config_loader.load_workload_patterns(
                    config_file)

            # Load patterns
            for workload_name, workload_data in self.config['patterns'].items():
                self.workload_patterns[workload_name] = workload_data['process_patterns']

            # Load fallback thresholds for resource-based classification
            self.fallback_thresholds = self.config.get(
                'fallback_thresholds', {})

            print(f"✅✅ Loaded {len(self.workload_patterns)} workload patterns")

        except Exception as e:
            print(f"⚠️ Failed to load workload patterns: {e}")

        self.fallback_thresholds = {
            'cpu_intensive': {'cpu_percent': 80},
            'memory_intensive': {'memory_percent': 50},
            'io_intensive': {'io_bytes_per_second': 1000000},
            'network_intensive': {'connection_count': 10}
        }
        print(
            f"✅ Loaded {len(self.workload_patterns)} default workload patterns")

    def classify_process(self, proc_info: WorkloadInfo) -> str:
        """
        Classify process based on name, command line, and resource usage

        Classification Strategy:
        1. First, try pattern matching against process name and command line
        2. If no pattern matches, use resource-based classification (CPU, memory, I/O, network)
        3. Default to 'general' if no classification matches
        4. Store classified workload in central data store

        Args:
            proc_info: WorkloadInfo object containing process details

        Returns:
            Workload type string (e.g., 'database', 'web_server', 'cpu_intensive', etc.)
        """
        full_cmd = f"{proc_info.name} {proc_info.cmdline}".lower()

        # Strategy 1: Pattern matching (most specific)
        for workload_type, patterns in self.workload_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, full_cmd):
                        proc_info.workload_type = workload_type
                        # Store in central data store
                        self.data_store.add_workload(proc_info)
                        return workload_type
                except re.error:
                    # Skip invalid regex patterns
                    continue

        # Strategy 2: Resource-based classification (fallback)
        workload_type = self._classify_by_resources(proc_info)
        proc_info.workload_type = workload_type
        
        # Store in central data store
        self.data_store.add_workload(proc_info)
        
        return workload_type

    def _classify_by_resources(self, proc_info: WorkloadInfo) -> str:
        """
        Classify process based on resource usage when pattern matching fails

        Args:
            proc_info: WorkloadInfo object containing process metrics

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
        print(
            f"✅ Added pattern '{pattern}' for workload type '{workload_type}'")

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
        print(f"✅ Set threshold for '{workload_type}.{metric}' to {value}")
