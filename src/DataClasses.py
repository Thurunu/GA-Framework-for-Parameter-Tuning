#!/usr/bin/env python3
"""
Data Classes for the Kernel Optimization Framework
Contains all shared data classes to avoid circular imports
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


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


@dataclass
class OptimizationProfile:
    """Optimization profile for specific workload types"""
    workload_type: str
    parameter_bounds: Dict[str, tuple]
    strategy: Any  # Will be OptimizationStrategy enum from WorkloadCharacterizer
    evaluation_budget: int
    time_budget: float
    performance_weights: Dict[str, float]

@dataclass
class ProcessInfo:
    """Information about a process and its priority"""
    pid: int
    name: str
    current_nice: int
    target_nice: int
    cpu_percent: float
    memory_percent: float


@dataclass
class WorkloadInfo:
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

@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_io_read: int
    disk_io_write: int
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: tuple
    context_switches: int
    interrupts: int
    tcp_connections: int
