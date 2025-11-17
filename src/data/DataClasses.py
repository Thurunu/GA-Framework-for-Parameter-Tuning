#!/usr/bin/env python3
"""
Data Classes for the Kernel Optimization Framework
Contains all shared data classes to avoid circular imports
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import os, sys

from WorkloadCharacterizer import OptimizationStrategy

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'workload'))

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population"""
    parameters: Dict[str, float]
    fitness: Optional[float] = None
    age: int = 0


@dataclass
class GAOptimizationResult:
    """Results from Genetic Algorithm optimization"""
    best_individual: Individual
    best_fitness: float
    generation_count: int
    population_history: List[List[Individual]]
    fitness_history: List[float]
    convergence_reached: bool
    optimization_time: float
    
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

@dataclass
class OptimizationResult:
    """
    Purpose: Stores and organizes the results of a Bayesian Optimization run.
    Use case: Returned after optimization to summarize best parameters, score, history, and convergence info.
    """
    best_parameters: Dict[str, Any]  # The parameter values which achieved the highest score
    best_score: float  # Highest score found during optimization
    iteration_count: int  # Total number of optimization iterations performed
    evaluation_history: List[Tuple[Dict[str, Any], float]]  # List of (parameters, score) tuples
    convergence_reached: bool  # Whether optimization met convergence criteria
    optimization_time: float  # Total time taken for optimization process

@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population"""
    parameters: Dict[str, float]
    fitness: Optional[float] = None
    age: int = 0


@dataclass
class HybridOptimizationResult:
    """Results from hybrid optimization"""
    best_parameters: Dict[str, Any]
    best_score: float
    strategy_used: OptimizationStrategy
    total_evaluations: int
    bayesian_results: Optional[OptimizationResult] = None
    genetic_results: Optional[GAOptimizationResult] = None
    optimization_time: float = 0.0
    convergence_reached: bool = False
    switch_points: List[Tuple[int, str]] = None

@dataclass
class OptimizationSession:
    """Complete optimization session data"""
    session_id: str
    workload_type: str
    optimization_strategy: str
    start_time: float
    end_time: Optional[float] = None
    best_parameters: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    total_evaluations: int = 0
    session_completed: bool = False