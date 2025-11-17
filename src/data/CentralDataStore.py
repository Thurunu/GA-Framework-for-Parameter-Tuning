#!/usr/bin/env python3
"""
Central Data Store - Singleton Pattern
Provides a centralized location for sharing data across all components
"""

import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from DataClasses import WorkloadInfo, PerformanceMetrics, ProcessInfo


class CentralDataStore:
    """
    Singleton class for centralized data storage and access
    Thread-safe implementation for concurrent access
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the data store (only once)"""
        if self._initialized:
            return
            
        # Thread lock for concurrent access
        self._data_lock = threading.RLock()
        
        # Workload Data
        self._active_workloads: Dict[int, WorkloadInfo] = {}
        self._current_workload_type: str = "unknown"
        self._workload_history: List[tuple] = []  # (timestamp, workload_type)
        
        # Performance Metrics
        self._current_metrics: Optional[PerformanceMetrics] = None
        self._metrics_history: List[PerformanceMetrics] = []
        self._max_history_size: int = 1000
        
        # Process Information
        self._active_processes: Dict[int, ProcessInfo] = {}
        
        # Kernel Parameters
        self._current_kernel_params: Dict[str, Any] = {}
        self._kernel_param_history: List[tuple] = []  # (timestamp, params_dict)
        
        # Optimization Status
        self._optimization_status: Dict[str, Any] = {
            "is_optimizing": False,
            "current_profile": None,
            "last_optimization_time": None,
            "optimization_count": 0
        }
        
        # Optimization Profiles (all available workload types)
        self._optimization_profiles: Dict[str, Any] = {}
        
        # Agent Status
        self._agent_status: Dict[str, Any] = {
            "agent_id": None,
            "registered": False,
            "last_heartbeat": None,
            "master_url": None
        }
        
        # Agent Reporter instance
        self._agent_reporter = None
        
        self._initialized = True
        print("✅ Central Data Store initialized (Singleton)")
    
    # ==================== Workload Data Methods ====================
    
    def set_active_workloads(self, workloads: Dict[int, WorkloadInfo]):
        """Store active workload information"""
        with self._data_lock:
            self._active_workloads = workloads.copy()
    
    def get_active_workloads(self) -> Dict[int, WorkloadInfo]:
        """Get current active workloads"""
        with self._data_lock:
            return self._active_workloads.copy()
    
    def add_workload(self, workload: WorkloadInfo):
        """Add or update a single workload"""
        with self._data_lock:
            self._active_workloads[workload.pid] = workload
    
    def remove_workload(self, pid: int):
        """Remove a workload by PID"""
        with self._data_lock:
            self._active_workloads.pop(pid, None)
    
    def set_current_workload_type(self, workload_type: str):
        """Set the current dominant workload type"""
        with self._data_lock:
            if workload_type != self._current_workload_type:
                self._current_workload_type = workload_type
                self._workload_history.append((datetime.now(), workload_type))
                # Keep history manageable
                if len(self._workload_history) > 100:
                    self._workload_history = self._workload_history[-100:]
    
    def get_current_workload_type(self) -> str:
        """Get current workload type"""
        with self._data_lock:
            return self._current_workload_type
    
    def get_workload_history(self) -> List[tuple]:
        """Get workload type history"""
        with self._data_lock:
            return self._workload_history.copy()
    
    def set_optimization_profiles(self, profiles: Dict[str, Any]):
        """Store optimization profiles (all available workload types)"""
        with self._data_lock:
            self._optimization_profiles = profiles.copy()
    
    def get_available_workload_types(self) -> List[str]:
        """Get list of all available workload types that the system can handle"""
        with self._data_lock:
            return list(self._optimization_profiles.keys())
    
    # ==================== Performance Metrics Methods ====================
    
    def set_current_metrics(self, metrics: PerformanceMetrics):
        """Store current performance metrics"""
        with self._data_lock:
            self._current_metrics = metrics
            self._metrics_history.append(metrics)
            # Limit history size
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history = self._metrics_history[-self._max_history_size:]
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics"""
        with self._data_lock:
            return self._current_metrics
    
    def get_metrics_history(self, limit: int = None) -> List[PerformanceMetrics]:
        """Get performance metrics history"""
        with self._data_lock:
            if limit:
                return self._metrics_history[-limit:]
            return self._metrics_history.copy()
    
    def clear_metrics_history(self):
        """Clear metrics history"""
        with self._data_lock:
            self._metrics_history.clear()
    
    
    # ==================== Process Information Methods ====================
    
    def set_active_processes(self, processes: Dict[int, ProcessInfo]):
        """Store active process information"""
        with self._data_lock:
            self._active_processes = processes.copy()
    
    def get_active_processes(self) -> Dict[int, ProcessInfo]:
        """Get current active processes"""
        with self._data_lock:
            return self._active_processes.copy()
    
    def add_process(self, process: ProcessInfo):
        """Add or update a single process"""
        with self._data_lock:
            self._active_processes[process.pid] = process
    
    def remove_process(self, pid: int):
        """Remove a process by PID"""
        with self._data_lock:
            self._active_processes.pop(pid, None)
    
    # ==================== Kernel Parameters Methods ====================
    
    def set_kernel_parameters(self, params: Dict[str, Any]):
        """Store current kernel parameters"""
        with self._data_lock:
            self._current_kernel_params = params.copy()
            self._kernel_param_history.append((datetime.now(), params.copy()))
            # Keep history manageable
            if len(self._kernel_param_history) > 50:
                self._kernel_param_history = self._kernel_param_history[-50:]
    
    def get_kernel_parameters(self) -> Dict[str, Any]:
        """Get current kernel parameters"""
        with self._data_lock:
            return self._current_kernel_params.copy()
    
    def get_kernel_parameter(self, param_name: str) -> Optional[Any]:
        """Get a specific kernel parameter value"""
        with self._data_lock:
            return self._current_kernel_params.get(param_name)
    
    def update_kernel_parameter(self, param_name: str, value: Any):
        """Update a single kernel parameter"""
        with self._data_lock:
            self._current_kernel_params[param_name] = value
    
    def get_kernel_param_history(self) -> List[tuple]:
        """Get kernel parameter change history"""
        with self._data_lock:
            return self._kernel_param_history.copy()
    
    # ==================== Optimization Status Methods ====================
    
    def set_optimization_status(self, key: str, value: Any):
        """Update optimization status"""
        with self._data_lock:
            self._optimization_status[key] = value
    
    def get_optimization_status(self, key: str = None) -> Any:
        """Get optimization status"""
        with self._data_lock:
            if key:
                return self._optimization_status.get(key)
            return self._optimization_status.copy()
    
    def start_optimization(self, profile: str):
        """Mark optimization as started"""
        with self._data_lock:
            self._optimization_status["is_optimizing"] = True
            self._optimization_status["current_profile"] = profile
    
    def end_optimization(self):
        """Mark optimization as completed"""
        with self._data_lock:
            self._optimization_status["is_optimizing"] = False
            self._optimization_status["last_optimization_time"] = datetime.now()
            self._optimization_status["optimization_count"] += 1
    
    # ==================== Agent Status Methods ====================
    
    def set_agent_reporter(self, reporter):
        """Store AgentReporter instance for global access"""
        with self._data_lock:
            self._agent_reporter = reporter
    
    def get_agent_reporter(self):
        """Get AgentReporter instance"""
        with self._data_lock:
            return getattr(self, '_agent_reporter', None)
    
    def set_agent_status(self, key: str, value: Any):
        """Update agent status"""
        with self._data_lock:
            self._agent_status[key] = value
    
    def get_agent_status(self, key: str = None) -> Any:
        """Get agent status"""
        with self._data_lock:
            if key:
                return self._agent_status.get(key)
            return self._agent_status.copy()
    
    def set_agent_registered(self, agent_id: str, master_url: str):
        """Mark agent as registered"""
        with self._data_lock:
            self._agent_status["agent_id"] = agent_id
            self._agent_status["master_url"] = master_url
            self._agent_status["registered"] = True
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp"""
        with self._data_lock:
            self._agent_status["last_heartbeat"] = datetime.now()
    
    # ==================== Workload Methods ====================

        
    # ==================== Utility Methods ====================
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get all data from the store (for debugging/monitoring)"""
        with self._data_lock:
            return {
                "workloads": {
                    "active": len(self._active_workloads),
                    "current_type": self._current_workload_type,
                    "history_count": len(self._workload_history)
                },
                "metrics": {
                    "current": self._current_metrics is not None,
                    "history_count": len(self._metrics_history)
                },
                "processes": {
                    "active": len(self._active_processes)
                },
                "kernel_params": {
                    "count": len(self._current_kernel_params),
                    "history_count": len(self._kernel_param_history)
                },
                "optimization": self._optimization_status.copy(),
                "agent": self._agent_status.copy()
            }
    
    def clear_all_data(self):
        """Clear all stored data (use with caution)"""
        with self._data_lock:
            self._active_workloads.clear()
            self._current_workload_type = "unknown"
            self._workload_history.clear()
            self._current_metrics = None
            self._metrics_history.clear()
            self._active_processes.clear()
            self._current_kernel_params.clear()
            self._kernel_param_history.clear()
            print("⚠️ Central Data Store cleared")
    
    def __repr__(self):
        """String representation for debugging"""
        return f"<CentralDataStore: {len(self._active_workloads)} workloads, {len(self._metrics_history)} metrics>"


# Convenience function to get the singleton instance
def get_data_store() -> CentralDataStore:
    """Get the singleton instance of CentralDataStore"""
    return CentralDataStore()
