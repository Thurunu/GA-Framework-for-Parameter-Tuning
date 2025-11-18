#!/usr/bin/env python3
"""
Process Priority Manager for EEVDF Scheduler Support
Handles process nice values and priority adjustments for kernel 6.6+ systems
"""

import subprocess
import logging
import os, sys
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum

from DataClasses import ProcessInfo
from LoadConfigs import LoadConfigs
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'utils'))
sys.path.insert(0, os.path.join(project_root, 'data'))
try:
    import psutil
except ImportError:
    print("Warning: psutil not installed. Install with: pip install psutil")
    psutil = None

class PriorityClass(Enum):
    """Process priority classes for different workload types"""
    CRITICAL = -20      # Highest priority (real-time-like)
    HIGH = -10         # High priority (interactive)
    NORMAL = 0         # Default priority
    LOW = 10          # Low priority (batch)
    BACKGROUND = 19    # Lowest priority (background tasks)
    
class ProcessPriorityManager:
    """
    Manages process priorities using nice/renice for EEVDF scheduler optimization
    """
    
    def __init__(self, log_level=logging.INFO, config_file: str = None):
        # Check for required dependencies
        if psutil is None:
            raise ImportError("psutil is required. Install with: pip install psutil")
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Load priority configuration from YAML
        self.workload_patterns = {}
        config_loader = LoadConfigs()
        self.config = config_loader.load_process_priorities(config_file)
        
        # Track managed processes
        self.managed_processes: Dict[int, ProcessInfo] = {}
        self.original_priorities: Dict[int, int] = {}
        
        # Process stability tracking for filtering short-lived processes
        # Format: {pid: {'first_seen': timestamp, 'count': observation_count}}
        self.process_observations: Dict[int, Dict] = {}
        self.last_cleanup_time = time.time()
    
   
    
    def _cleanup_old_observations(self):
        """Remove old process observations outside the tracking window"""
        current_time = time.time()
        
        # Only cleanup periodically to avoid overhead
        if current_time - self.last_cleanup_time < 60:  # Cleanup every 60 seconds
            return
        
        stability_config = self.config.get('filter_rules', {}).get('stability_tracking', {})
        observation_window = stability_config.get('observation_window', 30)
        
        # Remove observations older than the window
        expired_pids = [
            pid for pid, data in self.process_observations.items()
            if current_time - data['first_seen'] > observation_window
        ]
        
        for pid in expired_pids:
            del self.process_observations[pid]
        
        self.last_cleanup_time = current_time
    
    def _should_adjust_process(self, pid: int, process_age: float) -> bool:
        """
        Determine if a process should have its priority adjusted based on:
        1. Minimum process age (filter out very short-lived processes)
        2. Stability tracking (require multiple observations before adjustment)
        
        Args:
            pid: Process ID
            process_age: Process uptime in seconds
            
        Returns:
            True if process is eligible for priority adjustment
        """
        filter_rules = self.config.get('filter_rules', {})
        
        # Check minimum process age
        min_age = filter_rules.get('min_process_age', 5.0)
        if process_age < min_age:
            return False
        
        # Check stability tracking
        stability_config = filter_rules.get('stability_tracking', {})
        if not stability_config.get('enabled', True):
            return True  # Stability tracking disabled, process is eligible
        
        required_observations = stability_config.get('required_observations', 2)
        current_time = time.time()
        
        # Track this observation
        if pid not in self.process_observations:
            self.process_observations[pid] = {
                'first_seen': current_time,
                'count': 1
            }
            return False  # First time seeing this process
        else:
            # Increment observation count
            self.process_observations[pid]['count'] += 1
            
            # Check if we've seen it enough times
            if self.process_observations[pid]['count'] >= required_observations:
                return True
            else:
                return False
        
    def classify_process(self, process_name: str, cmdline: str) -> Tuple[str, PriorityClass]:
        """
        Classify a process based on its name and command line
        
        Returns:
            Tuple of (workload_type, priority_class)
        """
        full_command = f"{process_name} {' '.join(cmdline) if cmdline else ''}"
        
        for workload_type, config in self.workload_patterns.items():
            for pattern in config['patterns']:
                if pattern.lower() in full_command.lower():
                    return workload_type, config['priority_class']
        
        return 'general', PriorityClass.NORMAL
    
    def get_process_priority(self, pid: int) -> Optional[int]:
        """Get current nice value of a process"""
        try:
            process = psutil.Process(pid)
            return process.nice()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def set_process_priority(self, pid: int, nice_value: int) -> bool:
        """
        Set process priority using renice command
        
        Args:
            pid: Process ID
            nice_value: Nice value (-20 to 19)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate nice value range
            if not -20 <= nice_value <= 19:
                self.logger.error("Invalid nice value %s. Must be between -20 and 19", nice_value)
                return False
            
            # Store original priority if not already stored
            if pid not in self.original_priorities:
                original_nice = self.get_process_priority(pid)
                if original_nice is not None:
                    self.original_priorities[pid] = original_nice
            
            # Set new priority
            if os.name == 'nt':  # Windows simulation
                self.logger.info("Simulated: renice %s %s", nice_value, pid)
                return True
            else:
                subprocess.run(
                    ['renice', str(nice_value), str(pid)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.logger.info("Set process %s priority to %s", pid, nice_value)
                return True
                
        except subprocess.CalledProcessError as e:
            self.logger.error("Failed to set priority for process %s: %s", pid, e)
            return False
        except (OSError, ValueError) as e:
            self.logger.error("Unexpected error setting priority for %s: %s", pid, e)
            return False
    
    def scan_and_classify_processes(self) -> Dict[str, List[ProcessInfo]]:
        """
        Scan all running processes and classify them by workload type
        
        Returns:
            Dictionary mapping workload types to lists of ProcessInfo
        """
        classified_processes: Dict[str, List[ProcessInfo]] = {}
        
        # Cleanup old observations periodically
        self._cleanup_old_observations()
        
        # Get filter rules from configuration
        filter_rules = self.config.get('filter_rules', {})
        exclude_pids = filter_rules.get('exclude_pids', [0, 1, 2])
        exclude_prefixes = filter_rules.get('exclude_prefixes', ['[', 'kthreadd', 'ksoftirqd'])
        
        try:
            for process in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    process_info = process.info
                    pid = process_info['pid']
                    name = process_info['name'] or 'unknown'
                    cmdline = process_info['cmdline'] or []
                    
                    # Apply filter rules - exclude by PID
                    if pid in exclude_pids:
                        continue
                    
                    # Apply filter rules - exclude by name prefix
                    if any(name.startswith(prefix) for prefix in exclude_prefixes):
                        continue
                    
                    # Check process age and stability
                    try:
                        proc = psutil.Process(pid)
                        process_age = time.time() - proc.create_time()
                        
                        # Skip short-lived or unstable processes
                        if not self._should_adjust_process(pid, process_age):
                            continue
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    
                    # Get current priority
                    current_nice = self.get_process_priority(pid)
                    if current_nice is None:
                        continue
                    
                    # Classify the process
                    workload_type, priority_class = self.classify_process(name, cmdline)
                    target_nice = priority_class.value
                    
                    # Create ProcessInfo
                    proc_info = ProcessInfo(
                        pid=pid,
                        name=name,
                        current_nice=current_nice,
                        target_nice=target_nice,
                        cpu_percent=process_info.get('cpu_percent', 0.0) or 0.0,
                        memory_percent=process_info.get('memory_percent', 0.0) or 0.0
                    )
                    
                    # Add to classification
                    if workload_type not in classified_processes:
                        classified_processes[workload_type] = []
                    classified_processes[workload_type].append(proc_info)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except (psutil.Error, OSError) as e:
            self.logger.error("Error scanning processes: %s", e)
        
        return classified_processes
    
    def optimize_process_priorities(self, workload_focus: Optional[str] = None) -> Dict[str, int]:
        """
        Optimize process priorities based on current workload
        
        Args:
            workload_focus: Optional workload type to prioritize
            
        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            'processes_adjusted': 0,
            'high_priority_set': 0,
            'low_priority_set': 0,
            'errors': 0,
            'short_lived_filtered': len([p for p in self.process_observations.values() if p['count'] < self.config.get('filter_rules', {}).get('stability_tracking', {}).get('required_observations', 2)]),
            'processes_tracked': len(self.process_observations)
        }
        
        # Get configuration values
        boost_config = self.config.get('workload_focus_boost', {})
        boost_enabled = boost_config.get('enabled', True)
        boost_amount = boost_config.get('boost_amount', 5)
        max_priority = boost_config.get('max_priority', -20)
        
        classified_processes = self.scan_and_classify_processes()
        
        for workload_type, processes in classified_processes.items():
            for proc_info in processes:
                try:
                    # Determine target priority
                    target_nice = proc_info.target_nice
                    
                    # Apply workload focus boost if enabled
                    if boost_enabled and workload_focus and workload_type == workload_focus:
                        target_nice = max(target_nice - boost_amount, max_priority)  # Boost priority
                        self.logger.info(
                            "Boosting %s process %s (PID %s)",
                            workload_type, proc_info.name, proc_info.pid
                        )
                    
                    # Only adjust if different from current
                    if proc_info.current_nice != target_nice:
                        if self.set_process_priority(proc_info.pid, target_nice):
                            stats['processes_adjusted'] += 1
                            if target_nice < 0:
                                stats['high_priority_set'] += 1
                            elif target_nice > 0:
                                stats['low_priority_set'] += 1
                            
                            # Update managed processes
                            self.managed_processes[proc_info.pid] = proc_info
                        else:
                            stats['errors'] += 1
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError) as e:
                    self.logger.error("Error optimizing process %s: %s", proc_info.pid, e)
                    stats['errors'] += 1
        
        self.logger.info("Priority optimization complete: %s", stats)
        return stats
    
    def restore_original_priorities(self) -> int:
        """
        Restore original priorities for all managed processes
        
        Returns:
            Number of processes restored
        """
        restored_count = 0
        
        for pid, original_nice in self.original_priorities.items():
            try:
                if self.set_process_priority(pid, original_nice):
                    restored_count += 1
                    self.logger.info(
                        "Restored process %s to original priority %s",
                        pid, original_nice
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError) as e:
                self.logger.error("Failed to restore priority for process %s: %s", pid, e)
        
        # Clear tracking
        self.managed_processes.clear()
        self.original_priorities.clear()
        
        return restored_count
    
    def get_priority_statistics(self) -> Dict[str, any]:
        """Get current priority statistics"""
        classified = self.scan_and_classify_processes()
        
        stats = {
            'total_processes': sum(len(procs) for procs in classified.values()),
            'workload_distribution': {wl: len(procs) for wl, procs in classified.items()},
            'priority_distribution': {},
            'managed_processes': len(self.managed_processes)
        }
        
        # Calculate priority distribution
        for processes in classified.values():
            for proc in processes:
                nice_range = self._get_nice_range(proc.current_nice)
                stats['priority_distribution'][nice_range] = stats['priority_distribution'].get(nice_range, 0) + 1
        
        return stats
    
    def _get_nice_range(self, nice_value: int) -> str:
        """Convert nice value to human-readable range"""
        if nice_value <= -15:
            return 'Critical (-20 to -15)'
        elif nice_value <= -5:
            return 'High (-14 to -5)'
        elif nice_value <= 5:
            return 'Normal (-4 to 5)'
        elif nice_value <= 15:
            return 'Low (6 to 15)'
        else:
            return 'Background (16 to 19)'

def main():
    """Test the ProcessPriorityManager"""
    
    
    # Initialize manager
    manager = ProcessPriorityManager()
    
    print("Current process classification:")
    classified = manager.scan_and_classify_processes()
    
    for workload_type, processes in classified.items():
        if processes:  # Only show non-empty categories
            print(f"\n{workload_type.upper()} ({len(processes)} processes):")
            for proc in processes[:3]:  # Show first 3 processes
                print(f"  {proc.name} (PID {proc.pid}): nice={proc.current_nice}, "
                      f"CPU={proc.cpu_percent:.1f}%, Memory={proc.memory_percent:.1f}%")
            if len(processes) > 3:
                print(f"  ... and {len(processes) - 3} more")
    
    print("\nPriority Statistics:")
    stats = manager.get_priority_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
