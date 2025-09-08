#!/usr/bin/env python3
"""
Process Priority Manager for EEVDF Scheduler Support
Handles process nice values and priority adjustments for kernel 6.6+ systems
"""

import subprocess
import logging
import os
import psutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PriorityClass(Enum):
    """Process priority classes for different workload types"""
    CRITICAL = -20      # Highest priority (real-time-like)
    HIGH = -10         # High priority (interactive)
    NORMAL = 0         # Default priority
    LOW = 10          # Low priority (batch)
    BACKGROUND = 19    # Lowest priority (background tasks)

@dataclass
class ProcessInfo:
    """Information about a process and its priority"""
    pid: int
    name: str
    current_nice: int
    target_nice: int
    cpu_percent: float
    memory_percent: float
    
class ProcessPriorityManager:
    """
    Manages process priorities using nice/renice for EEVDF scheduler optimization
    """
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Process classification patterns
        self.workload_patterns = {
            'database': {
                'patterns': ['mysqld', 'postgres', 'redis', 'mongodb', 'sqlite'],
                'priority_class': PriorityClass.HIGH,
                'description': 'Database servers'
            },
            'web_server': {
                'patterns': ['nginx', 'apache', 'httpd', 'gunicorn', 'uwsgi'],
                'priority_class': PriorityClass.HIGH,
                'description': 'Web servers'
            },
            'compute_intensive': {
                'patterns': ['python.*numpy', 'python.*scipy', 'matlab', 'R '],
                'priority_class': PriorityClass.NORMAL,
                'description': 'Compute-intensive applications'
            },
            'background_tasks': {
                'patterns': ['cron', 'rsync', 'backup', 'updatedb', 'logrotate'],
                'priority_class': PriorityClass.BACKGROUND,
                'description': 'Background maintenance tasks'
            },
            'interactive': {
                'patterns': ['firefox', 'chrome', 'vim', 'emacs', 'code'],
                'priority_class': PriorityClass.HIGH,
                'description': 'Interactive applications'
            },
            'system_critical': {
                'patterns': ['systemd', 'kernel', 'init', 'ssh'],
                'priority_class': PriorityClass.CRITICAL,
                'description': 'Critical system processes'
            }
        }
        
        # Track managed processes
        self.managed_processes: Dict[int, ProcessInfo] = {}
        self.original_priorities: Dict[int, int] = {}
        
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
                self.logger.error(f"Invalid nice value {nice_value}. Must be between -20 and 19")
                return False
            
            # Store original priority if not already stored
            if pid not in self.original_priorities:
                original_nice = self.get_process_priority(pid)
                if original_nice is not None:
                    self.original_priorities[pid] = original_nice
            
            # Set new priority
            if os.name == 'nt':  # Windows simulation
                self.logger.info(f"Simulated: renice {nice_value} {pid}")
                return True
            else:
                result = subprocess.run(
                    ['renice', str(nice_value), str(pid)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self.logger.info(f"Set process {pid} priority to {nice_value}")
                return True
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set priority for process {pid}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error setting priority for {pid}: {e}")
            return False
    
    def scan_and_classify_processes(self) -> Dict[str, List[ProcessInfo]]:
        """
        Scan all running processes and classify them by workload type
        
        Returns:
            Dictionary mapping workload types to lists of ProcessInfo
        """
        classified_processes: Dict[str, List[ProcessInfo]] = {}
        
        try:
            for process in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    process_info = process.info
                    pid = process_info['pid']
                    name = process_info['name'] or 'unknown'
                    cmdline = process_info['cmdline'] or []
                    
                    # Skip kernel threads and system processes we shouldn't touch
                    if pid <= 2 or name.startswith('[') or name in ['kthreadd', 'ksoftirqd']:
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
                    
        except Exception as e:
            self.logger.error(f"Error scanning processes: {e}")
        
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
            'errors': 0
        }
        
        classified_processes = self.scan_and_classify_processes()
        
        for workload_type, processes in classified_processes.items():
            for proc_info in processes:
                try:
                    # Determine target priority
                    target_nice = proc_info.target_nice
                    
                    # Apply workload focus boost
                    if workload_focus and workload_type == workload_focus:
                        target_nice = max(target_nice - 5, -20)  # Boost priority
                        self.logger.info(f"Boosting {workload_type} process {proc_info.name} (PID {proc_info.pid})")
                    
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
                            
                except Exception as e:
                    self.logger.error(f"Error optimizing process {proc_info.pid}: {e}")
                    stats['errors'] += 1
        
        self.logger.info(f"Priority optimization complete: {stats}")
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
                    self.logger.info(f"Restored process {pid} to original priority {original_nice}")
            except Exception as e:
                self.logger.error(f"Failed to restore priority for process {pid}: {e}")
        
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
    print("Process Priority Manager - EEVDF Scheduler Support")
    print("=" * 60)
    
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
    
    print(f"\nPriority Statistics:")
    stats = manager.get_priority_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
