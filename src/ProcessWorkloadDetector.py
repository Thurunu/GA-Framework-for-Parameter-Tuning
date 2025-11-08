#!/usr/bin/env python3
"""
Dynamic Process Workload Detector
Monitors running processes and classifies workload types
"""

import psutil
import time
import threading
from typing import Dict, List
from collections import defaultdict, deque

# Import WorkloadClassifier from separate module
from WorkloadClassifier import WorkloadClassifier, ProcessInfo

class ProcessWorkloadDetector:
    """Monitors system processes and detects workload changes"""
    
    def __init__(self, monitoring_interval: float = 2.0, 
                 significance_threshold: float = 5.0,
                 config_file: str = None,
                 workload_stability_duration: float = 15.0,
                 workload_change_threshold: float = 0.3):
        """
        Initialize process detector
        
        Args:
            monitoring_interval: How often to check processes (seconds)
            significance_threshold: CPU% threshold for significant processes
            config_file: Path to workload patterns YAML file (optional)
            workload_stability_duration: Seconds to wait before confirming workload change
            workload_change_threshold: Minimum score difference (0-1) to trigger change
        """
        self.monitoring_interval = monitoring_interval
        self.significance_threshold = significance_threshold
        self.workload_stability_duration = workload_stability_duration
        self.workload_change_threshold = workload_change_threshold
        
        # Initialize classifier with patterns from YAML
        self.classifier = WorkloadClassifier(config_file)
        
        self.running = False
        self.monitor_thread = None
        
        # Process tracking
        self.current_processes: Dict[int, ProcessInfo] = {}
        self.workload_history = deque(maxlen=100)
        self.dominant_workload = "general"
        self.workload_change_callbacks = []
        
        # Workload stability tracking
        self.pending_workload_change = None
        self.pending_workload_since = None
        self.last_confirmed_workload = "general"
        
        # Statistics
        self.workload_stats = defaultdict(lambda: {
            'count': 0, 'cpu_time': 0, 'memory_usage': 0
        })
    
    def add_workload_change_callback(self, callback):
        """Add callback function to be called when workload changes"""
        self.workload_change_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous process monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Process workload monitoring started...")
    
    def stop_monitoring(self):
        """Stop process monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Process workload monitoring stopped.")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._scan_processes()
                self._analyze_workload_changes()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Error in process monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _scan_processes(self):
        """Scan and analyze current processes"""
        new_processes = {}
        significant_processes = []
        current_time = time.time()
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 
                                       'memory_percent', 'create_time']):
            try:
                proc_data = proc.info
                
                # Skip system processes and low-impact processes
                if (proc_data['cpu_percent'] < self.significance_threshold and 
                    proc_data['memory_percent'] < 1.0):
                    continue
                
                # Get I/O stats
                try:
                    io_stats = proc.io_counters()
                    io_read = io_stats.read_bytes
                    io_write = io_stats.write_bytes
                except:
                    io_read = io_write = 0
                
                # Calculate I/O rate (bytes per second) instead of using total accumulated bytes
                io_rate = 0
                if proc_data['pid'] in self.current_processes:
                    old_proc = self.current_processes[proc_data['pid']]
                    time_diff = current_time - old_proc.start_time
                    if time_diff > 0:
                        # Calculate rate since last scan (not since process start)
                        io_diff = (io_read + io_write) - (old_proc.io_read_bytes + old_proc.io_write_bytes)
                        io_rate = max(0, io_diff / self.monitoring_interval)  # bytes per second
                
                # Get network connections
                try:
                    connections = len(proc.connections())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    connections = 0
                
                proc_info = ProcessInfo(
                    pid=proc_data['pid'],
                    name=proc_data['name'],
                    cmdline=' '.join(proc_data['cmdline'] or []),
                    cpu_percent=proc_data['cpu_percent'],
                    memory_percent=proc_data['memory_percent'],
                    io_read_bytes=io_read,
                    io_write_bytes=io_write,
                    io_rate_bytes_per_sec=io_rate,  # Add I/O rate
                    network_connections=connections,
                    start_time=current_time  # Use current time for tracking
                )
                
                # Classify workload using instance method
                proc_info.workload_type = self.classifier.classify_process(proc_info)
                
                new_processes[proc_info.pid] = proc_info
                significant_processes.append(proc_info)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Update process tracking
        self.current_processes = new_processes
        
        # Update workload statistics
        self._update_workload_stats(significant_processes)
    
    def _update_workload_stats(self, processes: List[ProcessInfo]):
        """Update workload statistics and detect dominant workload"""
        current_workloads = defaultdict(lambda: {
            'count': 0, 'total_cpu': 0, 'total_memory': 0
        })
        
        for proc in processes:
            workload = proc.workload_type
            current_workloads[workload]['count'] += 1
            current_workloads[workload]['total_cpu'] += proc.cpu_percent
            current_workloads[workload]['total_memory'] += proc.memory_percent
        
        # Determine dominant workload based on weighted score
        max_score = 0
        new_dominant = "general"
        total_score = 0
        
        # Calculate scores for all workloads
        workload_scores = {}
        for workload, stats in current_workloads.items():
            # Score = CPU weight * 0.6 + Memory weight * 0.3 + Process count * 0.1
            score = (stats['total_cpu'] * 0.6 + 
                    stats['total_memory'] * 0.3 + 
                    stats['count'] * 0.1)
            workload_scores[workload] = score
            total_score += score
            
            if score > max_score:
                max_score = score
                new_dominant = workload
        
        # If no significant workload detected, default to general
        if max_score < 1.0:  # Minimum threshold to avoid noise
            new_dominant = "general"
        
        # Calculate relative dominance (0-1 scale)
        dominance_ratio = max_score / total_score if total_score > 0 else 0
        
        # Record workload history
        self.workload_history.append({
            'timestamp': time.time(),
            'dominant_workload': new_dominant,
            'workload_distribution': dict(current_workloads),
            'dominance_ratio': dominance_ratio
        })
        
        # Workload change detection with stability check
        current_time = time.time()
        
        # Check if we have a pending workload change
        if self.pending_workload_change and self.pending_workload_change != new_dominant:
            # Workload changed again before stabilizing - reset the timer
            print(f"📊 Workload fluctuation: {self.pending_workload_change} -> {new_dominant} (resetting timer)")
            self.pending_workload_change = new_dominant
            self.pending_workload_since = current_time
        elif self.pending_workload_change is None and new_dominant != self.last_confirmed_workload:
            # New workload detected - start stability timer
            if dominance_ratio > self.workload_change_threshold:
                print(f"📊 Potential workload change: {self.last_confirmed_workload} -> {new_dominant} (waiting {self.workload_stability_duration}s for stability)")
                self.pending_workload_change = new_dominant
                self.pending_workload_since = current_time
        elif self.pending_workload_change == new_dominant:
            # Same workload persists - check if it's been stable long enough
            duration = current_time - self.pending_workload_since
            if duration >= self.workload_stability_duration:
                # Workload has been stable - confirm the change
                old_workload = self.last_confirmed_workload
                self.last_confirmed_workload = new_dominant
                self.dominant_workload = new_dominant
                self.pending_workload_change = None
                self.pending_workload_since = None
                
                print(f"✅ Workload change CONFIRMED: {old_workload} -> {new_dominant} (stable for {duration:.1f}s)")
                
                # Notify callbacks
                for callback in self.workload_change_callbacks:
                    try:
                        callback(old_workload, new_dominant, current_workloads)
                    except Exception as e:
                        print(f"Error in workload change callback: {e}")
            else:
                # Still stabilizing
                remaining = self.workload_stability_duration - duration
                print(f"⏳ Workload stabilizing: {new_dominant} ({remaining:.1f}s remaining)")
        elif new_dominant == self.last_confirmed_workload:
            # Workload returned to previous state - cancel pending change
            if self.pending_workload_change:
                print(f"↩️  Workload returned to: {new_dominant} (cancelled pending change)")
                self.pending_workload_change = None
                self.pending_workload_since = None
    
    def _analyze_workload_changes(self):
        """Analyze workload change patterns"""
        if len(self.workload_history) < 5:
            return
        
        # Check for rapid workload switching (could indicate batch processing)
        recent_workloads = [entry['dominant_workload'] 
                          for entry in list(self.workload_history)[-5:]]
        
        if len(set(recent_workloads)) > 3:
            print("Rapid workload switching detected - using adaptive strategy")
    
    def get_current_workload_info(self) -> Dict:
        """Get current workload information"""
        return {
            'dominant_workload': self.dominant_workload,
            'active_processes': len(self.current_processes),
            'workload_history': list(self.workload_history)[-10:],  # Last 10 entries
            'process_details': [
                {
                    'name': proc.name,
                    'workload_type': proc.workload_type,
                    'cpu_percent': proc.cpu_percent,
                    'memory_percent': proc.memory_percent
                }
                for proc in self.current_processes.values()
            ]
        }

# Example usage
if __name__ == "__main__":
    detector = ProcessWorkloadDetector()
    
    def on_workload_change(old_workload, new_workload, stats):
        print(f"Workload changed from {old_workload} to {new_workload}")
        print(f"Current stats: {stats}")
    
    detector.add_workload_change_callback(on_workload_change)
    
    try:
        detector.start_monitoring()
        
        # Monitor for 30 seconds
        time.sleep(30)
        
        # Print current info
        info = detector.get_current_workload_info()
        print(f"Current dominant workload: {info['dominant_workload']}")
        print(f"Active processes: {info['active_processes']}")
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        detector.stop_monitoring()
