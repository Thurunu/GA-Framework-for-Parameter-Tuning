#!/usr/bin/env python3
"""
Dynamic Process Workload Detector
Monitors running processes and classifies workload types
"""

import psutil
import time
import threading
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import re

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
    """Classifies processes into workload types"""
    
    WORKLOAD_PATTERNS = {
        'database': [
            r'mysql.*', r'postgres.*', r'oracle.*', r'mongodb.*', 
            r'redis.*', r'memcached.*', r'cassandra.*', r'sqlite.*',
            r'mariadb.*', r'elasticsearch.*'
        ],
        'web_server': [
            r'nginx.*', r'apache.*', r'httpd.*', r'lighttpd.*',
            r'tomcat.*', r'jetty.*', r'gunicorn.*', r'uwsgi.*',
            r'node.*', r'npm.*'
        ],
        'hpc_compute': [
            r'matlab.*', r'python.*scipy.*', r'R.*', r'octave.*',
            r'mpirun.*', r'.*scientific.*', r'.*simulation.*',
            r'python.*numpy.*', r'tensorflow.*', r'pytorch.*'
        ],
        'media_processing': [
            r'ffmpeg.*', r'mencoder.*', r'handbrake.*', r'blender.*',
            r'gimp.*', r'imagemagick.*', r'convert.*', r'vlc.*'
        ],
        'compilation': [
            r'gcc.*', r'g\+\+.*', r'clang.*', r'make.*', r'cmake.*',
            r'rustc.*', r'javac.*', r'dotnet.*', r'cargo.*',
            r'mvn.*', r'gradle.*'
        ],
        'io_intensive': [
            r'rsync.*', r'cp.*', r'tar.*', r'gzip.*', r'unzip.*',
            r'backup.*', r'sync.*', r'dd.*', r'find.*'
        ]
    }
    
    @classmethod
    def classify_process(cls, proc_info: ProcessInfo) -> str:
        """Classify process based on name and command line"""
        full_cmd = f"{proc_info.name} {proc_info.cmdline}".lower()
        
        for workload_type, patterns in cls.WORKLOAD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, full_cmd):
                    return workload_type
        
        # Classify based on resource usage patterns
        if proc_info.cpu_percent > 80:
            return 'cpu_intensive'
        elif proc_info.memory_percent > 50:
            return 'memory_intensive'
        elif proc_info.io_read_bytes + proc_info.io_write_bytes > 1000000:  # 1MB/s
            return 'io_intensive'
        elif proc_info.network_connections > 10:
            return 'network_intensive'
        
        return 'general'

class ProcessWorkloadDetector:
    """Monitors system processes and detects workload changes"""
    
    def __init__(self, monitoring_interval: float = 2.0, 
                 significance_threshold: float = 5.0):
        """
        Initialize process detector
        
        Args:
            monitoring_interval: How often to check processes (seconds)
            significance_threshold: CPU% threshold for significant processes
        """
        self.monitoring_interval = monitoring_interval
        self.significance_threshold = significance_threshold
        
        self.running = False
        self.monitor_thread = None
        
        # Process tracking
        self.current_processes: Dict[int, ProcessInfo] = {}
        self.workload_history = deque(maxlen=100)
        self.dominant_workload = "general"
        self.workload_change_callbacks = []
        
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
                
                # Get network connections
                try:
                    connections = len(proc.connections())
                except:
                    connections = 0
                
                proc_info = ProcessInfo(
                    pid=proc_data['pid'],
                    name=proc_data['name'],
                    cmdline=' '.join(proc_data['cmdline'] or []),
                    cpu_percent=proc_data['cpu_percent'],
                    memory_percent=proc_data['memory_percent'],
                    io_read_bytes=io_read,
                    io_write_bytes=io_write,
                    network_connections=connections,
                    start_time=proc_data['create_time']
                )
                
                # Classify workload
                proc_info.workload_type = WorkloadClassifier.classify_process(proc_info)
                
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
        
        for workload, stats in current_workloads.items():
            # Score = CPU weight * 0.6 + Memory weight * 0.3 + Process count * 0.1
            score = (stats['total_cpu'] * 0.6 + 
                    stats['total_memory'] * 0.3 + 
                    stats['count'] * 0.1)
            
            if score > max_score:
                max_score = score
                new_dominant = workload
        
        # Record workload history
        self.workload_history.append({
            'timestamp': time.time(),
            'dominant_workload': new_dominant,
            'workload_distribution': dict(current_workloads)
        })
        
        # Check for workload change
        if new_dominant != self.dominant_workload:
            old_workload = self.dominant_workload
            self.dominant_workload = new_dominant
            
            print(f"Workload change detected: {old_workload} -> {new_dominant}")
            
            # Notify callbacks
            for callback in self.workload_change_callbacks:
                try:
                    callback(old_workload, new_dominant, current_workloads)
                except Exception as e:
                    print(f"Error in workload change callback: {e}")
    
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
