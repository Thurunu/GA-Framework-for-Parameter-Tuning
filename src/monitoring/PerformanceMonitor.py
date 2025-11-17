#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Performance Monitor
This module handles real-time system performance monitoring
"""

import psutil
import time
import json
import threading
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import subprocess
import os
import sys
from DataClasses import PerformanceMetrics
from CentralDataStore import get_data_store

project_root = os.path.join(os.path.dirname(__file__), '..')

sys.path.insert(0, os.path.join(project_root, 'data'))

class PerformanceMonitor:
    """Real-time system performance monitoring"""
    
    def __init__(self, sampling_interval: float = 1.0, history_size: int = 1000):
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.monitoring = False
        self.monitor_thread = None
        
        # Get central data store
        self.data_store = get_data_store()
        
        # Initialize baseline counters
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_cpu_times = psutil.cpu_times()
        
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Performance monitoring started...")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Performance monitoring stopped.")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Store in central data store
                self.data_store.set_current_metrics(metrics)
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(self.sampling_interval)
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes if disk_io else 0
        disk_write = disk_io.write_bytes if disk_io else 0
        
        # Network I/O metrics
        network_io = psutil.net_io_counters()
        net_sent = network_io.bytes_sent if network_io else 0
        net_recv = network_io.bytes_recv if network_io else 0
        
        # System load (handle Windows vs Linux difference)
        try:
            load_avg = os.getloadavg()
        except (OSError, AttributeError):
            # Windows doesn't have getloadavg, use CPU count approximation
            load_avg = (psutil.cpu_percent(interval=None) / 100.0 * psutil.cpu_count(), 0, 0)
        
        # Context switches and interrupts
        cpu_stats = psutil.cpu_stats()
        
        # TCP connections
        tcp_connections = len([c for c in psutil.net_connections() if c.type == 1])
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_bytes_sent=net_sent,
            network_bytes_recv=net_recv,
            load_average=load_avg,
            context_switches=cpu_stats.ctx_switches,
            interrupts=cpu_stats.interrupts,
            tcp_connections=tcp_connections
        )
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_seconds: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for specified duration"""
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, duration_seconds: int = 60) -> Dict:
        """Calculate average metrics over specified duration"""
        history = self.get_metrics_history(duration_seconds)
        
        if not history:
            return {}
        
        metrics_sum = {
            'cpu_percent': sum(m.cpu_percent for m in history),
            'memory_percent': sum(m.memory_percent for m in history),
            'disk_io_read_rate': 0,
            'disk_io_write_rate': 0,
            'network_sent_rate': 0,
            'network_recv_rate': 0,
            'load_average_1m': sum(m.load_average[0] for m in history),
            'context_switches_rate': 0,
            'tcp_connections': sum(m.tcp_connections for m in history)
        }
        
        count = len(history)
        
        # Calculate rates for cumulative metrics
        if count > 1:
            time_span = history[-1].timestamp - history[0].timestamp
            if time_span > 0:
                metrics_sum['disk_io_read_rate'] = (
                    (history[-1].disk_io_read - history[0].disk_io_read) / time_span
                )
                metrics_sum['disk_io_write_rate'] = (
                    (history[-1].disk_io_write - history[0].disk_io_write) / time_span
                )
                metrics_sum['network_sent_rate'] = (
                    (history[-1].network_bytes_sent - history[0].network_bytes_sent) / time_span
                )
                metrics_sum['network_recv_rate'] = (
                    (history[-1].network_bytes_recv - history[0].network_bytes_recv) / time_span
                )
                metrics_sum['context_switches_rate'] = (
                    (history[-1].context_switches - history[0].context_switches) / time_span
                )
        
        # Calculate averages
        averages = {
            'cpu_percent_avg': metrics_sum['cpu_percent'] / count,
            'memory_percent_avg': metrics_sum['memory_percent'] / count,
            'disk_read_rate_mb_s': metrics_sum['disk_io_read_rate'] / (1024 * 1024),
            'disk_write_rate_mb_s': metrics_sum['disk_io_write_rate'] / (1024 * 1024),
            'network_sent_rate_mb_s': metrics_sum['network_sent_rate'] / (1024 * 1024),
            'network_recv_rate_mb_s': metrics_sum['network_recv_rate'] / (1024 * 1024),
            'load_average_1m_avg': metrics_sum['load_average_1m'] / count,
            'context_switches_per_sec': metrics_sum['context_switches_rate'],
            'tcp_connections_avg': metrics_sum['tcp_connections'] / count,
            'sample_count': count
        }
        
        return averages
    
    def detect_performance_anomalies(self, threshold_multiplier: float = 2.0) -> Dict:
        """Detect performance anomalies based on historical data"""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_metrics = self.get_average_metrics(60)  # Last minute
        historical_metrics = self.get_average_metrics(600)  # Last 10 minutes
        
        if not recent_metrics or not historical_metrics:
            return {"status": "insufficient_data"}
        
        anomalies = {}
        
        # Check CPU anomaly
        if recent_metrics['cpu_percent_avg'] > historical_metrics['cpu_percent_avg'] * threshold_multiplier:
            anomalies['cpu_high'] = {
                'current': recent_metrics['cpu_percent_avg'],
                'baseline': historical_metrics['cpu_percent_avg']
            }
        
        # Check memory anomaly
        if recent_metrics['memory_percent_avg'] > historical_metrics['memory_percent_avg'] * threshold_multiplier:
            anomalies['memory_high'] = {
                'current': recent_metrics['memory_percent_avg'],
                'baseline': historical_metrics['memory_percent_avg']
            }
        
        # Check load average anomaly
        if recent_metrics['load_average_1m_avg'] > historical_metrics['load_average_1m_avg'] * threshold_multiplier:
            anomalies['load_high'] = {
                'current': recent_metrics['load_average_1m_avg'],
                'baseline': historical_metrics['load_average_1m_avg']
            }
        
        return {
            "status": "analyzed",
            "anomalies": anomalies,
            "anomaly_count": len(anomalies)
        }
    
    def export_metrics_json(self, filename: str, duration_seconds: int = 300):
        """Export metrics history to JSON file"""
        history = self.get_metrics_history(duration_seconds)
        data = {
            'export_timestamp': time.time(),
            'duration_seconds': duration_seconds,
            'metrics_count': len(history),
            'metrics': [asdict(m) for m in history]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(history)} metrics to {filename}")

# Example usage and testing
if __name__ == "__main__":
    monitor = PerformanceMonitor(sampling_interval=0.5)
    
    try:
        monitor.start_monitoring()
        
        # Let it collect data for a few seconds
        time.sleep(10)
        
        # Get current metrics
        current = monitor.get_current_metrics()
        if current:
            print(f"\nCurrent Performance:")
            print(f"CPU: {current.cpu_percent:.1f}%")
            print(f"Memory: {current.memory_percent:.1f}%")
            print(f"Load Average: {current.load_average[0]:.2f}")
            print(f"TCP Connections: {current.tcp_connections}")
        
        # Get averages
        averages = monitor.get_average_metrics(10)
        if averages:
            print(f"\n10-Second Averages:")
            print(f"CPU: {averages['cpu_percent_avg']:.1f}%")
            print(f"Memory: {averages['memory_percent_avg']:.1f}%")
            print(f"Disk Read: {averages['disk_read_rate_mb_s']:.2f} MB/s")
            print(f"Network Sent: {averages['network_sent_rate_mb_s']:.2f} MB/s")
        
        # Check for anomalies
        anomalies = monitor.detect_performance_anomalies()
        print(f"\nAnomaly Detection: {anomalies}")
        
        # Export data
        monitor.export_metrics_json("performance_metrics.json", 10)
        
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        monitor.stop_monitoring()