#!/usr/bin/env python3
"""
Continuous Kernel Parameter Optimization System
Runs continuously and adapts to workload changes
"""

import time
import threading
import json
import sys
import signal
import os
import yaml
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import queue
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ProcessWorkloadDetector import ProcessWorkloadDetector
from HybridOptimizationEngine import HybridOptimizationEngine
from WorkloadCharacterizer import OptimizationStrategy
from PerformanceMonitor import PerformanceMonitor
from KernelParameterInterface import KernelParameterInterface
from ProcessPriorityManager import ProcessPriorityManager

# Import centralized management agent reporter
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from agent_reporter import AgentReporter
    CENTRALIZED_MANAGEMENT_ENABLED = True
except ImportError:
    print("Warning: AgentReporter not available. Running in standalone mode.")
    CENTRALIZED_MANAGEMENT_ENABLED = False
    AgentReporter = None

@dataclass
class OptimizationProfile:
    """Optimization profile for specific workload types"""
    workload_type: str
    parameter_bounds: Dict[str, tuple]
    strategy: OptimizationStrategy
    evaluation_budget: int
    time_budget: float
    performance_weights: Dict[str, float]

class ContinuousOptimizer:
    """Continuous optimization system that adapts to workload changes"""
    
    @staticmethod
    def _load_optimization_profiles(config_file: str = None) -> Dict[str, OptimizationProfile]:
        """Load optimization profiles from YAML configuration file"""
        if config_file is None:
            # Default to config/optimization_profiles.yml relative to project root
            current_dir = Path(__file__).parent
            config_file = current_dir.parent / "config" / "optimization_profiles.yml"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            profiles = {}
            for profile_name, profile_data in config['profiles'].items():
                # Convert parameter bounds from lists to tuples
                parameter_bounds = {
                    param: tuple(bounds) 
                    for param, bounds in profile_data['parameter_bounds'].items()
                }
                
                # Convert strategy string to enum
                strategy = OptimizationStrategy[profile_data['strategy']]
                
                profiles[profile_name] = OptimizationProfile(
                    workload_type=profile_data['workload_type'],
                    parameter_bounds=parameter_bounds,
                    strategy=strategy,
                    evaluation_budget=profile_data['evaluation_budget'],
                    time_budget=profile_data['time_budget'],
                    performance_weights=profile_data['performance_weights']
                )
            
            return profiles
            
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found. Using default profiles.")
            return ContinuousOptimizer._get_default_profiles()
        except Exception as e:
            print(f"Error loading optimization profiles: {e}")
            print("Falling back to default profiles.")
            return ContinuousOptimizer._get_default_profiles()
    
    @staticmethod
    def _get_default_profiles() -> Dict[str, OptimizationProfile]:
        """Get default optimization profiles as fallback"""
        return {
            'general': OptimizationProfile(
                workload_type='general',
                parameter_bounds={
                    'vm.swappiness': (10, 80),
                    'vm.dirty_ratio': (10, 30),
                },
                strategy=OptimizationStrategy.ADAPTIVE,
                evaluation_budget=10,
                time_budget=120.0,
                performance_weights={
                    'cpu_efficiency': 0.25,
                    'memory_efficiency': 0.25,
                    'io_throughput': 0.25,
                    'network_throughput': 0.25
                }
            )
        }
    
    def __init__(self, 
                 adaptation_delay: float = 30.0,
                 stability_period: float = 180.0,
                 log_file: str = "/var/log/continuous_optimizer.log",
                 config_file: str = None):
        """
        Initialize continuous optimizer
        
        Args:
            adaptation_delay: Wait time before optimizing after workload change
            stability_period: Minimum time between optimizations
            log_file: Log file path
            config_file: Path to optimization profiles YAML file (optional)
        """
        self.adaptation_delay = adaptation_delay
        self.stability_period = stability_period
        self.log_file = log_file
        
        # Load optimization profiles from YAML
        self.OPTIMIZATION_PROFILES = self._load_optimization_profiles(config_file)
        print(f"Loaded {len(self.OPTIMIZATION_PROFILES)} optimization profiles")
        
        # Initialize components
        self.process_detector = ProcessWorkloadDetector()
        self.performance_monitor = PerformanceMonitor()
        self.kernel_interface = KernelParameterInterface()
        self.priority_manager = ProcessPriorityManager()  # New EEVDF support
        
        # Initialize centralized management reporter (optional)
        self.reporter = None
        if CENTRALIZED_MANAGEMENT_ENABLED:
            try:
                master_url = os.getenv('MASTER_URL')
                # api_key = os.getenv('API_KEY')
                if master_url:
                    self.reporter = AgentReporter(
                        agent_id=os.getenv('AGENT_ID', os.uname().nodename),
                        master_url=master_url,
                        # api_key=api_key
                    )
                    # Register with master on startup
                    print("🌍 Connecting to centralized management...")
                    self.reporter.register()
                    print(f"✓ Connected to centralized management at {master_url}")
                else:
                    print("Info: MASTER_URL not set. Running in standalone mode.")
            except Exception as e:
                print(f"Warning: Could not connect to centralized management: {e}")
                self.reporter = None
        
        # Optimization tracking
        self.current_profile: Optional[OptimizationProfile] = None
        self.last_optimization_time = 0
        self.optimization_in_progress = False
        self.optimization_queue = queue.Queue()
        
        # Threading
        self.running = False
        self.optimization_thread = None
        
        # Setup callbacks
        self.process_detector.add_workload_change_callback(self._on_workload_change)
    
    def start_continuous_optimization(self):
        """Start the continuous optimization system"""
        print("Starting Continuous Kernel Optimization System...")
        
        self.running = True
        
        # Start monitoring components
        self.process_detector.start_monitoring()
        self.performance_monitor.start_monitoring()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_worker, daemon=True
        )
        self.optimization_thread.start()
        
        # Initial optimization with general profile
        self._schedule_optimization('general')
        
        print("Continuous optimization system started!")
        print("Monitoring processes and adapting parameters...")
        print(f"Logs: {self.log_file}")
    
    def stop_continuous_optimization(self):
        """Stop the continuous optimization system"""
        print("Stopping continuous optimization...")
        
        self.running = False
        
        # Stop monitoring
        self.process_detector.stop_monitoring()
        self.performance_monitor.stop_monitoring()
        
        # Wait for optimization thread
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        
        print("Continuous optimization stopped.")
    
    def _on_workload_change(self, old_workload: str, new_workload: str, stats: Dict):
        """Handle workload change event"""
        self._log(f"Workload change detected: {old_workload} -> {new_workload}")
        
        # Schedule optimization after delay
        threading.Timer(
            self.adaptation_delay, 
            lambda: self._schedule_optimization(new_workload)
        ).start()
    
    def _schedule_optimization(self, workload_type: str):
        """Schedule optimization for specific workload type"""
        if self.optimization_in_progress:
            self._log(f"Optimization already in progress, skipping {workload_type}")
            return
        
        # Check stability period
        time_since_last = time.time() - self.last_optimization_time
        if time_since_last < self.stability_period:
            self._log(f"Stability period not met, delaying optimization for {workload_type}")
            return
        
        # Add to queue
        self.optimization_queue.put(workload_type)
        self._log(f"Scheduled optimization for workload: {workload_type}")
    
    # this is for connecting to centralized management and handling commands
    def _optimization_worker(self):
        """Worker thread for running optimizations"""
        last_heartbeat = 0
        heartbeat_interval = 30.0  # Send heartbeat every 30 seconds
        
        while self.running:
            try:
                # Send heartbeat to centralized management
                current_time = time.time()
                if self.reporter and (current_time - last_heartbeat) >= heartbeat_interval:
                    try:
                        # Get current metrics and convert to dictionary
                        metrics_obj = self.performance_monitor.get_current_metrics()
                        
                        if metrics_obj:
                            # Convert PerformanceMetrics dataclass to dict
                            metrics = asdict(metrics_obj)
                        else:
                            # Fallback: use average metrics if no current metrics
                            metrics = self.performance_monitor.get_average_metrics(30) or {}
                        
                        # Get current workload from detector (if available)
                        current_workload = None
                        if hasattr(self.process_detector, 'current_workload_type'):
                            current_workload = self.process_detector.current_workload_type
                        
                        self.reporter.send_heartbeat(
                            metrics=metrics,
                            workload_type=current_workload,
                            optimization_score=None  # Could add optimization score tracking
                        )
                        last_heartbeat = current_time
                    except Exception as e:
                        self._log(f"Failed to send heartbeat: {e}")
                
                # Poll for commands from centralized management
                if self.reporter:
                    try:
                        commands = self.reporter.poll_commands()
                        for cmd in commands:
                            self._execute_remote_command(cmd)
                    except Exception as e:
                        self._log(f"Failed to poll commands: {e}")
                
                # Wait for optimization request
                workload_type = self.optimization_queue.get(timeout=1)
                
                if workload_type in self.OPTIMIZATION_PROFILES:
                    self._run_optimization(workload_type)
                else:
                    self._log(f"Unknown workload type: {workload_type}")
                
                self.optimization_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self._log(f"Error in optimization worker: {e}")
    
    def _run_optimization(self, workload_type: str):
        """Run optimization for specific workload type"""
        if self.optimization_in_progress:
            return
        
        self.optimization_in_progress = True
        start_time = time.time()
        
        try:
            profile = self.OPTIMIZATION_PROFILES[workload_type]
            self.current_profile = profile
            
            self._log(f"🚀 Starting optimization for {workload_type} workload")
            self._log(f"🎯 Strategy: {profile.strategy.value}, 💰 Budget: {profile.evaluation_budget}")
            
            # Create backup
            backup_file = self.kernel_interface.backup_current_parameters()
            self._log(f"Created parameter backup: {backup_file}")
            
            # Initialize optimization engine
            engine = HybridOptimizationEngine(
                parameter_bounds=profile.parameter_bounds,
                strategy=profile.strategy,
                evaluation_budget=profile.evaluation_budget,
                time_budget=profile.time_budget
            )
            
            # Create workload-specific objective function
            objective_function = self._create_objective_function(profile)
            
            # Run optimization
            result = engine.optimize(objective_function)
            
            # Apply best parameters
            if result.best_parameters:
                self._log(f"Applying optimized parameters: {result.best_parameters}")
                apply_results = self.kernel_interface.apply_parameter_set(result.best_parameters)
                
                failed_params = [name for name, success in apply_results.items() if not success]
                if failed_params:
                    self._log(f"Failed to apply parameters: {failed_params}")
                else:
                    self._log("All parameters applied successfully")
                
                # Apply process priority optimizations for EEVDF scheduler
                try:
                    priority_stats = self.priority_manager.optimize_process_priorities(
                        workload_focus=workload_type
                    )
                    self._log(f"Process priority optimization: {priority_stats}")
                except Exception as e:
                    self._log(f"Process priority optimization failed: {e}")
            
            # Log results
            optimization_time = time.time() - start_time
            self._log(f"Optimization completed in {optimization_time:.2f}s")
            self._log(f"Best score: {result.best_score:.6f}")
            self._log(f"Total evaluations: {result.total_evaluations}")
            
            # Export results
            results_file = f"continuous_opt_{workload_type}_{int(start_time)}.json"
            engine.export_results(results_file, result)
            
            self.last_optimization_time = time.time()
            
        except Exception as e:
            self._log(f"Optimization failed for {workload_type}: {e}")
            
        finally:
            self.optimization_in_progress = False
    
    def _create_objective_function(self, profile: OptimizationProfile):
        """Create objective function for specific workload profile"""
        def objective_function(params: Dict[str, float]) -> float:
            try:
                # Apply parameters
                results = self.kernel_interface.apply_parameter_set(params)
                
                # Separate failed params into unavailable vs actual failures
                failed_params = []
                unavailable_params = []
                
                for name, success in results.items():
                    if not success:
                        # Check if parameter is unavailable on this system
                        if not self.kernel_interface.check_parameter_availability(name):
                            unavailable_params.append(name)
                        else:
                            failed_params.append(name)
                
                # If there are actual failures (not just unavailable params), return penalty
                if failed_params:
                    self._log(f"Parameter application failed: {failed_params}")
                    return -1000.0  # Penalty for actual failures
                
                # If only unavailable params, continue with available ones
                if unavailable_params:
                    self._log(f"Skipped unavailable parameters: {unavailable_params} (continuing with available params)")
                
                # Allow system to stabilize
                time.sleep(2)
                
                # Monitor performance for shorter duration in continuous mode
                monitor_duration = 10  # Reduced for continuous operation
                time.sleep(monitor_duration)
                
                # Get performance metrics
                metrics = self.performance_monitor.get_average_metrics(monitor_duration)
                
                if not metrics:
                    return -500.0
                
                # Calculate weighted score based on workload profile
                score = self._calculate_workload_score(metrics, profile.performance_weights)
                
                return score
                
            except Exception as e:
                self._log(f"Error in objective function: {e}")
                return -1000.0
        
        return objective_function
    
    def _calculate_workload_score(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate performance score based on workload-specific weights"""
        score = 0.0
        
        # CPU efficiency (lower usage is better)
        if 'cpu_efficiency' in weights:
            cpu_score = max(0, 100 - metrics.get('cpu_percent_avg', 100))
            score += weights['cpu_efficiency'] * cpu_score
        
        # Memory efficiency
        if 'memory_efficiency' in weights:
            memory_score = max(0, 100 - metrics.get('memory_percent_avg', 100))
            score += weights['memory_efficiency'] * memory_score
        
        # I/O throughput
        if 'io_throughput' in weights:
            io_read = metrics.get('disk_read_rate_mb_s', 0)
            io_write = metrics.get('disk_write_rate_mb_s', 0)
            io_score = min(100, (io_read + io_write) * 5)
            score += weights['io_throughput'] * io_score
        
        # Network throughput
        if 'network_throughput' in weights:
            net_sent = metrics.get('network_sent_rate_mb_s', 0)
            net_recv = metrics.get('network_recv_rate_mb_s', 0)
            net_score = min(100, (net_sent + net_recv) * 5)
            score += weights['network_throughput'] * net_score
        
        return score
# this is for connecting to centralized management and handling commands
    def _execute_remote_command(self, command: Dict):
        """Execute command received from centralized management"""
        cmd_id = command.get('id')
        cmd_type = command.get('command_type')
        params = command.get('parameters', {})
        
        self._log(f"Executing remote command: {cmd_type} (ID: {cmd_id})")
        
        try:
            if cmd_type == 'update_parameters':
                # Apply kernel parameters
                self.kernel_interface.apply_parameter_set(params)
                if self.reporter:
                    self.reporter.report_command_result(
                        cmd_id, 
                        status='success',
                        result={"message": "Parameters updated successfully"}
                    )
            
            elif cmd_type == 'trigger_optimization':
                # Manually trigger optimization
                workload_type = params.get('workload_type', 'general')
                self._schedule_optimization(workload_type)
                if self.reporter:
                    self.reporter.report_command_result(
                        cmd_id,
                        status='success',
                        result={"message": f"Optimization scheduled for {workload_type}"}
                    )
            
            elif cmd_type == 'get_metrics':
                # Return current metrics
                metrics = self.performance_monitor.get_current_metrics()
                if self.reporter:
                    self.reporter.report_command_result(
                        cmd_id,
                        status='success',
                        result=metrics
                    )
            
            elif cmd_type == 'restart_monitoring':
                # Restart monitoring components
                self.process_detector.stop_monitoring()
                self.performance_monitor.stop_monitoring()
                time.sleep(2)
                self.process_detector.start_monitoring()
                self.performance_monitor.start_monitoring()
                if self.reporter:
                    self.reporter.report_command_result(
                        cmd_id,
                        status='success',
                        result={"message": "Monitoring restarted"}
                    )
            
            else:
                if self.reporter:
                    self.reporter.report_command_result(
                        cmd_id,
                        status='failed',
                        error=f"Unknown command type: {cmd_type}"
                    )
        
        except Exception as e:
            self._log(f"Command execution failed: {e}")
            if self.reporter:
                self.reporter.report_command_result(
                    cmd_id,
                    status='error',
                    error=str(e)
                )
    
    def _log(self, message: str):
        """Log message to file and console"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except:
            pass  # Fail silently if can't write to log
    
    def get_status(self) -> Dict:
        """Get current status of continuous optimizer"""
        workload_info = self.process_detector.get_current_workload_info()
        current_params = self.kernel_interface.get_current_configuration()
        
        # Determine the active profile based on current workload
        detected_workload = workload_info['dominant_workload']
        active_profile = detected_workload if detected_workload in self.OPTIMIZATION_PROFILES else 'general'
        
        return {
            'running': self.running,
            'optimization_in_progress': self.optimization_in_progress,
            'current_workload': detected_workload,
            'active_profile': active_profile,  # Profile based on detected workload
            'last_optimized_profile': self.current_profile.workload_type if self.current_profile else None,
            'last_optimization': self.last_optimization_time,
            'active_processes': workload_info['active_processes'],
            'current_parameters': current_params,
            'queue_size': self.optimization_queue.qsize()
        }

# Main execution script
if __name__ == "__main__":
    optimizer = ContinuousOptimizer()
    
    def signal_handler(sig, frame):
        print('\nShutdown signal received...')
        optimizer.stop_continuous_optimization()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        optimizer.start_continuous_optimization()
        
        # Keep running until interrupted
        while True:
            time.sleep(60)  # Check status every minute
            
            status = optimizer.get_status()
            print("\nStatus Update:")
            print(f"  Current workload: {status['current_workload']}")
            print(f"  Active profile: {status['active_profile']}")
            print(f"  Last optimized profile: {status['last_optimized_profile']}")
            print(f"  Active processes: {status['active_processes']}")
            print(f"  Queue size: {status['queue_size']}")
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        optimizer.stop_continuous_optimization()
