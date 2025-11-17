#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Main Integration Module
This module integrates all components for complete kernel optimization
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os, sys

# Import framework components
from HybridOptimizationEngine import HybridOptimizationEngine, HybridOptimizationResult
from WorkloadCharacterizer import OptimizationStrategy
from PerformanceMonitor import PerformanceMonitor
from KernelParameterInterface import KernelParameterInterface

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'core'))
sys.path.insert(0, os.path.join(project_root, 'data'))
sys.path.insert(0, os.path.join(project_root, 'workload'))
sys.path.insert(0, os.path.join(project_root, 'monitoring'))
sys.path.insert(0, os.path.join(project_root, 'system'))



class KernelOptimizationFramework:
    """Main framework class integrating all components"""
    
    def __init__(self, 
                 parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 monitoring_interval: float = 1.0,
                 backup_dir: str = "/tmp/kernel_optimizer"):
        """
        Initialize the complete optimization framework
        
        Args:
            parameter_bounds: Custom parameter bounds (uses defaults if None)
            monitoring_interval: Performance monitoring sampling interval
            backup_dir: Directory for backups and logs
        """
        
        # Initialize kernel parameter interface
        self.kernel_interface = KernelParameterInterface(backup_dir=backup_dir)
        
        # Use provided bounds or get default optimization parameters
        if parameter_bounds is None:
            # Extract bounds from kernel interface
            self.parameter_bounds = {}
            for name, param in self.kernel_interface.get_all_parameters().items():
                if param.min_value is not None and param.max_value is not None:
                    self.parameter_bounds[name] = (param.min_value, param.max_value)
        else:
            self.parameter_bounds = parameter_bounds
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(sampling_interval=monitoring_interval)
        
        # Initialize optimization engine
        self.optimization_engine = None
        
        # Session tracking
        self.current_session = None
        self.session_history = []
        
        print(f"Kernel Optimization Framework initialized")
        print(f"Optimizable parameters: {len(self.parameter_bounds)}")
        print(f"Parameter bounds: {list(self.parameter_bounds.keys())}")
    
    def create_performance_objective(self, 
                                   evaluation_duration: int = 30,
                                   metrics_weights: Optional[Dict[str, float]] = None) -> callable:
        """
        Create objective function based on system performance metrics
        
        Args:
            evaluation_duration: How long to monitor performance (seconds)
            metrics_weights: Weights for different performance metrics
        
        Returns:
            Objective function that evaluates kernel parameters
        """
        
        if metrics_weights is None:
            # Default weights favoring throughput and low latency
            metrics_weights = {
                'cpu_efficiency': 0.3,     # Lower CPU usage is better
                'memory_efficiency': 0.2,   # Lower memory usage is better
                'io_throughput': 0.25,      # Higher I/O throughput is better
                'network_throughput': 0.15, # Higher network throughput is better
                'response_time': 0.1        # Lower response time is better
            }
        
        def objective_function(parameters: Dict[str, float]) -> float:
            """
            Objective function that applies parameters and measures performance
            
            Args:
                parameters: Kernel parameters to test
                
            Returns:
                Performance score (higher is better)
            """
            try:
                # Apply kernel parameters
                print(f"Testing parameters: {parameters}")
                results = self.kernel_interface.apply_parameter_set(parameters)
                
                # Check if parameters were applied successfully
                failed_params = [name for name, success in results.items() if not success]
                if failed_params:
                    print(f"Warning: Failed to apply parameters: {failed_params}")
                    # Return penalty score for failed parameter application
                    return -1000.0
                
                # Allow system to stabilize
                time.sleep(2)
                
                # Start performance monitoring
                self.performance_monitor.start_monitoring()
                
                # Monitor for specified duration
                print(f"Monitoring performance for {evaluation_duration} seconds...")
                time.sleep(evaluation_duration)
                
                # Get performance metrics
                metrics = self.performance_monitor.get_average_metrics(evaluation_duration)
                
                if not metrics:
                    print("Warning: No performance metrics collected")
                    return -500.0
                
                # Calculate performance score
                score = self._calculate_performance_score(metrics, metrics_weights)
                
                print(f"Performance score: {score:.6f}")
                print(f"Key metrics - CPU: {metrics.get('cpu_percent_avg', 0):.1f}%, "
                      f"Memory: {metrics.get('memory_percent_avg', 0):.1f}%, "
                      f"Load: {metrics.get('load_average_1m_avg', 0):.2f}")
                
                return score
                
            except Exception as e:
                print(f"Error in objective function: {e}")
                return -1000.0
            
        return objective_function
    
    def _calculate_performance_score(self, 
                                   metrics: Dict[str, float], 
                                   weights: Dict[str, float]) -> float:
        """Calculate weighted performance score from metrics"""
        
        score = 0.0
        
        # CPU efficiency (lower usage is better, normalize to 0-100 scale)
        cpu_score = max(0, 100 - metrics.get('cpu_percent_avg', 100))
        score += weights.get('cpu_efficiency', 0) * cpu_score
        
        # Memory efficiency (lower usage is better)
        memory_score = max(0, 100 - metrics.get('memory_percent_avg', 100))
        score += weights.get('memory_efficiency', 0) * memory_score
        
        # I/O throughput (higher is better, normalize based on typical values)
        io_read = metrics.get('disk_read_rate_mb_s', 0)
        io_write = metrics.get('disk_write_rate_mb_s', 0)
        io_score = min(100, (io_read + io_write) * 10)  # Scale to 0-100
        score += weights.get('io_throughput', 0) * io_score
        
        # Network throughput (higher is better)
        net_sent = metrics.get('network_sent_rate_mb_s', 0)
        net_recv = metrics.get('network_recv_rate_mb_s', 0)
        net_score = min(100, (net_sent + net_recv) * 10)  # Scale to 0-100
        score += weights.get('network_throughput', 0) * net_score
        
        # Response time (lower load average is better)
        load_avg = metrics.get('load_average_1m_avg', 0)
        response_score = max(0, 100 - load_avg * 20)  # Scale load average
        score += weights.get('response_time', 0) * response_score
        
        return score
    
    def start_optimization_session(self,
                                 workload_type: str = "general",
                                 strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
                                 evaluation_budget: int = 50,
                                 time_budget: float = 1800.0,  # 30 minutes
                                 evaluation_duration: int = 30) -> str:
        """
        Start a new optimization session
        
        Args:
            workload_type: Type of workload being optimized
            strategy: Optimization strategy to use
            evaluation_budget: Maximum number of parameter evaluations
            time_budget: Maximum optimization time in seconds
            evaluation_duration: Duration to monitor each parameter set
            
        Returns:
            Session ID
        """
        
        # Create session
        session_id = f"opt_session_{int(time.time())}"
        self.current_session = OptimizationSession(
            session_id=session_id,
            workload_type=workload_type,
            optimization_strategy=strategy.value,
            start_time=time.time()
        )
        
        print(f"Starting optimization session: {session_id}")
        print(f"Workload type: {workload_type}")
        print(f"Strategy: {strategy.value}")
        print(f"Evaluation budget: {evaluation_budget}")
        print(f"Time budget: {time_budget:.0f} seconds")
        
        # Create backup of current parameters
        backup_file = self.kernel_interface.backup_current_parameters()
        print(f"Parameter backup created: {backup_file}")
        
        # Initialize optimization engine
        self.optimization_engine = HybridOptimizationEngine(
            parameter_bounds=self.parameter_bounds,
            strategy=strategy,
            evaluation_budget=evaluation_budget,
            time_budget=time_budget
        )
        
        # Create objective function
        objective_function = self.create_performance_objective(
            evaluation_duration=evaluation_duration
        )
        
        # Run optimization
        try:
            result = self.optimization_engine.optimize(objective_function)
            
            # Update session with results
            self.current_session.end_time = time.time()
            self.current_session.best_parameters = result.best_parameters
            self.current_session.best_score = result.best_score
            self.current_session.total_evaluations = result.total_evaluations
            self.current_session.session_completed = True
            
            # Store session in history
            self.session_history.append(self.current_session)
            
            print(f"Optimization session completed successfully!")
            print(f"Best score: {result.best_score:.6f}")
            print(f"Best parameters: {result.best_parameters}")
            
            # Export results
            self.export_session_results(session_id, result)
            
        except Exception as e:
            print(f"Optimization session failed: {e}")
            self.current_session.session_completed = False
            self.session_history.append(self.current_session)
        
        finally:
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
        
        return session_id
    
    def export_session_results(self, session_id: str, result: HybridOptimizationResult):
        """Export complete session results"""
        
        export_data = {
            "session_info": {
                "session_id": session_id,
                "workload_type": self.current_session.workload_type,
                "strategy": self.current_session.optimization_strategy,
                "start_time": self.current_session.start_time,
                "end_time": self.current_session.end_time,
                "duration": self.current_session.end_time - self.current_session.start_time if self.current_session.end_time else 0
            },
            "optimization_results": {
                "best_score": result.best_score,
                "best_parameters": result.best_parameters,
                "total_evaluations": result.total_evaluations,
                "strategy_used": result.strategy_used.value,
                "convergence_reached": result.convergence_reached
            },
            "parameter_bounds": self.parameter_bounds,
            "current_kernel_config": self.kernel_interface.get_current_configuration()
        }
        
        filename = f"session_{session_id}_results.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Session results exported to {filename}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization sessions"""
        
        if not self.session_history:
            return {"status": "no_sessions"}
        
        completed_sessions = [s for s in self.session_history if s.session_completed]
        
        if not completed_sessions:
            return {"status": "no_completed_sessions"}
        
        best_session = max(completed_sessions, key=lambda x: x.best_score or -np.inf)
        
        return {
            "total_sessions": len(self.session_history),
            "completed_sessions": len(completed_sessions),
            "best_session": {
                "session_id": best_session.session_id,
                "workload_type": best_session.workload_type,
                "best_score": best_session.best_score,
                "best_parameters": best_session.best_parameters,
                "strategy": best_session.optimization_strategy
            },
            "average_evaluations": np.mean([s.total_evaluations for s in completed_sessions]),
            "average_improvement": np.mean([s.best_score for s in completed_sessions if s.best_score is not None])
        }

# Example usage
if __name__ == "__main__":
    print("Linux Kernel Optimization Framework - Integration Test")
    print("=" * 60)
    
    # For testing purposes, we'll use a mock objective function
    # In real usage, you would use the performance-based objective
    
    # Define a subset of parameters for testing
    test_parameter_bounds = {
        'vm.swappiness': (0, 100),
        'vm.dirty_ratio': (5, 40),
        'kernel.sched_min_granularity_ns': (1000000, 20000000)
    }
    
    # Initialize framework
    framework = KernelOptimizationFramework(
        parameter_bounds=test_parameter_bounds,
        monitoring_interval=0.5
    )
    
    # Mock objective function for testing (replace with real performance measurement)
    def mock_objective_function(params: Dict[str, float]) -> float:
        """Mock function for testing - replace with real performance evaluation"""
        swappiness = params.get('vm.swappiness', 60)
        dirty_ratio = params.get('vm.dirty_ratio', 20)
        sched_gran = params.get('kernel.sched_min_granularity_ns', 10000000)
        
        # Simulate performance score
        score = (
            -0.01 * (swappiness - 30)**2 +
            -0.1 * (dirty_ratio - 15)**2 +
            -0.000000001 * (sched_gran - 6000000)**2
        ) + np.random.normal(0, 0.5)
        
        print(f"Mock evaluation - Score: {score:.6f}")
        return score
    
    # Replace the framework's objective function for testing
    framework.create_performance_objective = lambda **kwargs: mock_objective_function
    
    print("Starting test optimization session...")
    
    # Test optimization with different strategies
    strategies_to_test = [
        OptimizationStrategy.BAYESIAN_ONLY,
        OptimizationStrategy.ADAPTIVE
    ]
    
    for strategy in strategies_to_test:
        print(f"\nTesting strategy: {strategy.value}")
        print("-" * 40)
        
        try:
            session_id = framework.start_optimization_session(
                workload_type="test",
                strategy=strategy,
                evaluation_budget=20,
                time_budget=120.0,  # 2 minutes for testing
                evaluation_duration=1  # Short evaluation for testing
            )
            
            print(f"Session {session_id} completed")
            
        except Exception as e:
            print(f"Error in optimization session: {e}")
    
    # Get summary
    summary = framework.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Total sessions: {summary.get('total_sessions', 0)}")
    print(f"Completed sessions: {summary.get('completed_sessions', 0)}")
    
    if 'best_session' in summary:
        best = summary['best_session']
        print(f"Best session: {best['session_id']}")
        print(f"Best score: {best['best_score']:.6f}")
        print(f"Best parameters: {best['best_parameters']}")
    
    print("\nFramework integration test completed!")
