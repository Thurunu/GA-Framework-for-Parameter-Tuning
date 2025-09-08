# Complete Continuous Optimization Workflow

## 1. System Initialization
```
ContinuousOptimizer.__init__()
├── Initialize components:
│   ├── ProcessWorkloadDetector()
│   ├── PerformanceMonitor()  
│   ├── KernelParameterInterface()
│   └── ProcessPriorityManager()
├── Setup optimization tracking variables
├── Create optimization_queue (Queue object)
└── Register callback: process_detector.add_workload_change_callback(_on_workload_change)
```

## 2. Starting Continuous Optimization
```
ContinuousOptimizer.start_continuous_optimization()
├── Set self.running = True
├── Start monitoring services (parallel threads):
│   ├── process_detector.start_monitoring() [Thread 1]
│   └── performance_monitor.start_monitoring() [Thread 2]
├── Start optimization worker:
│   └── optimization_thread = Thread(target=_optimization_worker) [Thread 3]
└── Initial optimization: _schedule_optimization('general')
```

## 3. Workload Detection & Change Handling
```
ProcessWorkloadDetector (Thread 1)
├── Continuously monitors running processes
├── Classifies workload type (database, web_server, hpc_compute, io_intensive, general)
├── On workload change detected:
│   └── Triggers callback: _on_workload_change(old_workload, new_workload, stats)
│       ├── Logs workload change
│       └── Schedules delayed optimization:
│           └── Timer(adaptation_delay, lambda: _schedule_optimization(new_workload))
```

## 4. Optimization Scheduling
```
_schedule_optimization(workload_type)
├── Check if optimization already in progress → Skip if true
├── Check stability_period (time since last optimization) → Skip if too soon
├── Add workload_type to optimization_queue
└── Log: "Scheduled optimization for workload: {workload_type}"
```

## 5. Optimization Worker Loop (Thread 3)
```
_optimization_worker() [Continuous loop while self.running]
├── Wait for item from optimization_queue.get(timeout=1)
├── Check if workload_type exists in OPTIMIZATION_PROFILES
├── If valid workload_type:
│   └── Call _run_optimization(workload_type)
├── Mark queue task as done: optimization_queue.task_done()
└── Handle exceptions and continue loop
```

## 6. Running Optimization Process
```
_run_optimization(workload_type)
├── Set optimization_in_progress = True
├── Get OptimizationProfile for workload_type containing:
│   ├── parameter_bounds (kernel parameters to optimize)
│   ├── strategy (BAYESIAN_ONLY, GENETIC_ONLY, or ADAPTIVE)
│   ├── evaluation_budget (number of iterations)
│   ├── time_budget (max optimization time)
│   └── performance_weights (cpu, memory, io, network priorities)
├── Create parameter backup: kernel_interface.backup_current_parameters()
├── Initialize HybridOptimizationEngine with profile settings
├── Create objective_function using _create_objective_function(profile)
└── Run optimization: engine.optimize(objective_function)
```

## 7. HybridOptimizationEngine.optimize()
```
HybridOptimizationEngine.optimize(objective_function)
├── Based on strategy, choose optimization method:
│   ├── BAYESIAN_ONLY → Use BayesianOptimization
│   ├── GENETIC_ONLY → Use GeneticAlgorithm  
│   └── ADAPTIVE → Use both algorithms in sequence/parallel
├── For each parameter combination:
│   └── Call objective_function(parameters)
├── Return optimization result with:
│   ├── best_parameters
│   ├── best_score
│   └── total_evaluations
```

## 8. Objective Function Evaluation
```
_create_objective_function(profile) returns objective_function(params)
├── Apply parameters: kernel_interface.apply_parameter_set(params)
├── Check application success → Return penalty (-1000) if failed
├── Allow system stabilization: sleep(2)
├── Monitor performance: sleep(monitor_duration=10)
├── Get metrics: performance_monitor.get_average_metrics(monitor_duration)
├── Calculate weighted score: _calculate_workload_score(metrics, weights)
└── Return performance score
```

## 9. Parameter Application & Process Optimization
```
After optimization completes:
├── Apply best parameters: kernel_interface.apply_parameter_set(best_parameters)
├── Apply process priority optimizations:
│   └── priority_manager.optimize_process_priorities(workload_focus=workload_type)
├── Log results (optimization time, best score, evaluations)
├── Export results to JSON file
├── Set last_optimization_time = current_time
└── Set optimization_in_progress = False
```

## 10. Performance Monitoring (Thread 2)
```
PerformanceMonitor.start_monitoring() [Continuous background monitoring]
├── Collects system metrics:
│   ├── CPU usage, memory usage
│   ├── I/O throughput, network throughput
│   └── Process-specific metrics
├── Maintains rolling averages
└── Provides metrics via get_average_metrics() when requested
```

## Key Threading Architecture:
- **Thread 1**: ProcessWorkloadDetector - monitors processes and detects workload changes
- **Thread 2**: PerformanceMonitor - continuously collects system performance metrics  
- **Thread 3**: Optimization Worker - processes optimization requests from queue
- **Timer Threads**: Delayed optimization scheduling after workload changes
- **Main Thread**: System coordination and user interface

## Queue-based Communication:
- **optimization_queue**: Thread-safe queue for workload optimization requests
- **Callbacks**: ProcessWorkloadDetector → ContinuousOptimizer communication
- **Synchronization**: optimization_in_progress flag prevents concurrent optimizations