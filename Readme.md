# Linux Kernel Optimization Framework

A comprehensive framework that combines Bayesian Optimization and Genetic Algorithms to automatically optimize Linux kernel parameters for improved system performance.

## 🚨 Important: Linux Kernel 6.6+ EEVDF Scheduler Support

**If you're seeing parameter warnings like:**
```
INFO:KernelParameterInterface:Parameter kernel.sched_min_granularity_ns not available on this system, using default
INFO:KernelParameterInterface:Parameter kernel.sched_wakeup_granularity_ns not available on this system, using default
INFO:KernelParameterInterface:Parameter kernel.sched_migration_cost_ns not available on this system, using default
```

**This is normal for Linux kernel 6.6+** which uses the new **EEVDF (Earliest Eligible Virtual Deadline First)** scheduler instead of the older CFS scheduler.

### What Changed in Kernel 6.6+

1. **Scheduler Parameters Moved**: The old CFS parameters were moved from `/proc/sys/kernel/` to `/sys/kernel/debug/sched/`
2. **New Scheduler**: EEVDF scheduler replaces CFS for better latency and performance
3. **Different Tuning Approach**: Focus on process priorities and new bandwidth control parameters

### Updated Framework Features

✅ **EEVDF-Compatible Parameters**:
- `kernel.sched_cfs_bandwidth_slice_us` - Time slice control for EEVDF
- `kernel.sched_latency_ns` - Target preemption latency  
- `kernel.sched_rt_period_us` - RT scheduler period
- `kernel.sched_rt_runtime_us` - RT scheduler runtime

✅ **Process Priority Management**:
- Automatic process classification by workload type
- Dynamic nice value adjustments using `renice`
- Workload-specific priority optimization

✅ **Backward Compatibility**:
- Works on both older and newer kernels
- Graceful fallback for unavailable parameters
- Simulation mode for testing

### Testing EEVDF Support

Run the EEVDF compatibility test:
```bash
python3 TestEEVDFSupport.py
```

This will:
- Check for EEVDF scheduler parameters
- Test process priority management
- Verify continuous optimization compatibility
- Show kernel version compatibility

## System Architecture

Here's the system architecture diagram for your Linux kernel optimization framework! The design shows:

### Key Components:

1. **Workload Input Layer** - Handles different types of workloads (databases, web apps, HPC, etc.)
2. **Performance Monitor** - Tracks system metrics (CPU, memory, I/O, network, response time)
3. **Workload Analyzer** - Analyzes patterns and classifies workload characteristics
4. **Parameter Selector** - Identifies relevant kernel parameters for optimization
5. **Hybrid Optimization Engine** - The core component with:
   - **Bayesian Optimization**: For sample-efficient fine-tuning
   - **Genetic Algorithm**: For global exploration of parameter space
   - **Decision Logic**: Determines when to use BO vs GA

6. **Kernel Interface Layer** - Interfaces with kernel parameter systems (/proc/sys, sysctl, etc.)
7. **Feedback Loop** - Continuously monitors performance and adjusts optimization strategy

### Data Flow:

1. Workloads feed into monitoring and analysis components
2. Analysis results guide the optimization engine
3. Optimized parameters are applied through kernel interfaces
4. Performance feedback creates a continuous improvement loop

## Project Structure

```
📁 Linux Kernel Optimization Framework/
├── 📄 BayesianOptimzation.py          # Bayesian optimization implementation
├── 📄 GeneticAlgorithm.py             # Genetic algorithm implementation
├── 📄 HybridOptimizationEngine.py     # Hybrid optimization engine
├── 📄 PerformanceMonitor.py           # System performance monitoring
├── 📄 KernelParameterInterface.py     # Kernel parameter management
├── 📄 MainIntegration.py              # Complete framework integration
├── 📄 ProcessWorkloadDetector.py      # Process monitoring and workload detection
├── 📄 ContinuousOptimizer.py          # Continuous optimization engine
├── 📄 SystemTest.py                   # System testing and validation
├── 📄 QuickStart.py                   # Easy startup script
├── 📄 install_service.sh              # System service installation script
├── 📄 kernel-optimizer.service        # Systemd service file
├── 📄 requirements.txt                # Python dependencies
├── 📄 Readme.md                       # This file
└── 📄 Linux Kernel Optimization Framework Architecture.svg
```

## Features

### 🔧 Optimization Algorithms
- **Bayesian Optimization**: Efficient parameter exploration with Gaussian Process models
- **Genetic Algorithm**: Global optimization with population-based search
- **Hybrid Strategy**: Intelligently combines both algorithms for optimal results
- **Adaptive Selection**: Automatically chooses the best strategy based on problem characteristics

### 📊 Performance Monitoring
- Real-time system metric collection
- CPU, memory, I/O, and network performance tracking
- Anomaly detection and baseline comparison
- Historical data analysis and export

### ⚙️ Kernel Parameter Management
- Safe parameter modification with automatic backups
- Validation of parameter bounds and constraints
- Support for major kernel subsystems (memory, CPU, network, filesystem)
- Cross-platform compatibility (Linux primary, Windows simulation)

### 🔍 Workload Analysis
- Automatic workload characterization
- Performance pattern recognition
- Optimization strategy recommendation
- Multi-objective optimization support

### 🔄 Continuous Operation
- **Real-time Process Detection**: Monitors running processes and identifies workload types
- **Dynamic Adaptation**: Automatically adjusts optimization when workload changes
- **System Service**: Runs as a background daemon for 24/7 operation
- **Intelligent Scheduling**: Prevents over-optimization with stability controls
- **Workload Profiles**: Pre-configured optimization strategies for different application types

## Quick Start

### 1. Installation

First, ensure you have Python 3.7+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### 2. System Test

Run the system test to verify all components work correctly:

```bash
python SystemTest.py
```

### 3. Quick Demo

Run the quick start script for a guided demonstration:

```bash
python QuickStart.py
```

### 4. Example Usage

```python
from MainIntegration import KernelOptimizationFramework
from HybridOptimizationEngine import OptimizationStrategy

# Initialize the framework
framework = KernelOptimizationFramework()

# Start an optimization session
session_id = framework.start_optimization_session(
    workload_type="database",
    strategy=OptimizationStrategy.ADAPTIVE,
    evaluation_budget=50,
    time_budget=1800.0  # 30 minutes
)

# Get optimization summary
summary = framework.get_optimization_summary()
print(f"Best score: {summary['best_session']['best_score']}")
```

### 5. Continuous Operation (Recommended)

For 24/7 automatic optimization that adapts to changing workloads:

```bash
# Install and start the continuous optimizer service
sudo python install_service.py
sudo systemctl start kernel-optimizer
sudo systemctl enable kernel-optimizer  # Start on boot

# Monitor the service
sudo systemctl status kernel-optimizer
sudo journalctl -u kernel-optimizer -f  # View live logs
```

The continuous optimizer will:
- Monitor running processes automatically
- Detect workload changes (database → web server → HPC, etc.)
- Apply optimal kernel parameters for each workload type
- Run safely in the background with automatic backups

## Component Documentation

### Bayesian Optimization (`BayesianOptimzation.py`)

Implements Bayesian optimization using:
- Gaussian Process surrogate models
- Multiple acquisition functions (EI, UCB, PI)
- Automatic hyperparameter tuning
- Convergence detection

**Key Parameters:**
- `parameter_bounds`: Dict of parameter ranges
- `acquisition_function`: 'ei', 'ucb', or 'pi'
- `initial_samples`: Number of random initial evaluations
- `max_iterations`: Maximum optimization iterations

### Genetic Algorithm (`GeneticAlgorithm.py`)

Features both standard and advanced genetic algorithms:
- Population-based global optimization
- Multiple crossover and mutation strategies
- Elitism and tournament selection
- Adaptive parameter control
- Diversity injection mechanisms

**Key Parameters:**
- `population_size`: Size of the population
- `max_generations`: Maximum number of generations
- `mutation_rate`: Probability of mutation
- `crossover_rate`: Probability of crossover

### Hybrid Optimization Engine (`HybridOptimizationEngine.py`)

Intelligent combination of optimization strategies:
- **Bayesian Only**: For small parameter spaces and limited budgets
- **Genetic Only**: For large, complex parameter spaces
- **Sequential Hybrid**: Bayesian exploration followed by genetic exploitation
- **Adaptive**: Dynamic strategy selection based on problem characteristics

### Performance Monitor (`PerformanceMonitor.py`)

Real-time system monitoring:
- Multi-threaded performance data collection
- Configurable sampling intervals
- Statistical analysis and anomaly detection
- Data export capabilities

### Kernel Parameter Interface (`KernelParameterInterface.py`)

Safe kernel parameter management:
- Automatic parameter validation
- Backup and restore functionality
- Cross-platform compatibility
- Subsystem-based parameter organization

## Optimized Kernel Parameters

The framework optimizes key kernel parameters across multiple subsystems:

### Memory Management
- `vm.swappiness`: Controls swap usage tendency (0-100)
- `vm.dirty_ratio`: Dirty page write-back threshold (1-90%)
- `vm.dirty_background_ratio`: Background writeback threshold
- `vm.vfs_cache_pressure`: VFS cache reclaim pressure

### CPU Scheduling
- `kernel.sched_min_granularity_ns`: Minimum CPU time slice
- `kernel.sched_wakeup_granularity_ns`: Wakeup granularity
- `kernel.sched_migration_cost_ns`: Process migration cost

### Network
- `net.core.rmem_max`: Maximum receive buffer size
- `net.core.wmem_max`: Maximum send buffer size
- `net.core.netdev_max_backlog`: Network device queue size
- `net.ipv4.tcp_rmem`: TCP receive buffer sizes
- `net.ipv4.tcp_wmem`: TCP send buffer sizes

### Filesystem
- `fs.file-max`: Maximum number of file handles
- `fs.nr_open`: Maximum file descriptors per process

## Safety Features

### 🛡️ Backup and Restore
- Automatic parameter backup before optimization
- Timestamped backup files
- One-click restore functionality
- Parameter validation before application

### 🔒 Safe Parameter Bounds
- Hard limits on all parameter ranges
- Validation against system constraints
- Graceful handling of invalid values
- Non-destructive testing mode

### 📈 Monitoring and Alerts
- Real-time performance monitoring
- Anomaly detection
- Automatic rollback on system instability
- Comprehensive logging

## Advanced Usage

### Custom Objective Functions

Define custom performance metrics:

```python
def custom_objective(params):
    # Apply parameters
    interface.apply_parameter_set(params)
    
    # Run workload and measure performance
    score = run_benchmark()
    
    return score

# Use with optimizer
optimizer = HybridOptimizationEngine(parameter_bounds)
result = optimizer.optimize(custom_objective)
```

### Multi-Workload Optimization

Optimize for multiple workload types:

```python
workloads = ['database', 'web_server', 'hpc']
results = {}

for workload in workloads:
    session_id = framework.start_optimization_session(
        workload_type=workload,
        strategy=OptimizationStrategy.ADAPTIVE
    )
    results[workload] = framework.get_session_results(session_id)
```

### Custom Parameter Bounds

Define custom parameter ranges:

```python
custom_bounds = {
    'vm.swappiness': (10, 60),        # Conservative swap usage
    'vm.dirty_ratio': (5, 30),        # Limited dirty pages
    'custom.parameter': (0, 1000)     # Custom parameter
}

framework = KernelOptimizationFramework(
    parameter_bounds=custom_bounds
)
```

## Continuous Operation Mode

For production environments, the framework can run continuously and automatically adapt to changing workloads:

### Process Monitoring
The system continuously monitors running processes and classifies them into workload types:
- **Database**: MySQL, PostgreSQL, MongoDB, Redis, etc.
- **Web Server**: Nginx, Apache, Node.js, etc.
- **HPC Compute**: Scientific applications, ML frameworks
- **I/O Intensive**: File operations, backups, data processing
- **General**: Mixed or unidentified workloads

### Automatic Adaptation
When the system detects a workload change (e.g., database → web server), it:
1. Waits for a stability period to avoid thrashing
2. Selects the appropriate optimization profile
3. Runs targeted optimization for the new workload
4. Applies optimized parameters safely with automatic backup

### Installation as System Service

**Step 1: Install the Service**
```bash
# Install as system service (requires root)
sudo python3 install_service.py

# Start the service
sudo systemctl start kernel-optimizer
sudo systemctl enable kernel-optimizer  # Start on boot
```

**Step 2: Monitor Operation**
```bash
# Check service status
sudo systemctl status kernel-optimizer

# View live logs
sudo journalctl -u kernel-optimizer -f

# View optimization logs
sudo tail -f /var/log/kernel-optimizer/continuous_optimizer.log
```

**Step 3: Manual Control**
```bash
# Stop service
sudo systemctl stop kernel-optimizer

# Restart service
sudo systemctl restart kernel-optimizer

# Uninstall service
sudo python3 install_service.py uninstall
```

### Manual Continuous Mode

For testing or custom setups, run manually:

```bash
# Run continuous optimizer directly
sudo python3 ContinuousOptimizer.py

# The system will:
# - Monitor processes every 2 seconds
# - Detect workload changes automatically
# - Optimize parameters when workload changes
# - Log all activities
```

### Workload Profiles

The system includes pre-configured profiles for different workload types:

| Workload Type | Focus Parameters | Strategy | Optimization Time |
|---------------|------------------|----------|-------------------|
| Database | Memory, I/O, Network | Bayesian | 3 minutes |
| Web Server | Network, CPU | Adaptive | 2.5 minutes |
| HPC Compute | CPU, Memory | Genetic | 4 minutes |
| I/O Intensive | Disk I/O, Cache | Bayesian | 2 minutes |
| General | Balanced | Adaptive | 2 minutes |

### Safety Features for Continuous Operation

- **Automatic Backups**: Parameters backed up before each optimization
- **Stability Periods**: Minimum 3 minutes between optimizations
- **Resource Limits**: CPU and memory usage capped for the optimizer
- **Graceful Shutdown**: Proper cleanup on system shutdown/restart
- **Error Recovery**: Continues operation even if individual optimizations fail

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure root privileges for kernel parameter modification
2. **Import Errors**: Install all dependencies with `pip install -r requirements.txt`
3. **Platform Issues**: Some features are Linux-specific, Windows users see simulation mode
4. **Performance Issues**: Reduce evaluation budget or time limits for faster execution

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Run system diagnostics:

```bash
python SystemTest.py
```

### Validation

Verify parameter changes:

```bash
# Check current parameters
sysctl vm.swappiness
sysctl vm.dirty_ratio

# View optimization logs
tail -f /var/log/kernel_optimizer.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass with `python SystemTest.py`
5. Submit a pull request

## Requirements

### System Requirements
- Linux kernel 3.10+ (recommended 4.0+)
- Python 3.7+
- Root privileges for parameter modification
- Minimum 4GB RAM
- 1GB free disk space

### Python Dependencies
- numpy >= 1.20.0
- scipy >= 1.7.0
- psutil >= 5.8.0

### Optional Dependencies
- matplotlib (for visualization)
- scikit-learn (for additional ML algorithms)
- pandas (for data analysis)

## License

This project is released under the MIT License. See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{kernel_optimization_framework,
  title={Linux Kernel Optimization Framework: A Hybrid Approach Using Bayesian Optimization and Genetic Algorithms},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/kernel-optimization-framework}}
}
```

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Run `python SystemTest.py` for diagnostics

---

**⚠️ Important**: This framework modifies kernel parameters that can affect system stability. Always test on non-production systems first and maintain proper backups.