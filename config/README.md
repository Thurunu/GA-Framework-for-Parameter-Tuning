# Optimization Profiles Configuration

This directory contains configuration files for the Continuous Kernel Optimization Framework.

## optimization_profiles.yml

Defines optimization profiles for different workload types. Each profile specifies:

### Profile Structure

```yaml
profiles:
  <workload_name>:
    workload_type: <string>              # Name of the workload type
    strategy: <STRATEGY_NAME>            # Optimization strategy
    evaluation_budget: <int>             # Number of evaluations
    time_budget: <float>                 # Time budget in seconds
    
    parameter_bounds:                    # Kernel parameters to optimize
      <parameter_name>: [min, max]       # Parameter bounds as [min, max]
    
    performance_weights:                 # Performance metric weights (sum to 1.0)
      cpu_efficiency: <float>            # Weight for CPU efficiency
      memory_efficiency: <float>         # Weight for memory efficiency
      io_throughput: <float>             # Weight for I/O throughput
      network_throughput: <float>        # Weight for network throughput
```

### Available Strategies

- `BAYESIAN_ONLY`: Use Bayesian optimization only
- `GENETIC_ONLY`: Use genetic algorithm only
- `ADAPTIVE`: Adaptive strategy that switches between methods

### Available Workload Types

1. **database**: Database servers (MySQL, PostgreSQL, MongoDB, etc.)
   - Focus: I/O throughput and memory efficiency
   - Strategy: Bayesian optimization for precise tuning

2. **web_server**: Web servers (Nginx, Apache, Node.js, etc.)
   - Focus: Balanced performance with network throughput
   - Strategy: Adaptive for varying workloads

3. **hpc_compute**: High-performance computing workloads
   - Focus: CPU efficiency and memory
   - Strategy: Genetic algorithm for exploration

4. **io_intensive**: I/O-heavy workloads (backup, file operations)
   - Focus: I/O throughput
   - Strategy: Bayesian optimization

5. **general**: General purpose workloads
   - Focus: Balanced performance
   - Strategy: Adaptive

### Common Kernel Parameters

- `vm.swappiness`: Swappiness setting (0-100)
- `vm.dirty_ratio`: Dirty page ratio (0-100)
- `vm.dirty_background_ratio`: Background dirty page ratio (0-100)
- `vm.vfs_cache_pressure`: VFS cache pressure (0-500)
- `kernel.sched_cfs_bandwidth_slice_us`: EEVDF scheduler bandwidth slice (microseconds)
- `kernel.sched_latency_ns`: EEVDF scheduler latency (nanoseconds)
- `kernel.sched_rt_runtime_us`: Real-time scheduler runtime (microseconds)
- `net.core.rmem_max`: Maximum receive socket buffer size
- `net.core.wmem_max`: Maximum send socket buffer size
- `net.core.netdev_max_backlog`: Network device backlog size

### Customization

To add a new workload profile:

1. Add a new entry under `profiles:`
2. Define all required fields
3. Choose appropriate parameter bounds for your use case
4. Set performance weights based on workload priorities
5. Select an optimization strategy

Example:
```yaml
  my_custom_workload:
    workload_type: my_custom_workload
    strategy: ADAPTIVE
    evaluation_budget: 12
    time_budget: 150.0
    
    parameter_bounds:
      vm.swappiness: [20, 40]
      vm.dirty_ratio: [10, 25]
    
    performance_weights:
      cpu_efficiency: 0.3
      memory_efficiency: 0.3
      io_throughput: 0.2
      network_throughput: 0.2
```

### Notes

- All parameter bounds must be valid for your Linux kernel version
- Performance weights should sum to approximately 1.0
- Evaluation budget affects optimization time (higher = longer but better results)
- Time budget is a hard limit on optimization duration

---

## workload_patterns.yml

Defines patterns for classifying running processes into workload types.

### Pattern Structure

```yaml
patterns:
  <workload_name>:
    description: <string>           # Description of this workload type
    process_patterns:               # List of regex patterns to match
      - '<regex_pattern>'
      - '<regex_pattern>'

fallback_thresholds:               # Used when no pattern matches
  cpu_intensive:
    cpu_percent: <int>              # CPU usage threshold
  memory_intensive:
    memory_percent: <int>           # Memory usage threshold
  io_intensive:
    io_bytes_per_second: <int>     # I/O bytes per second threshold
  network_intensive:
    connection_count: <int>         # Network connection count threshold
```

### Adding New Patterns

To add detection for a new workload type:

1. Add a new entry under `patterns:`
2. Provide a description
3. List regex patterns that match process names/command lines
4. Make sure a corresponding profile exists in `optimization_profiles.yml`

Example:
```yaml
my_custom_workload:
  description: "My custom application workload"
  process_patterns:
    - 'myapp.*'
    - 'custom-process.*'
    - '.*mycustompattern.*'
```

### Pattern Matching

- Patterns are regex expressions (Python `re` module)
- Matched against: `{process_name} {command_line}` (lowercase)
- First matching pattern determines workload type
- If no pattern matches, fallback thresholds are used

### Fallback Classification

When a process doesn't match any pattern, it's classified based on resource usage:
- High CPU → `cpu_intensive`
- High memory → `memory_intensive`
- High I/O → `io_intensive`
- Many network connections → `network_intensive`
- Otherwise → `general`

---

## kernel_parameters.yml

Defines all kernel parameters that can be optimized by the framework.

### Parameter Structure

```yaml
parameters:
  <parameter_name>:
    default_value: <value>          # Default system value
    min_value: <value>              # Minimum allowed value
    max_value: <value>              # Maximum allowed value
    description: <string>           # Parameter description
    subsystem: <string>             # Subsystem (memory, cpu, network, filesystem)
    writable: <boolean>             # Whether parameter can be modified
    requires_reboot: <boolean>      # Whether change requires reboot
```

### Parameter Subsystems

**Memory Management** (`subsystem: memory`):
- `vm.swappiness` - Swap usage tendency
- `vm.dirty_ratio` - Dirty page writeback threshold
- `vm.dirty_background_ratio` - Background writeback threshold
- `vm.vfs_cache_pressure` - VFS cache reclaim pressure

**CPU Scheduling** (`subsystem: cpu`):
- `kernel.sched_cfs_bandwidth_slice_us` - EEVDF bandwidth slice
- `kernel.sched_latency_ns` - Target preemption latency
- `kernel.sched_rt_period_us` - RT task bandwidth period
- `kernel.sched_rt_runtime_us` - RT task runtime quota

**Network** (`subsystem: network`):
- `net.core.rmem_max` - Max receive buffer size
- `net.core.wmem_max` - Max send buffer size
- `net.core.netdev_max_backlog` - Network device queue size
- `net.ipv4.tcp_rmem` - TCP receive buffer sizes
- `net.ipv4.tcp_wmem` - TCP send buffer sizes
- `net.ipv4.tcp_congestion_control` - TCP congestion algorithm

**File System** (`subsystem: filesystem`):
- `fs.file-max` - Maximum file handles
- `fs.nr_open` - Max file descriptors per process

### Adding New Parameters

To add a new kernel parameter to optimize:

1. Add entry to `kernel_parameters.yml`:
```yaml
kernel.my_new_param:
  default_value: 100
  min_value: 10
  max_value: 1000
  description: "My new kernel parameter"
  subsystem: "cpu"
  writable: true
  requires_reboot: false
```

2. Add to relevant optimization profile bounds in `optimization_profiles.yml`
3. Restart the optimizer - the new parameter is now available!

### Value Types

- **Integer**: `default_value: 60`
- **String**: `default_value: "cubic"`
- **Null** (no bounds): `min_value: null`

### Important Notes

- Only add parameters that are safe to modify dynamically
- Test parameter changes on non-production systems first
- Some parameters may require specific kernel versions
- Use `requires_reboot: true` for parameters that need restart
- Validate min/max values match your kernel's constraints

---

## process_priorities.yml

Defines process priority assignments for EEVDF scheduler optimization.

### Priority Classes

```yaml
CRITICAL: -20    # Highest priority (real-time-like)
HIGH: -10       # High priority (interactive)
NORMAL: 0       # Default priority
LOW: 10         # Low priority (batch)
BACKGROUND: 19  # Lowest priority (background tasks)
```

### Priority Mapping Structure

```yaml
priority_mappings:
  <workload_name>:
    description: <string>              # Workload description
    priority_class: <PRIORITY_CLASS>   # CRITICAL, HIGH, NORMAL, LOW, or BACKGROUND
    patterns:                          # List of process name patterns
      - "<pattern1>"
      - "<pattern2>"
```

### Configuration Sections

**Priority Mappings**: Maps workload types to priority classes
- `database` - Database servers (HIGH priority)
- `web_server` - Web servers (HIGH priority)
- `compute_intensive` - Scientific computing (NORMAL priority)
- `background_tasks` - Maintenance tasks (BACKGROUND priority)
- `interactive` - User applications (HIGH priority)
- `system_critical` - Critical system processes (CRITICAL priority)
- `media_processing` - Media encoding (LOW priority)
- `compilation` - Build processes (LOW priority)

**Workload Focus Boost**: Temporary priority boost for dominant workload
```yaml
workload_focus_boost:
  enabled: true
  boost_amount: 5          # Reduce nice value by this amount
  max_priority: -20        # Never exceed this priority
```

**Filter Rules**: Which processes to exclude
```yaml
filter_rules:
  exclude_pids: [0, 1, 2]           # Never touch these PIDs
  exclude_prefixes: ["[", "kthreadd"]  # Exclude by name prefix
  min_cpu_threshold: 0.5             # Minimum CPU % to consider
  min_memory_threshold: 0.1          # Minimum memory % to consider
```

**Safety Settings**: Protection mechanisms
```yaml
safety:
  confirm_critical: true              # Confirm critical process changes
  max_processes_per_run: 100         # Max processes to adjust at once
  enable_restore: true               # Allow restoring original priorities
  dry_run: false                     # Log changes without applying
```

### Adding New Workload Priority Mappings

To add priority mapping for a new workload type:

```yaml
my_custom_workload:
  description: "My custom applications"
  priority_class: HIGH  # or NORMAL, LOW, BACKGROUND, CRITICAL
  patterns:
    - "myapp"
    - "custom-service"
    - "myprocess.*"  # Regex patterns supported
```

### Pattern Matching

- Patterns are matched against process names and command lines
- Case-insensitive matching
- Regex patterns supported (e.g., `python.*numpy`)
- First matching pattern determines workload classification

### Priority Assignment Strategy

1. **Process Scanning**: All processes are scanned and classified
2. **Pattern Matching**: Each process is matched against workload patterns
3. **Priority Assignment**: Matched processes get the workload's priority class
4. **Focus Boost**: If a workload is dominant, its processes get additional boost
5. **Safety Checks**: Critical processes and filter rules are respected

### Use Cases

**Database Server Optimization**:
```yaml
database:
  priority_class: CRITICAL  # Change from HIGH to CRITICAL
  patterns:
    - "mysqld"
    - "my-custom-db"  # Add your custom database
```

**Background Task Management**:
```yaml
background_tasks:
  priority_class: BACKGROUND
  patterns:
    - "backup"
    - "my-batch-job"  # Add your batch jobs
```

**Interactive Application Priority**:
```yaml
interactive:
  priority_class: HIGH
  patterns:
    - "firefox"
    - "my-app"  # Add your interactive app
```

### Best Practices

1. **Test Carefully**: Priority changes affect system stability
2. **Start Conservative**: Use NORMAL priority for new workloads
3. **Monitor Performance**: Check system responsiveness after changes
4. **Use Dry Run**: Test with `dry_run: true` first
5. **Document Changes**: Add descriptive comments
6. **Backup Original**: Original priorities are automatically backed up
7. **Critical Processes**: Be extremely careful with CRITICAL priority

### Safety Considerations

⚠️ **CRITICAL Priority**: Reserved for essential system processes
⚠️ **System Stability**: Too many high-priority processes can cause issues
⚠️ **Testing**: Always test on non-production systems first
⚠️ **Monitoring**: Watch system load and responsiveness
⚠️ **Restore**: Use `restore_original_priorities()` if issues occur
