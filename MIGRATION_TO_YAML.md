# Migration to YAML-based Configuration

## Overview
Moved optimization profiles AND workload detection patterns from hardcoded Python dictionaries to YAML configuration files for better maintainability and flexibility.

## Changes Made

### 1. Created Configuration Files

#### `config/optimization_profiles.yml`
- Contains all 5 optimization profiles (database, web_server, hpc_compute, io_intensive, general)
- Easy-to-read YAML format
- Well-commented with parameter explanations

#### `config/workload_patterns.yml`
- Contains all workload detection patterns (database, web_server, hpc_compute, media_processing, compilation, io_intensive)
- Regex patterns for process name/command matching
- Fallback thresholds for resource-based classification

### 2. Updated ContinuousOptimizer.py

#### Added Dependencies
```python
import yaml
from pathlib import Path
```

#### New Methods
- `_load_optimization_profiles(config_file)`: Loads profiles from YAML
- `_get_default_profiles()`: Provides fallback profiles if YAML loading fails

#### Updated Constructor
- Added `config_file` parameter (optional)
- Loads profiles on initialization
- Prints confirmation of loaded profiles

#### Benefits
- ✅ Cleaner code (removed ~100 lines of hardcoded profiles)
- ✅ Easier to maintain and modify profiles
- ✅ No code changes needed to adjust parameters
- ✅ Graceful fallback to default profile on errors
- ✅ Better separation of configuration and logic

### 3. Updated ProcessWorkloadDetector.py

#### Added Dependencies
```python
import yaml
from pathlib import Path
```

#### Modified WorkloadClassifier
- Changed from static class with hardcoded patterns to instance-based
- New `__init__(config_file)`: Initializes and loads patterns from YAML
- New `_load_patterns(config_file)`: Loads patterns from YAML
- New `_load_default_patterns()`: Provides fallback patterns if YAML loading fails
- Updated `classify_process()`: Now instance method using loaded patterns

#### Updated ProcessWorkloadDetector
- Added `config_file` parameter to constructor
- Creates `WorkloadClassifier` instance with patterns from YAML
- Changed `WorkloadClassifier.classify_process()` to `self.classifier.classify_process()`

#### Benefits
- ✅ Cleaner code (removed ~30 lines of hardcoded patterns)
- ✅ Easy to add new workload types without code changes
- ✅ Customizable thresholds for fallback classification
- ✅ Graceful fallback to default patterns on errors
- ✅ Better separation of detection logic and pattern data

### 4. Updated Dependencies
**File**: `requirements.txt`
- Added `PyYAML>=5.4.0`

### 5. Created Documentation
**File**: `config/README.md`
- Complete guide to YAML structure
- Examples of customization
- List of available parameters and strategies
- Instructions for adding new profiles

### 6. Updated Test Script
**File**: `test_config.py`
- Validates both YAML files load correctly
- Test 1: Optimization profiles validation
- Test 2: Workload patterns validation
- Test 3: Cross-validation (ensures patterns match profiles)
- Displays comprehensive configuration summary

## How to Use

### Install Dependencies
```bash
pip install PyYAML
# or
pip install -r requirements.txt
```

### Test Configuration Loading
```bash
python test_config.py
```

### Run with Default Config
```python
optimizer = ContinuousOptimizer()  # Uses config/optimization_profiles.yml
```

### Run with Custom Config
```python
optimizer = ContinuousOptimizer(config_file="/path/to/custom_profiles.yml")
```

## Customizing Profiles

To modify a profile, simply edit `config/optimization_profiles.yml`:

```yaml
database:
  workload_type: database
  strategy: BAYESIAN_ONLY
  evaluation_budget: 20  # Changed from 15
  time_budget: 180.0
  
  parameter_bounds:
    vm.swappiness: [1, 25]  # Changed from [1, 30]
    # ... other parameters
```

No code changes needed! Just restart the optimizer.

## Adding New Profiles

1. Add new entry to `optimization_profiles.yml`
2. Update `ProcessWorkloadDetector` to detect the workload
3. Restart the optimizer

Example:
```yaml
my_workload:
  workload_type: my_workload
  strategy: ADAPTIVE
  evaluation_budget: 12
  time_budget: 150.0
  parameter_bounds:
    vm.swappiness: [20, 60]
  performance_weights:
    cpu_efficiency: 0.4
    memory_efficiency: 0.3
    io_throughput: 0.2
    network_throughput: 0.1
```

## Error Handling

- If YAML file not found → Falls back to default general profile
- If YAML parsing fails → Falls back to default general profile
- If strategy name invalid → Error with clear message
- All errors are logged to console

## Adding New Workload Types

To add a completely new workload type:

1. **Add to `workload_patterns.yml`**:
```yaml
my_workload:
  description: "My custom workload"
  process_patterns:
    - 'myapp.*'
    - 'custom-process.*'
```

2. **Add to `optimization_profiles.yml`**:
```yaml
my_workload:
  workload_type: my_workload
  strategy: ADAPTIVE
  evaluation_budget: 12
  time_budget: 150.0
  parameter_bounds:
    vm.swappiness: [20, 60]
  performance_weights:
    cpu_efficiency: 0.4
    memory_efficiency: 0.3
    io_throughput: 0.2
    network_throughput: 0.1
```

3. Restart the optimizer - no code changes needed!

## Migration Checklist

- ✅ Created `config/optimization_profiles.yml`
- ✅ Created `config/workload_patterns.yml`
- ✅ Updated `ContinuousOptimizer.py` to load profiles from YAML
- ✅ Updated `ProcessWorkloadDetector.py` to load patterns from YAML
- ✅ Converted `WorkloadClassifier` from static to instance-based
- ✅ Added PyYAML to `requirements.txt`
- ✅ Created comprehensive configuration documentation
- ✅ Updated test script with cross-validation
- ✅ Maintained backward compatibility (fallback defaults)
- ✅ Fixed profile display bug (active_profile vs last_optimized_profile)

## Next Steps

1. Install PyYAML: `pip install PyYAML`
2. Test configuration: `python test_config.py`
3. Run optimizer: `python quick_start_continuous.py`

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Profiles** | ~100 lines of hardcoded dict | Clean YAML file |
| **Patterns** | ~30 lines of hardcoded dict | Clean YAML file |
| **Modifications** | Code changes required | Edit YAML only |
| **Maintainability** | Difficult | Easy |
| **Documentation** | None | Comprehensive |
| **Organization** | Scattered in code | Centralized config |
| **Adding workloads** | Modify 2 Python files | Edit 2 YAML files |
| **Testing** | Manual | Automated validation |
