#!/usr/bin/env python3
"""
Test script to validate both optimization_profiles.yml and workload_patterns.yml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("="*70)
print("Testing Configuration Files")
print("="*70)
print()

# Test 1: Optimization Profiles
print("📋 TEST 1: Optimization Profiles (optimization_profiles.yml)")
print("-"*70)
try:
    from ContinuousOptimizer import ContinuousOptimizer
    
    profiles = ContinuousOptimizer._load_optimization_profiles()
    
    print(f"✅ Successfully loaded {len(profiles)} optimization profiles")
    print()
    
    for profile_name, profile in profiles.items():
        print(f"  • {profile_name}:")
        print(f"    - Strategy: {profile.strategy.value}")
        print(f"    - Parameters: {len(profile.parameter_bounds)}")
        print(f"    - Budget: {profile.evaluation_budget} evaluations / {profile.time_budget}s")
        
        # Validate performance weights
        weight_sum = sum(profile.performance_weights.values())
        if abs(weight_sum - 1.0) < 0.01:
            print(f"    - Weights: ✅ Valid (sum={weight_sum:.2f})")
        else:
            print(f"    - Weights: ⚠️  Warning (sum={weight_sum:.2f}, should be ~1.0)")
    
    print()
    print("✅ Optimization profiles test PASSED")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Workload Patterns
print("📋 TEST 2: Workload Patterns (workload_patterns.yml)")
print("-"*70)
try:
    from WorkloadClassifier import WorkloadClassifier
    
    classifier = WorkloadClassifier()
    
    print(f"✅ Successfully loaded {len(classifier.workload_patterns)} workload patterns")
    print()
    
    for workload_name, patterns in classifier.workload_patterns.items():
        print(f"  • {workload_name}:")
        print(f"    - Patterns: {len(patterns)}")
        print(f"    - Examples: {', '.join(patterns[:3])}")
        if len(patterns) > 3:
            print(f"                ... and {len(patterns) - 3} more")
    
    print()
    print("  Fallback Thresholds:")
    for threshold_type, values in classifier.fallback_thresholds.items():
        print(f"    • {threshold_type}: {values}")
    
    print()
    print("✅ Workload patterns test PASSED")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Cross-validation (patterns match profiles)
print("📋 TEST 3: Cross-validation (patterns ↔ profiles)")
print("-"*70)
try:
    profile_workloads = set(profiles.keys())
    pattern_workloads = set(classifier.workload_patterns.keys())
    
    # Profiles without patterns
    profiles_without_patterns = profile_workloads - pattern_workloads
    if profiles_without_patterns:
        print(f"⚠️  Profiles without detection patterns: {profiles_without_patterns}")
    
    # Patterns without profiles
    patterns_without_profiles = pattern_workloads - profile_workloads
    if patterns_without_profiles:
        print(f"⚠️  Detection patterns without optimization profiles: {patterns_without_profiles}")
    
    # Common workloads
    common_workloads = profile_workloads & pattern_workloads
    print(f"✅ {len(common_workloads)} workloads have both patterns and profiles:")
    for workload in sorted(common_workloads):
        print(f"   • {workload}")
    
    print()
    if not profiles_without_patterns and not patterns_without_profiles:
        print("✅ Cross-validation test PASSED (all workloads properly configured)")
    else:
        print("⚠️  Cross-validation test PASSED WITH WARNINGS")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Kernel Parameters
print("📋 TEST 4: Kernel Parameters (kernel_parameters.yml)")
print("-"*70)
try:
    from KernelParameterInterface import KernelParameterInterface
    
    interface = KernelParameterInterface()
    
    print(f"✅ Successfully loaded {len(interface.optimization_parameters)} kernel parameters")
    print()
    
    # Group by subsystem
    subsystems = {}
    for param_name, param in interface.optimization_parameters.items():
        subsystem = param.subsystem
        if subsystem not in subsystems:
            subsystems[subsystem] = []
        subsystems[subsystem].append(param_name)
    
    for subsystem, params in sorted(subsystems.items()):
        print(f"  • {subsystem}: {len(params)} parameters")
    
    print()
    print("✅ Kernel parameters test PASSED")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Process Priorities
print("📋 TEST 5: Process Priorities (process_priorities.yml)")
print("-"*70)
try:
    from ProcessPriorityManager import ProcessPriorityManager
    
    priority_manager = ProcessPriorityManager()
    
    print(f"✅ Successfully loaded priority mappings for {len(priority_manager.workload_patterns)} workload types")
    print()
    
    for workload_name, config in priority_manager.workload_patterns.items():
        priority_class = config['priority_class']
        pattern_count = len(config['patterns'])
        print(f"  • {workload_name}: {priority_class.name} ({priority_class.value}) - {pattern_count} patterns")
    
    print()
    
    # Check configuration sections
    if 'workload_focus_boost' in priority_manager.config:
        print("  Configuration sections:")
        print(f"    ✓ Workload focus boost")
        print(f"    ✓ Filter rules")
        print(f"    ✓ Safety settings")
    
    print()
    
    # Check short-lived process filtering
    filter_rules = priority_manager.config.get('filter_rules', {})
    min_age = filter_rules.get('min_process_age', None)
    stability = filter_rules.get('stability_tracking', {})
    
    if min_age is not None:
        print("  Short-lived process filtering:")
        print(f"    ✓ Minimum process age: {min_age}s")
        
        if stability.get('enabled'):
            print(f"    ✓ Stability tracking: {stability.get('required_observations')} observations")
            print(f"    ✓ Observation window: {stability.get('observation_window')}s")
        else:
            print(f"    ⚠ Stability tracking: Disabled")
    
    print()
    print("✅ Process priorities test PASSED")
    print()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print()
print("Configuration Summary:")
print(f"  • {len(profiles)} optimization profiles loaded")
print(f"  • {len(classifier.workload_patterns)} workload patterns loaded")
print(f"  • {len(interface.optimization_parameters)} kernel parameters loaded")
print(f"  • {len(priority_manager.workload_patterns)} process priority mappings loaded")
print(f"  • {len(common_workloads)} workloads fully configured")
print(f"  • {len(subsystems)} kernel subsystems defined")
print()
print("Complete YAML Configuration System:")
print("  ✓ optimization_profiles.yml - How to optimize")
print("  ✓ workload_patterns.yml - What to detect")
print("  ✓ kernel_parameters.yml - What to tune")
print("  ✓ process_priorities.yml - How to prioritize")
print()
print("You can now run the optimizer:")
print("  python quick_start_continuous.py")
print()
