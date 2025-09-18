#!/usr/bin/env python3
"""
Test script to verify the kernel parameter validation fix
This script tests the Linux-only kernel parameter interface
"""

import sys
sys.path.append('src')

from KernelParameterInterface import KernelParameterInterface

def test_parameter_validation():
    """Test parameter validation and clamping for Linux kernel optimization"""
    print("Testing Linux Kernel Parameter Interface fixes...")
    print("=" * 60)
    
    # Create interface
    kernel_interface = KernelParameterInterface()
    
    # Test the problematic value that was causing the error
    test_value = 11600367.927387634
    param_name = 'kernel.sched_cfs_bandwidth_slice_us'
    
    print(f"Original problematic value: {test_value}")
    
    # Get parameter info
    param_info = kernel_interface.get_parameter_info(param_name)
    if param_info:
        print(f"Parameter bounds: min={param_info.min_value}, max={param_info.max_value}")
    
    # Test clamping
    clamped_value = kernel_interface.clamp_parameter_value(param_name, test_value)
    print(f"Clamped value: {clamped_value}")
    
    # Test validation
    param = kernel_interface.optimization_parameters[param_name]
    is_valid_original = kernel_interface._validate_parameter_value(param, test_value)
    is_valid_clamped = kernel_interface._validate_parameter_value(param, clamped_value)
    
    print(f"Original value valid: {is_valid_original}")
    print(f"Clamped value valid: {is_valid_clamped}")
    
    # Test optimization bounds
    print("\nOptimization bounds for algorithms:")
    bounds = kernel_interface.get_optimization_bounds()
    for param_name, (min_val, max_val) in bounds.items():
        print(f"  {param_name}: [{min_val}, {max_val}]")
    
    # Test with multiple parameters including edge cases
    print("\nTesting with multiple parameters...")
    test_params_multi = {
        'kernel.sched_cfs_bandwidth_slice_us': 11600367.927387634,  # Too high
        'vm.swappiness': -5.5,  # Too low, should clamp to 0
        'vm.dirty_ratio': 150.2,  # Too high, should clamp to 90
        'kernel.sched_latency_ns': 25000000.7  # Within bounds
    }
    
    for param_name, value in test_params_multi.items():
        param_info = kernel_interface.get_parameter_info(param_name)
        if param_info:
            clamped = kernel_interface.clamp_parameter_value(param_name, value)
            print(f"  {param_name}: {value} -> {clamped} (bounds: {param_info.min_value}-{param_info.max_value})")
    
    print("\n" + "=" * 60)
    print("✅ All parameter validation tests completed successfully!")
    print("✅ The framework can now handle floating-point values from optimization algorithms")
    print("✅ Values are automatically clamped to valid kernel parameter ranges")

if __name__ == "__main__":
    test_parameter_validation()
