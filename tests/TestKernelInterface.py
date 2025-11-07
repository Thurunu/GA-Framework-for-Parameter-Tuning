#!/usr/bin/env python3
"""
Test script specifically for KernelParameterInterface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_kernel_interface():
    """Test KernelParameterInterface with error handling"""
    print("Testing Kernel Parameter Interface...")
    
    try:
        from src.KernelParameterInterface import KernelParameterInterface
        
        # Initialize interface with error handling
        print("  ✓ Initializing KernelParameterInterface...")
        interface = KernelParameterInterface()
        
        # Test getting current configuration
        print("  ✓ Getting current configuration...")
        config = interface.get_current_configuration()
        print(f"    Found {len(config)} parameters")
        
        # Show some example parameters
        print("  ✓ Sample parameters:")
        count = 0
        for param_name, value in config.items():
            if count < 3:  # Show first 3 parameters
                param_info = interface.get_parameter_info(param_name)
                subsystem = param_info.subsystem if param_info else "unknown"
                print(f"    {param_name}: {value} [{subsystem}]")
                count += 1
        
        # Test getting parameters by subsystem
        print("  ✓ Testing subsystem queries...")
        subsystems = ['memory', 'cpu', 'network', 'filesystem']
        for subsystem in subsystems:
            params = interface.get_parameters_by_subsystem(subsystem)
            print(f"    {subsystem}: {len(params)} parameters")
        
        # Test parameter validation (without actually changing anything)
        print("  ✓ Testing parameter validation...")
        test_params = {
            'vm.swappiness': 30,
            'vm.dirty_ratio': 15
        }
        
        for param_name, value in test_params.items():
            param_info = interface.get_parameter_info(param_name)
            if param_info:
                is_valid = interface._validate_parameter_value(param_info, value)
                print(f"    {param_name} = {value}: {'valid' if is_valid else 'invalid'}")
        
        print("  ✓ Kernel Parameter Interface test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ✗ Kernel Interface error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Kernel Parameter Interface - Standalone Test")
    print("=" * 50)
    
    if test_kernel_interface():
        print("\n✅ Test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Test FAILED!")
        sys.exit(1)
