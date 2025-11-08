#!/usr/bin/env python3
"""
Test script to check which kernel parameters are available on this system
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from KernelParameterInterface import KernelParameterInterface

def main():
    print("=" * 80)
    print("Kernel Parameter Availability Check")
    print("=" * 80)
    
    # Initialize kernel interface
    kernel_interface = KernelParameterInterface()
    
    print(f"\nTotal parameters loaded: {len(kernel_interface.optimization_parameters)}")
    print("\nChecking availability of each parameter...\n")
    
    available_params = []
    unavailable_params = []
    
    for param_name, param in kernel_interface.optimization_parameters.items():
        is_available = kernel_interface.check_parameter_availability(param_name)
        
        if is_available:
            available_params.append(param_name)
            print(f"✅ {param_name:<40} [Available] - Current: {param.current_value}")
        else:
            unavailable_params.append(param_name)
            print(f"❌ {param_name:<40} [NOT Available]")
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Available:     {len(available_params)}")
    print(f"  Not Available: {len(unavailable_params)}")
    print("=" * 80)
    
    if unavailable_params:
        print("\n⚠️  Unavailable parameters:")
        for param in unavailable_params:
            param_info = kernel_interface.optimization_parameters[param]
            print(f"  - {param} ({param_info.subsystem}): {param_info.description}")
        
        print("\nℹ️  Note: Some parameters (like EEVDF scheduler parameters) are only")
        print("   available in Linux kernel 6.6+. Your kernel version may not support them.")
    
    print("\n✅ Optimization will use only the available parameters.")

if __name__ == "__main__":
    main()
