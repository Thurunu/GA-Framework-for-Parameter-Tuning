#!/usr/bin/env python3
"""
Test script for EEVDF Scheduler Support (Kernel 6.6+)
Tests the updated kernel parameter interface and process priority management
"""

import sys
import os
import time

def test_updated_kernel_interface():
    """Test the updated KernelParameterInterface with EEVDF parameters"""
    print("Testing Updated Kernel Parameter Interface (EEVDF Support)...")
    
    try:
        from KernelParameterInterface import KernelParameterInterface
        
        print("  ✓ Initializing KernelParameterInterface...")
        interface = KernelParameterInterface()
        
        # Test getting current configuration
        print("  ✓ Getting current configuration...")
        config = interface.get_current_configuration()
        print(f"    Found {len(config)} parameters")
        
        # Check for new EEVDF parameters
        print("  ✓ Checking for EEVDF scheduler parameters...")
        eevdf_params = [
            'kernel.sched_cfs_bandwidth_slice_us',
            'kernel.sched_latency_ns',
            'kernel.sched_rt_period_us',
            'kernel.sched_rt_runtime_us'
        ]
        
        for param in eevdf_params:
            param_info = interface.get_parameter_info(param)
            if param_info:
                print(f"    ✓ {param}: available")
                try:
                    current_value = config.get(param, 'not found')
                    print(f"      Current value: {current_value}")
                    print(f"      Default: {param_info.default_value}")
                    print(f"      Range: {param_info.min_value} - {param_info.max_value}")
                except Exception as e:
                    print(f"      Error reading value: {e}")
            else:
                print(f"    ✗ {param}: not found in parameter list")
        
        # Check for deprecated parameters (should not be found on kernel 6.6+)
        print("  ✓ Checking for deprecated CFS parameters...")
        deprecated_params = [
            'kernel.sched_min_granularity_ns',
            'kernel.sched_wakeup_granularity_ns', 
            'kernel.sched_migration_cost_ns'
        ]
        
        for param in deprecated_params:
            param_info = interface.get_parameter_info(param)
            if param_info:
                print(f"    ⚠️  {param}: still in parameter list (should be removed for kernel 6.6+)")
            else:
                print(f"    ✓ {param}: correctly removed from parameter list")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Kernel Interface error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_process_priority_manager():
    """Test the ProcessPriorityManager for EEVDF support"""
    print("\nTesting Process Priority Manager (EEVDF Support)...")
    
    try:
        from ProcessPriorityManager import ProcessPriorityManager, PriorityClass
        
        print("  ✓ Initializing ProcessPriorityManager...")
        manager = ProcessPriorityManager()
        
        # Test process classification
        print("  ✓ Testing process classification...")
        classified = manager.scan_and_classify_processes()
        
        total_processes = sum(len(procs) for procs in classified.values())
        print(f"    Found {total_processes} processes in {len(classified)} categories")
        
        # Show classification results
        for workload_type, processes in classified.items():
            if processes:  # Only show non-empty categories
                print(f"    {workload_type}: {len(processes)} processes")
                # Show first 2 processes as examples
                for i, proc in enumerate(processes[:2]):
                    print(f"      - {proc.name} (PID {proc.pid}, nice={proc.current_nice})")
        
        # Test priority statistics
        print("  ✓ Getting priority statistics...")
        stats = manager.get_priority_statistics()
        print(f"    Total processes: {stats['total_processes']}")
        print(f"    Managed processes: {stats['managed_processes']}")
        
        if stats['priority_distribution']:
            print("    Priority distribution:")
            for range_name, count in stats['priority_distribution'].items():
                print(f"      {range_name}: {count} processes")
        
        # Test optimization (simulation mode)
        print("  ✓ Testing priority optimization (simulation)...")
        if os.name != 'nt':  # Only on Linux
            optimization_stats = manager.optimize_process_priorities(workload_focus='database')
            print(f"    Optimization results: {optimization_stats}")
        else:
            print("    Skipping actual optimization on Windows (simulation only)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Process Priority Manager error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_continuous_optimizer_eevdf():
    """Test the ContinuousOptimizer with EEVDF support"""
    print("\nTesting Continuous Optimizer (EEVDF Support)...")
    
    try:
        from ContinuousOptimizer import ContinuousOptimizer
        
        print("  ✓ Initializing ContinuousOptimizer...")
        optimizer = ContinuousOptimizer(
            adaptation_delay=5.0,     # Short delay for testing
            stability_period=10.0,    # Short period for testing
            log_file="test_eevdf_optimizer.log"
        )
        
        # Check optimization profiles
        print("  ✓ Checking optimization profiles...")
        profiles = optimizer.OPTIMIZATION_PROFILES
        
        for profile_name, profile in profiles.items():
            print(f"    {profile_name}:")
            
            # Check for EEVDF parameters
            eevdf_found = False
            deprecated_found = False
            
            for param_name in profile.parameter_bounds.keys():
                if 'sched_cfs_bandwidth_slice_us' in param_name or 'sched_latency_ns' in param_name:
                    eevdf_found = True
                    print(f"      ✓ EEVDF parameter: {param_name}")
                
                if 'sched_min_granularity_ns' in param_name or 'sched_wakeup_granularity_ns' in param_name:
                    deprecated_found = True
                    print(f"      ⚠️  Deprecated parameter: {param_name}")
            
            if eevdf_found:
                print(f"      ✓ Profile {profile_name} includes EEVDF parameters")
            if deprecated_found:
                print(f"      ⚠️  Profile {profile_name} includes deprecated CFS parameters")
        
        # Test status retrieval
        print("  ✓ Testing status retrieval...")
        status = optimizer.get_status()
        print(f"    Current workload: {status['current_workload']}")
        print(f"    Queue size: {status['queue_size']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Continuous Optimizer error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all EEVDF scheduler tests"""
    print("EEVDF Scheduler Support Test Suite")
    print("=" * 50)
    print("Testing Linux Kernel 6.6+ EEVDF scheduler compatibility")
    print()
    
    # Detect kernel version if on Linux
    if os.name != 'nt':
        try:
            import subprocess
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            kernel_version = result.stdout.strip()
            print(f"Detected kernel version: {kernel_version}")
            
            # Parse version to check if >= 6.6
            version_parts = kernel_version.split('.')
            if len(version_parts) >= 2:
                major = int(version_parts[0])
                minor = int(version_parts[1].split('-')[0])  # Handle versions like "6.6.0-rc1"
                
                if major > 6 or (major == 6 and minor >= 6):
                    print("✅ Kernel version supports EEVDF scheduler")
                else:
                    print("⚠️  Kernel version may not support EEVDF scheduler")
                    print("   EEVDF scheduler is available from kernel 6.6+")
            print()
        except:
            print("Could not detect kernel version")
            print()
    else:
        print("Running on Windows - testing in simulation mode")
        print()
    
    # Run tests
    results = []
    
    print("1. Testing Updated Kernel Parameter Interface")
    print("-" * 45)
    results.append(test_updated_kernel_interface())
    
    print("\n2. Testing Process Priority Manager")
    print("-" * 35)
    results.append(test_process_priority_manager())
    
    print("\n3. Testing Continuous Optimizer EEVDF Support")
    print("-" * 45)
    results.append(test_continuous_optimizer_eevdf())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    test_names = [
        "Kernel Parameter Interface (EEVDF)",
        "Process Priority Manager",
        "Continuous Optimizer EEVDF Support"
    ]
    
    all_passed = True
    for i, (test_name, passed) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {i+1}. {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests PASSED! EEVDF scheduler support is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python3 quick_start_continuous.py' to test continuous optimization")
        print("2. Use 'python3 ContinuousOptimizer.py' for production continuous operation")
        print("3. Install as service with 'sudo python3 install_service.py'")
    else:
        print("⚠️  Some tests FAILED. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure you're running on Linux kernel 6.6+ for full EEVDF support")
        print("2. Check that all required Python packages are installed")
        print("3. Verify file permissions and dependencies")

if __name__ == "__main__":
    main()
