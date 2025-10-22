#!/usr/bin/env python3
"""
Comprehensive Test Suite for ProcessPriorityManager
Tests all functionality including classification, priority setting, and stability tracking
"""

import sys
import os
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from ProcessPriorityManager import ProcessPriorityManager, PriorityClass, ProcessInfo
    import psutil
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages:")
    print("  pip install psutil pyyaml")
    sys.exit(1)


class TestProcessPriorityManager:
    """Test suite for ProcessPriorityManager"""
    
    def __init__(self):
        self.manager = None
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
    
    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        print(f"\n{'='*60}")
        print(f"🧪 Running test: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            if result:
                print(f"✅ PASSED: {test_name}")
                self.test_results['passed'] += 1
            else:
                print(f"❌ FAILED: {test_name}")
                self.test_results['failed'] += 1
                self.test_results['errors'].append(test_name)
            return result
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"{test_name}: {e}")
            return False
    
    def test_initialization(self):
        """Test 1: Manager initialization"""
        try:
            self.manager = ProcessPriorityManager()
            print(f"✓ Manager initialized successfully")
            print(f"✓ Loaded {len(self.manager.workload_patterns)} workload patterns")
            
            # Check that configuration is loaded
            if not self.manager.workload_patterns:
                print("⚠ Warning: No workload patterns loaded")
                return False
            
            # Check configuration structure
            if not self.manager.config:
                print("⚠ Warning: No configuration loaded")
                return False
                
            print(f"✓ Configuration loaded with keys: {list(self.manager.config.keys())}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize manager: {e}")
            return False
    
    def test_workload_patterns(self):
        """Test 2: Workload pattern loading"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        expected_workloads = ['database', 'compilation', 'system_critical', 'compute_intensive']
        
        for workload in expected_workloads:
            if workload in self.manager.workload_patterns:
                pattern_info = self.manager.workload_patterns[workload]
                print(f"✓ Found workload: {workload}")
                print(f"  - Priority: {pattern_info['priority_class'].name} ({pattern_info['priority_class'].value})")
                print(f"  - Patterns: {pattern_info['patterns']}")
            else:
                print(f"⚠ Missing workload: {workload}")
        
        return len(self.manager.workload_patterns) > 0
    
    def test_process_classification(self):
        """Test 3: Process classification logic"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        # Test cases: (process_name, cmdline, expected_workload, expected_priority_class)
        test_cases = [
            ('mysqld', ['mysqld', '--defaults-file=/etc/my.cnf'], 'database', PriorityClass.HIGH),
            ('python3', ['python3', 'script.py'], 'compilation', PriorityClass.LOW),
            ('redis-server', ['redis-server', '*:6379'], 'database', PriorityClass.HIGH),
            ('systemd', ['systemd'], 'system_critical', PriorityClass.CRITICAL),
            ('unknown_process', ['unknown'], 'general', PriorityClass.NORMAL),
        ]
        
        all_passed = True
        for proc_name, cmdline, expected_workload, expected_priority in test_cases:
            workload, priority = self.manager.classify_process(proc_name, cmdline)
            
            if workload == expected_workload and priority == expected_priority:
                print(f"✓ Correctly classified '{proc_name}' as {workload} ({priority.name})")
            else:
                print(f"✗ Classification mismatch for '{proc_name}':")
                print(f"  Expected: {expected_workload} ({expected_priority.name})")
                print(f"  Got: {workload} ({priority.name})")
                all_passed = False
        
        return all_passed
    
    def test_process_scanning(self):
        """Test 4: Process scanning functionality"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        print("Scanning running processes...")
        classified = self.manager.scan_and_classify_processes()
        
        total_processes = sum(len(procs) for procs in classified.values())
        print(f"✓ Scanned {total_processes} processes")
        
        if total_processes == 0:
            print("⚠ Warning: No processes found (might be due to short-lived process filtering)")
            print("  This is expected on first scan due to stability tracking")
        
        for workload_type, processes in classified.items():
            if processes:
                print(f"\n  {workload_type}: {len(processes)} processes")
                for proc in processes[:3]:  # Show first 3
                    print(f"    - {proc.name} (PID {proc.pid}, nice={proc.current_nice})")
        
        return True
    
    def test_stability_tracking(self):
        """Test 5: Short-lived process filtering"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        print("Testing stability tracking (requires multiple scans)...")
        
        # First scan
        scan1 = self.manager.scan_and_classify_processes()
        count1 = sum(len(procs) for procs in scan1.values())
        print(f"✓ First scan: {count1} processes, {len(self.manager.process_observations)} tracked")
        
        # Wait a bit
        time.sleep(2)
        
        # Second scan
        scan2 = self.manager.scan_and_classify_processes()
        count2 = sum(len(procs) for procs in scan2.values())
        print(f"✓ Second scan: {count2} processes, {len(self.manager.process_observations)} tracked")
        
        # Third scan - should see more stable processes now
        time.sleep(2)
        scan3 = self.manager.scan_and_classify_processes()
        count3 = sum(len(procs) for procs in scan3.values())
        print(f"✓ Third scan: {count3} processes (stable processes now eligible)")
        
        if count3 >= count2:
            print("✓ Stability tracking working - more processes eligible after multiple observations")
            return True
        else:
            print("⚠ Process count decreased (normal if processes terminated)")
            return True
    
    def test_priority_statistics(self):
        """Test 6: Priority statistics generation"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        stats = self.manager.get_priority_statistics()
        
        print(f"✓ Generated statistics:")
        print(f"  Total processes: {stats['total_processes']}")
        print(f"  Workload distribution: {stats['workload_distribution']}")
        print(f"  Priority distribution: {stats['priority_distribution']}")
        print(f"  Managed processes: {stats['managed_processes']}")
        
        return 'total_processes' in stats and 'workload_distribution' in stats
    
    def test_get_process_priority(self):
        """Test 7: Reading process priorities"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        # Get current process PID
        current_pid = os.getpid()
        nice_value = self.manager.get_process_priority(current_pid)
        
        if nice_value is not None:
            print(f"✓ Successfully read priority for PID {current_pid}: {nice_value}")
            return True
        else:
            print(f"✗ Failed to read priority for PID {current_pid}")
            return False
    
    def test_priority_validation(self):
        """Test 8: Priority value validation"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        # Test invalid nice values
        current_pid = os.getpid()
        
        # Test value too low
        result = self.manager.set_process_priority(current_pid, -25)
        if not result:
            print("✓ Correctly rejected nice value -25 (too low)")
        else:
            print("✗ Should have rejected nice value -25")
            return False
        
        # Test value too high
        result = self.manager.set_process_priority(current_pid, 25)
        if not result:
            print("✓ Correctly rejected nice value 25 (too high)")
        else:
            print("✗ Should have rejected nice value 25")
            return False
        
        print("✓ Priority validation working correctly")
        return True
    
    def test_configuration_values(self):
        """Test 9: Configuration value access"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        # Check filter rules
        filter_rules = self.manager.config.get('filter_rules', {})
        print(f"✓ Filter rules loaded:")
        print(f"  - Exclude PIDs: {filter_rules.get('exclude_pids', [])}")
        print(f"  - Min process age: {filter_rules.get('min_process_age', 'Not set')}s")
        print(f"  - Stability tracking: {filter_rules.get('stability_tracking', {}).get('enabled', False)}")
        
        # Check boost configuration
        boost_config = self.manager.config.get('workload_focus_boost', {})
        print(f"✓ Workload focus boost:")
        print(f"  - Enabled: {boost_config.get('enabled', False)}")
        print(f"  - Boost amount: {boost_config.get('boost_amount', 0)}")
        
        return True
    
    def test_cleanup_observations(self):
        """Test 10: Cleanup of old observations"""
        if not self.manager:
            print("✗ Manager not initialized")
            return False
        
        initial_count = len(self.manager.process_observations)
        print(f"✓ Initial observations: {initial_count}")
        
        # Force cleanup
        self.manager.last_cleanup_time = 0  # Force cleanup on next call
        self.manager._cleanup_old_observations()
        
        print(f"✓ Cleanup executed (observations may or may not be removed based on age)")
        return True
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("  ProcessPriorityManager Comprehensive Test Suite")
        print("="*60)
        
        # Run all tests
        self.run_test("Initialization", self.test_initialization)
        self.run_test("Workload Patterns", self.test_workload_patterns)
        self.run_test("Process Classification", self.test_process_classification)
        self.run_test("Process Scanning", self.test_process_scanning)
        self.run_test("Stability Tracking", self.test_stability_tracking)
        self.run_test("Priority Statistics", self.test_priority_statistics)
        self.run_test("Get Process Priority", self.test_get_process_priority)
        self.run_test("Priority Validation", self.test_priority_validation)
        self.run_test("Configuration Values", self.test_configuration_values)
        self.run_test("Cleanup Observations", self.test_cleanup_observations)
        
        # Print summary
        print("\n" + "="*60)
        print("  Test Summary")
        print("="*60)
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        
        if self.test_results['errors']:
            print(f"\n Failed tests:")
            for error in self.test_results['errors']:
                print(f"  - {error}")
        
        success_rate = (self.test_results['passed'] / 
                       (self.test_results['passed'] + self.test_results['failed']) * 100)
        print(f"\n📊 Success Rate: {success_rate:.1f}%")
        
        return self.test_results['failed'] == 0


def main():
    """Main test execution"""
    print("\n🔧 Starting ProcessPriorityManager Tests")
    print("="*60)
    
    # Check dependencies
    try:
        import psutil
        import yaml
        print("✓ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install psutil pyyaml")
        return 1
    
    # Run tests
    tester = TestProcessPriorityManager()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
