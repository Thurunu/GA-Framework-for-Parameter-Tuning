#!/usr/bin/env python3
"""
Quick test to verify ProcessPriorityManager can adjust MySQL priority
Run with: sudo python3 test_mysql_priority.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from ProcessPriorityManager import ProcessPriorityManager
    import psutil
except ImportError as e:
    print(f"Error: {e}")
    print("Install: pip install psutil pyyaml")
    sys.exit(1)

def main():
    print("="*60)
    print("MySQL Process Priority Test")
    print("="*60)
    
    # Find MySQL process
    mysql_pid = None
    mysql_nice = None
    
    for proc in psutil.process_iter(['pid', 'name', 'nice']):
        try:
            if 'mysqld' in proc.info['name']:
                mysql_pid = proc.info['pid']
                mysql_nice = proc.info['nice']
                print(f"\n✓ Found MySQL process:")
                print(f"  PID: {mysql_pid}")
                print(f"  Current nice value: {mysql_nice}")
                print(f"  Current priority (PR): {20 + mysql_nice}")
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not mysql_pid:
        print("\n❌ ERROR: MySQL process (mysqld) not found!")
        print("Make sure MySQL is running.")
        sys.exit(1)
    
    # Initialize ProcessPriorityManager
    print("\n" + "="*60)
    print("Initializing ProcessPriorityManager...")
    print("="*60)
    
    manager = ProcessPriorityManager()
    
    # Scan and classify
    print("\nScanning and classifying processes...")
    classified = manager.scan_and_classify_processes()
    
    print(f"\nClassified processes:")
    for workload_type, processes in classified.items():
        print(f"  {workload_type}: {len(processes)} processes")
        for proc in processes:
            if 'mysql' in proc.name.lower():
                print(f"    → {proc.name} (PID {proc.pid}): "
                      f"current_nice={proc.current_nice}, "
                      f"target_nice={proc.target_nice}")
    
    # Check if MySQL was classified
    mysql_found_in_classification = False
    for processes in classified.values():
        for proc in processes:
            if proc.pid == mysql_pid:
                mysql_found_in_classification = True
                break
    
    if not mysql_found_in_classification:
        print(f"\n⚠️  WARNING: MySQL (PID {mysql_pid}) was NOT classified!")
        print("This means it was filtered out. Checking why...")
        
        # Check process age
        try:
            mysql_proc = psutil.Process(mysql_pid)
            import time
            age = time.time() - mysql_proc.create_time()
            print(f"  Process age: {age:.1f} seconds")
            print(f"  Min age requirement: {manager.config.get('filter_rules', {}).get('min_process_age', 5.0)}s")
            
            if age < manager.config.get('filter_rules', {}).get('min_process_age', 5.0):
                print(f"  ❌ Filtered by age (too young)")
            else:
                print(f"  ✓ Age check passed")
                
            # Check stability tracking
            stability = manager.config.get('filter_rules', {}).get('stability_tracking', {})
            if stability.get('enabled', True):
                print(f"  ⚠️  Stability tracking ENABLED")
                print(f"     Required observations: {stability.get('required_observations', 2)}")
                print(f"     Current observations: {manager.process_observations.get(mysql_pid, {}).get('count', 0)}")
            else:
                print(f"  ✓ Stability tracking disabled")
                
        except Exception as e:
            print(f"  Error checking: {e}")
    
    # Optimize priorities
    print("\n" + "="*60)
    print("Optimizing process priorities for DATABASE workload...")
    print("="*60)
    
    stats = manager.optimize_process_priorities(workload_focus='database')
    
    print(f"\nOptimization Results:")
    print(f"  Processes adjusted: {stats['processes_adjusted']}")
    print(f"  High priority set: {stats['high_priority_set']}")
    print(f"  Low priority set: {stats['low_priority_set']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Short-lived filtered: {stats['short_lived_filtered']}")
    print(f"  Processes tracked: {stats['processes_tracked']}")
    
    # Check MySQL priority again
    try:
        mysql_proc = psutil.Process(mysql_pid)
        new_nice = mysql_proc.nice()
        new_pr = 20 + new_nice
        
        print(f"\n" + "="*60)
        print("MySQL Process Status After Optimization:")
        print("="*60)
        print(f"  PID: {mysql_pid}")
        print(f"  Old nice: {mysql_nice} (PR: {20 + mysql_nice})")
        print(f"  New nice: {new_nice} (PR: {new_pr})")
        
        if new_nice < mysql_nice:
            print(f"  ✅ SUCCESS! Priority INCREASED (nice reduced from {mysql_nice} to {new_nice})")
        elif new_nice == mysql_nice:
            print(f"  ⚠️  No change (nice still {mysql_nice})")
        else:
            print(f"  ❌ Priority DECREASED? (nice increased from {mysql_nice} to {new_nice})")
            
    except Exception as e:
        print(f"\n❌ Error checking MySQL process: {e}")
    
    print("\n" + "="*60)
    print("Expected for DATABASE workload:")
    print("  MySQL should have nice = -10 (PR = 10)")
    print("  This gives HIGH priority for database processes")
    print("="*60)

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("⚠️  WARNING: Not running as root!")
        print("You may not be able to set negative nice values.")
        print("Run with: sudo python3 test_mysql_priority.py")
        print("\nContinuing anyway...\n")
    
    main()
