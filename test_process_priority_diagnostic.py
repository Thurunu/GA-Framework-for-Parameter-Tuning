#!/usr/bin/env python3
"""
Diagnostic script to check ProcessPriorityManager
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ProcessPriorityManager import ProcessPriorityManager
import psutil

def main():
    print("=" * 60)
    print("Process Priority Manager Diagnostic")
    print("=" * 60)
    
    # Initialize manager
    print("\n1. Initializing ProcessPriorityManager...")
    try:
        manager = ProcessPriorityManager()
        print("✅ Manager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Show configuration
    print("\n2. Configuration:")
    print(f"   Workload patterns loaded: {list(manager.workload_patterns.keys())}")
    print(f"   Filter rules: {manager.config.get('filter_rules', {})}")
    
    # Scan for processes
    print("\n3. Scanning for processes...")
    classified = manager.scan_and_classify_processes()
    
    print(f"\n   Total workload types found: {len(classified)}")
    for workload, procs in classified.items():
        print(f"\n   📊 {workload} workload: {len(procs)} processes")
        for proc in procs[:5]:  # Show first 5
            print(f"      - PID {proc.pid}: {proc.name}")
            print(f"        Current nice: {proc.current_nice}, Target: {proc.target_nice}")
            print(f"        CPU: {proc.cpu_percent:.1f}%, Memory: {proc.memory_percent:.1f}%")
    
    # Check observation tracking
    print(f"\n4. Process Observations:")
    print(f"   Tracked processes: {len(manager.process_observations)}")
    for pid, data in list(manager.process_observations.items())[:5]:
        print(f"      PID {pid}: count={data['count']}, first_seen={data['first_seen']}")
    
    # Try optimization
    print("\n5. Testing optimization with 'general' workload...")
    stats = manager.optimize_process_priorities(workload_focus='general')
    print(f"\n   📊 Optimization Results:")
    print(f"      Processes adjusted: {stats['processes_adjusted']}")
    print(f"      High priority set: {stats['high_priority_set']}")
    print(f"      Low priority set: {stats['low_priority_set']}")
    print(f"      Errors: {stats['errors']}")
    print(f"      Short-lived filtered: {stats['short_lived_filtered']}")
    print(f"      Processes tracked: {stats['processes_tracked']}")
    
    # Show all running processes (sample)
    print("\n6. Sample of ALL running processes (first 10):")
    for proc in list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))[:10]:
        try:
            info = proc.info
            print(f"   PID {info['pid']}: {info['name']} - CPU: {info.get('cpu_percent', 0)}%, Mem: {info.get('memory_percent', 0)}%")
        except:
            pass
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
