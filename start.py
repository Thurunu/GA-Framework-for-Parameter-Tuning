#!/usr/bin/env python3
"""
Quick Start Script for Continuous Kernel Optimization
Run this to test the continuous optimization system
"""

import sys
import time
import signal
from pathlib import Path
import os
import threading

# Add src directory and all subdirectories to path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'api'))
sys.path.insert(0, os.path.join(project_root, 'src', 'core'))
sys.path.insert(0, os.path.join(project_root, 'src', 'data'))
sys.path.insert(0, os.path.join(project_root, 'src', 'monitoring'))
sys.path.insert(0, os.path.join(project_root, 'src', 'optimization'))
sys.path.insert(0, os.path.join(project_root, 'src', 'system'))
sys.path.insert(0, os.path.join(project_root, 'src', 'utils'))
sys.path.insert(0, os.path.join(project_root, 'src', 'workload'))
def main():
    print("="*60)
    print("   Continuous Kernel Optimization Framework")
    print("="*60)

    
    # Check if running as root
    try:
        if hasattr(os, 'geteuid') and os.geteuid() != 0:
            print("\n⚠️  Warning: Not running as root.")
            print("   Kernel parameter changes will be simulated only.")

    except:
        print("ℹ️  Running on Windows - using simulation mode\n")

    
    # Import continuous optimizer
    try:
        from ContinuousOptimizer import ContinuousOptimizer
        from agentServer import start_server
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required files are in the current directory:")
        print(sys.path)
        print(os.listdir())

        required_files = [
            "ContinuousOptimizer.py",
            "ProcessWorkloadDetector.py", 
            "HybridOptimizationEngine.py",
            "PerformanceMonitor.py",
            "KernelParameterInterface.py",
            "BayesianOptimzation.py",
            "GeneticAlgorithm.py",
            "agentServer.py"
        ]
        for file_name in required_files:
            exists = "✅" if Path(f"src/{file_name}").exists() else "❌"
            print(f"  {exists} {file_name}")
        sys.exit(1)
    
    print("🚀 Starting Continuous Optimization Test...")

    print("\nThis will:")
    print("  • Start Agent Metrics Server on port 9300")
    print("  • Monitor running processes continuously")
    print("  • Detect workload type changes automatically") 
    print("  • Optimize kernel parameters for detected workloads")
    print("  • Log all optimization activities")
    
    # Wait for user confirmation
    # try:
    #     input("Press Enter to start, or Ctrl+C to cancel...")
    # except KeyboardInterrupt:
    #     print("\nCancelled by user.")
    #     sys.exit(0)
    
    print("\nStarting agent server in background...")
    
    # Start agent server in a background thread
    agent_thread = threading.Thread(
        target=start_server,
        kwargs={'HOST': '0.0.0.0', 'PORT': 9300},
        daemon=True,
        name="AgentServer"
    )
    agent_thread.start()
    
    # Give the server a moment to start
    time.sleep(2)
    
    print("✅ Agent server started on http://0.0.0.0:9300\n")
    print("Starting continuous optimization...")
    
    # Initialize optimizer with shorter intervals for testing
    optimizer = ContinuousOptimizer(
        adaptation_delay=15.0,    # Wait 15 seconds after workload change
        stability_period=60.0,    # Minimum 1 minute between optimizations
        log_file="continuous_optimizer_test.log"
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\n\n🛑 Shutdown signal received...')
        print('Stopping continuous optimization...')
        optimizer.stop_continuous_optimization()
        print('✅ Shutdown complete.')
        sys.exit(0)
    
    print("Setting up signal handlers for graceful shutdown...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the continuous optimizer
        optimizer.start_continuous_optimization()
        
        
        print("\n▶️ Continuous optimization is now running!")
        print("="*50)
        print("Agent API available at: http://localhost:9300")
        print("  • Health: http://localhost:9300/health")
        print("  • Metrics: http://localhost:9300/metrics")
        print("  • Status: http://localhost:9300/status")
        print("\n🕛Status updates will appear every 30 seconds...")
        print("View detailed logs: tail -f results/logs/continuous_optimizer_test.log\n")
        
        # Status update loop
        status_counter = 0
        while True:
            time.sleep(30)  # Status update every 30 seconds
            status_counter += 1
            
            status = optimizer.get_status()
            
            print(f"\n📋 Status Update #{status_counter}") 
            print(f"  Current workload: {status['current_workload']}")
            print(f"  Active profile: {status['active_profile']}")
            print(f"  Last optimized profile: {status['last_optimized_profile'] or 'None'}")
            print(f"  Active processes: {status['active_processes']}")
            print(f"  Optimization in progress: {'Yes' if status['optimization_in_progress'] else 'No'}")
            print(f"  Queue size: {status['queue_size']}")
            
            if status['last_optimization']:
                last_opt_time = time.time() - status['last_optimization']
                print(f"  Last optimization: {last_opt_time:.0f} seconds ago")
            else:
                print("  Last optimization: Never")
            
            # Show some current parameters
            params = status['current_parameters']
            if params:
                print("  Current key parameters:")
                key_params = ['vm.swappiness', 'vm.dirty_ratio', 'kernel.sched_min_granularity_ns']
                for param in key_params:
                    if param in params:
                        print(f"    {param}: {params[param]}")
            
    except KeyboardInterrupt:
        print('\n\n🛑 Interrupted by user...')
    except Exception as e:
        print(f'\n\n❌ Error occurred: {e}')
    finally:
        print('Stopping continuous optimization...')
        optimizer.stop_continuous_optimization()
        print('✅ Test completed.\n')
        print("📄 Check the log file for detailed information:")
        print("   cat continuous_optimizer_test.log")

if __name__ == "__main__":
    main()
