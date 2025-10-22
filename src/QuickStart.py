#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Quick Start
This script provides a simple interface to run the optimization system
"""

import os
import sys
import time
from typing import Dict

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['numpy', 'scipy', 'psutil']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} found")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    return True

def run_system_test():
    """Run the system test to verify everything works"""
    print("\nRunning system test...")
    try:
        import SystemTest
        return SystemTest.main() == 0
    except Exception as e:
        print(f"System test failed: {e}")
        return False

def run_example_optimization():
    """Run an example optimization to demonstrate the system"""
    print("\nRunning example optimization...")
    
    try:
        # Import necessary components
        from HybridOptimizationEngine import HybridOptimizationEngine
        from WorkloadCharacterizer import OptimizationStrategy
        import numpy as np
        
        # Define test parameter bounds
        parameter_bounds = {
            'vm.swappiness': (0, 100),
            'vm.dirty_ratio': (5, 40),
            'kernel.sched_min_granularity_ns': (1000000, 20000000)
        }
        
        # Mock objective function for demonstration
        def demo_objective(params: Dict[str, float]) -> float:
            """Demo objective function (not real performance measurement)"""
            swappiness = params.get('vm.swappiness', 60)
            dirty_ratio = params.get('vm.dirty_ratio', 20)
            sched_gran = params.get('kernel.sched_min_granularity_ns', 10000000)
            
            # Simulate performance score with some optimal points
            score = (
                -0.01 * (swappiness - 30)**2 +  # Optimal around 30
                -0.1 * (dirty_ratio - 15)**2 +   # Optimal around 15
                -0.000000001 * (sched_gran - 6000000)**2  # Optimal around 6ms
            ) + np.random.normal(0, 0.5)
            
            print(f"  Testing: swappiness={swappiness:.1f}, dirty_ratio={dirty_ratio:.1f}, sched_gran={sched_gran/1e6:.1f}ms")
            print(f"  Score: {score:.4f}")
            
            return score
        
        # Initialize hybrid engine
        print("\nInitializing Hybrid Optimization Engine...")
        engine = HybridOptimizationEngine(
            parameter_bounds=parameter_bounds,
            strategy=OptimizationStrategy.ADAPTIVE,
            evaluation_budget=15,  # Small budget for demo
            time_budget=60.0  # 1 minute for demo
        )
        
        # Run optimization
        print("\nStarting optimization...")
        result = engine.optimize(demo_objective)
        
        # Display results
        print(f"\n{'='*50}")
        print("OPTIMIZATION RESULTS:")
        print(f"{'='*50}")
        print(f"Strategy Used: {result.strategy_used.value}")
        print(f"Best Score: {result.best_score:.6f}")
        print(f"Total Evaluations: {result.total_evaluations}")
        print(f"Optimization Time: {result.optimization_time:.2f} seconds")
        print(f"Converged: {result.convergence_reached}")
        
        print(f"\nBest Parameters Found:")
        for param_name, value in result.best_parameters.items():
            if 'ns' in param_name:
                print(f"  {param_name}: {value/1e6:.2f} ms")
            else:
                print(f"  {param_name}: {value:.2f}")
        
        # Export results
        engine.export_results("demo_optimization_results.json", result)
        print(f"\nResults exported to: demo_optimization_results.json")
        
        return True
        
    except Exception as e:
        print(f"Example optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_instructions():
    """Show instructions for using the framework"""
    print(f"\n{'='*60}")
    print("LINUX KERNEL OPTIMIZATION FRAMEWORK")
    print("Usage Instructions")
    print(f"{'='*60}")
    
    print("\n1. BASIC USAGE:")
    print("   - Run SystemTest.py to verify all components work")
    print("   - Use MainIntegration.py for complete optimization sessions")
    print("   - Individual components can be used separately")
    
    print("\n2. INDIVIDUAL COMPONENTS:")
    print("   - BayesianOptimzation.py: Bayesian optimization for fine-tuning")
    print("   - GeneticAlgorithm.py: Genetic algorithm for global search")
    print("   - HybridOptimizationEngine.py: Combines both algorithms intelligently")
    print("   - PerformanceMonitor.py: Real-time system performance monitoring")
    print("   - KernelParameterInterface.py: Kernel parameter management")
    
    print("\n3. EXAMPLE USAGE:")
    print("   from MainIntegration import KernelOptimizationFramework")
    print("   framework = KernelOptimizationFramework()")
    print("   session_id = framework.start_optimization_session()")
    
    print("\n4. IMPORTANT NOTES:")
    print("   - Requires root privileges to modify kernel parameters")
    print("   - Creates backups before making changes")
    print("   - Test on non-production systems first")
    print("   - Monitor system stability during optimization")
    
    print(f"\n{'='*60}")

def main():
    """Main startup function"""
    print("Linux Kernel Optimization Framework - Quick Start")
    print("=" * 60)
    
    # Check system requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        return 1
    
    # Run system test
    print("\n2. Running system test...")
    if not run_system_test():
        print("⚠️ System test failed. Please check the errors above.")
        return 1
    
    print("\n✅ System test passed!")
    
    # Ask user what to do
    print("\nChoose an option:")
    print("1. Run example optimization (demo)")
    print("2. Show usage instructions")
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            if run_example_optimization():
                print("\n✅ Example optimization completed successfully!")
            else:
                print("\n❌ Example optimization failed.")
                
        elif choice == '2':
            show_usage_instructions()
            
        elif choice == '3':
            print("Goodbye!")
            
        else:
            print("Invalid choice. Showing usage instructions...")
            show_usage_instructions()
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
