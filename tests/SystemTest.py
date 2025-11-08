#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - System Test
This script tests all components to ensure they work together without errors
"""

import sys
import traceback
import numpy as np
from typing import Dict

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        print("  ✓ Importing BayesianOptimzation...")
        from src.BayesianOptimzation import BayesianOptimizer, OptimizationResult
        
        print("  ✓ Importing GeneticAlgorithm...")
        from src.GeneticAlgorithm import GeneticAlgorithm, AdvancedGeneticAlgorithm, GAOptimizationResult
        
        print("  ✓ Importing HybridOptimizationEngine...")
        from src.HybridOptimizationEngine import HybridOptimizationEngine, OptimizationStrategy
        
        print("  ✓ Importing PerformanceMonitor...")
        from src.PerformanceMonitor import PerformanceMonitor
        
        print("  ✓ Importing KernelParameterInterface...")
        from src.KernelParameterInterface import KernelParameterInterface
        
        print("  ✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_bayesian_optimizer():
    """Test Bayesian Optimizer functionality"""
    print("\nTesting Bayesian Optimizer...")
    
    try:
        from src.BayesianOptimzation import BayesianOptimizer
        
        # Define simple parameter bounds
        parameter_bounds = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        # Simple objective function (2D Gaussian)
        def test_objective(params: Dict[str, float]) -> float:
            x, y = params['x'], params['y']
            return -(x**2 + y**2)  # Maximum at (0,0)
        
        # Initialize optimizer
        optimizer = BayesianOptimizer(
            parameter_bounds=parameter_bounds,
            initial_samples=3,
            max_iterations=5
        )
        
        # Run optimization
        result = optimizer.optimize(test_objective)
        
        print(f"  ✓ Best score: {result.best_score:.4f}")
        print(f"  ✓ Best parameters: {result.best_parameters}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Bayesian Optimizer error: {e}")
        traceback.print_exc()
        return False

def test_genetic_algorithm():
    """Test Genetic Algorithm functionality"""
    print("\nTesting Genetic Algorithm...")
    
    try:
        from src.GeneticAlgorithm import GeneticAlgorithm
        
        # Define simple parameter bounds
        parameter_bounds = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        # Simple objective function
        def test_objective(params: Dict[str, float]) -> float:
            x, y = params['x'], params['y']
            return -(x**2 + y**2)  # Maximum at (0,0)
        
        # Initialize genetic algorithm
        ga = GeneticAlgorithm(
            parameter_bounds=parameter_bounds,
            population_size=10,
            max_generations=5
        )
        
        # Run optimization
        result = ga.optimize(test_objective)
        
        print(f"  ✓ Best fitness: {result.best_fitness:.4f}")
        print(f"  ✓ Best parameters: {result.best_individual.parameters}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Genetic Algorithm error: {e}")
        traceback.print_exc()
        return False

def test_hybrid_optimization():
    """Test Hybrid Optimization Engine"""
    print("\nTesting Hybrid Optimization Engine...")
    
    try:
        from src.HybridOptimizationEngine import HybridOptimizationEngine, OptimizationStrategy
        
        # Define simple parameter bounds
        parameter_bounds = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        # Simple objective function
        def test_objective(params: Dict[str, float]) -> float:
            x, y = params['x'], params['y']
            return -(x**2 + y**2)  # Maximum at (0,0)
        
        # Initialize hybrid engine
        engine = HybridOptimizationEngine(
            parameter_bounds=parameter_bounds,
            strategy=OptimizationStrategy.BAYESIAN_ONLY,
            evaluation_budget=10,
            time_budget=30.0
        )
        
        # Run optimization
        result = engine.optimize(test_objective)
        
        print(f"  ✓ Strategy: {result.strategy_used.value}")
        print(f"  ✓ Best score: {result.best_score:.4f}")
        print(f"  ✓ Best parameters: {result.best_parameters}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Hybrid Optimization error: {e}")
        traceback.print_exc()
        return False

def test_performance_monitor():
    """Test Performance Monitor"""
    print("\nTesting Performance Monitor...")
    
    try:
        from src.PerformanceMonitor import PerformanceMonitor
        import time
        
        # Initialize monitor
        monitor = PerformanceMonitor(sampling_interval=0.1)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Let it collect some data
        time.sleep(1)
        
        # Get current metrics
        current = monitor.get_current_metrics()
        if current:
            print(f"  ✓ CPU: {current.cpu_percent:.1f}%")
            print(f"  ✓ Memory: {current.memory_percent:.1f}%")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        print("  ✓ Performance monitoring successful!")
        return True
        
    except Exception as e:
        print(f"  ✗ Performance Monitor error: {e}")
        traceback.print_exc()
        return False

def test_kernel_interface():
    """Test Kernel Parameter Interface"""
    print("\nTesting Kernel Parameter Interface...")
    
    try:
        from src.KernelParameterInterface import KernelParameterInterface
        
        # Initialize interface
        interface = KernelParameterInterface()
        
        # Get current configuration
        config = interface.get_current_configuration()
        print(f"  ✓ Found {len(config)} parameters")
        
        # Get parameters by subsystem
        memory_params = interface.get_parameters_by_subsystem('memory')
        print(f"  ✓ Memory parameters: {len(memory_params)}")
        
        print("  ✓ Kernel interface successful!")
        return True
        
    except Exception as e:
        print(f"  ✗ Kernel Interface error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Linux Kernel Optimization Framework - System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Bayesian Optimizer Test", test_bayesian_optimizer),
        ("Genetic Algorithm Test", test_genetic_algorithm),
        ("Hybrid Optimization Test", test_hybrid_optimization),
        ("Performance Monitor Test", test_performance_monitor),
        ("Kernel Interface Test", test_kernel_interface)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! The system is ready to run.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
