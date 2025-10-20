#!/usr/bin/env python3
"""
Test script for Genetic Algorithm with simple mathematical function
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def simple_objective_function(params):
    """Simple quadratic function with known optimum at (5, 3)"""
    x = params['x']
    y = params['y']
    return -(x - 5)**2 - (y - 3)**2 + 25  # Max at 25

def main():
    # Define parameter bounds for testing
    parameter_bounds = {
        'x': (0, 10),  # Optimum at 5
        'y': (0, 10)   # Optimum at 3
    }
    
    print("Testing Genetic Algorithm with Simple Function:")
    print("=" * 50)
    
    # Initialize and run GA
    ga = GeneticAlgorithm(
        parameter_bounds=parameter_bounds,
        population_size=20,
        max_generations=30,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_ratio=0.2
    )
    
    # Run optimization
    result = ga.optimize(simple_objective_function)
    
    print(f"\nResults:")
    print(f"Best Fitness: {result.best_fitness:.6f} (Expected: ~25.0)")
    print(f"Best Parameters: {result.best_individual.parameters} (Expected: x~5, y~3)")
    print(f"Generations: {result.generation_count}")
    print(f"Converged: {result.convergence_reached}")
    
    # Verify correctness
    expected_optimum = 25.0
    tolerance = 0.1
    if abs(result.best_fitness - expected_optimum) < tolerance:
        print("✓ Test PASSED: Algorithm found the optimum!")
    else:
        print("✗ Test FAILED: Algorithm did not converge to optimum.")

if __name__ == "__main__":
    main()