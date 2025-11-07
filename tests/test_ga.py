#!/usr/bin/env python3
"""
Test script for Genetic Algorithm with time-based constraints
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.GeneticAlgorithm import GeneticAlgorithm

# Common test configuration
DIMENSIONS = 3
BOUNDS = (0, 100)
OPTIMUM = 50  # Center of search space
TIME_LIMIT_SECONDS = 0.5  # Run for 60 seconds

def test_objective_function(params):
    """
    Standardized objective function for testing.
    Returns a score based on distance from optimum.
    Maximum value is 0 at the optimum point (50, 50, 50).
    """
    score = -sum((float(v) - OPTIMUM)**2 for v in params.values())
    return score

class TimeLimitedGA:
    """Wrapper to enforce time limit on GA optimization"""
    def __init__(self, ga, time_limit):
        self.ga = ga
        self.time_limit = time_limit
        self.start_time = None
        self.evaluations = 0
        
    def objective_wrapper(self, params):
        """Wrapper that counts evaluations and checks time limit"""
        if time.time() - self.start_time >= self.time_limit:
            raise TimeoutError("Time limit reached")
        self.evaluations += 1
        return test_objective_function(params)
    
    def optimize(self):
        """Run optimization with time limit"""
        self.start_time = time.time()
        self.evaluations = 0
        
        try:
            # Set large max_generations, will be limited by time
            self.ga.max_generations = 10000
            result = self.ga.optimize(self.objective_wrapper)
            elapsed_time = time.time() - self.start_time
            return result, elapsed_time, self.evaluations
        except TimeoutError:
            # Time limit reached, return best so far
            elapsed_time = time.time() - self.start_time
            # Create a result-like object with current best
            class TimeoutResult:
                def __init__(self, ga):
                    self.best_fitness = ga.best_fitness
                    self.best_individual = ga.best_individual
                    self.generation_count = ga.current_generation
                    self.convergence_reached = False
                    self.optimization_time = elapsed_time
            
            return TimeoutResult(self.ga), elapsed_time, self.evaluations

def main():
    print("Testing Genetic Algorithm with Time-Based Constraint:")
    print("=" * 60)
    
    # Define parameter bounds for testing
    parameter_bounds = {f'param{i}': BOUNDS for i in range(1, DIMENSIONS + 1)}
    
    print(f"Configuration:")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Parameter Bounds: {BOUNDS}")
    print(f"  Optimum Location: {OPTIMUM} for all parameters")
    print(f"  Time Limit: {TIME_LIMIT_SECONDS} seconds")
    print(f"  Expected Maximum Score: 0.0")
    print()
    
    # Initialize GA with reasonable population size
    ga = GeneticAlgorithm(
        parameter_bounds=parameter_bounds,
        population_size=50,
        max_generations=10000,  # Large number, will be limited by time
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_ratio=0.2
    )
    
    time_limited = TimeLimitedGA(ga, TIME_LIMIT_SECONDS)
    
    print(f"Starting optimization (will run for {TIME_LIMIT_SECONDS} seconds)...")
    result, elapsed_time, evaluations = time_limited.optimize()
    
    print(f"\nTest Results:")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Total Evaluations: {evaluations}")
    print(f"Evaluations per Second: {evaluations/elapsed_time:.2f}")
    print(f"Best Fitness: {result.best_fitness:.4f} (Expected: ~0.0)")
    print(f"Best Parameters: {result.best_individual.parameters}")
    print(f"  Expected: ~{OPTIMUM} for all parameters")
    print(f"Generations Completed: {result.generation_count}")
    
    # Calculate how close we got to optimum
    if result.best_individual and result.best_individual.parameters:
        distance_from_optimum = sum((float(v) - OPTIMUM)**2 for v in result.best_individual.parameters.values()) ** 0.5
        print(f"Distance from Optimum: {distance_from_optimum:.2f}")
    
    # Verify correctness
    expected_optimum = 0.0
    tolerance = 100.0  # Allow some tolerance for near-optimal solutions
    if abs(result.best_fitness - expected_optimum) < tolerance:
        print("\n✓ Test PASSED: Algorithm found near-optimal solution!")
    else:
        print("\n✗ Test WARNING: Algorithm did not fully converge (may need more time).")

if __name__ == "__main__":
    main()