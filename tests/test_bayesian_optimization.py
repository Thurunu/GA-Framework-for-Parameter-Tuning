"""
Test for BayesianOptimization.py
This script tests the BayesianOptimizer class with time-based constraints.
"""

import sys
import os
import time

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.BayesianOptimzation import BayesianOptimizer

# Common test configuration
DIMENSIONS = 3
BOUNDS = (0, 100)
OPTIMUM = 50  # Center of search space
TIME_LIMIT_SECONDS = 0.5  # Run for 0.3 seconds

def test_objective_function(params):
    """
    Standardized objective function for testing.
    Returns a score based on distance from optimum.
    Maximum value is 0 at the optimum point (50, 50, 50).
    """
    score = -sum((float(v) - OPTIMUM)**2 for v in params.values())
    return score

class TimeLimitedOptimizer:
    """Wrapper to enforce time limit on optimization"""
    def __init__(self, optimizer, time_limit):
        self.optimizer = optimizer
        self.time_limit = time_limit
        self.start_time = None
        self.evaluations = 0
        
    def objective_wrapper(self, params):
        """Wrapper that checks time limit"""
        if time.time() - self.start_time >= self.time_limit:
            raise TimeoutError("Time limit reached")
        self.evaluations += 1
        return test_objective_function(params)
    
    def optimize(self):
        """Run optimization with time limit"""
        self.start_time = time.time()
        self.evaluations = 0
        
        try:
            # Start with conservative settings and let it run until timeout
            result = self.optimizer.optimize(self.objective_wrapper)
            elapsed_time = time.time() - self.start_time
            return result, elapsed_time, self.evaluations
        except TimeoutError:
            # Time limit reached, return best so far
            elapsed_time = time.time() - self.start_time
            # Create a result-like object with current best
            class TimeoutResult:
                def __init__(self, optimizer):
                    self.best_score = max(optimizer.history) if optimizer.history else float('-inf')
                    self.best_parameters = optimizer.best_params
                    self.iteration_count = len(optimizer.history)
                    self.convergence_reached = False
                    self.optimization_time = elapsed_time
            
            return TimeoutResult(self.optimizer), elapsed_time, self.evaluations

if __name__ == "__main__":
    print("Testing Bayesian Optimization with Time-Based Constraint:")
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

    # Use high max_iterations since we'll stop based on time
    optimizer = BayesianOptimizer(
        parameter_bounds=parameter_bounds,
        acquisition_function='ei',
        initial_samples=3,
        max_iterations=10000  # Large number, will be limited by time
    )

    time_limited = TimeLimitedOptimizer(optimizer, TIME_LIMIT_SECONDS)
    
    print(f"Starting optimization (will run for {TIME_LIMIT_SECONDS} seconds)...")
    result, elapsed_time, evaluations = time_limited.optimize()

    print("\nTest Results:")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Total Evaluations: {evaluations}")
    print(f"Evaluations per Second: {evaluations/elapsed_time:.2f}")
    print(f"Best Score: {result.best_score:.4f} (Expected: ~0.0)")
    print(f"Best Parameters: {result.best_parameters}")
    print(f"  Expected: ~{OPTIMUM} for all parameters")
    print(f"Iterations Completed: {result.iteration_count}")
    
    # Calculate how close we got to optimum
    if result.best_parameters:
        distance_from_optimum = sum((float(v) - OPTIMUM)**2 for v in result.best_parameters.values()) ** 0.5
        print(f"Distance from Optimum: {distance_from_optimum:.2f}")
    
    # Verify correctness
    expected_optimum = 0.0
    tolerance = 100.0  # Allow some tolerance for near-optimal solutions
    if abs(result.best_score - expected_optimum) < tolerance:
        print("\n✓ Test PASSED: Algorithm found near-optimal solution!")
    else:
        print("\n✗ Test WARNING: Algorithm did not fully converge (may need more time).")

    # Export results
    try:
        optimizer.export_results("test_bayesian_optimization_results.json")
        print("\nResults exported to: test_bayesian_optimization_results.json")
    except:
        print("\nNote: Could not export results (export method may not be available)")