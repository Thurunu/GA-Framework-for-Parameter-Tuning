"""
Test for BayesianOptimzation.py
This script tests the BayesianOptimizer class with a mock objective function.
"""

from src.BayesianOptimzation import BayesianOptimizer
import random

def mock_objective_function(params):
    """
    Simulates an objective function for testing.
    Returns a score based on a simple mathematical relationship.
    """
    # Example: maximize negative sum of squares (minimum at all zeros)
    score = -sum((float(v) - 50)**2 for v in params.values())
    # Add some noise
    score += random.uniform(-5, 5)
    return score

if __name__ == "__main__":
    # Define parameter bounds for testing
    parameter_bounds = {
        'param1': (0, 100),
        'param2': (0, 100),
        'param3': (0, 100)
    }

    optimizer = BayesianOptimizer(
        parameter_bounds=parameter_bounds,
        acquisition_function='ei',
        initial_samples=5,
        max_iterations=100
    )

    result = optimizer.optimize(mock_objective_function)

    print("\nTest Results:")
    print(f"Best Score: {result.best_score:.4f}")
    print(f"Best Parameters: {result.best_parameters}")
    print(f"Iterations: {result.iteration_count}")
    print(f"Converged: {result.convergence_reached}")
    print(f"Time: {result.optimization_time:.2f} seconds")

    # Optionally export results
    optimizer.export_results("test_bayesian_optimization_results.json")
