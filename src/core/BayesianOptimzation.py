#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Bayesian Optimization
This module implements Bayesian Optimization for kernel parameter tuning using scikit-optimize (skopt)
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import warnings
import os
import sys


# Scikit-optimize for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import dump, load
from DataClasses import OptimizationResult
warnings.filterwarnings('ignore')


project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'data'))


class BayesianOptimizer:
    """
    Purpose: Implements Bayesian Optimization for kernel parameter tuning using scikit-optimize.
    Use case: Manages the optimization loop, parameter suggestions, and result collection.
    """

    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 acquisition_function: str = 'ei',
                 initial_samples: int = 5,
                 max_iterations: int = 50,
                 convergence_threshold: float = 1e-6,
                 random_seed: int = 42):
        """
        Purpose: Initialize optimizer settings and parameter bounds.
        Use case: Called when creating a new BayesianOptimizer instance.
        
        Args:
            parameter_bounds: Dict of parameter names to (min, max) bounds
            acquisition_function: 'ei' (Expected Improvement), 'gp_hedge', or 'EI'
            initial_samples: Number of random initial samples
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for optimization
            random_seed: Random seed for reproducibility
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        
        # Convert bounds to skopt space
        self.space = [
            Real(low=bounds[0], high=bounds[1], name=name) 
            for name, bounds in parameter_bounds.items()
        ]
        
        # Map acquisition function names
        acq_map = {
            'ei': 'EI',
            'ucb': 'LCB',  # skopt uses LCB (Lower Confidence Bound) for minimization
            'pi': 'PI',
            'gp_hedge': 'gp_hedge'
        }
        self.acquisition_function = acq_map.get(acquisition_function.lower(), 'EI')
        
        self.initial_samples = initial_samples
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.random_seed = random_seed
        
        # Storage for optimization history
        self.best_score = -np.inf
        self.best_parameters = None
        self.optimization_result = None

    def _parameters_to_dict(self, param_list: List[float]) -> Dict[str, float]:
        """
        Purpose: Convert parameter list to dictionary format with integer rounding.
        Use case: Used for compatibility with objective functions.
        Note: Kernel parameters only accept integer values, so all outputs are rounded.
        """
        return {name: int(round(value)) for name, value in zip(self.parameter_names, param_list)}

    def optimize(self, objective_function: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        """
        Purpose: Run the full Bayesian Optimization loop using scikit-optimize.
        Use case: Main method to perform optimization and return results.
        """
        start_time = time.time()
        evaluation_history = []
        
        print(f"Starting Bayesian Optimization with {self.max_iterations} iterations...")
        print(f"Using acquisition function: {self.acquisition_function}")
        
        # Wrapper for objective function (skopt minimizes, so we negate the score)
        @use_named_args(self.space)
        def objective_wrapper(**params):
            try:
                # Evaluate objective function (higher is better)
                score = objective_function(params)
                
                # Store in history
                evaluation_history.append((params.copy(), score))
                
                # Update best result
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = params.copy()
                
                print(f"Iteration {len(evaluation_history)}: Score = {score:.6f}")
                
                # Return negative score (skopt minimizes)
                return -score
                
            except Exception as e:
                print(f"Error evaluating parameters: {e}")
                return 0.0  # Return neutral score on error
        
        # Run Bayesian Optimization
        self.optimization_result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.space,
            acq_func=self.acquisition_function,
            n_calls=self.max_iterations,
            n_initial_points=self.initial_samples,
            random_state=self.random_seed,
            verbose=False
        )
        
        optimization_time = time.time() - start_time
        
        # Check for convergence (simple heuristic: improvement in last few iterations)
        convergence_reached = False
        if len(evaluation_history) >= 6:
            recent_scores = [score for _, score in evaluation_history[-6:]]
            improvement = max(recent_scores[-3:]) - max(recent_scores[:3])
            convergence_reached = abs(improvement) < self.convergence_threshold
        
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Best score: {self.best_score:.6f}")
        print(f"Best parameters: {self.best_parameters}")
        
        return OptimizationResult(
            best_parameters=self.best_parameters or {},
            best_score=self.best_score,
            iteration_count=len(evaluation_history),
            evaluation_history=evaluation_history,
            convergence_reached=convergence_reached,
            optimization_time=optimization_time
        )
    
    def adaptive_optimize(self, objective_function: Callable[[Dict[str, float]], float], 
                         max_iterations: int = 50, 
                         max_restarts: int = 10, 
                         required_stable_runs: int = 3, 
                         tolerance: float = 1e-4) -> OptimizationResult:
        """
        Purpose: Run adaptive optimization with multiple restarts for stability.
        Use case: Used when robustness is more important than speed.
        """
        stable_count = 0
        previous_best = None
        all_results = []
        self.max_iterations = max_iterations
        
        for restart in range(max_restarts):
            print(f"\n--- Adaptive Restart {restart + 1}/{max_restarts} ---")
            result = self.optimize(objective_function)
            all_results.append(result)
            
            if previous_best and self._is_similar(result.best_parameters, previous_best, tolerance):
                stable_count += 1
                print(f"Stable count: {stable_count}/{required_stable_runs}")
            else:
                stable_count = 1
                previous_best = result.best_parameters
            
            if stable_count >= required_stable_runs:
                print(f"Convergence achieved after {restart + 1} restarts")
                break
        
        return result

    def _is_similar(self, params1: Dict[str, float], params2: Dict[str, float], tolerance: float) -> bool:
        """
        Purpose: Check if two parameter sets are similar within tolerance.
        Use case: Used in adaptive optimization to detect convergence.
        """
        return all(abs(params1.get(k, 0) - params2.get(k, 0)) < tolerance for k in params1)

    def get_posterior_statistics(self) -> Dict:
        """
        Purpose: Get statistics from the Gaussian Process model.
        Use case: Used for analysis and visualization of the model's predictions.
        """
        if self.optimization_result is None:
            return {"error": "No optimization results available"}
        
        # Extract information from skopt result
        return {
            "n_evaluations": len(self.optimization_result.x_iters),
            "best_score": -self.optimization_result.fun,  # Negate back to original score
            "best_params": self._parameters_to_dict(self.optimization_result.x),
            "model_type": str(type(self.optimization_result.models[-1]).__name__) if self.optimization_result.models else "Unknown"
        }

    def export_results(self, filename: str):
        """
        Purpose: Export optimization results to a JSON file.
        Use case: Used to save results for later analysis or reporting.
        """
        if self.optimization_result is None:
            print("No optimization results to export")
            return
        
        results_data = {
            "optimization_settings": {
                "parameter_bounds": self.parameter_bounds,
                "acquisition_function": self.acquisition_function,
                "initial_samples": self.initial_samples,
                "max_iterations": self.max_iterations
            },
            "best_result": {
                "parameters": self.best_parameters,
                "score": self.best_score
            },
            "evaluation_history": [
                {
                    "iteration": i + 1,
                    "parameters": self._parameters_to_dict(x),
                    "score": -f  # Negate back to original score
                }
                for i, (x, f) in enumerate(zip(
                    self.optimization_result.x_iters,
                    self.optimization_result.func_vals
                ))
            ],
            "posterior_statistics": self.get_posterior_statistics()
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results exported to {filename}")
    
    def save_model(self, filename: str):
        """
        Purpose: Save the trained model to a file for later use.
        Use case: Allows resuming optimization or using the model for predictions.
        """
        if self.optimization_result is None:
            print("No optimization results to save")
            return
        
        dump(self.optimization_result, filename)
        print(f"Model saved to {filename}")
    
    def suggest_next_parameters(self) -> Dict[str, float]:
        """
        Purpose: Suggest next parameters based on current model (for manual iteration).
        Use case: Allows manual control over the optimization loop.
        Note: Returns integer values for kernel parameters.
        """
        if self.optimization_result is None:
            # Generate random initial sample with integer rounding
            return {
                name: int(round(np.random.uniform(bounds[0], bounds[1])))
                for name, bounds in self.parameter_bounds.items()
            }
        
        # Use the model to suggest next point
        # Note: This is a simplified approach; full implementation would use ask/tell interface
        print("Warning: suggest_next_parameters is simplified. Use optimize() for full functionality.")
        return self.best_parameters


# Example usage and testing
if __name__ == "__main__":
    # Define parameter bounds for kernel optimization
    parameter_bounds = {
        'vm.swappiness': (0, 100),
        'vm.dirty_ratio': (1, 90),
        'net.core.rmem_max': (8192, 1048576),
        'kernel.sched_min_granularity_ns': (1000000, 50000000)
    }

    # Create mock objective function (in real implementation, this would evaluate system performance)
    def mock_objective_function(params: Dict[str, float]) -> float:
        """
        Mock objective function for testing.
        In reality, this would run benchmarks and return performance metrics.
        """
        swappiness = params['vm.swappiness']
        dirty_ratio = params['vm.dirty_ratio']
        rmem_max = params['net.core.rmem_max']
        sched_gran = params['kernel.sched_min_granularity_ns']

        # Mock scoring function (higher is better)
        score = (
            -0.01 * (swappiness - 30)**2 +  # Optimal around 30
            -0.001 * (dirty_ratio - 20)**2 +  # Optimal around 20
            0.00001 * rmem_max +  # Higher is generally better
            -0.000000001 * (sched_gran - 10000000)**2  # Optimal around 10ms
        ) + np.random.normal(0, 0.1)  # Add some noise

        return score

    # Initialize and run Bayesian Optimizer
    print("=" * 80)
    print("Bayesian Optimization with scikit-optimize (skopt)")
    print("=" * 80)
    
    optimizer = BayesianOptimizer(
        parameter_bounds=parameter_bounds,
        acquisition_function='ei',
        initial_samples=5,
        max_iterations=20,
        random_seed=42
    )

    # Run optimization
    result = optimizer.optimize(mock_objective_function)

    print("\n" + "=" * 80)
    print("Optimization Results:")
    print("=" * 80)
    print(f"Best Score: {result.best_score:.6f}")
    print(f"Best Parameters:")
    for param, value in result.best_parameters.items():
        print(f"  {param}: {value:.2f}")
    print(f"Iterations: {result.iteration_count}")
    print(f"Converged: {result.convergence_reached}")
    print(f"Time: {result.optimization_time:.2f} seconds")

    # Export results
    optimizer.export_results("bayesian_optimization_results.json")
    print("\nResults exported to bayesian_optimization_results.json")

    # Get posterior statistics
    stats = optimizer.get_posterior_statistics()
    print("\n" + "=" * 80)
    print("Model Statistics:")
    print("=" * 80)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.2f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:.6f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    # Save the model for future use
    optimizer.save_model("bayesian_model.pkl")
    print("\nModel saved to bayesian_model.pkl")

