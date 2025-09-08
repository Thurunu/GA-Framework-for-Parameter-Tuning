#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Bayesian Optimization
This module implements Bayesian Optimization for kernel parameter tuning
"""

from scipy.optimize import minimize
from scipy.stats import norm
import math
import numpy as np
import json
import time
# For type hints to improve code clarity and safety
from typing import Dict, List, Tuple, Optional, Callable, Any
# For defining simple classes to store structured results (OptimizationResult)
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# For Gaussian Process - using simplified implementation


@dataclass
class OptimizationResult:
    """
    Purpose: Stores and organizes the results of a Bayesian Optimization run.
    Use case: Returned after optimization to summarize best parameters, score, history, and convergence info.
    """
    best_parameters: Dict[str,
                          Any]  # the parameter values which acheived the highest score
    best_score: float  # highest score find during optimization
    iteration_count: int  # total number of optimization iterations performed
    # a list of parameter sets evaluated with their corresponding scores
    evaluation_history: List[Tuple[Dict[str, Any], float]]
    convergence_reached: bool  # checked the optimization met the convergence criteria
    optimization_time: float  # total time taken to optimization process


class GaussianProcess:
    """
    Purpose: Models the relationship between parameters and performance scores using Gaussian Process regression.
    Use case: Acts as a surrogate model in Bayesian Optimization, predicting expected score and uncertainty for new parameter sets.
    """

    def __init__(self, kernel_variance=1.0, kernel_lengthscale=1.0, noise_variance=1e-6):
        """
        Purpose: Initialize GP hyperparameters and storage for training data.
        Use case: Called when creating a new GaussianProcess instance.
        """
        self.kernel_variance = kernel_variance
        self.kernel_lengthscale = kernel_lengthscale
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _rbf_kernel(self, X1, X2):
        """
        Purpose: Compute the RBF (squared exponential) kernel for covariance calculations.
        Use case: Used internally for measuring similarity between parameter sets.
        """
        """(Radial Basis Function) RBF (Squared Exponential) kernel, this measures similarity between parameter sets. 
        this is for GP convariance calculations"""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
            np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)

        return self.kernel_variance * np.exp(-0.5 / self.kernel_lengthscale**2 * sqdist)

    def fit(self, X, y):
        """
        Purpose: Fit the GP to training data (parameters and scores).
        Use case: Called to train the model before making predictions.
        """
        """Fit the Gaussian Process to training data"""
        """Fits the GP to training data (parameter sets and their scores). 
        It computes the kernel matrix and its inverse, which are used for making predictions."""
        self.X_train = np.atleast_2d(X)
        self.y_train = np.array(y).flatten()

        # Compute kernel matrix
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(len(self.X_train))

        # Compute inverse (with regularization for numerical stability)
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Add regularization if matrix is singular
            K += 1e-6 * np.eye(len(self.X_train))
            self.K_inv = np.linalg.inv(K)

    def predict(self, X_test, return_std=True):
        """
        Purpose: Predict mean and uncertainty for new parameter sets using the trained GP.
        Use case: Used by the optimizer to evaluate candidate parameters.
        """
        """Make predictions with uncertainty estimates"""
        """Given new parameter sets, predicts the mean (expected score) and standard deviation (uncertainty)
        using the trained GP. This is essential for acquisition functions to balance exploration and exploitation."""
        X_test = np.atleast_2d(X_test)

        if self.X_train is None:
            # No training data yet
            mean = np.zeros(len(X_test))
            if return_std:
                std = np.sqrt(self.kernel_variance) * np.ones(len(X_test))
                return mean, std
            return mean

        # Compute mean prediction
        K_s = self._rbf_kernel(self.X_train, X_test)
        mean = np.dot(K_s.T, np.dot(self.K_inv, self.y_train))

        if not return_std:
            return mean

        # Compute variance prediction
        K_ss = self._rbf_kernel(X_test, X_test)
        variance = np.diag(K_ss) - np.sum(K_s * np.dot(self.K_inv, K_s), axis=0)
        variance = np.maximum(variance, 1e-10)  # Ensure positive variance
        std = np.sqrt(variance)

        return mean, std


class AcquisitionFunction:
    """
    Purpose: Provides acquisition functions for Bayesian Optimization. used to decide which candidate parameters should be evaluated next. 
    Use case: Used to select the next candidate parameters to evaluate based on GP predictions.
    """

    @staticmethod
    def expected_improvement(mean, std, f_best, xi=0.01):
        """
        Purpose: Calculate Expected Improvement for candidate parameters. 
        Use case: Guides optimizer to balance exploration and exploitation.
        """
        improvement = mean - f_best - xi
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0.0] = 0.0
        return ei

    @staticmethod
    def upper_confidence_bound(mean, std, kappa=2.576):
        """
        Purpose: Calculate Upper Confidence Bound for candidate parameters.
        Use case: Encourages exploration of uncertain regions.
        """
        return mean + kappa * std

    @staticmethod
    def probability_of_improvement(mean, std, f_best, xi=0.01):
        """
        Purpose: Calculate Probability of Improvement for candidate parameters.
        Use case: Selects parameters likely to improve over current best.
        """
        improvement = mean - f_best - xi
        Z = improvement / std
        return norm.cdf(Z)


class BayesianOptimizer:
    """
    Purpose: Implements Bayesian Optimization for kernel parameter tuning.
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
        """
        """
        Initialize Bayesian Optimizer

        Args:
            parameter_bounds: Dict of parameter names to (min, max) bounds
            acquisition_function: 'ei', 'ucb', or 'pi'
            initial_samples: Number of random initial samples
            max_iterations: Maximum optimization iterations
            convergence_threshold: Convergence threshold for optimization
            random_seed: Random seed for reproducibility
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.bounds_array = np.array(
            [parameter_bounds[name] for name in self.parameter_names])

        self.acquisition_function = acquisition_function
        self.initial_samples = initial_samples
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # Initialize random number generator
        np.random.seed(random_seed)

        # Initialize Gaussian Process
        self.gp = GaussianProcess()

        # Storage for optimization history
        self.X_history = []
        self.y_history = []
        self.best_score = -np.inf
        self.best_parameters = None

        # Acquisition function mapping
        self.acq_functions = {
            'ei': AcquisitionFunction.expected_improvement,
            'ucb': AcquisitionFunction.upper_confidence_bound,
            'pi': AcquisitionFunction.probability_of_improvement
        }

    def _normalize_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Purpose: Normalize parameters to [0, 1] range for optimization.
        Use case: Used internally before optimization steps.
        """
        """Normalize parameters to [0, 1] range"""
        return (parameters - self.bounds_array[:, 0]) / (self.bounds_array[:, 1] - self.bounds_array[:, 0])

    def _denormalize_parameters(self, normalized_params: np.ndarray) -> np.ndarray:
        """
        Purpose: Convert normalized parameters back to their original scale.
        Use case: Used internally to interpret results in real parameter space.
        """
        """Denormalize parameters from [0, 1] to original range"""
        return normalized_params * (self.bounds_array[:, 1] - self.bounds_array[:, 0]) + self.bounds_array[:, 0]

    def _parameters_to_dict(self, param_array: np.ndarray) -> Dict[str, float]:
        """
        Purpose: Convert parameter array to dictionary format.
        Use case: Used for compatibility with objective functions and result storage.
        """
        """Convert parameter array to dictionary"""
        return {name: float(param_array[i]) for i, name in enumerate(self.parameter_names)}

    def _dict_to_parameters(self, param_dict: Dict[str, float]) -> np.ndarray:
        """
        Purpose: Convert parameter dictionary to array format.
        Use case: Used internally for optimization calculations.
        """
        """Convert parameter dictionary to array"""
        return np.array([param_dict[name] for name in self.parameter_names])

    def _generate_initial_samples(self) -> List[np.ndarray]:
        """
        Purpose: Generate initial random samples for optimization.
        Use case: Used to start the optimization process with diverse parameter sets.
        """
        """Generate initial random samples"""
        samples = []
        for _ in range(self.initial_samples):
            # Generate random samples within bounds
            sample = np.random.uniform(
                self.bounds_array[:, 0],
                self.bounds_array[:, 1]
            )
            samples.append(sample)
        return samples

    def _optimize_acquisition(self, acquisition_func: Callable) -> np.ndarray:
        """
        Purpose: Find parameters that maximize the acquisition function.
        Use case: Used to suggest the next candidate parameters to evaluate.
        """
        """Optimize acquisition function to find next evaluation point"""

        def objective(x_norm):
            """Objective function to minimize (negative acquisition)"""
            x = self._denormalize_parameters(x_norm)
            mean, std = self.gp.predict([self._normalize_parameters(x)])

            if self.acquisition_function == 'ei':
                acq_val = acquisition_func(mean[0], std[0], self.best_score)
            elif self.acquisition_function == 'ucb':
                acq_val = acquisition_func(mean[0], std[0])
            else:  # pi
                acq_val = acquisition_func(mean[0], std[0], self.best_score)

            return -acq_val  # Minimize negative acquisition

        # Multiple random starts for global optimization
        best_x = None
        best_val = np.inf

        for _ in range(10):  # 10 random starts
            x0 = np.random.uniform(0, 1, len(self.parameter_names))

            try:
                result = minimize(
                    objective, x0,
                    bounds=[(0, 1)] * len(self.parameter_names),
                    method='L-BFGS-B'
                )

                if result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x
            except:
                continue

        if best_x is None:
            # Fallback to random sampling
            best_x = np.random.uniform(0, 1, len(self.parameter_names))

        return self._denormalize_parameters(best_x)

    def suggest_next_parameters(self) -> Dict[str, float]:
        """
        Purpose: Suggest the next parameters to evaluate based on acquisition function.
        Use case: Called during each optimization iteration.
        """
        """Suggest next parameters to evaluate"""

        if len(self.X_history) < self.initial_samples:
            # Generate initial random samples
            remaining_samples = self.initial_samples - len(self.X_history)
            initial_samples = self._generate_initial_samples()
            next_params = initial_samples[len(self.X_history)]
        else:
            # Use acquisition function to suggest next point
            acquisition_func = self.acq_functions[self.acquisition_function]
            next_params = self._optimize_acquisition(acquisition_func)

        return self._parameters_to_dict(next_params)

    def update_with_result(self, parameters: Dict[str, float], score: float):
        """
        Purpose: Update the optimizer with the result of an evaluated parameter set.
        Use case: Called after each objective function evaluation.
        """
        """Update optimizer with evaluation result"""
        param_array = self._dict_to_parameters(parameters)

        # Store in history
        self.X_history.append(param_array)
        self.y_history.append(score)

        # Update best result
        if score > self.best_score:
            self.best_score = score
            self.best_parameters = parameters.copy()

        # Update Gaussian Process
        if len(self.X_history) >= 2:
            X_normalized = np.array(
                [self._normalize_parameters(x) for x in self.X_history])
            self.gp.fit(X_normalized, self.y_history)

    def optimize(self, objective_function: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        """
        Purpose: Run the full Bayesian Optimization loop.
        Use case: Main method to perform optimization and return results.
        """
        """Run complete Bayesian Optimization"""

        start_time = time.time()
        iteration_count = 0
        convergence_reached = False
        evaluation_history = []

        print(
            f"Starting Bayesian Optimization with {self.max_iterations} iterations...")

        for iteration in range(self.max_iterations):
            iteration_count += 1

            # Get next parameters to evaluate
            next_params = self.suggest_next_parameters()

            # Evaluate objective function
            try:
                score = objective_function(next_params)
                print(
                    f"Iteration {iteration + 1}: Score = {score:.6f}, Params = {next_params}")

                # Update optimizer
                self.update_with_result(next_params, score)
                evaluation_history.append((next_params.copy(), score))

                # Check for convergence
                if len(self.y_history) >= 3:
                    recent_improvement = max(self.y_history[-3:]) - max(
                        self.y_history[-6:-3]) if len(self.y_history) >= 6 else float('inf')
                    if abs(recent_improvement) < self.convergence_threshold:
                        convergence_reached = True
                        print(
                            f"Convergence reached at iteration {iteration + 1}")
                        break

            except Exception as e:
                print(
                    f"Error evaluating parameters at iteration {iteration + 1}: {e}")
                continue

        optimization_time = time.time() - start_time

        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best score: {self.best_score:.6f}")
        print(f"Best parameters: {self.best_parameters}")

        return OptimizationResult(
            best_parameters=self.best_parameters or {},
            best_score=self.best_score,
            iteration_count=iteration_count,
            evaluation_history=evaluation_history,
            convergence_reached=convergence_reached,
            optimization_time=optimization_time
        )
    
    def adaptive_optimize(self, objective_function, max_iterations=50, max_restarts=10, required_stable_runs=3, tolerance=1e-4):
        stable_count = 0
        previous_best = None
        all_results = []
        self.max_iterations = max_iterations
        for _ in range(max_restarts):
            result = self.optimize(objective_function)
            all_results.append(result)
            if previous_best and self.is_similar(result.best_parameters, previous_best, tolerance):
                stable_count += 1
            else:
                stable_count = 1
                previous_best = result.best_parameters
            if stable_count >= required_stable_runs:
                break
        return result

    def is_similar(self, params1, params2, tolerance):
        return all(abs(params1[k] - params2[k]) < tolerance for k in params1)

    def get_posterior_statistics(self, test_points: Optional[List[Dict[str, float]]] = None) -> Dict:
        """
        Purpose: Get statistics (mean, std) from the GP for given test points.
        Use case: Used for analysis and visualization of the model's predictions.
        """
        """Get posterior statistics from Gaussian Process"""
        if not self.X_history:
            return {"error": "No training data available"}

        if test_points is None:
            # Generate test points
            n_test = 100
            test_points_array = []
            for _ in range(n_test):
                test_point = np.random.uniform(
                    self.bounds_array[:, 0],
                    self.bounds_array[:, 1]
                )
                test_points_array.append(test_point)
        else:
            test_points_array = [
                self._dict_to_parameters(tp) for tp in test_points]

        # Get predictions
        X_test_norm = np.array([self._normalize_parameters(x)
                               for x in test_points_array])
        means, stds = self.gp.predict(X_test_norm)

        return {
            "mean_prediction": np.mean(means),
            "std_prediction": np.mean(stds),
            "max_uncertainty": np.max(stds),
            "min_uncertainty": np.min(stds),
            "predicted_best": np.max(means),
            "predicted_best_params": self._parameters_to_dict(test_points_array[np.argmax(means)])
        }

    def export_results(self, filename: str):
        """
        Purpose: Export optimization results to a file.
        Use case: Used to save results for later analysis or reporting.
        """
        """Export optimization results to JSON file"""
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
                {"parameters": params, "score": score}
                for params, score in zip(
                    [self._parameters_to_dict(x) for x in self.X_history],
                    self.y_history
                )
            ],
            "posterior_statistics": self.get_posterior_statistics()
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results exported to {filename}")


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
        """Mock objective function for testing"""
        # Simulate performance score based on parameters
        # In reality, this would run benchmarks and return performance metrics

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
    optimizer = BayesianOptimizer(
        parameter_bounds=parameter_bounds,
        acquisition_function='ei',
        initial_samples=5,
        max_iterations=20
    )

    # Run optimization
    result = optimizer.optimize(mock_objective_function)

    print("\nOptimization Results:")
    print(f"Best Score: {result.best_score:.6f}")
    print(f"Best Parameters: {result.best_parameters}")
    print(f"Iterations: {result.iteration_count}")
    print(f"Converged: {result.convergence_reached}")
    print(f"Time: {result.optimization_time:.2f} seconds")

    # Export results
    optimizer.export_results("bayesian_optimization_results.json")

    # Get posterior statistics
    stats = optimizer.get_posterior_statistics()
    print(f"\nPosterior Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
