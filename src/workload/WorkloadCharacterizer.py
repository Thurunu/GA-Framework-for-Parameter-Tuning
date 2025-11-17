#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Workload Characterizer
Analyzes workload characteristics to suggest optimization strategy
"""

import numpy as np
from typing import Dict, Tuple, Any
from enum import Enum


class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    BAYESIAN_ONLY = "bayesian_only"
    GENETIC_ONLY = "genetic_only"
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    HYBRID_PARALLEL = "hybrid_parallel"
    ADAPTIVE = "adaptive"


class WorkloadCharacterizer:
    """Analyzes workload characteristics to suggest optimization strategy"""

    @staticmethod
    def analyze_parameter_space(parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze characteristics of parameter space
        
        Args:
            parameter_bounds: Dictionary of parameter bounds
            
        Returns:
            Dictionary containing analysis metrics:
            - num_parameters: Number of parameters
            - space_size: Total parameter space size
            - avg_range_ratio: Average ratio of parameter ranges
            - dimensionality: Classification of dimensionality (low/medium/high)
            - complexity: Classification of complexity (low/medium/high)
        """
        num_params = len(parameter_bounds)
        
        # Calculate parameter space size
        space_size = 1
        range_ratios = []

        for _, (min_val, max_val) in parameter_bounds.items():
            param_range = max_val - min_val
            space_size *= param_range

            # Calculate range ratio (how large is the range)
            if min_val != 0:
                range_ratios.append(param_range / abs(min_val))
            else:
                range_ratios.append(param_range)

        avg_range_ratio = np.mean(range_ratios) if range_ratios else 1.0

        return {
            "num_parameters": num_params,
            "space_size": space_size,
            "avg_range_ratio": avg_range_ratio,
            "dimensionality": "high" if num_params > 10 else "medium" if num_params > 5 else "low",
            "complexity": "high" if space_size > 1e12 else "medium" if space_size > 1e6 else "low"
        }

    @staticmethod
    def suggest_strategy(parameter_bounds: Dict[str, Tuple[float, float]],
                         evaluation_budget: int = 100,
                         time_budget: float = 300.0) -> OptimizationStrategy:
        """
        Suggests an optimization strategy based on the characteristics of the parameter space 
        and resource constraints.
        
        This function analyzes the provided parameter bounds using WorkloadCharacterizer to 
        determine the number of parameters and the complexity of the optimization problem. 
        Based on this analysis, along with the evaluation and time budgets, it selects an 
        appropriate optimization strategy from the following options:
        
        Strategy Selection Logic:
        - BAYESIAN_ONLY: For small parameter spaces (<= 5 parameters) and limited evaluation 
          budget (<= 50), or when the time budget is very short (< 60 seconds).
        - GENETIC_ONLY: For large parameter spaces (> 15 parameters) or when the problem 
          complexity is high.
        - HYBRID_SEQUENTIAL: For cases with a large evaluation budget (> 200).
        - ADAPTIVE: Default strategy for other scenarios.
        
        Args:
            parameter_bounds: Dictionary specifying the bounds for each parameter.
            evaluation_budget: Maximum number of allowed evaluations. Defaults to 100.
            time_budget: Maximum allowed optimization time in seconds. Defaults to 300.0.
        
        Returns:
            OptimizationStrategy: The suggested optimization strategy based on the problem 
            characteristics.
        """
        analysis = WorkloadCharacterizer.analyze_parameter_space(parameter_bounds)

        # Decision logic based on problem characteristics
        if analysis["num_parameters"] <= 5 and evaluation_budget <= 50:
            return OptimizationStrategy.BAYESIAN_ONLY

        elif analysis["num_parameters"] > 15 or analysis["complexity"] == "high":
            return OptimizationStrategy.GENETIC_ONLY

        elif time_budget < 60:  # Short time budget
            return OptimizationStrategy.BAYESIAN_ONLY

        elif evaluation_budget > 200:  # Large budget
            return OptimizationStrategy.HYBRID_SEQUENTIAL

        else:
            return OptimizationStrategy.ADAPTIVE

    @staticmethod
    def get_recommended_population_size(num_parameters: int, 
                                       evaluation_budget: int) -> int:
        """
        Get recommended GA population size based on parameters
        
        Args:
            num_parameters: Number of optimization parameters
            evaluation_budget: Available evaluation budget
            
        Returns:
            Recommended population size
        """
        # Rule of thumb: 10-20 individuals per parameter
        min_pop = max(10, num_parameters * 10)
        max_pop = min(100, evaluation_budget // 4)
        
        return min(max(min_pop, 20), max_pop)

    @staticmethod
    def get_recommended_generations(population_size: int, 
                                   evaluation_budget: int) -> int:
        """
        Get recommended number of GA generations
        
        Args:
            population_size: GA population size
            evaluation_budget: Available evaluation budget
            
        Returns:
            Recommended number of generations
        """
        if population_size == 0:
            return 1
        
        return max(1, evaluation_budget // population_size)

    @staticmethod
    def get_recommended_bayesian_samples(num_parameters: int,
                                        evaluation_budget: int) -> int:
        """
        Get recommended number of initial Bayesian samples
        
        Args:
            num_parameters: Number of optimization parameters
            evaluation_budget: Available evaluation budget
            
        Returns:
            Recommended number of initial samples
        """
        # Rule of thumb: 2-5 samples per parameter
        min_samples = max(5, num_parameters * 2)
        max_samples = min(20, evaluation_budget // 5)
        
        return min(min_samples, max_samples)


