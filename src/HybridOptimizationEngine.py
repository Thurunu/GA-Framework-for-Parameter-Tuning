#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Hybrid Optimization Engine
This module combines Bayesian Optimization and Genetic Algorithm intelligently
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import copy

# Import our optimization components
from BayesianOptimzation import BayesianOptimizer, OptimizationResult
from GeneticAlgorithm import GeneticAlgorithm, AdvancedGeneticAlgorithm, GAOptimizationResult, Individual
from ProcessPriorityManager import ProcessPriorityManager, PriorityClass

class OptimizationStrategy(Enum):
    """Optimization strategy types"""
    BAYESIAN_ONLY = "bayesian_only"
    GENETIC_ONLY = "genetic_only" 
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    HYBRID_PARALLEL = "hybrid_parallel"
    ADAPTIVE = "adaptive"

@dataclass
class HybridOptimizationResult:
    """Results from hybrid optimization"""
    best_parameters: Dict[str, Any]
    best_score: float
    strategy_used: OptimizationStrategy
    total_evaluations: int
    bayesian_results: Optional[OptimizationResult] = None
    genetic_results: Optional[GAOptimizationResult] = None
    optimization_time: float = 0.0
    convergence_reached: bool = False
    switch_points: List[Tuple[int, str]] = None

class WorkloadCharacterizer:
    """Analyzes workload characteristics to suggest optimization strategy"""
    
    @staticmethod
    def analyze_parameter_space(parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze characteristics of parameter space"""
        num_params = len(parameter_bounds)
        
        # Calculate parameter space size
        space_size = 1
        range_ratios = []
        
        for param_name, (min_val, max_val) in parameter_bounds.items():
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
        """Suggest optimization strategy based on problem characteristics"""
        
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

class HybridOptimizationEngine:
    """Intelligent hybrid optimization engine"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
                 evaluation_budget: int = 100,
                 time_budget: float = 300.0,
                 bayesian_config: Optional[Dict] = None,
                 genetic_config: Optional[Dict] = None,
                 random_seed: int = 42):
        """
        Initialize Hybrid Optimization Engine
        
        Args:
            parameter_bounds: Dictionary of parameter bounds
            strategy: Optimization strategy to use
            evaluation_budget: Maximum number of evaluations
            time_budget: Maximum time in seconds
            bayesian_config: Configuration for Bayesian Optimizer
            genetic_config: Configuration for Genetic Algorithm
            random_seed: Random seed for reproducibility
        """
        self.parameter_bounds = parameter_bounds
        self.strategy = strategy
        self.evaluation_budget = evaluation_budget
        self.time_budget = time_budget
        self.random_seed = random_seed
        
        # Initialize configurations
        self.bayesian_config = bayesian_config or {
            'acquisition_function': 'ei',
            'initial_samples': min(10, evaluation_budget // 10),
            'max_iterations': evaluation_budget // 2
        }
        
        self.genetic_config = genetic_config or {
            'population_size': min(50, evaluation_budget // 4),
            'max_generations': evaluation_budget // (evaluation_budget // 4),
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }
        
        # Auto-suggest strategy if adaptive
        if self.strategy == OptimizationStrategy.ADAPTIVE:
            self.strategy = WorkloadCharacterizer.suggest_strategy(
                parameter_bounds, evaluation_budget, time_budget
            )
            print(f"Auto-selected strategy: {self.strategy.value}")
        
        # Initialize optimizers
        self.bayesian_optimizer = None
        self.genetic_optimizer = None
        
        # Tracking
        self.total_evaluations = 0
        self.evaluation_history = []
        self.switch_points = []
        
    def _initialize_bayesian_optimizer(self) -> BayesianOptimizer:
        """Initialize Bayesian Optimizer"""
        return BayesianOptimizer(
            parameter_bounds=self.parameter_bounds,
            acquisition_function=self.bayesian_config['acquisition_function'],
            initial_samples=self.bayesian_config['initial_samples'],
            max_iterations=self.bayesian_config['max_iterations'],
            random_seed=self.random_seed
        )
    
    def _initialize_genetic_optimizer(self) -> AdvancedGeneticAlgorithm:
        """Initialize Genetic Algorithm"""
        return AdvancedGeneticAlgorithm(
            parameter_bounds=self.parameter_bounds,
            population_size=self.genetic_config['population_size'],
            max_generations=self.genetic_config['max_generations'],
            mutation_rate=self.genetic_config['mutation_rate'],
            crossover_rate=self.genetic_config['crossover_rate'],
            random_seed=self.random_seed
        )
    
    def _create_budget_limited_objective(self, original_objective: Callable, 
                                       start_time: float) -> Callable:
        """Create objective function wrapper with budget limits"""
        def limited_objective(params: Dict[str, float]) -> float:
            # Check time budget
            if time.time() - start_time > self.time_budget:
                raise RuntimeError("Time budget exceeded")
            
            # Check evaluation budget
            if self.total_evaluations >= self.evaluation_budget:
                raise RuntimeError("Evaluation budget exceeded")
            
            # Evaluate original objective
            score = original_objective(params)
            
            # Track evaluation
            self.total_evaluations += 1
            self.evaluation_history.append((params.copy(), score))
            
            return score
        
        return limited_objective
    
    def _optimize_bayesian_only(self, objective_function: Callable) -> OptimizationResult:
        """Run Bayesian Optimization only"""
        print("Running Bayesian Optimization...")
        self.bayesian_optimizer = self._initialize_bayesian_optimizer()
        return self.bayesian_optimizer.optimize(objective_function)
    
    def _optimize_genetic_only(self, objective_function: Callable) -> GAOptimizationResult:
        """Run Genetic Algorithm only"""
        print("Running Genetic Algorithm...")
        self.genetic_optimizer = self._initialize_genetic_optimizer()
        return self.genetic_optimizer.optimize(objective_function)
    
    def _optimize_hybrid_sequential(self, objective_function: Callable) -> Tuple[OptimizationResult, GAOptimizationResult]:
        """Run Bayesian then Genetic sequentially"""
        print("Running Hybrid Sequential Optimization...")
        
        # Phase 1: Bayesian Optimization (exploration)
        print("Phase 1: Bayesian Optimization for exploration...")
        bayesian_budget = min(self.evaluation_budget // 3, 30)
        
        self.bayesian_config['max_iterations'] = bayesian_budget
        self.bayesian_optimizer = self._initialize_bayesian_optimizer()
        
        try:
            bayesian_result = self.bayesian_optimizer.optimize(objective_function)
            self.switch_points.append((self.total_evaluations, "bayesian_to_genetic"))
        except RuntimeError as e:
            print(f"Bayesian phase ended early: {e}")
            bayesian_result = OptimizationResult(
                best_parameters=self.bayesian_optimizer.best_parameters or {},
                best_score=self.bayesian_optimizer.best_score,
                iteration_count=len(self.bayesian_optimizer.y_history),
                evaluation_history=[],
                convergence_reached=False,
                optimization_time=0
            )
        
        # Phase 2: Genetic Algorithm (exploitation + further exploration)
        print("Phase 2: Genetic Algorithm for exploitation...")
        remaining_budget = self.evaluation_budget - self.total_evaluations
        
        if remaining_budget > 0:
            # Initialize GA population with Bayesian results
            genetic_pop_size = min(remaining_budget // 10, self.genetic_config['population_size'])
            genetic_generations = remaining_budget // genetic_pop_size if genetic_pop_size > 0 else 1
            
            self.genetic_config.update({
                'population_size': genetic_pop_size,
                'max_generations': genetic_generations
            })
            
            self.genetic_optimizer = self._initialize_genetic_optimizer()
            
            # Seed GA population with BO results
            if bayesian_result.best_parameters:
                best_individual = Individual(parameters=bayesian_result.best_parameters)
                self.genetic_optimizer.population = [best_individual] + self.genetic_optimizer.population[1:]
            
            try:
                genetic_result = self.genetic_optimizer.optimize(objective_function)
            except RuntimeError as e:
                print(f"Genetic phase ended early: {e}")
                genetic_result = GAOptimizationResult(
                    best_individual=self.genetic_optimizer.best_individual,
                    best_fitness=self.genetic_optimizer.best_fitness,
                    generation_count=self.genetic_optimizer.generation_count,
                    population_history=[],
                    fitness_history=self.genetic_optimizer.fitness_history,
                    convergence_reached=False,
                    optimization_time=0
                )
        else:
            genetic_result = None
        
        return bayesian_result, genetic_result
    
    def _optimize_adaptive(self, objective_function: Callable) -> Union[OptimizationResult, GAOptimizationResult, Tuple]:
        """Adaptive optimization that switches strategies based on progress"""
        print("Running Adaptive Optimization...")
        
        # Start with Bayesian for initial exploration
        initial_budget = min(20, self.evaluation_budget // 4)
        self.bayesian_config['max_iterations'] = initial_budget
        self.bayesian_optimizer = self._initialize_bayesian_optimizer()
        
        print(f"Starting with Bayesian Optimization ({initial_budget} evaluations)...")
        
        try:
            # Run initial Bayesian phase
            bayesian_result = self.bayesian_optimizer.optimize(objective_function)
            
            # Analyze progress
            if len(self.bayesian_optimizer.y_history) >= 10:
                recent_improvement = max(self.bayesian_optimizer.y_history[-5:]) - max(self.bayesian_optimizer.y_history[-10:-5])
                
                # If good improvement, continue with Bayesian
                if recent_improvement > 0.1:  # Threshold for "good" improvement
                    print("Good progress with Bayesian, continuing...")
                    remaining_budget = self.evaluation_budget - self.total_evaluations
                    if remaining_budget > 0:
                        self.bayesian_config['max_iterations'] = remaining_budget
                        additional_result = self.bayesian_optimizer.optimize(objective_function)
                        return additional_result
                
                # Otherwise switch to Genetic Algorithm
                print("Switching to Genetic Algorithm for broader exploration...")
                self.switch_points.append((self.total_evaluations, "bayesian_to_genetic"))
                
                remaining_budget = self.evaluation_budget - self.total_evaluations
                if remaining_budget > 0:
                    genetic_pop_size = min(remaining_budget // 8, 25)
                    genetic_generations = remaining_budget // genetic_pop_size if genetic_pop_size > 0 else 1
                    
                    self.genetic_config.update({
                        'population_size': genetic_pop_size,
                        'max_generations': genetic_generations
                    })
                    
                    self.genetic_optimizer = self._initialize_genetic_optimizer()
                    genetic_result = self.genetic_optimizer.optimize(objective_function)
                    
                    return bayesian_result, genetic_result
            
            return bayesian_result
            
        except RuntimeError as e:
            print(f"Adaptive optimization ended early: {e}")
            if self.bayesian_optimizer and self.bayesian_optimizer.best_parameters:
                return OptimizationResult(
                    best_parameters=self.bayesian_optimizer.best_parameters,
                    best_score=self.bayesian_optimizer.best_score,
                    iteration_count=len(self.bayesian_optimizer.y_history),
                    evaluation_history=[],
                    convergence_reached=False,
                    optimization_time=0
                )
    
    def optimize(self, objective_function: Callable[[Dict[str, float]], float]) -> HybridOptimizationResult:
        """Run hybrid optimization"""
        start_time = time.time()
        
        print(f"Starting Hybrid Optimization with strategy: {self.strategy.value}")
        print(f"Parameter bounds: {self.parameter_bounds}")
        print(f"Evaluation budget: {self.evaluation_budget}")
        print(f"Time budget: {self.time_budget}s")
        
        # Create budget-limited objective function
        limited_objective = self._create_budget_limited_objective(objective_function, start_time)
        
        bayesian_result = None
        genetic_result = None
        
        try:
            # Execute strategy
            if self.strategy == OptimizationStrategy.BAYESIAN_ONLY:
                bayesian_result = self._optimize_bayesian_only(limited_objective)
                
            elif self.strategy == OptimizationStrategy.GENETIC_ONLY:
                genetic_result = self._optimize_genetic_only(limited_objective)
                
            elif self.strategy == OptimizationStrategy.HYBRID_SEQUENTIAL:
                bayesian_result, genetic_result = self._optimize_hybrid_sequential(limited_objective)
                
            elif self.strategy == OptimizationStrategy.ADAPTIVE:
                result = self._optimize_adaptive(limited_objective)
                if isinstance(result, tuple):
                    bayesian_result, genetic_result = result
                elif hasattr(result, 'best_parameters'):  # Bayesian result
                    bayesian_result = result
                else:  # Genetic result
                    genetic_result = result
            
        except RuntimeError as e:
            print(f"Optimization stopped: {e}")
        
        # Determine best result
        best_score = -np.inf
        best_parameters = {}
        
        if bayesian_result and bayesian_result.best_score > best_score:
            best_score = bayesian_result.best_score
            best_parameters = bayesian_result.best_parameters
        
        if genetic_result and genetic_result.best_fitness > best_score:
            best_score = genetic_result.best_fitness
            best_parameters = genetic_result.best_individual.parameters
        
        # Check convergence
        convergence_reached = False
        if bayesian_result and bayesian_result.convergence_reached:
            convergence_reached = True
        elif genetic_result and genetic_result.convergence_reached:
            convergence_reached = True
        
        optimization_time = time.time() - start_time
        
        print(f"\nHybrid Optimization Complete!")
        print(f"Strategy: {self.strategy.value}")
        print(f"Best Score: {best_score:.6f}")
        print(f"Best Parameters: {best_parameters}")
        print(f"Total Evaluations: {self.total_evaluations}")
        print(f"Time: {optimization_time:.2f}s")
        print(f"Converged: {convergence_reached}")
        
        return HybridOptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            strategy_used=self.strategy,
            total_evaluations=self.total_evaluations,
            bayesian_results=bayesian_result,
            genetic_results=genetic_result,
            optimization_time=optimization_time,
            convergence_reached=convergence_reached,
            switch_points=self.switch_points
        )
    
    def export_results(self, filename: str, result: HybridOptimizationResult):
        """Export hybrid optimization results"""
        export_data = {
            "hybrid_optimization": {
                "strategy": result.strategy_used.value,
                "best_score": result.best_score,
                "best_parameters": result.best_parameters,
                "total_evaluations": result.total_evaluations,
                "optimization_time": result.optimization_time,
                "convergence_reached": result.convergence_reached,
                "switch_points": result.switch_points
            },
            "bayesian_results": None,
            "genetic_results": None,
            "evaluation_history": [
                {"parameters": params, "score": score}
                for params, score in self.evaluation_history
            ]
        }
        
        # Add Bayesian results if available
        if result.bayesian_results:
            export_data["bayesian_results"] = {
                "best_score": result.bayesian_results.best_score,
                "best_parameters": result.bayesian_results.best_parameters,
                "iterations": result.bayesian_results.iteration_count,
                "convergence": result.bayesian_results.convergence_reached
            }
        
        # Add Genetic results if available
        if result.genetic_results:
            export_data["genetic_results"] = {
                "best_fitness": result.genetic_results.best_fitness,
                "best_parameters": result.genetic_results.best_individual.parameters if result.genetic_results.best_individual else None,
                "generations": result.genetic_results.generation_count,
                "convergence": result.genetic_results.convergence_reached
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
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
    
    # Create mock objective function
    def mock_objective_function(params: Dict[str, float]) -> float:
        """Mock objective function for testing"""
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
    
    print("Testing Hybrid Optimization Engine:")
    print("=" * 50)
    
    # Test different strategies
    strategies = [
        OptimizationStrategy.BAYESIAN_ONLY,
        OptimizationStrategy.GENETIC_ONLY,
        OptimizationStrategy.HYBRID_SEQUENTIAL,
        OptimizationStrategy.ADAPTIVE
    ]
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value}:")
        print("-" * 30)
        
        # Initialize Hybrid Engine
        engine = HybridOptimizationEngine(
            parameter_bounds=parameter_bounds,
            strategy=strategy,
            evaluation_budget=50,
            time_budget=30.0
        )
        
        # Run optimization
        result = engine.optimize(mock_objective_function)
        
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Best Score: {result.best_score:.6f}")
        print(f"Best Parameters: {result.best_parameters}")
        print(f"Total Evaluations: {result.total_evaluations}")
        print(f"Time: {result.optimization_time:.2f}s")
        print(f"Converged: {result.convergence_reached}")
        
        # Export results
        engine.export_results(f"hybrid_{strategy.value}_results.json", result)
        
        if result.switch_points:
            print(f"Strategy Switches: {result.switch_points}")
    
    print("\nHybrid Optimization Engine testing completed!")
    print("Check the generated JSON files for detailed results.")