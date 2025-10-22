#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Genetic Algorithm
This module implements Genetic Algorithm for kernel parameter optimization using PyGAD
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import copy
import pygad


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population"""
    parameters: Dict[str, float]
    fitness: Optional[float] = None
    age: int = 0


@dataclass
class GAOptimizationResult:
    """Results from Genetic Algorithm optimization"""
    best_individual: Individual
    best_fitness: float
    generation_count: int
    population_history: List[List[Individual]]
    fitness_history: List[float]
    convergence_reached: bool
    optimization_time: float


class GeneticAlgorithm:
    """Genetic Algorithm for kernel parameter optimization using PyGAD"""

    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 population_size: int = 50,
                 max_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.1,
                 tournament_size: int = 3,
                 convergence_threshold: float = 1e-6,
                 convergence_patience: int = 10,
                 adaptive_parameters: bool = False,
                 local_search: bool = False,
                 diversity_injection: bool = False,
                 random_seed: int = 42):
        """
        Initialize Genetic Algorithm using PyGAD

        Args:
            parameter_bounds: Dict of parameter names to (min, max) bounds
            population_size: Size of the population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation (percentage)
            crossover_rate: Probability of crossover (not directly used in PyGAD)
            elitism_ratio: Ratio of best individuals to preserve
            tournament_size: Size of tournament selection
            convergence_threshold: Convergence threshold
            convergence_patience: Generations to wait for improvement
            adaptive_parameters: Enable adaptive mutation rate adjustment
            local_search: Enable local search optimization (reserved for future)
            diversity_injection: Enable periodic diversity injection (reserved for future)
            random_seed: Random seed for reproducibility
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())

        # Extract bounds for PyGAD
        self.gene_space = [
            {'low': bounds[0], 'high': bounds[1]}
            for bounds in parameter_bounds.values()
        ]

        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience

        # Advanced features (stored for compatibility)
        self.adaptive_parameters = adaptive_parameters
        self.local_search = local_search
        self.diversity_injection = diversity_injection

        self.random_seed = random_seed

        # Tracking variables
        self.best_individual = None
        self.best_fitness = -np.inf
        self.generation_count = 0
        self.fitness_history = []
        self.population_history = []
        self.stagnation_count = 0
        self.objective_function = None

        # PyGAD instance (will be created during optimization)
        self.ga_instance = None

        # Elite size calculation
        self.elite_size = max(
            1, int(self.population_size * self.elitism_ratio))

    def _solution_to_dict(self, solution: np.ndarray) -> Dict[str, float]:
        """Convert PyGAD solution array to parameter dictionary"""
        return {name: float(value) for name, value in zip(self.parameter_names, solution)}

    def _dict_to_solution(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to PyGAD solution array"""
        return np.array([params[name] for name in self.parameter_names])

    def _fitness_wrapper(self, ga_instance, solution, solution_idx):
        """Wrapper for objective function to match PyGAD interface"""
        try:
            params = self._solution_to_dict(solution)
            fitness = self.objective_function(params)
            return fitness
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return -np.inf

    def _on_generation(self, ga_instance):
        """Callback called after each generation"""
        generation = ga_instance.generations_completed

        # Get best solution
        solution, fitness, _ = ga_instance.best_solution()

        # Update tracking
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = Individual(
                parameters=self._solution_to_dict(solution),
                fitness=fitness
            )

        self.fitness_history.append(self.best_fitness)
        self.generation_count = generation

        # Store population snapshot (convert to Individual objects)
        population_snapshot = []
        for idx in range(len(ga_instance.population)):
            ind_params = self._solution_to_dict(ga_instance.population[idx])
            ind_fitness = ga_instance.last_generation_fitness[idx] if hasattr(
                ga_instance, 'last_generation_fitness') else None
            population_snapshot.append(Individual(
                parameters=ind_params, fitness=ind_fitness))
        self.population_history.append(population_snapshot)

        # Print progress
        stats = self._get_population_statistics(ga_instance)
        print(f"Generation {generation}: "
              f"Best = {self.best_fitness:.6f}, "
              f"Mean = {stats['mean']:.6f}, "
              f"Std = {stats['std']:.6f}")

        # Adaptive mutation (if enabled)
        if self.adaptive_parameters:
            self.adaptive_mutation(generation, ga_instance)

        # Check convergence
        if self._check_convergence():
            print(f"Convergence reached at generation {generation}")
            return "stop"  # Signal PyGAD to stop

    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        if len(self.fitness_history) < self.convergence_patience:
            return False

        recent_best = max(self.fitness_history[-self.convergence_patience:])

        prev_start = -2 * self.convergence_patience
        prev_end = -self.convergence_patience
        prev_slice = self.fitness_history[prev_start:prev_end]

        if len(prev_slice) == 0:
            previous_best = recent_best
        else:
            previous_best = max(prev_slice)

        improvement = recent_best - previous_best

        if abs(improvement) < self.convergence_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        return self.stagnation_count >= self.convergence_patience

    def _get_population_statistics(self, ga_instance=None) -> Dict[str, float]:
        """Get statistics about current population"""
        if ga_instance is None or not hasattr(ga_instance, 'last_generation_fitness'):
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}

        fitnesses = ga_instance.last_generation_fitness

        return {
            "mean": np.mean(fitnesses),
            "std": np.std(fitnesses),
            "min": np.min(fitnesses),
            "max": np.max(fitnesses),
            "median": np.median(fitnesses)
        }

    def optimize(self, objective_function: Callable[[Dict[str, float]], float],
                 use_advanced_features: bool = None) -> GAOptimizationResult:
        """
        Run genetic algorithm optimization using PyGAD

        Args:
            objective_function: Function to maximize (higher is better)
            use_advanced_features: Enable adaptive parameters (PyGAD has built-in features)

        Returns:
            GAOptimizationResult with optimization results
        """
        start_time = time.time()
        self.objective_function = objective_function

        # Auto-detect if we should use advanced features
        if use_advanced_features is None:
            use_advanced_features = self.adaptive_parameters

        print(
            f"Starting Genetic Algorithm with {self.max_generations} generations...")
        print(f"Population size: {self.population_size}")
        print(f"Mutation rate: {self.mutation_rate * 100}%")
        print(f"Elite size: {self.elite_size}")
        if use_advanced_features:
            print(f"  • Adaptive mutation enabled")

        # Configure PyGAD
        self.ga_instance = pygad.GA(
            num_generations=self.max_generations,
            num_parents_mating=max(2, self.population_size - self.elite_size),
            fitness_func=self._fitness_wrapper,
            sol_per_pop=self.population_size,
            num_genes=len(self.parameter_names),
            gene_space=self.gene_space,
            parent_selection_type="tournament",
            K_tournament=self.tournament_size,
            keep_elitism=self.elite_size,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=int(
                self.mutation_rate * 100),  # PyGAD uses percentage
            random_seed=self.random_seed,
            on_generation=self._on_generation,
            suppress_warnings=True
        )

        # Run optimization
        self.ga_instance.run()

        # Get final results
        solution, fitness, _ = self.ga_instance.best_solution()

        optimization_time = time.time() - start_time
        convergence_reached = self.stagnation_count >= self.convergence_patience

        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Best fitness: {self.best_fitness:.6f}")
        print(
            f"Best parameters: {self.best_individual.parameters if self.best_individual else None}")

        return GAOptimizationResult(
            best_individual=self.best_individual,
            best_fitness=self.best_fitness,
            generation_count=self.generation_count,
            population_history=self.population_history,
            fitness_history=self.fitness_history,
            convergence_reached=convergence_reached,
            optimization_time=optimization_time
        )

    def get_diversity_metrics(self) -> Dict[str, float]:
        """Calculate population diversity metrics"""
        if self.ga_instance is None or not hasattr(self.ga_instance, 'population'):
            return {"overall_diversity": 0, "parameter_diversities": {}, "fitness_diversity": 0}

        population = self.ga_instance.population

        # Calculate parameter diversity
        diversities = []
        param_diversities = {}

        for i, param_name in enumerate(self.parameter_names):
            values = population[:, i]
            param_range = self.parameter_bounds[param_name][1] - \
                self.parameter_bounds[param_name][0]
            diversity = np.std(values) / param_range if param_range > 0 else 0
            diversities.append(diversity)
            param_diversities[param_name] = diversity

        # Fitness diversity
        fitness_diversity = 0
        if hasattr(self.ga_instance, 'last_generation_fitness'):
            fitness_diversity = np.std(
                self.ga_instance.last_generation_fitness)

        return {
            "overall_diversity": np.mean(diversities),
            "parameter_diversities": param_diversities,
            "fitness_diversity": fitness_diversity
        }

    def export_results(self, filename: str):
        """Export optimization results to JSON file"""
        diversity = self.get_diversity_metrics()

        results_data = {
            "optimization_settings": {
                "parameter_bounds": self.parameter_bounds,
                "population_size": self.population_size,
                "max_generations": self.max_generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elitism_ratio": self.elitism_ratio
            },
            "best_result": {
                "parameters": self.best_individual.parameters if self.best_individual else None,
                "fitness": self.best_fitness,
                "generation_found": self.fitness_history.index(self.best_fitness) + 1 if self.best_fitness in self.fitness_history else 0
            },
            "optimization_progress": {
                "generations": self.generation_count,
                "fitness_history": self.fitness_history,
                "final_diversity": diversity
            },
            "population_statistics": self._get_population_statistics(self.ga_instance)
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results exported to {filename}")

    def adaptive_mutation(self, generation: int, ga_instance=None):
        """Adaptive mutation rate based on generation"""
        if not self.adaptive_parameters:
            return

        # Decrease mutation rate over time
        initial_rate = 0.2
        final_rate = 0.05
        decay_factor = (final_rate / initial_rate) ** (1 /
                                                       self.max_generations)
        new_rate = initial_rate * (decay_factor ** generation)

        self.mutation_rate = new_rate

        # Update PyGAD instance if available
        if ga_instance is not None:
            ga_instance.mutation_percent_genes = int(new_rate * 100)

    def inject_diversity(self, diversity_threshold: float = 0.01):
        """
        Inject diversity if population becomes too similar.
        Note: PyGAD handles diversity internally, this is kept for API compatibility.
        """
        diversity = self.get_diversity_metrics()

        if diversity["overall_diversity"] < diversity_threshold:
            print(
                f"Low diversity detected ({diversity['overall_diversity']:.4f})")
            # PyGAD handles this through its mutation and crossover mechanisms

    def save_model(self, filename: str):
        """Save the PyGAD instance to a file"""
        if self.ga_instance is None:
            print("No GA instance to save")
            return

        self.ga_instance.save(filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str):
        """Load a saved PyGAD instance"""
        self.ga_instance = pygad.load(filename)
        print(f"Model loaded from {filename}")


# Advanced Genetic Algorithm with additional features
class AdvancedGeneticAlgorithm(GeneticAlgorithm):
    """
    Enhanced Genetic Algorithm with advanced features.
    This class uses PyGAD's built-in advanced capabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable advanced features by default
        self.adaptive_parameters = True
        self.diversity_injection = False  # PyGAD handles diversity internally
        self.local_search = False  # Reserved for future hybrid approaches

    def _local_search_optimization(self, individual: Individual, objective_function: Callable) -> Individual:
        """
        Apply local search to improve individual (hill climbing).
        Note: This is kept for API compatibility but not actively used with PyGAD.
        """
        if not self.local_search:
            return individual

        # Simple hill climbing (can be extended)
        current_individual = copy.deepcopy(individual)
        best_individual = copy.deepcopy(individual)

        for _ in range(5):  # 5 local search iterations
            for param_name in self.parameter_names:
                min_val, max_val = self.parameter_bounds[param_name]
                current_val = current_individual.parameters[param_name]

                # Try small modifications
                range_size = max_val - min_val
                step_size = range_size * 0.01  # 1% of range

                for delta in [-step_size, step_size]:
                    new_val = np.clip(current_val + delta, min_val, max_val)

                    # Create test individual
                    test_params = current_individual.parameters.copy()
                    test_params[param_name] = new_val

                    # Evaluate
                    test_fitness = objective_function(test_params)

                    if test_fitness > (best_individual.fitness or -np.inf):
                        best_individual = Individual(
                            parameters=test_params, fitness=test_fitness)

        return best_individual


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

        # Mock scoring function with multiple peaks to test global search
        score1 = -0.01 * (swappiness - 30)**2 + 50
        score2 = -0.001 * (dirty_ratio - 20)**2 + 30
        score3 = 0.00001 * rmem_max
        score4 = -0.000000001 * (sched_gran - 10000000)**2 + 20

        # Add interaction terms
        interaction = 0.001 * (swappiness * dirty_ratio / 1000)

        total_score = score1 + score2 + score3 + score4 + interaction

        # Add some noise
        total_score += np.random.normal(0, 0.5)

        return total_score

    print("=" * 80)
    print("Testing Genetic Algorithm with PyGAD")
    print("=" * 80)

    # Initialize and run standard GA
    ga = GeneticAlgorithm(
        parameter_bounds=parameter_bounds,
        population_size=30,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_ratio=0.2,
        random_seed=42
    )

    # Run optimization
    result = ga.optimize(mock_objective_function)

    print(f"\n{'='*80}")
    print("Standard GA Results:")
    print(f"{'='*80}")
    print(f"Best Fitness: {result.best_fitness:.6f}")
    print("Best Parameters:")
    for param, value in result.best_individual.parameters.items():
        print(f"  {param}: {value:.2f}")
    print(f"Generations: {result.generation_count}")
    print(f"Converged: {result.convergence_reached}")
    print(f"Time: {result.optimization_time:.2f} seconds")

    # Export results
    ga.export_results("genetic_algorithm_results.json")
    print("\nResults exported to genetic_algorithm_results.json")

    # Save model
    ga.save_model("ga_model.pkl")

    print(f"\n{'='*80}")
    print("Testing Advanced Genetic Algorithm")
    print(f"{'='*80}")

    # Initialize and run advanced GA
    advanced_ga = AdvancedGeneticAlgorithm(
        parameter_bounds=parameter_bounds,
        population_size=30,
        max_generations=50,
        mutation_rate=0.15,  # Start higher for adaptive mutation
        crossover_rate=0.8,
        elitism_ratio=0.2,
        random_seed=42
    )

    # Run optimization
    advanced_result = advanced_ga.optimize(mock_objective_function)

    print(f"\n{'='*80}")
    print(f"Advanced GA Results:")
    print(f"{'='*80}")
    print(f"Best Fitness: {advanced_result.best_fitness:.6f}")
    print(f"Best Parameters:")
    for param, value in advanced_result.best_individual.parameters.items():
        print(f"  {param}: {value:.2f}")
    print(f"Generations: {advanced_result.generation_count}")
    print(f"Converged: {advanced_result.convergence_reached}")
    print(f"Time: {advanced_result.optimization_time:.2f} seconds")

    # Export results
    advanced_ga.export_results("advanced_genetic_algorithm_results.json")

    # Compare results
    print(f"\n{'='*80}")
    print(f"Comparison:")
    print(f"{'='*80}")
    print(f"Standard GA Best: {result.best_fitness:.6f}")
    print(f"Advanced GA Best: {advanced_result.best_fitness:.6f}")
    improvement = ((advanced_result.best_fitness - result.best_fitness) /
                   abs(result.best_fitness) * 100) if result.best_fitness != 0 else 0
    print(f"Improvement: {improvement:.2f}%")

    # Get final diversity metrics
    diversity = advanced_ga.get_diversity_metrics()
    print(f"\nFinal Population Diversity:")
    print(f"Overall: {diversity['overall_diversity']:.4f}")
    for param, div in diversity['parameter_diversities'].items():
        print(f"  {param}: {div:.4f}")
