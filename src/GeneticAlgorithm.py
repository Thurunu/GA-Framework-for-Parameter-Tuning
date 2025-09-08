#!/usr/bin/env python3
"""
Linux Kernel Optimization Framework - Genetic Algorithm
This module implements Genetic Algorithm for kernel parameter optimization
"""

import numpy as np
import json
import time
import random
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import copy

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
    """Genetic Algorithm for kernel parameter optimization"""
    
    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]],
                 population_size: int = 50,
                 max_generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.1,
                 tournament_size: int = 3,
                 convergence_threshold: float = 1e-6,
                 convergence_patience: int = 10,
                 random_seed: int = 42):
        """
        Initialize Genetic Algorithm
        
        Args:
            parameter_bounds: Dict of parameter names to (min, max) bounds
            population_size: Size of the population
            max_generations: Maximum number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_ratio: Ratio of best individuals to preserve
            tournament_size: Size of tournament selection
            convergence_threshold: Convergence threshold
            convergence_patience: Generations to wait for improvement
            random_seed: Random seed for reproducibility
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.bounds_array = np.array([parameter_bounds[name] for name in self.parameter_names])
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        
        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize population and tracking variables
        self.population = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.generation_count = 0
        self.fitness_history = []
        self.population_history = []
        self.stagnation_count = 0
        
        # Elite size calculation
        self.elite_size = max(1, int(self.population_size * self.elitism_ratio))
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual within parameter bounds"""
        parameters = {}
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.bounds_array[i]
            parameters[param_name] = np.random.uniform(min_val, max_val)
        
        return Individual(parameters=parameters)
    
    def _initialize_population(self):
        """Initialize the population with random individuals"""
        self.population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            self.population.append(individual)
    
    def _evaluate_individual(self, individual: Individual, objective_function: Callable) -> float:
        """Evaluate fitness of an individual"""
        if individual.fitness is None:
            try:
                individual.fitness = objective_function(individual.parameters)
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                individual.fitness = -np.inf
        return individual.fitness
    
    def _evaluate_population(self, objective_function: Callable):
        """Evaluate fitness for entire population"""
        for individual in self.population:
            self._evaluate_individual(individual, objective_function)
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection to choose parent"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness if x.fitness is not None else -np.inf)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover operation to create offspring"""
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Arithmetic crossover
        alpha = random.random()
        
        child1_params = {}
        child2_params = {}
        
        for param_name in self.parameter_names:
            p1_val = parent1.parameters[param_name]
            p2_val = parent2.parameters[param_name]
            
            child1_params[param_name] = alpha * p1_val + (1 - alpha) * p2_val
            child2_params[param_name] = (1 - alpha) * p1_val + alpha * p2_val
            
            # Ensure bounds are respected
            min_val, max_val = self.parameter_bounds[param_name]
            child1_params[param_name] = np.clip(child1_params[param_name], min_val, max_val)
            child2_params[param_name] = np.clip(child2_params[param_name], min_val, max_val)
        
        child1 = Individual(parameters=child1_params)
        child2 = Individual(parameters=child2_params)
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutation operation"""
        mutated_individual = copy.deepcopy(individual)
        
        for param_name in self.parameter_names:
            if random.random() < self.mutation_rate:
                min_val, max_val = self.parameter_bounds[param_name]
                current_val = mutated_individual.parameters[param_name]
                
                # Gaussian mutation with adaptive step size
                range_size = max_val - min_val
                mutation_strength = range_size * 0.1  # 10% of parameter range
                
                new_val = current_val + np.random.normal(0, mutation_strength)
                new_val = np.clip(new_val, min_val, max_val)
                
                mutated_individual.parameters[param_name] = new_val
        
        # Reset fitness as parameters have changed
        mutated_individual.fitness = None
        return mutated_individual
    
    def _select_survivors(self, population: List[Individual], objective_function: Callable) -> List[Individual]:
        """Select survivors for next generation using elitism and tournament selection"""
        # Evaluate all individuals
        for individual in population:
            if individual.fitness is None:
                self._evaluate_individual(individual, objective_function)
        
        # Sort population by fitness (descending)
        population.sort(key=lambda x: x.fitness if x.fitness is not None else -np.inf, reverse=True)
        
        # Keep elite individuals
        survivors = population[:self.elite_size].copy()
        
        # Fill remaining spots with tournament selection
        while len(survivors) < self.population_size:
            selected = self._tournament_selection()
            survivors.append(copy.deepcopy(selected))
        
        # Age individuals
        for individual in survivors:
            individual.age += 1
        
        return survivors[:self.population_size]
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        if len(self.fitness_history) < self.convergence_patience:
            return False
        
        recent_best = max(
            self.fitness_history[
                -self.convergence_patience:
            ]
        )
    
        prev_start = -2 * self.convergence_patience
        prev_end = -self.convergence_patience
        prev_slice = self.fitness_history[prev_start:prev_end]
       
        if len(prev_slice) == 0:
            previous_best = recent_best
        else:
            previous_best = max(prev_slice)
        improvement = recent_best - previous_best
        print("Imrpovment: ", improvement)
        
        if abs(improvement) < self.convergence_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        return self.stagnation_count >= self.convergence_patience
    
    def _update_best(self):
        """Update best individual and fitness"""
        current_best = max(self.population, key=lambda x: x.fitness if x.fitness is not None else -np.inf)
        
        if current_best.fitness is not None and current_best.fitness > self.best_fitness:
            self.best_individual = copy.deepcopy(current_best)
            self.best_fitness = current_best.fitness
    
    def _get_population_statistics(self) -> Dict[str, float]:
        """Get statistics about current population"""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        
        if not fitnesses:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        return {
            "mean": np.mean(fitnesses),
            "std": np.std(fitnesses),
            "min": np.min(fitnesses),
            "max": np.max(fitnesses),
            "median": np.median(fitnesses)
        }
    
    def optimize(self, objective_function: Callable[[Dict[str, float]], float]) -> GAOptimizationResult:
        """Run the genetic algorithm optimization"""
        start_time = time.time()
        
        print(f"Starting Genetic Algorithm with {self.max_generations} generations...")
        print(f"Population size: {self.population_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Crossover rate: {self.crossover_rate}")
        
        # Initialize population
        self._initialize_population()
        self._evaluate_population(objective_function)
        self._update_best()
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation_count = generation + 1
            
            # Create next generation
            new_population = []
            
            # Generate offspring
            while len(new_population) < self.population_size - self.elite_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Combine with current population for survival selection
            combined_population = self.population + new_population
            self.population = self._select_survivors(combined_population, objective_function)
            
            # Update tracking
            self._update_best()
            self.fitness_history.append(self.best_fitness)
            self.population_history.append([copy.deepcopy(ind) for ind in self.population])
            
            # Print progress
            stats = self._get_population_statistics()
            print(f"Generation {generation + 1}: "
                  f"Best = {self.best_fitness:.6f}, "
                  f"Mean = {stats['mean']:.6f}, "
                  f"Std = {stats['std']:.6f}")
            
            # Check convergence
            if self._check_convergence():
                print(f"Convergence reached at generation {generation + 1}")
                break
        
        optimization_time = time.time() - start_time
        convergence_reached = self.stagnation_count >= self.convergence_patience
        
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best fitness: {self.best_fitness:.6f}")
        print(f"Best parameters: {self.best_individual.parameters if self.best_individual else None}")
        
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
        if not self.population:
            return {"diversity": 0}
        
        # Calculate parameter diversity
        diversities = []
        
        for param_name in self.parameter_names:
            values = [ind.parameters[param_name] for ind in self.population]
            diversity = np.std(values) / (self.parameter_bounds[param_name][1] - self.parameter_bounds[param_name][0])
            diversities.append(diversity)
        
        return {
            "overall_diversity": np.mean(diversities),
            "parameter_diversities": dict(zip(self.parameter_names, diversities)),
            "fitness_diversity": np.std([ind.fitness for ind in self.population if ind.fitness is not None])
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
                "generation_found": self.fitness_history.index(self.best_fitness) + 1 if self.fitness_history else 0
            },
            "optimization_progress": {
                "generations": self.generation_count,
                "fitness_history": self.fitness_history,
                "final_diversity": diversity
            },
            "population_statistics": self._get_population_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results exported to {filename}")
    
    def adaptive_mutation(self, generation: int):
        """Adaptive mutation rate based on generation"""
        # Decrease mutation rate over time
        initial_rate = 0.2
        final_rate = 0.05
        decay_factor = (final_rate / initial_rate) ** (1 / self.max_generations)
        self.mutation_rate = initial_rate * (decay_factor ** generation)
    
    def inject_diversity(self, diversity_threshold: float = 0.01):
        """Inject diversity if population becomes too similar"""
        diversity = self.get_diversity_metrics()
        
        if diversity["overall_diversity"] < diversity_threshold:
            print(f"Low diversity detected ({diversity['overall_diversity']:.4f}), injecting new individuals...")
            
            # Replace worst 25% of population with random individuals
            self.population.sort(key=lambda x: x.fitness if x.fitness is not None else -np.inf)
            num_to_replace = max(1, self.population_size // 4)
            
            for i in range(num_to_replace):
                self.population[i] = self._create_random_individual()

# Advanced Genetic Algorithm with additional features
class AdvancedGeneticAlgorithm(GeneticAlgorithm):
    """Enhanced Genetic Algorithm with advanced features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_parameters = True
        self.diversity_injection = True
        self.local_search = False  # Can be enabled for hybrid approach
    
    def _local_search_optimization(self, individual: Individual, objective_function: Callable) -> Individual:
        """Apply local search to improve individual (hill climbing)"""
        if not self.local_search:
            return individual
        
        current_individual = copy.deepcopy(individual)
        best_individual = copy.deepcopy(individual)
        
        for _ in range(5):  # 5 local search iterations
            # Try small modifications
            for param_name in self.parameter_names:
                min_val, max_val = self.parameter_bounds[param_name]
                current_val = current_individual.parameters[param_name]
                
                # Try small positive and negative changes
                range_size = max_val - min_val
                step_size = range_size * 0.01  # 1% of range
                
                for delta in [-step_size, step_size]:
                    new_val = np.clip(current_val + delta, min_val, max_val)
                    
                    # Create test individual
                    test_params = current_individual.parameters.copy()
                    test_params[param_name] = new_val
                    test_individual = Individual(parameters=test_params)
                    
                    # Evaluate
                    test_fitness = objective_function(test_params)
                    
                    if test_fitness > (best_individual.fitness or -np.inf):
                        best_individual = Individual(parameters=test_params, fitness=test_fitness)
        
        return best_individual
    
    def optimize(self, objective_function: Callable[[Dict[str, float]], float]) -> GAOptimizationResult:
        """Enhanced optimization with adaptive features"""
        start_time = time.time()
        
        print(f"Starting Advanced Genetic Algorithm with {self.max_generations} generations...")
        
        # Initialize population
        self._initialize_population()
        self._evaluate_population(objective_function)
        self._update_best()
        
        # Evolution loop with enhancements
        for generation in range(self.max_generations):
            self.generation_count = generation + 1
            
            # Adaptive parameters
            if self.adaptive_parameters:
                self.adaptive_mutation(generation)
            
            # Create next generation
            new_population = []
            
            # Generate offspring
            while len(new_population) < self.population_size - self.elite_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # Apply local search to some offspring
                if self.local_search and random.random() < 0.1:  # 10% chance
                    child1 = self._local_search_optimization(child1, objective_function)
                
                new_population.extend([child1, child2])
            
            # Combine with current population for survival selection
            combined_population = self.population + new_population
            self.population = self._select_survivors(combined_population, objective_function)
            
            # Diversity injection
            if self.diversity_injection and generation % 20 == 0:
                self.inject_diversity()
            
            # Update tracking
            self._update_best()
            self.fitness_history.append(self.best_fitness)
            self.population_history.append([copy.deepcopy(ind) for ind in self.population])
            
            # Print progress with additional info
            stats = self._get_population_statistics()
            diversity = self.get_diversity_metrics()
            print(f"Generation {generation + 1}: "
                  f"Best = {self.best_fitness:.6f}, "
                  f"Mean = {stats['mean']:.6f}, "
                  f"Diversity = {diversity['overall_diversity']:.4f}, "
                  f"MutRate = {self.mutation_rate:.4f}")
            
            # Check convergence
            if self._check_convergence():
                print(f"Convergence reached at generation {generation + 1}")
                break
        
        optimization_time = time.time() - start_time
        convergence_reached = self.stagnation_count >= self.convergence_patience
        
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Best fitness: {self.best_fitness:.6f}")
        print(f"Best parameters: {self.best_individual.parameters if self.best_individual else None}")
        
        return GAOptimizationResult(
            best_individual=self.best_individual,
            best_fitness=self.best_fitness,
            generation_count=self.generation_count,
            population_history=self.population_history,
            fitness_history=self.fitness_history,
            convergence_reached=convergence_reached,
            optimization_time=optimization_time
        )

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
    
    print("Testing Standard Genetic Algorithm:")
    print("=" * 50)
    
    # Initialize and run standard GA
    ga = GeneticAlgorithm(
        parameter_bounds=parameter_bounds,
        population_size=30,
        max_generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_ratio=0.2
    )
    
    # Run optimization
    result = ga.optimize(mock_objective_function)
    
    print(f"\nStandard GA Results:")
    print(f"Best Fitness: {result.best_fitness:.6f}")
    print(f"Best Parameters: {result.best_individual.parameters}")
    print(f"Generations: {result.generation_count}")
    print(f"Converged: {result.convergence_reached}")
    print(f"Time: {result.optimization_time:.2f} seconds")
    
    # Export results
    ga.export_results("genetic_algorithm_results.json")
    
    print("\n" + "="*50)
    print("Testing Advanced Genetic Algorithm:")
    print("=" * 50)
    
    # Initialize and run advanced GA
    advanced_ga = AdvancedGeneticAlgorithm(
        parameter_bounds=parameter_bounds,
        population_size=30,
        max_generations=50,
        mutation_rate=0.15,  # Start higher for adaptive mutation
        crossover_rate=0.8,
        elitism_ratio=0.2
    )
    
    # Enable advanced features
    advanced_ga.adaptive_parameters = True
    advanced_ga.diversity_injection = True
    advanced_ga.local_search = False  # Disabled for speed
    
    # Run optimization
    advanced_result = advanced_ga.optimize(mock_objective_function)
    
    print(f"\nAdvanced GA Results:")
    print(f"Best Fitness: {advanced_result.best_fitness:.6f}")
    print(f"Best Parameters: {advanced_result.best_individual.parameters}")
    print(f"Generations: {advanced_result.generation_count}")
    print(f"Converged: {advanced_result.convergence_reached}")
    print(f"Time: {advanced_result.optimization_time:.2f} seconds")
    
    # Export results
    advanced_ga.export_results("advanced_genetic_algorithm_results.json")
    
    # Compare results
    print(f"\nComparison:")
    print(f"Standard GA Best: {result.best_fitness:.6f}")
    print(f"Advanced GA Best: {advanced_result.best_fitness:.6f}")
    print(f"Improvement: {((advanced_result.best_fitness - result.best_fitness) / abs(result.best_fitness) * 100):.2f}%")
    
    # Get final diversity metrics
    diversity = advanced_ga.get_diversity_metrics()
    print(f"\nFinal Population Diversity:")
    print(f"Overall: {diversity['overall_diversity']:.4f}")
    for param, div in diversity['parameter_diversities'].items():
        print(f"{param}: {div:.4f}")