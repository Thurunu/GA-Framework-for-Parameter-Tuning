from HybridOptimizationEngine import HybridOptimizationResult
import json
from typing import List, Tuple, Dict

def export_results(filename: str, result: HybridOptimizationResult, evaluation_history: List[Tuple[Dict, float]] = None):
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
            for params, score in (evaluation_history or [])
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

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

    print(f"Results exported to {filename}")
