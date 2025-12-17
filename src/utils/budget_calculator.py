"""
Automatic Budget Calculator
Calculates optimal evaluation and time budgets based on strategy and workload complexity
"""

from typing import Dict, Tuple
import yaml
from pathlib import Path

class BudgetCalculator:
    """Calculate optimal budgets based on strategy and complexity"""
    
    # Strategy presets
    STRATEGY_PRESETS = {
        "BAYESIAN_ONLY": {
            "base_eval": 8,
            "base_time": 100,
            "scaling": 0.8,
            "name": "Quick Convergence"
        },
        "HYBRID_SEQUENTIAL": {
            "base_eval": 15,
            "base_time": 180,
            "scaling": 1.0,
            "name": "Balanced Hybrid"
        },
        "ADAPTIVE": {
            "base_eval": 12,
            "base_time": 150,
            "scaling": 0.9,
            "name": "Intelligent Adaptive"
        },
        "GENETIC_ONLY": {
            "base_eval": 20,
            "base_time": 240,
            "scaling": 1.2,
            "name": "Thorough Search"
        }
    }
    
    # Complexity multipliers based on parameter count
    COMPLEXITY_TIERS = {
        "simple": {"params": (1, 3), "eval_mult": 1.0, "time_mult": 1.0},
        "moderate": {"params": (4, 6), "eval_mult": 1.3, "time_mult": 1.2},
        "complex": {"params": (7, 10), "eval_mult": 1.6, "time_mult": 1.5},
        "very_complex": {"params": (11, 20), "eval_mult": 2.0, "time_mult": 1.8}
    }
    
    @staticmethod
    def calculate_budgets(
        strategy: str,
        parameter_count: int,
        custom_multiplier: float = 1.0
    ) -> Tuple[int, float]:
        """
        Calculate optimal budgets based on strategy and complexity
        
        Args:
            strategy: Optimization strategy (BAYESIAN_ONLY, ADAPTIVE, etc.)
            parameter_count: Number of parameters to optimize
            custom_multiplier: Optional custom scaling factor (default: 1.0)
            
        Returns:
            (evaluation_budget, time_budget)
            
        Example:
            >>> calc = BudgetCalculator()
            >>> eval_budget, time_budget = calc.calculate_budgets("ADAPTIVE", 5)
            >>> print(f"Eval: {eval_budget}, Time: {time_budget}s")
            Eval: 16, Time: 180.0s
        """
        # Get strategy preset
        if strategy not in BudgetCalculator.STRATEGY_PRESETS:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        preset = BudgetCalculator.STRATEGY_PRESETS[strategy]
        
        # Determine complexity tier
        complexity = BudgetCalculator._get_complexity_tier(parameter_count)
        
        # Calculate budgets
        eval_budget = int(
            preset["base_eval"] * 
            complexity["eval_mult"] * 
            preset["scaling"] * 
            custom_multiplier
        )
        
        time_budget = float(
            preset["base_time"] * 
            complexity["time_mult"] * 
            preset["scaling"] * 
            custom_multiplier
        )
        
        # Enforce minimum budgets
        eval_budget = max(5, eval_budget)  # Minimum 5 evaluations
        time_budget = max(60.0, time_budget)  # Minimum 60 seconds
        
        return eval_budget, time_budget
    
    @staticmethod
    def _get_complexity_tier(param_count: int) -> Dict:
        """Determine complexity tier based on parameter count"""
        for tier_name, tier_data in BudgetCalculator.COMPLEXITY_TIERS.items():
            min_params, max_params = tier_data["params"]
            if min_params <= param_count <= max_params:
                return tier_data
        
        # If >20 parameters, use very_complex tier
        return BudgetCalculator.COMPLEXITY_TIERS["very_complex"]
    
    @staticmethod
    def get_budget_recommendation(
        strategy: str,
        parameter_count: int
    ) -> Dict:
        """
        Get budget recommendation with explanation
        
        Returns:
            {
                'evaluation_budget': int,
                'time_budget': float,
                'complexity': str,
                'reasoning': str,
                'can_customize': bool
            }
        """
        eval_budget, time_budget = BudgetCalculator.calculate_budgets(
            strategy, parameter_count
        )
        
        complexity_tier = BudgetCalculator._get_complexity_tier(parameter_count)
        complexity_name = [
            k for k, v in BudgetCalculator.COMPLEXITY_TIERS.items() 
            if v == complexity_tier
        ][0]
        
        preset = BudgetCalculator.STRATEGY_PRESETS[strategy]
        
        reasoning = (
            f"Based on '{preset['name']}' strategy with {parameter_count} parameters "
            f"({complexity_name} complexity), recommended budgets are "
            f"{eval_budget} evaluations and {time_budget:.0f} seconds."
        )
        
        return {
            'evaluation_budget': eval_budget,
            'time_budget': time_budget,
            'complexity': complexity_name,
            'reasoning': reasoning,
            'can_customize': True,
            'min_eval_budget': 5,
            'max_eval_budget': eval_budget * 2,
            'min_time_budget': 60,
            'max_time_budget': time_budget * 2
        }


# Example usage
if __name__ == "__main__":
    calc = BudgetCalculator()
    
    # Test different scenarios
    scenarios = [
        ("BAYESIAN_ONLY", 3, "Quick - Simple"),
        ("HYBRID_SEQUENTIAL", 6, "Balanced - Moderate"),
        ("ADAPTIVE", 8, "Adaptive - Complex"),
        ("GENETIC_ONLY", 12, "Thorough - Very Complex")
    ]
    
    print("=" * 70)
    print("AUTOMATIC BUDGET RECOMMENDATIONS")
    print("=" * 70)
    
    for strategy, params, desc in scenarios:
        eval_b, time_b = calc.calculate_budgets(strategy, params)
        rec = calc.get_budget_recommendation(strategy, params)
        
        print(f"\n{desc}:")
        print(f"  Strategy: {strategy}")
        print(f"  Parameters: {params}")
        print(f"  Evaluation Budget: {eval_b}")
        print(f"  Time Budget: {time_b:.0f}s ({time_b/60:.1f} min)")
        print(f"  Complexity: {rec['complexity']}")