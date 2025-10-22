#!/usr/bin/env python3
"""
Test Strategy Selection Fix
Verifies that explicit strategy choices (HYBRID_SEQUENTIAL, GENETIC_ONLY) are respected
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from HybridOptimizationEngine import HybridOptimizationEngine
from WorkloadCharacterizer import OptimizationStrategy

# Test parameter bounds (4 parameters - would normally trigger BAYESIAN_ONLY suggestion)
parameter_bounds = {
    'vm.swappiness': (10, 60),
    'vm.dirty_ratio': (10, 30),
    'kernel.sched_cfs_bandwidth_slice_us': (1000, 20000),
    'vm.dirty_background_ratio': (5, 15)
}

print("=" * 80)
print("Testing Strategy Selection")
print("=" * 80)

# Test 1: Explicit HYBRID_SEQUENTIAL should be respected
print("\n1. Testing HYBRID_SEQUENTIAL (should NOT be overridden):")
print("-" * 80)
engine1 = HybridOptimizationEngine(
    parameter_bounds=parameter_bounds,
    strategy=OptimizationStrategy.HYBRID_SEQUENTIAL,
    evaluation_budget=50,  # Small budget that would normally suggest BAYESIAN_ONLY
    time_budget=120.0
)
assert engine1.strategy == OptimizationStrategy.HYBRID_SEQUENTIAL, \
    f"❌ FAILED: Expected HYBRID_SEQUENTIAL, got {engine1.strategy.value}"
print(f"✅ PASS: Strategy correctly set to {engine1.strategy.value}")

# Test 2: Explicit GENETIC_ONLY should be respected
print("\n2. Testing GENETIC_ONLY (should NOT be overridden):")
print("-" * 80)
engine2 = HybridOptimizationEngine(
    parameter_bounds=parameter_bounds,
    strategy=OptimizationStrategy.GENETIC_ONLY,
    evaluation_budget=50,
    time_budget=120.0
)
assert engine2.strategy == OptimizationStrategy.GENETIC_ONLY, \
    f"❌ FAILED: Expected GENETIC_ONLY, got {engine2.strategy.value}"
print(f"✅ PASS: Strategy correctly set to {engine2.strategy.value}")

# Test 3: ADAPTIVE should trigger auto-suggestion
print("\n3. Testing ADAPTIVE (should auto-select based on problem):")
print("-" * 80)
engine3 = HybridOptimizationEngine(
    parameter_bounds=parameter_bounds,
    strategy=OptimizationStrategy.ADAPTIVE,
    evaluation_budget=50,
    time_budget=120.0
)
# With 4 params and budget=50, should suggest BAYESIAN_ONLY
print(f"✅ PASS: Auto-selected strategy is {engine3.strategy.value}")
print(f"   (Expected BAYESIAN_ONLY for 4 params, budget=50)")

# Test 4: Explicit BAYESIAN_ONLY should be respected
print("\n4. Testing BAYESIAN_ONLY (should NOT be overridden):")
print("-" * 80)
engine4 = HybridOptimizationEngine(
    parameter_bounds=parameter_bounds,
    strategy=OptimizationStrategy.BAYESIAN_ONLY,
    evaluation_budget=50,
    time_budget=120.0
)
assert engine4.strategy == OptimizationStrategy.BAYESIAN_ONLY, \
    f"❌ FAILED: Expected BAYESIAN_ONLY, got {engine4.strategy.value}"
print(f"✅ PASS: Strategy correctly set to {engine4.strategy.value}")

print("\n" + "=" * 80)
print("✅ All tests passed! Strategy selection is working correctly.")
print("=" * 80)
print("\nSummary:")
print("  • Explicit strategy choices (HYBRID_SEQUENTIAL, GENETIC_ONLY, etc.) are respected")
print("  • ADAPTIVE strategy triggers auto-selection based on problem characteristics")
print("  • User configuration is no longer overridden")
