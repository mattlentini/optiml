"""
Example: Using Batch Acquisition for Parallel Experiments
===========================================================

This example demonstrates how to use batch acquisition functions to suggest
multiple experimental conditions simultaneously, enabling parallel experimentation.
"""

import numpy as np
from optiml import BayesianOptimizer, Space, Real, Integer, Categorical
from optiml.batch import suggest_batch


def chromatography_objective(params):
    """
    Simulated chromatography optimization.
    
    Parameters:
    - Organic %: Mobile phase composition
    - Temperature: Column temperature
    - Flow Rate: mL/min
    - pH: Mobile phase pH
    
    Returns: Resolution score (higher is better)
    """
    organic, temp, flow, ph = params
    
    # Simulate a complex response surface
    resolution = (
        10 * np.sin(organic / 10) 
        + 5 * np.cos(temp / 10)
        - 0.5 * (flow - 1.0)**2
        - 0.3 * (ph - 6.5)**2
        + np.random.normal(0, 0.5)  # Experimental noise
    )
    
    return resolution


def main():
    """Demonstrate batch acquisition for parallel experiments."""
    
    # Define the search space
    space = Space([
        Real(5, 95, name='Organic %'),
        Real(25, 60, name='Temperature'),
        Real(0.2, 2.0, name='Flow Rate'),
        Real(2.0, 9.0, name='pH'),
    ])
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        space, 
        maximize=True,
        n_initial=5,  # Random initial experiments
        random_state=42
    )
    
    print("=" * 70)
    print("Chromatography Method Optimization with Batch Acquisition")
    print("=" * 70)
    
    # Phase 1: Initial experiments (can be done in parallel)
    print("\nPhase 1: Initial Experiments (Sequential for Demo)")
    print("-" * 70)
    
    for i in range(5):
        x = optimizer.suggest()
        y = chromatography_objective(x)
        optimizer.tell(x, y)
        
        print(f"Trial {i+1}:")
        print(f"  Organic: {x[0]:.1f}%, Temp: {x[1]:.1f}°C, "
              f"Flow: {x[2]:.2f} mL/min, pH: {x[3]:.1f}")
        print(f"  Resolution: {y:.2f}")
    
    # Phase 2: Bayesian optimization with batch suggestions
    print("\n" + "=" * 70)
    print("Phase 2: Bayesian Optimization with Batch Acquisition")
    print("=" * 70)
    
    n_iterations = 3
    batch_size = 3
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}: Requesting {batch_size} parallel experiments")
        print("-" * 70)
        
        # Get batch suggestions using different strategies
        strategies = ['local_penalization', 'constant_liar', 'qei']
        strategy = strategies[iteration % len(strategies)]
        
        print(f"Strategy: {strategy}")
        
        batch = suggest_batch(
            optimizer,
            n_points=batch_size,
            strategy=strategy,
        )
        
        # "Evaluate" all points in the batch (simulating parallel experiments)
        print("\nSuggested experiments:")
        for i, x in enumerate(batch):
            y = chromatography_objective(x)
            optimizer.tell(x, y)
            
            print(f"\n  Experiment {i+1}:")
            print(f"    Organic: {x[0]:.1f}%, Temp: {x[1]:.1f}°C, "
                  f"Flow: {x[2]:.2f} mL/min, pH: {x[3]:.1f}")
            print(f"    Resolution: {y:.2f}")
    
    # Final results
    result = optimizer.get_result()
    
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"\nBest parameters found:")
    print(f"  Organic: {result.x_best[0]:.1f}%")
    print(f"  Temperature: {result.x_best[1]:.1f}°C")
    print(f"  Flow Rate: {result.x_best[2]:.2f} mL/min")
    print(f"  pH: {result.x_best[3]:.1f}")
    print(f"\nBest Resolution: {result.y_best:.2f}")
    print(f"Total experiments: {result.n_iterations}")
    
    # Demonstrate different batch strategies
    print("\n" + "=" * 70)
    print("Comparing Batch Strategies")
    print("=" * 70)
    
    print("\n1. Local Penalization:")
    print("   - Promotes diversity by penalizing points near existing suggestions")
    print("   - Good for broad exploration of the space")
    
    batch_lp = suggest_batch(optimizer, n_points=3, strategy='local_penalization')
    print(f"   Suggested {len(batch_lp)} diverse points")
    
    print("\n2. Constant Liar (pessimistic):")
    print("   - Assumes suggested points will perform poorly")
    print("   - Encourages exploration of different regions")
    
    batch_cl = suggest_batch(
        optimizer, 
        n_points=3, 
        strategy='constant_liar',
        liar_strategy='min'
    )
    print(f"   Suggested {len(batch_cl)} exploratory points")
    
    print("\n3. q-Expected Improvement:")
    print("   - Optimizes the joint acquisition value")
    print("   - Balances diversity and exploitation")
    
    batch_qei = suggest_batch(optimizer, n_points=3, strategy='qei')
    print(f"   Suggested {len(batch_qei)} balanced points")
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("• Batch acquisition enables parallel experimentation")
    print("• Saves time when experiments can be run simultaneously")
    print("• Different strategies balance exploration vs exploitation")
    print("• Local penalization generally provides good diversity")
    print("• Use 'constant_liar' for more aggressive exploration")
    print("• Use 'qei' when you want optimal batch selection (slower)")


if __name__ == "__main__":
    main()
