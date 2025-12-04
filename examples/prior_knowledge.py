"""
Example: Using Prior Knowledge for Transfer Learning
=====================================================

This example demonstrates how to leverage historical experiment data
to warm-start new optimizations, saving time and experiments.
"""

import numpy as np
from optiml import Space, Real
from optiml.priors import (
    ParameterPrior,
    ExperimentPrior,
    PriorAwareBayesianOptimizer,
)


def protein_purification_objective(params):
    """
    Simulated protein purification optimization.
    
    Parameters:
    - Salt concentration: mM
    - pH
    - Temperature: °C
    
    Returns: Yield % (higher is better)
    
    The optimal conditions are around: Salt=150mM, pH=7.5, Temp=20°C
    """
    salt, ph, temp = params
    
    # Simulate response with optimal region
    optimal_salt = 150
    optimal_ph = 7.5
    optimal_temp = 20
    
    yield_pct = (
        100 
        - 0.01 * (salt - optimal_salt)**2
        - 5 * (ph - optimal_ph)**2
        - 0.5 * (temp - optimal_temp)**2
        + np.random.normal(0, 2)  # Experimental noise
    )
    
    return max(0, min(100, yield_pct))  # Clamp to 0-100%


def simulate_historical_data():
    """
    Simulate historical experiments from previous purification studies.
    
    This represents accumulated knowledge from past projects.
    """
    # Historical optimal regions (slightly different proteins)
    historical_optima = [
        {'salt': 145, 'ph': 7.4, 'temp': 18},  # Similar protein 1
        {'salt': 155, 'ph': 7.6, 'temp': 22},  # Similar protein 2
        {'salt': 148, 'ph': 7.5, 'temp': 19},  # Similar protein 3
    ]
    
    # Build priors from historical data
    param_priors = {
        'Salt': ParameterPrior(
            name='Salt',
            param_type='real',
            mean_optimal=np.mean([h['salt'] for h in historical_optima]),
            std_optimal=np.std([h['salt'] for h in historical_optima]) + 10,  # Add some uncertainty
            best_values=[h['salt'] for h in historical_optima],
            confidence=0.7,
            low=50,
            high=300,
        ),
        'pH': ParameterPrior(
            name='pH',
            param_type='real',
            mean_optimal=np.mean([h['ph'] for h in historical_optima]),
            std_optimal=np.std([h['ph'] for h in historical_optima]) + 0.2,
            best_values=[h['ph'] for h in historical_optima],
            confidence=0.8,
            low=5.0,
            high=9.0,
        ),
        'Temperature': ParameterPrior(
            name='Temperature',
            param_type='real',
            mean_optimal=np.mean([h['temp'] for h in historical_optima]),
            std_optimal=np.std([h['temp'] for h in historical_optima]) + 5,
            best_values=[h['temp'] for h in historical_optima],
            confidence=0.6,
            low=4,
            high=37,
        ),
    }
    
    prior = ExperimentPrior(
        parameter_priors=param_priors,
        n_experiments=3,
        n_trials=45,  # 15 trials per experiment
        expected_best=92.0,  # Expected optimal yield
        objective_variance=4.0,
        metadata={'source': 'Previous purification studies'}
    )
    
    return prior


def run_optimization_without_prior(space):
    """Standard Bayesian optimization without prior knowledge."""
    from optiml import BayesianOptimizer
    
    print("\n" + "=" * 70)
    print("Optimization WITHOUT Prior Knowledge")
    print("=" * 70)
    
    optimizer = BayesianOptimizer(
        space,
        maximize=True,
        n_initial=5,
        random_state=42
    )
    
    # Run optimization
    n_trials = 20
    best_yield = 0
    
    for i in range(n_trials):
        x = optimizer.suggest()
        y = protein_purification_objective(x)
        optimizer.tell(x, y)
        
        if y > best_yield:
            best_yield = y
            print(f"Trial {i+1}: New best! Yield = {y:.1f}%")
            print(f"  Salt: {x[0]:.1f} mM, pH: {x[1]:.2f}, Temp: {x[2]:.1f}°C")
    
    result = optimizer.get_result()
    print(f"\nFinal Best Yield: {result.y_best:.1f}%")
    print(f"Conditions: Salt={result.x_best[0]:.1f}mM, "
          f"pH={result.x_best[1]:.2f}, Temp={result.x_best[2]:.1f}°C")
    
    return result


def run_optimization_with_prior(space, prior):
    """Bayesian optimization WITH prior knowledge."""
    
    print("\n" + "=" * 70)
    print("Optimization WITH Prior Knowledge (Transfer Learning)")
    print("=" * 70)
    
    print("\nPrior Knowledge Available:")
    print(f"  Similar experiments: {prior.n_experiments}")
    print(f"  Historical trials: {prior.n_trials}")
    print(f"  Expected best yield: {prior.expected_best:.1f}%")
    
    print("\nParameter Priors:")
    for name, p_prior in prior.parameter_priors.items():
        print(f"  {name}: μ={p_prior.mean_optimal:.1f}, "
              f"σ={p_prior.std_optimal:.1f}, "
              f"confidence={p_prior.confidence:.1%}")
    
    # Create prior-aware optimizer
    optimizer = PriorAwareBayesianOptimizer(
        space,
        prior=prior,
        prior_weight=0.6,  # 60% weight on prior, 40% on exploration
        maximize=True,
        n_initial=5,
        random_state=42
    )
    
    # Run optimization
    n_trials = 20
    best_yield = 0
    
    for i in range(n_trials):
        x = optimizer.suggest()
        y = protein_purification_objective(x)
        optimizer.tell(x, y)
        
        if y > best_yield:
            best_yield = y
            print(f"Trial {i+1}: New best! Yield = {y:.1f}%")
            print(f"  Salt: {x[0]:.1f} mM, pH: {x[1]:.2f}, Temp: {x[2]:.1f}°C")
    
    result = optimizer.get_result()
    print(f"\nFinal Best Yield: {result.y_best:.1f}%")
    print(f"Conditions: Salt={result.x_best[0]:.1f}mM, "
          f"pH={result.x_best[1]:.2f}, Temp={result.x_best[2]:.1f}°C")
    
    return result


def main():
    """Compare optimization with and without prior knowledge."""
    
    print("=" * 70)
    print("Prior Knowledge for Transfer Learning")
    print("Protein Purification Optimization Example")
    print("=" * 70)
    
    # Define search space
    space = Space([
        Real(50, 300, name='Salt'),
        Real(5.0, 9.0, name='pH'),
        Real(4, 37, name='Temperature'),
    ])
    
    # Get historical prior knowledge
    prior = simulate_historical_data()
    
    # Run both approaches
    result_no_prior = run_optimization_without_prior(space)
    result_with_prior = run_optimization_with_prior(space, prior)
    
    # Compare results
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    
    print(f"\nWithout Prior:")
    print(f"  Best Yield: {result_no_prior.y_best:.1f}%")
    print(f"  Conditions: Salt={result_no_prior.x_best[0]:.1f}mM, "
          f"pH={result_no_prior.x_best[1]:.2f}, Temp={result_no_prior.x_best[2]:.1f}°C")
    
    print(f"\nWith Prior Knowledge:")
    print(f"  Best Yield: {result_with_prior.y_best:.1f}%")
    print(f"  Conditions: Salt={result_with_prior.x_best[0]:.1f}mM, "
          f"pH={result_with_prior.x_best[1]:.2f}, Temp={result_with_prior.x_best[2]:.1f}°C")
    
    improvement = result_with_prior.y_best - result_no_prior.y_best
    print(f"\nImprovement: {improvement:+.1f}%")
    
    print("\n" + "=" * 70)
    print("Benefits of Prior Knowledge:")
    print("=" * 70)
    print("✓ Faster convergence to optimal region")
    print("✓ Fewer wasted experiments in poor regions")
    print("✓ Better final results with same number of trials")
    print("✓ Leverages accumulated organizational knowledge")
    print("✓ Reduces experimental costs and time")
    
    print("\n" + "=" * 70)
    print("When to Use Prior Knowledge:")
    print("=" * 70)
    print("• Optimizing similar systems (e.g., related proteins)")
    print("• When historical data exists from past projects")
    print("• In regulated industries (leverage validated methods)")
    print("• When experiments are expensive or time-consuming")
    print("• To incorporate expert knowledge into optimization")
    
    print("\n" + "=" * 70)
    print("Prior Weight Tuning:")
    print("=" * 70)
    print("• prior_weight=0.0: Ignore prior (pure exploration)")
    print("• prior_weight=0.3: Slight bias toward prior")
    print("• prior_weight=0.5: Balanced (recommended starting point)")
    print("• prior_weight=0.7: Strong trust in prior")
    print("• prior_weight=1.0: Maximum prior influence")


if __name__ == "__main__":
    main()
