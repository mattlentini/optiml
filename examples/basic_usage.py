"""
OptiML Basic Usage Example
==========================

This example demonstrates how to use the Bayesian optimizer
to find the minimum of a simple function.
"""

from optiml import BayesianOptimizer, Space, Real


def objective_function(params):
    """A simple 2D function with a known minimum at (2, 3).

    f(x, y) = (x - 2)^2 + (y - 3)^2

    Minimum value is 0 at (2, 3).
    """
    x, y = params
    return (x - 2) ** 2 + (y - 3) ** 2


def main():
    # Define the search space
    space = Space([
        Real(0, 5, name="x"),
        Real(0, 5, name="y"),
    ])

    # Create the optimizer (minimize mode)
    optimizer = BayesianOptimizer(
        space,
        n_initial=5,      # Number of random initial points
        maximize=False,   # We want to minimize
        random_state=42,  # For reproducibility
    )

    # Run optimization
    print("Starting optimization...")
    print("-" * 40)

    def callback(params, value, iteration):
        print(f"Iteration {iteration + 1:2d}: f({params[0]:.4f}, {params[1]:.4f}) = {value:.6f}")

    result = optimizer.optimize(
        objective_function,
        n_iterations=25,
        callback=callback,
    )

    # Print results
    print("-" * 40)
    print(f"\nOptimization completed!")
    print(f"Best parameters: x={result.x_best[0]:.4f}, y={result.x_best[1]:.4f}")
    print(f"Best value: {result.y_best:.6f}")
    print(f"True optimum: x=2.0, y=3.0, f=0.0")
    print(f"Error: {abs(result.x_best[0] - 2):.4f}, {abs(result.x_best[1] - 3):.4f}")


if __name__ == "__main__":
    main()
