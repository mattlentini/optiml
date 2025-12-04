"""
Hyperparameter Tuning Example
=============================

This example shows how to use OptiML for hyperparameter tuning
of a machine learning model using scikit-learn.
"""

from optiml import BayesianOptimizer, Space, Real, Integer, Categorical
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def main():
    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    # Define the hyperparameter search space
    space = Space([
        Integer(10, 200, name="n_estimators"),
        Integer(2, 20, name="max_depth"),
        Integer(2, 20, name="min_samples_split"),
        Integer(1, 10, name="min_samples_leaf"),
        Categorical(["gini", "entropy"], name="criterion"),
    ])

    def objective(params):
        """Train and evaluate a Random Forest with given hyperparameters."""
        n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion = params

        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            criterion=criterion,
            random_state=42,
            n_jobs=-1,
        )

        # 5-fold cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        return scores.mean()

    # Create optimizer
    optimizer = BayesianOptimizer(
        space,
        n_initial=10,     # More initial samples for high-dimensional space
        maximize=True,    # Maximize accuracy
        random_state=42,
    )

    # Run optimization
    print("Hyperparameter Optimization for Random Forest")
    print("=" * 50)
    print("\nStarting optimization...")

    best_so_far = float("-inf")

    def callback(params, value, iteration):
        nonlocal best_so_far
        is_best = value > best_so_far
        if is_best:
            best_so_far = value
        marker = " *" if is_best else ""
        print(f"Iter {iteration + 1:2d}: accuracy={value:.4f}{marker}")

    result = optimizer.optimize(
        objective,
        n_iterations=30,
        callback=callback,
    )

    # Print final results
    print("\n" + "=" * 50)
    print("Optimization Complete!")
    print("=" * 50)

    param_names = space.dimension_names
    print("\nBest hyperparameters:")
    for name, value in zip(param_names, result.x_best):
        print(f"  {name}: {value}")

    print(f"\nBest cross-validation accuracy: {result.y_best:.4f}")


if __name__ == "__main__":
    main()
