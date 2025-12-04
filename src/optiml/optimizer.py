"""Main Bayesian optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from optiml.acquisition import AcquisitionFunction, ExpectedImprovement
from optiml.space import Space
from optiml.surrogate import GaussianProcessSurrogate, SurrogateModel


@dataclass
class OptimizationResult:
    """Container for optimization results.

    Attributes
    ----------
    x_best : list[Any]
        Best parameters found.
    y_best : float
        Best objective value found.
    x_history : list[list[Any]]
        History of all evaluated parameter configurations.
    y_history : list[float]
        History of all objective values.
    n_iterations : int
        Number of iterations performed.
    """

    x_best: list[Any]
    y_best: float
    x_history: list[list[Any]] = field(default_factory=list)
    y_history: list[float] = field(default_factory=list)
    n_iterations: int = 0


class BayesianOptimizer:
    """Easy-to-use Bayesian Optimizer for black-box optimization.

    This optimizer uses a Gaussian Process surrogate model with an acquisition
    function to efficiently search for the optimum of expensive-to-evaluate
    objective functions.

    Parameters
    ----------
    space : Space
        The search space defining the parameters to optimize.
    surrogate : SurrogateModel, optional
        The surrogate model to use. Defaults to GaussianProcessSurrogate.
    acquisition : AcquisitionFunction, optional
        The acquisition function to use. Defaults to ExpectedImprovement.
    n_initial : int, default=5
        Number of random initial samples before starting Bayesian optimization.
    maximize : bool, default=True
        If True, maximize the objective. If False, minimize it.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    Basic usage with a simple function:

    >>> from optiml import BayesianOptimizer, Space, Real
    >>>
    >>> def objective(x):
    ...     return -(x[0] - 2)**2 - (x[1] - 3)**2
    >>>
    >>> space = Space([
    ...     Real(0, 5, name="x"),
    ...     Real(0, 5, name="y"),
    ... ])
    >>>
    >>> optimizer = BayesianOptimizer(space, maximize=True)
    >>> result = optimizer.optimize(objective, n_iterations=20)
    >>> print(f"Best parameters: {result.x_best}")
    >>> print(f"Best value: {result.y_best}")

    Hyperparameter tuning example:

    >>> from optiml import BayesianOptimizer, Space, Real, Integer, Categorical
    >>>
    >>> space = Space([
    ...     Real(1e-5, 1e-1, name="learning_rate", log_scale=True),
    ...     Integer(1, 5, name="n_layers"),
    ...     Categorical(["relu", "tanh", "sigmoid"], name="activation"),
    ... ])
    >>>
    >>> def train_model(params):
    ...     # Your model training code here
    ...     lr, n_layers, activation = params
    ...     # Return validation accuracy
    ...     return accuracy
    >>>
    >>> optimizer = BayesianOptimizer(space, maximize=True)
    >>> result = optimizer.optimize(train_model, n_iterations=50)
    """

    def __init__(
        self,
        space: Space,
        surrogate: SurrogateModel | None = None,
        acquisition: AcquisitionFunction | None = None,
        n_initial: int = 5,
        maximize: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.space = space
        self.surrogate = surrogate or GaussianProcessSurrogate()
        self.acquisition = acquisition or ExpectedImprovement()
        self.n_initial = n_initial
        self.maximize = maximize
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._X: list[list[Any]] = []
        self._y: list[float] = []

    def _suggest_initial(self) -> list[Any]:
        """Suggest a random initial point."""
        return self.space.sample(1, self._rng)[0]

    def _suggest_acquisition(self) -> list[Any]:
        """Suggest the next point by optimizing the acquisition function."""
        # Transform observed data to normalized space
        X_normalized = self.space.transform(self._X)
        y_array = np.array(self._y)

        # Flip sign if minimizing
        if not self.maximize:
            y_array = -y_array

        # Fit the surrogate model
        self.surrogate.fit(X_normalized, y_array)

        # Best observed value
        y_best = np.max(y_array)

        # Optimize acquisition function using multi-start L-BFGS-B
        best_x = None
        best_acq = -np.inf

        n_restarts = 20
        n_dims = len(self.space)

        for _ in range(n_restarts):
            # Random starting point in normalized space
            x0 = self._rng.uniform(0, 1, n_dims)

            def neg_acquisition(x: np.ndarray) -> float:
                # Clip to bounds
                x = np.clip(x, 0, 1)
                return -self.acquisition(x.reshape(1, -1), self.surrogate, y_best)[0]

            try:
                result = minimize(
                    neg_acquisition,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(0, 1)] * n_dims,
                )

                if -result.fun > best_acq:
                    best_acq = -result.fun
                    best_x = result.x

            except Exception:
                continue

        if best_x is None:
            # Fallback to random sampling
            return self._suggest_initial()

        # Transform back to original space
        best_x_clipped = np.clip(best_x, 0, 1)
        return self.space.inverse_transform(best_x_clipped.reshape(1, -1))[0]

    def suggest(self) -> list[Any]:
        """Suggest the next point to evaluate.

        Returns
        -------
        list[Any]
            The suggested parameter configuration.

        Examples
        --------
        >>> params = optimizer.suggest()
        >>> value = objective(params)
        >>> optimizer.tell(params, value)
        """
        if len(self._X) < self.n_initial:
            return self._suggest_initial()
        return self._suggest_acquisition()

    def tell(self, x: list[Any], y: float) -> None:
        """Record an observation.

        Parameters
        ----------
        x : list[Any]
            The evaluated parameter configuration.
        y : float
            The objective value.

        Examples
        --------
        >>> optimizer.tell([0.1, 2, "relu"], 0.95)
        """
        self._X.append(x)
        self._y.append(y)

    def optimize(
        self,
        objective: Callable[[list[Any]], float],
        n_iterations: int = 20,
        callback: Callable[[list[Any], float, int], None] | None = None,
    ) -> OptimizationResult:
        """Run the full optimization loop.

        Parameters
        ----------
        objective : Callable[[list[Any]], float]
            The objective function to optimize. Takes a list of parameters
            and returns a scalar value.
        n_iterations : int, default=20
            Total number of function evaluations.
        callback : Callable, optional
            A callback function called after each iteration with
            (params, value, iteration).

        Returns
        -------
        OptimizationResult
            The optimization results including best parameters and history.

        Examples
        --------
        >>> def objective(params):
        ...     x, y = params
        ...     return -(x - 2)**2 - (y - 3)**2
        >>>
        >>> result = optimizer.optimize(objective, n_iterations=50)
        >>> print(f"Best: {result.x_best} -> {result.y_best}")
        """
        for i in range(n_iterations):
            # Suggest next point
            x = self.suggest()

            # Evaluate objective
            y = objective(x)

            # Record observation
            self.tell(x, y)

            # Callback
            if callback is not None:
                callback(x, y, i)

        # Find best result
        y_array = np.array(self._y)
        if self.maximize:
            best_idx = np.argmax(y_array)
        else:
            best_idx = np.argmin(y_array)

        return OptimizationResult(
            x_best=self._X[best_idx],
            y_best=self._y[best_idx],
            x_history=self._X.copy(),
            y_history=self._y.copy(),
            n_iterations=n_iterations,
        )

    def get_result(self) -> OptimizationResult:
        """Get the current optimization result.

        Returns
        -------
        OptimizationResult
            The current optimization state.
        """
        if not self._X:
            raise RuntimeError("No observations recorded yet.")

        y_array = np.array(self._y)
        if self.maximize:
            best_idx = np.argmax(y_array)
        else:
            best_idx = np.argmin(y_array)

        return OptimizationResult(
            x_best=self._X[best_idx],
            y_best=self._y[best_idx],
            x_history=self._X.copy(),
            y_history=self._y.copy(),
            n_iterations=len(self._X),
        )

    def reset(self) -> None:
        """Reset the optimizer state, clearing all observations."""
        self._X = []
        self._y = []
        self._rng = np.random.default_rng(self.random_state)
