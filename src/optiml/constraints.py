"""
Constraint Handling for Bayesian Optimization.

This module provides constraint handling mechanisms for
optimization with constraints, including:
- Known (analytical) constraints
- Unknown (black-box) constraints modeled with GP
- Probabilistic feasibility constraints
- Penalty-based approaches
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from optiml.surrogate import GaussianProcessSurrogate


class Constraint(ABC):
    """Abstract base class for constraints.

    Constraints define feasible regions of the search space.
    A point x is feasible if constraint(x) <= 0.
    """

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate constraint at given points.

        Parameters
        ----------
        X : np.ndarray
            Points to evaluate, shape (n_samples, n_dims) or (n_dims,).

        Returns
        -------
        np.ndarray
            Constraint values, shape (n_samples,). Negative = feasible.
        """
        pass

    @abstractmethod
    def is_feasible(self, X: np.ndarray) -> np.ndarray:
        """Check if points are feasible.

        Parameters
        ----------
        X : np.ndarray
            Points to check.

        Returns
        -------
        np.ndarray
            Boolean array, True if feasible.
        """
        pass


class LinearConstraint(Constraint):
    """Linear constraint: a^T * x <= b.

    Parameters
    ----------
    a : np.ndarray
        Coefficient vector.
    b : float
        Upper bound.
    name : str, optional
        Constraint name.

    Examples
    --------
    >>> # x1 + 2*x2 <= 10
    >>> constraint = LinearConstraint(a=[1, 2], b=10)
    >>> constraint.is_feasible([[1, 1], [5, 5]])
    array([ True, False])
    """

    def __init__(self, a: np.ndarray | List[float], b: float, name: str = "linear"):
        self.a = np.asarray(a)
        self.b = float(b)
        self.name = name

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return X @ self.a - self.b

    def is_feasible(self, X: np.ndarray) -> np.ndarray:
        return self(X) <= 0

    def __repr__(self) -> str:
        return f"LinearConstraint({self.a} · x <= {self.b})"


class NonlinearConstraint(Constraint):
    """Nonlinear constraint defined by a function: g(x) <= 0.

    Parameters
    ----------
    func : callable
        Constraint function. Returns scalar or array.
    name : str, optional
        Constraint name.

    Examples
    --------
    >>> # x1^2 + x2^2 <= 1 (unit circle)
    >>> constraint = NonlinearConstraint(lambda x: x[:, 0]**2 + x[:, 1]**2 - 1)
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name: str = "nonlinear"):
        self.func = func
        self.name = name

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        result = self.func(X)
        return np.asarray(result).flatten()

    def is_feasible(self, X: np.ndarray) -> np.ndarray:
        return self(X) <= 0

    def __repr__(self) -> str:
        return f"NonlinearConstraint(g(x) <= 0, name='{self.name}')"


class BoundConstraint(Constraint):
    """Box constraints: low <= x <= high.

    Parameters
    ----------
    low : np.ndarray
        Lower bounds for each dimension.
    high : np.ndarray
        Upper bounds for each dimension.

    Examples
    --------
    >>> constraint = BoundConstraint(low=[0, 0], high=[10, 10])
    """

    def __init__(self, low: np.ndarray | List[float], high: np.ndarray | List[float]):
        self.low = np.asarray(low)
        self.high = np.asarray(high)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        # Constraint satisfied when all dimensions in bounds
        # Return max violation (positive = infeasible)
        low_violation = self.low - X  # Positive when below lower bound
        high_violation = X - self.high  # Positive when above upper bound
        
        max_low = np.max(low_violation, axis=1)
        max_high = np.max(high_violation, axis=1)
        
        return np.maximum(max_low, max_high)

    def is_feasible(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.all((X >= self.low) & (X <= self.high), axis=1)

    def __repr__(self) -> str:
        return f"BoundConstraint({self.low} <= x <= {self.high})"


class SumConstraint(Constraint):
    """Sum constraint: sum(x[indices]) <=/>=/>= bound.

    Parameters
    ----------
    bound : float
        Bound value.
    constraint_type : str
        One of '<=', '>=', '=='.
    indices : list[int], optional
        Indices to sum. If None, sums all.
    tolerance : float
        Tolerance for equality constraints.

    Examples
    --------
    >>> # x1 + x2 + x3 <= 100 (budget constraint)
    >>> constraint = SumConstraint(bound=100, constraint_type='<=')
    """

    def __init__(
        self,
        bound: float,
        constraint_type: str = "<=",
        indices: List[int] | None = None,
        tolerance: float = 1e-6,
        name: str = "sum",
    ):
        if constraint_type not in ["<=", ">=", "=="]:
            raise ValueError("constraint_type must be '<=', '>=', or '=='")
        
        self.bound = float(bound)
        self.constraint_type = constraint_type
        self.indices = indices
        self.tolerance = tolerance
        self.name = name

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        
        if self.indices is not None:
            sums = X[:, self.indices].sum(axis=1)
        else:
            sums = X.sum(axis=1)
        
        if self.constraint_type == "<=":
            return sums - self.bound
        elif self.constraint_type == ">=":
            return self.bound - sums
        else:  # ==
            return np.abs(sums - self.bound) - self.tolerance

    def is_feasible(self, X: np.ndarray) -> np.ndarray:
        return self(X) <= 0

    def __repr__(self) -> str:
        return f"SumConstraint(sum(x) {self.constraint_type} {self.bound})"


@dataclass
class BlackBoxConstraint:
    """Black-box constraint modeled with a Gaussian Process.

    For expensive-to-evaluate constraints where we build a surrogate
    model to predict feasibility.

    Attributes
    ----------
    surrogate : GaussianProcessSurrogate
        GP model for constraint function.
    X_observed : np.ndarray
        Observed constraint evaluation points.
    g_observed : np.ndarray
        Observed constraint values (g(x) <= 0 is feasible).
    threshold : float
        Feasibility threshold (typically 0).
    name : str
        Constraint name.
    """

    surrogate: Optional["GaussianProcessSurrogate"] = None
    X_observed: np.ndarray = field(default_factory=lambda: np.array([]))
    g_observed: np.ndarray = field(default_factory=lambda: np.array([]))
    threshold: float = 0.0
    name: str = "blackbox"

    def __post_init__(self):
        if len(self.X_observed) == 0:
            self.X_observed = np.array([]).reshape(0, 0)
        if len(self.g_observed) == 0:
            self.g_observed = np.array([])

    def add_observation(self, x: np.ndarray, g: float) -> None:
        """Add a constraint observation.

        Parameters
        ----------
        x : np.ndarray
            Point where constraint was evaluated.
        g : float
            Constraint value.
        """
        x = np.atleast_2d(x)
        
        if self.X_observed.size == 0:
            self.X_observed = x
        else:
            self.X_observed = np.vstack([self.X_observed, x])
        
        self.g_observed = np.append(self.g_observed, g)

    def fit(self) -> None:
        """Fit the GP surrogate to observed data."""
        if self.surrogate is None:
            from optiml.surrogate import GaussianProcessSurrogate
            self.surrogate = GaussianProcessSurrogate()
        
        if len(self.g_observed) > 0:
            self.surrogate.fit(self.X_observed, self.g_observed)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict constraint value and uncertainty.

        Parameters
        ----------
        X : np.ndarray
            Points to predict.

        Returns
        -------
        mean : np.ndarray
            Predicted constraint values.
        std : np.ndarray
            Prediction uncertainty.
        """
        if self.surrogate is None or len(self.g_observed) == 0:
            X = np.atleast_2d(X)
            return np.zeros(len(X)), np.ones(len(X)) * np.inf
        
        return self.surrogate.predict(X)

    def probability_of_feasibility(self, X: np.ndarray) -> np.ndarray:
        """Compute probability that constraint is satisfied.

        P(g(x) <= threshold)

        Parameters
        ----------
        X : np.ndarray
            Points to evaluate.

        Returns
        -------
        np.ndarray
            Probability of feasibility for each point.
        """
        mean, std = self.predict(X)
        
        # Handle zero std (at observed points)
        std = np.maximum(std, 1e-10)
        
        # P(g <= threshold) = Phi((threshold - mean) / std)
        z = (self.threshold - mean) / std
        return norm.cdf(z)


class ConstraintHandler:
    """Handler for managing multiple constraints.

    Combines multiple constraints and provides methods for
    checking feasibility and computing constraint violations.

    Parameters
    ----------
    constraints : list[Constraint], optional
        List of constraints.
    black_box_constraints : list[BlackBoxConstraint], optional
        List of black-box constraints.

    Examples
    --------
    >>> handler = ConstraintHandler([
    ...     LinearConstraint([1, 1], 10),
    ...     NonlinearConstraint(lambda x: x[:, 0]**2 + x[:, 1]**2 - 4),
    ... ])
    >>> handler.is_feasible([[1, 1], [5, 5]])
    array([ True, False])
    """

    def __init__(
        self,
        constraints: List[Constraint] | None = None,
        black_box_constraints: List[BlackBoxConstraint] | None = None,
    ):
        self.constraints = constraints or []
        self.black_box_constraints = black_box_constraints or []

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a known constraint."""
        self.constraints.append(constraint)

    def add_black_box_constraint(self, constraint: BlackBoxConstraint) -> None:
        """Add a black-box constraint."""
        self.black_box_constraints.append(constraint)

    def is_feasible(self, X: np.ndarray, include_black_box: bool = True) -> np.ndarray:
        """Check if points are feasible.

        Parameters
        ----------
        X : np.ndarray
            Points to check.
        include_black_box : bool, default=True
            Whether to include black-box constraint predictions.

        Returns
        -------
        np.ndarray
            Boolean array, True if feasible.
        """
        X = np.atleast_2d(X)
        feasible = np.ones(len(X), dtype=bool)
        
        for constraint in self.constraints:
            feasible &= constraint.is_feasible(X)
        
        if include_black_box:
            for bb_constraint in self.black_box_constraints:
                mean, _ = bb_constraint.predict(X)
                feasible &= (mean <= bb_constraint.threshold)
        
        return feasible

    def constraint_violation(self, X: np.ndarray) -> np.ndarray:
        """Compute total constraint violation.

        Parameters
        ----------
        X : np.ndarray
            Points to evaluate.

        Returns
        -------
        np.ndarray
            Total violation (0 if feasible, positive if infeasible).
        """
        X = np.atleast_2d(X)
        violation = np.zeros(len(X))
        
        for constraint in self.constraints:
            cv = constraint(X)
            violation += np.maximum(0, cv)
        
        for bb_constraint in self.black_box_constraints:
            mean, _ = bb_constraint.predict(X)
            violation += np.maximum(0, mean - bb_constraint.threshold)
        
        return violation

    def probability_of_feasibility(self, X: np.ndarray) -> np.ndarray:
        """Compute probability of feasibility for black-box constraints.

        For known constraints, returns 1 if feasible, 0 otherwise.
        For black-box constraints, returns P(g(x) <= 0).

        Parameters
        ----------
        X : np.ndarray
            Points to evaluate.

        Returns
        -------
        np.ndarray
            Probability of feasibility (product over all constraints).
        """
        X = np.atleast_2d(X)
        prob = np.ones(len(X))
        
        # Known constraints: binary feasibility
        for constraint in self.constraints:
            prob *= constraint.is_feasible(X).astype(float)
        
        # Black-box constraints: probabilistic
        for bb_constraint in self.black_box_constraints:
            prob *= bb_constraint.probability_of_feasibility(X)
        
        return prob

    def filter_feasible(self, X: np.ndarray) -> np.ndarray:
        """Filter to only feasible points.

        Parameters
        ----------
        X : np.ndarray
            Points to filter.

        Returns
        -------
        np.ndarray
            Only feasible points.
        """
        X = np.atleast_2d(X)
        mask = self.is_feasible(X)
        return X[mask]

    @property
    def n_constraints(self) -> int:
        """Total number of constraints."""
        return len(self.constraints) + len(self.black_box_constraints)

    def fit_black_box(self) -> None:
        """Fit all black-box constraint surrogates."""
        for bb_constraint in self.black_box_constraints:
            bb_constraint.fit()


class PenaltyMethod:
    """Penalty-based constraint handling.

    Converts constrained optimization to unconstrained by adding
    penalty terms for constraint violations.

    Parameters
    ----------
    constraint_handler : ConstraintHandler
        Handler containing constraints.
    penalty_weight : float, default=1000.0
        Penalty multiplier for violations.
    penalty_type : str, default="quadratic"
        Type of penalty: "linear", "quadratic", or "log_barrier".

    Examples
    --------
    >>> handler = ConstraintHandler([LinearConstraint([1, 1], 10)])
    >>> penalty = PenaltyMethod(handler, penalty_weight=100)
    >>> penalized_y = penalty.apply(y, X)
    """

    def __init__(
        self,
        constraint_handler: ConstraintHandler,
        penalty_weight: float = 1000.0,
        penalty_type: str = "quadratic",
    ):
        self.constraint_handler = constraint_handler
        self.penalty_weight = penalty_weight
        
        if penalty_type not in ["linear", "quadratic", "log_barrier"]:
            raise ValueError("penalty_type must be 'linear', 'quadratic', or 'log_barrier'")
        self.penalty_type = penalty_type

    def compute_penalty(self, X: np.ndarray) -> np.ndarray:
        """Compute penalty for constraint violations.

        Parameters
        ----------
        X : np.ndarray
            Points to evaluate.

        Returns
        -------
        np.ndarray
            Penalty values.
        """
        violation = self.constraint_handler.constraint_violation(X)
        
        if self.penalty_type == "linear":
            return self.penalty_weight * violation
        elif self.penalty_type == "quadratic":
            return self.penalty_weight * violation ** 2
        else:  # log_barrier
            # Log barrier: -log(-g(x)) for g(x) < 0
            # For violations, use large penalty
            feasible = violation <= 0
            penalty = np.zeros_like(violation)
            penalty[~feasible] = self.penalty_weight * violation[~feasible] ** 2
            penalty[feasible] = -self.penalty_weight * 0.01 * np.log(1e-6 - violation[feasible] + 1e-10)
            return penalty

    def apply(
        self,
        y: np.ndarray,
        X: np.ndarray,
        minimize: bool = True,
    ) -> np.ndarray:
        """Apply penalty to objective values.

        Parameters
        ----------
        y : np.ndarray
            Original objective values.
        X : np.ndarray
            Corresponding parameter values.
        minimize : bool, default=True
            Whether optimization is minimizing.

        Returns
        -------
        np.ndarray
            Penalized objective values.
        """
        penalty = self.compute_penalty(X)
        
        if minimize:
            return y + penalty
        else:
            return y - penalty


class ConstrainedExpectedImprovement:
    """Expected Improvement weighted by probability of feasibility.

    cEI(x) = EI(x) * P(feasible(x))

    This is a common approach for handling black-box constraints
    in Bayesian Optimization.

    Parameters
    ----------
    constraint_handler : ConstraintHandler
        Handler containing constraints.

    Examples
    --------
    >>> handler = ConstraintHandler(black_box_constraints=[bb_constraint])
    >>> cei = ConstrainedExpectedImprovement(handler)
    >>> weighted_ei = cei.apply(ei_values, X)
    """

    def __init__(self, constraint_handler: ConstraintHandler):
        self.constraint_handler = constraint_handler

    def apply(self, acquisition_values: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Weight acquisition values by probability of feasibility.

        Parameters
        ----------
        acquisition_values : np.ndarray
            Original acquisition values (e.g., EI).
        X : np.ndarray
            Corresponding points.

        Returns
        -------
        np.ndarray
            Weighted acquisition values.
        """
        pof = self.constraint_handler.probability_of_feasibility(X)
        return acquisition_values * pof

    def __call__(self, acquisition_values: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Alias for apply()."""
        return self.apply(acquisition_values, X)


def sample_feasible_points(
    constraint_handler: ConstraintHandler,
    bounds: np.ndarray,
    n_samples: int,
    max_attempts: int = 10000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample uniformly from the feasible region.

    Uses rejection sampling.

    Parameters
    ----------
    constraint_handler : ConstraintHandler
        Handler containing constraints.
    bounds : np.ndarray
        Bounds for each dimension, shape (n_dims, 2).
    n_samples : int
        Number of feasible samples desired.
    max_attempts : int, default=10000
        Maximum sampling attempts.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Feasible samples, shape (n_samples, n_dims).

    Raises
    ------
    RuntimeError
        If unable to generate enough feasible samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    bounds = np.asarray(bounds)
    n_dims = len(bounds)
    
    feasible_samples = []
    attempts = 0
    
    while len(feasible_samples) < n_samples and attempts < max_attempts:
        # Generate batch
        batch_size = min(1000, max_attempts - attempts)
        samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(batch_size, n_dims))
        
        # Filter feasible
        mask = constraint_handler.is_feasible(samples, include_black_box=False)
        feasible = samples[mask]
        
        feasible_samples.extend(feasible[:n_samples - len(feasible_samples)])
        attempts += batch_size
    
    if len(feasible_samples) < n_samples:
        raise RuntimeError(
            f"Could only generate {len(feasible_samples)} feasible samples "
            f"out of {n_samples} requested after {attempts} attempts."
        )
    
    return np.array(feasible_samples)
