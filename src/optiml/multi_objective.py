"""
Multi-Objective Bayesian Optimization.

This module provides tools for optimizing multiple objectives
simultaneously, including:
- Pareto front computation
- Multi-objective acquisition functions (EHVI, ParEGO, NSGA-II-based)
- Scalarization methods
- Hypervolume calculations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from optiml.surrogate import GaussianProcessSurrogate


@dataclass
class ParetoFront:
    """Container for Pareto front solutions.

    Attributes
    ----------
    X : np.ndarray
        Pareto-optimal parameter configurations.
    Y : np.ndarray
        Corresponding objective values.
    n_objectives : int
        Number of objectives.
    """

    X: np.ndarray
    Y: np.ndarray
    n_objectives: int

    @property
    def n_points(self) -> int:
        """Number of Pareto-optimal points."""
        return len(self.X)

    def dominated_hypervolume(
        self,
        reference_point: np.ndarray | None = None,
    ) -> float:
        """Compute hypervolume dominated by Pareto front.

        Parameters
        ----------
        reference_point : np.ndarray, optional
            Reference point for hypervolume. If None, uses max + margin.

        Returns
        -------
        float
            Hypervolume indicator.
        """
        if reference_point is None:
            reference_point = np.max(self.Y, axis=0) + 1.0
        
        return compute_hypervolume(self.Y, reference_point)

    def crowding_distance(self) -> np.ndarray:
        """Compute crowding distance for diversity preservation.

        Returns
        -------
        np.ndarray
            Crowding distance for each point.
        """
        return compute_crowding_distance(self.Y)

    def select_by_crowding(self, n_select: int) -> Tuple[np.ndarray, np.ndarray]:
        """Select points using crowding distance.

        Parameters
        ----------
        n_select : int
            Number of points to select.

        Returns
        -------
        X_selected : np.ndarray
            Selected parameter configurations.
        Y_selected : np.ndarray
            Selected objective values.
        """
        if n_select >= self.n_points:
            return self.X, self.Y
        
        distances = self.crowding_distance()
        selected_idx = np.argsort(-distances)[:n_select]  # Highest distance first
        
        return self.X[selected_idx], self.Y[selected_idx]


def is_pareto_optimal(
    Y: np.ndarray,
    minimize: bool = True,
) -> np.ndarray:
    """Determine which points are Pareto optimal.

    Parameters
    ----------
    Y : np.ndarray
        Objective values, shape (n_samples, n_objectives).
    minimize : bool, default=True
        Whether objectives are to be minimized.

    Returns
    -------
    np.ndarray
        Boolean mask of Pareto-optimal points.

    Examples
    --------
    >>> Y = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [2.5, 2.5]])
    >>> mask = is_pareto_optimal(Y, minimize=True)
    >>> print(mask)  # [True, True, True, True, False]
    """
    Y = np.atleast_2d(Y)
    n_samples = len(Y)
    is_optimal = np.ones(n_samples, dtype=bool)
    
    for i in range(n_samples):
        if is_optimal[i]:
            # Check if any other point dominates point i
            for j in range(n_samples):
                if i != j and is_optimal[j]:
                    if minimize:
                        # j dominates i if j <= i in all and j < i in at least one
                        if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                            is_optimal[i] = False
                            break
                    else:
                        if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                            is_optimal[i] = False
                            break
    
    return is_optimal


def compute_pareto_front(
    X: np.ndarray,
    Y: np.ndarray,
    minimize: bool = True,
) -> ParetoFront:
    """Extract Pareto front from observed data.

    Parameters
    ----------
    X : np.ndarray
        Parameter configurations.
    Y : np.ndarray
        Objective values.
    minimize : bool, default=True
        Whether objectives are minimized.

    Returns
    -------
    ParetoFront
        Pareto front container.
    """
    Y = np.atleast_2d(Y)
    X = np.atleast_2d(X)
    
    mask = is_pareto_optimal(Y, minimize)
    
    return ParetoFront(
        X=X[mask],
        Y=Y[mask],
        n_objectives=Y.shape[1],
    )


def compute_hypervolume(
    Y: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """Compute hypervolume indicator (exact for 2D, approximate for higher).

    Parameters
    ----------
    Y : np.ndarray
        Pareto front objective values, shape (n_points, n_objectives).
    reference_point : np.ndarray
        Reference point (should dominate all Pareto points).

    Returns
    -------
    float
        Hypervolume indicator.

    Notes
    -----
    For 2D, uses exact computation. For higher dimensions, uses
    Monte Carlo approximation.
    """
    Y = np.atleast_2d(Y)
    reference_point = np.asarray(reference_point)
    
    n_points, n_objectives = Y.shape
    
    if n_points == 0:
        return 0.0
    
    if n_objectives == 2:
        return _hypervolume_2d(Y, reference_point)
    else:
        return _hypervolume_monte_carlo(Y, reference_point, n_samples=10000)


def _hypervolume_2d(Y: np.ndarray, reference_point: np.ndarray) -> float:
    """Exact 2D hypervolume computation."""
    # Sort by first objective
    sorted_idx = np.argsort(Y[:, 0])
    Y_sorted = Y[sorted_idx]
    
    hv = 0.0
    prev_y2 = reference_point[1]
    
    for i in range(len(Y_sorted)):
        y1, y2 = Y_sorted[i]
        if y1 < reference_point[0] and y2 < prev_y2:
            width = reference_point[0] - y1 if i == len(Y_sorted) - 1 else Y_sorted[i + 1, 0] - y1
            height = prev_y2 - y2
            hv += width * height
            prev_y2 = y2
    
    # Handle last segment
    if len(Y_sorted) > 0:
        hv += (reference_point[0] - Y_sorted[-1, 0]) * (prev_y2 - Y_sorted[-1, 1])
    
    return max(0.0, hv)


def _hypervolume_monte_carlo(
    Y: np.ndarray,
    reference_point: np.ndarray,
    n_samples: int = 10000,
) -> float:
    """Monte Carlo hypervolume approximation for higher dimensions."""
    n_objectives = len(reference_point)
    
    # Find dominated region bounds
    lower_bounds = np.min(Y, axis=0)
    
    # Total box volume
    box_volume = np.prod(reference_point - lower_bounds)
    
    if box_volume <= 0:
        return 0.0
    
    # Sample uniformly in box
    rng = np.random.default_rng(42)
    samples = rng.uniform(
        lower_bounds,
        reference_point,
        size=(n_samples, n_objectives),
    )
    
    # Count samples dominated by Pareto front
    dominated_count = 0
    for sample in samples:
        # Check if sample is dominated by any Pareto point
        for y in Y:
            if np.all(y <= sample):
                dominated_count += 1
                break
    
    # Estimate hypervolume
    return box_volume * dominated_count / n_samples


def compute_crowding_distance(Y: np.ndarray) -> np.ndarray:
    """Compute crowding distance for diversity preservation.

    Parameters
    ----------
    Y : np.ndarray
        Objective values, shape (n_points, n_objectives).

    Returns
    -------
    np.ndarray
        Crowding distance for each point.
    """
    Y = np.atleast_2d(Y)
    n_points, n_objectives = Y.shape
    
    if n_points <= 2:
        return np.full(n_points, np.inf)
    
    distances = np.zeros(n_points)
    
    for obj_idx in range(n_objectives):
        # Sort by this objective
        sorted_idx = np.argsort(Y[:, obj_idx])
        obj_range = Y[sorted_idx[-1], obj_idx] - Y[sorted_idx[0], obj_idx]
        
        if obj_range == 0:
            continue
        
        # Boundary points get infinite distance
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        
        # Interior points
        for i in range(1, n_points - 1):
            distances[sorted_idx[i]] += (
                Y[sorted_idx[i + 1], obj_idx] - Y[sorted_idx[i - 1], obj_idx]
            ) / obj_range
    
    return distances


@dataclass
class Scalarization:
    """Base class for scalarization methods.

    Scalarization converts multi-objective problems into
    single-objective problems using weights.
    """

    weights: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    
    def __call__(self, Y: np.ndarray) -> np.ndarray:
        """Scalarize objectives.

        Parameters
        ----------
        Y : np.ndarray
            Objective values, shape (n_samples, n_objectives).

        Returns
        -------
        np.ndarray
            Scalarized values, shape (n_samples,).
        """
        raise NotImplementedError


class WeightedSum(Scalarization):
    """Weighted sum scalarization.

    f(Y) = sum(w_i * Y_i)

    Parameters
    ----------
    weights : np.ndarray
        Weight for each objective.

    Examples
    --------
    >>> scalarize = WeightedSum(weights=[0.5, 0.5])
    >>> scalar_values = scalarize(Y)
    """

    def __call__(self, Y: np.ndarray) -> np.ndarray:
        Y = np.atleast_2d(Y)
        return Y @ self.weights


class Chebyshev(Scalarization):
    """Chebyshev (Tchebycheff) scalarization.

    f(Y) = max(w_i * |Y_i - ideal_i|)

    Better at finding Pareto points in non-convex regions.

    Parameters
    ----------
    weights : np.ndarray
        Weight for each objective.
    ideal_point : np.ndarray, optional
        Ideal point (best possible for each objective).

    Examples
    --------
    >>> scalarize = Chebyshev(weights=[0.5, 0.5], ideal_point=[0, 0])
    >>> scalar_values = scalarize(Y)
    """

    ideal_point: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    def __init__(
        self,
        weights: np.ndarray | List[float],
        ideal_point: np.ndarray | List[float] | None = None,
    ):
        self.weights = np.asarray(weights)
        if ideal_point is None:
            ideal_point = np.zeros_like(weights)
        self.ideal_point = np.asarray(ideal_point)

    def __call__(self, Y: np.ndarray) -> np.ndarray:
        Y = np.atleast_2d(Y)
        weighted_dist = self.weights * np.abs(Y - self.ideal_point)
        return np.max(weighted_dist, axis=1)


class AugmentedChebyshev(Scalarization):
    """Augmented Chebyshev scalarization.

    Adds a small weighted sum term to break ties and ensure
    Pareto optimality of solutions.

    f(Y) = max(w_i * |Y_i - ideal_i|) + rho * sum(w_i * |Y_i - ideal_i|)

    Parameters
    ----------
    weights : np.ndarray
        Weight for each objective.
    ideal_point : np.ndarray, optional
        Ideal point.
    rho : float, default=0.05
        Augmentation parameter.
    """

    def __init__(
        self,
        weights: np.ndarray | List[float],
        ideal_point: np.ndarray | List[float] | None = None,
        rho: float = 0.05,
    ):
        self.weights = np.asarray(weights)
        if ideal_point is None:
            ideal_point = np.zeros_like(weights)
        self.ideal_point = np.asarray(ideal_point)
        self.rho = rho

    def __call__(self, Y: np.ndarray) -> np.ndarray:
        Y = np.atleast_2d(Y)
        weighted_dist = self.weights * np.abs(Y - self.ideal_point)
        chebyshev = np.max(weighted_dist, axis=1)
        augmentation = self.rho * np.sum(weighted_dist, axis=1)
        return chebyshev + augmentation


class ParEGO:
    """ParEGO (Pareto Efficient Global Optimization) method.

    Uses random scalarization weights to explore the Pareto front.

    Parameters
    ----------
    n_objectives : int
        Number of objectives.
    rho : float, default=0.05
        Augmented Chebyshev parameter.
    rng : np.random.Generator, optional
        Random number generator.

    Examples
    --------
    >>> parego = ParEGO(n_objectives=2)
    >>> weights = parego.sample_weights()
    >>> scalarized = parego.scalarize(Y, weights)

    References
    ----------
    Knowles, J. (2006). ParEGO: A hybrid algorithm with on-line
    landscape approximation for expensive multiobjective optimization
    problems.
    """

    def __init__(
        self,
        n_objectives: int,
        rho: float = 0.05,
        rng: np.random.Generator | None = None,
    ):
        self.n_objectives = n_objectives
        self.rho = rho
        self.rng = rng or np.random.default_rng()

    def sample_weights(self) -> np.ndarray:
        """Sample random weights from simplex.

        Returns
        -------
        np.ndarray
            Weight vector that sums to 1.
        """
        # Sample from Dirichlet distribution (uniform on simplex)
        weights = self.rng.dirichlet(np.ones(self.n_objectives))
        return weights

    def scalarize(
        self,
        Y: np.ndarray,
        weights: np.ndarray | None = None,
        ideal_point: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scalarize objectives using augmented Chebyshev.

        Parameters
        ----------
        Y : np.ndarray
            Objective values.
        weights : np.ndarray, optional
            Weight vector. If None, samples new weights.
        ideal_point : np.ndarray, optional
            Ideal point. If None, uses column minimums.

        Returns
        -------
        np.ndarray
            Scalarized values.
        """
        Y = np.atleast_2d(Y)
        
        if weights is None:
            weights = self.sample_weights()
        
        if ideal_point is None:
            ideal_point = np.min(Y, axis=0)
        
        scalarizer = AugmentedChebyshev(weights, ideal_point, self.rho)
        return scalarizer(Y)


class ExpectedHypervolumeImprovement:
    """Expected Hypervolume Improvement acquisition function.

    Computes the expected improvement in hypervolume for
    multi-objective optimization.

    Parameters
    ----------
    pareto_front : ParetoFront
        Current Pareto front.
    reference_point : np.ndarray
        Reference point for hypervolume.
    n_samples : int, default=128
        Number of Monte Carlo samples.

    Notes
    -----
    Uses Monte Carlo sampling for approximation when n_objectives > 2.
    """

    def __init__(
        self,
        pareto_front: ParetoFront,
        reference_point: np.ndarray,
        n_samples: int = 128,
    ):
        self.pareto_front = pareto_front
        self.reference_point = np.asarray(reference_point)
        self.n_samples = n_samples
        
        # Pre-compute current hypervolume
        self.current_hv = pareto_front.dominated_hypervolume(reference_point)

    def __call__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Compute EHVI for candidate points.

        Parameters
        ----------
        mean : np.ndarray
            Predicted mean for each objective, shape (n_points, n_objectives).
        std : np.ndarray
            Predicted std for each objective, shape (n_points, n_objectives).
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            EHVI values, shape (n_points,).
        """
        if rng is None:
            rng = np.random.default_rng()
        
        mean = np.atleast_2d(mean)
        std = np.atleast_2d(std)
        
        n_points = len(mean)
        n_objectives = mean.shape[1]
        
        ehvi = np.zeros(n_points)
        
        for i in range(n_points):
            # Sample from predictive distribution
            samples = rng.normal(mean[i], std[i], size=(self.n_samples, n_objectives))
            
            hv_improvements = []
            for sample in samples:
                # Compute hypervolume with this new point
                new_front = np.vstack([self.pareto_front.Y, sample])
                
                # Filter to Pareto optimal
                pareto_mask = is_pareto_optimal(new_front)
                new_pareto = new_front[pareto_mask]
                
                new_hv = compute_hypervolume(new_pareto, self.reference_point)
                hv_improvements.append(max(0, new_hv - self.current_hv))
            
            ehvi[i] = np.mean(hv_improvements)
        
        return ehvi


class MultiObjectiveOptimizer:
    """Multi-objective Bayesian Optimization.

    Parameters
    ----------
    n_objectives : int
        Number of objectives.
    surrogates : list[GaussianProcessSurrogate], optional
        GP surrogates for each objective.
    method : str, default="parego"
        Multi-objective method: "parego", "ehvi", or "random_scalarization".

    Examples
    --------
    >>> optimizer = MultiObjectiveOptimizer(n_objectives=2)
    >>> optimizer.fit(X_train, Y_train)
    >>> next_x = optimizer.suggest(bounds)
    """

    def __init__(
        self,
        n_objectives: int,
        surrogates: List["GaussianProcessSurrogate"] | None = None,
        method: str = "parego",
        rng: np.random.Generator | None = None,
    ):
        self.n_objectives = n_objectives
        self.method = method
        self.rng = rng or np.random.default_rng()
        
        if surrogates is not None:
            self.surrogates = surrogates
        else:
            self.surrogates = []
        
        self.X_observed: np.ndarray = np.array([])
        self.Y_observed: np.ndarray = np.array([])
        self._pareto_front: Optional[ParetoFront] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit surrogate models to observed data.

        Parameters
        ----------
        X : np.ndarray
            Parameter configurations.
        Y : np.ndarray
            Objective values, shape (n_samples, n_objectives).
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        
        self.X_observed = X
        self.Y_observed = Y
        
        # Initialize surrogates if needed
        if len(self.surrogates) == 0:
            from optiml.surrogate import GaussianProcessSurrogate
            self.surrogates = [GaussianProcessSurrogate() for _ in range(self.n_objectives)]
        
        # Fit each surrogate
        for i, surrogate in enumerate(self.surrogates):
            surrogate.fit(X, Y[:, i])
        
        # Update Pareto front
        self._pareto_front = compute_pareto_front(X, Y, minimize=True)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict objectives for new points.

        Parameters
        ----------
        X : np.ndarray
            Points to predict.

        Returns
        -------
        mean : np.ndarray
            Predicted means, shape (n_points, n_objectives).
        std : np.ndarray
            Predicted stds, shape (n_points, n_objectives).
        """
        X = np.atleast_2d(X)
        n_points = len(X)
        
        means = np.zeros((n_points, self.n_objectives))
        stds = np.zeros((n_points, self.n_objectives))
        
        for i, surrogate in enumerate(self.surrogates):
            m, s = surrogate.predict(X)
            means[:, i] = m
            stds[:, i] = s
        
        return means, stds

    def suggest(
        self,
        bounds: np.ndarray,
        n_candidates: int = 1000,
    ) -> np.ndarray:
        """Suggest next point to evaluate.

        Parameters
        ----------
        bounds : np.ndarray
            Parameter bounds, shape (n_dims, 2).
        n_candidates : int, default=1000
            Number of random candidates.

        Returns
        -------
        np.ndarray
            Suggested point.
        """
        bounds = np.asarray(bounds)
        n_dims = len(bounds)
        
        # Generate candidates
        candidates = self.rng.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(n_candidates, n_dims),
        )
        
        if self.method == "parego":
            return self._suggest_parego(candidates)
        elif self.method == "ehvi":
            return self._suggest_ehvi(candidates)
        else:
            # Random scalarization
            return self._suggest_random_scalarization(candidates)

    def _suggest_parego(self, candidates: np.ndarray) -> np.ndarray:
        """Suggest using ParEGO method."""
        parego = ParEGO(self.n_objectives, rng=self.rng)
        
        # Scalarize observed data
        weights = parego.sample_weights()
        scalarized_y = parego.scalarize(self.Y_observed, weights)
        
        # Fit single surrogate to scalarized objective
        from optiml.surrogate import GaussianProcessSurrogate
        surrogate = GaussianProcessSurrogate()
        surrogate.fit(self.X_observed, scalarized_y)
        
        # Predict on candidates
        mean, std = surrogate.predict(candidates)
        
        # Use standard EI
        best_so_far = np.min(scalarized_y)
        improvement = best_so_far - mean
        z = improvement / (std + 1e-10)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei[std < 1e-10] = 0.0
        
        return candidates[np.argmax(ei)]

    def _suggest_ehvi(self, candidates: np.ndarray) -> np.ndarray:
        """Suggest using Expected Hypervolume Improvement."""
        if self._pareto_front is None or self._pareto_front.n_points == 0:
            # No Pareto front yet, return random candidate
            return candidates[self.rng.integers(len(candidates))]
        
        # Reference point
        reference_point = np.max(self.Y_observed, axis=0) + 1.0
        
        ehvi_acq = ExpectedHypervolumeImprovement(
            self._pareto_front,
            reference_point,
            n_samples=64,  # Fewer samples for speed
        )
        
        # Predict on candidates
        means, stds = self.predict(candidates)
        
        # Compute EHVI
        ehvi_values = ehvi_acq(means, stds, self.rng)
        
        return candidates[np.argmax(ehvi_values)]

    def _suggest_random_scalarization(self, candidates: np.ndarray) -> np.ndarray:
        """Suggest using random weighted sum."""
        weights = self.rng.dirichlet(np.ones(self.n_objectives))
        
        # Scalarize observed data
        scalarized_y = self.Y_observed @ weights
        
        # Fit single surrogate
        from optiml.surrogate import GaussianProcessSurrogate
        surrogate = GaussianProcessSurrogate()
        surrogate.fit(self.X_observed, scalarized_y)
        
        # Use EI
        mean, std = surrogate.predict(candidates)
        best_so_far = np.min(scalarized_y)
        improvement = best_so_far - mean
        z = improvement / (std + 1e-10)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        ei[std < 1e-10] = 0.0
        
        return candidates[np.argmax(ei)]

    @property
    def pareto_front(self) -> Optional[ParetoFront]:
        """Get current Pareto front."""
        return self._pareto_front

    def get_pareto_optimal_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get Pareto-optimal X and Y values.

        Returns
        -------
        X_pareto : np.ndarray
            Pareto-optimal parameter configurations.
        Y_pareto : np.ndarray
            Corresponding objective values.
        """
        if self._pareto_front is None:
            if len(self.Y_observed) > 0:
                self._pareto_front = compute_pareto_front(
                    self.X_observed, self.Y_observed, minimize=True
                )
            else:
                return np.array([]), np.array([])
        
        return self._pareto_front.X, self._pareto_front.Y


def generate_weight_vectors(
    n_objectives: int,
    n_divisions: int = 5,
) -> np.ndarray:
    """Generate uniformly distributed weight vectors.

    Uses Das and Dennis's method for generating reference directions.

    Parameters
    ----------
    n_objectives : int
        Number of objectives.
    n_divisions : int, default=5
        Number of divisions along each axis.

    Returns
    -------
    np.ndarray
        Weight vectors, shape (n_vectors, n_objectives).

    Examples
    --------
    >>> weights = generate_weight_vectors(3, n_divisions=4)
    >>> print(weights.shape)  # (15, 3)
    """
    from itertools import combinations_with_replacement
    
    # Generate all combinations that sum to n_divisions
    weights = []
    for combo in combinations_with_replacement(range(n_divisions + 1), n_objectives - 1):
        # Add boundaries
        values = [0] + list(combo) + [n_divisions]
        # Compute differences
        diffs = [values[i+1] - values[i] for i in range(len(values) - 1)]
        weight = np.array(diffs) / n_divisions
        weights.append(weight)
    
    return np.array(weights)
