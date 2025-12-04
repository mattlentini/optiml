"""Sensitivity analysis for Bayesian optimization.

This module provides tools for analyzing the sensitivity of objective functions
to input parameters, helping identify which parameters are most important.

Features:
- Sobol sensitivity indices (first-order and total)
- Morris screening method for parameter ranking
- Local sensitivity analysis
- ANOVA-based sensitivity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.stats import norm, spearmanr


@dataclass
class SobolIndices:
    """Results from Sobol sensitivity analysis.
    
    Attributes
    ----------
    first_order : np.ndarray
        First-order sensitivity indices (main effects).
    total : np.ndarray
        Total sensitivity indices (including interactions).
    second_order : dict, optional
        Second-order interaction indices.
    confidence_first : np.ndarray, optional
        Confidence intervals for first-order indices.
    confidence_total : np.ndarray, optional
        Confidence intervals for total indices.
    parameter_names : list of str, optional
        Names of parameters.
    """
    first_order: np.ndarray
    total: np.ndarray
    second_order: Optional[Dict[Tuple[int, int], float]] = None
    confidence_first: Optional[np.ndarray] = None
    confidence_total: Optional[np.ndarray] = None
    parameter_names: Optional[List[str]] = None
    
    def get_ranking(self, by: str = "total") -> List[Tuple[int, float]]:
        """Get parameter ranking by sensitivity.
        
        Parameters
        ----------
        by : str, default="total"
            Which index to rank by: "first_order" or "total".
            
        Returns
        -------
        list of (index, value) tuples
            Sorted from most to least sensitive.
        """
        if by == "first_order":
            indices = self.first_order
        else:
            indices = self.total
        
        ranking = sorted(enumerate(indices), key=lambda x: x[1], reverse=True)
        return ranking
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy inspection."""
        result = {
            "first_order": self.first_order.tolist(),
            "total": self.total.tolist(),
        }
        if self.parameter_names:
            result["parameter_names"] = self.parameter_names
        if self.second_order:
            result["second_order"] = {
                f"{i}_{j}": v for (i, j), v in self.second_order.items()
            }
        return result


@dataclass
class MorrisResult:
    """Results from Morris screening analysis.
    
    Attributes
    ----------
    mu : np.ndarray
        Mean of elementary effects (indicates overall importance).
    mu_star : np.ndarray
        Mean of absolute elementary effects (recommended for ranking).
    sigma : np.ndarray
        Standard deviation of elementary effects (indicates non-linearity/interactions).
    elementary_effects : list of np.ndarray
        Raw elementary effects for each trajectory.
    parameter_names : list of str, optional
        Names of parameters.
    """
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    elementary_effects: List[np.ndarray]
    parameter_names: Optional[List[str]] = None
    
    def get_ranking(self) -> List[Tuple[int, float]]:
        """Get parameter ranking by mu_star.
        
        Returns
        -------
        list of (index, value) tuples
            Sorted from most to least important.
        """
        ranking = sorted(enumerate(self.mu_star), key=lambda x: x[1], reverse=True)
        return ranking
    
    def classify_parameters(
        self,
        mu_star_threshold: float = 0.1,
        sigma_threshold: float = 0.1,
    ) -> Dict[str, List[int]]:
        """Classify parameters by their behavior.
        
        Parameters
        ----------
        mu_star_threshold : float
            Threshold for considering a parameter important.
        sigma_threshold : float
            Threshold for considering a parameter to have interactions.
            
        Returns
        -------
        dict
            Dictionary with categories: "negligible", "linear", "nonlinear".
        """
        # Normalize
        mu_star_norm = self.mu_star / (self.mu_star.max() + 1e-10)
        sigma_norm = self.sigma / (self.sigma.max() + 1e-10)
        
        negligible = []
        linear = []
        nonlinear = []
        
        for i in range(len(self.mu_star)):
            if mu_star_norm[i] < mu_star_threshold:
                negligible.append(i)
            elif sigma_norm[i] < sigma_threshold:
                linear.append(i)
            else:
                nonlinear.append(i)
        
        return {
            "negligible": negligible,
            "linear": linear,
            "nonlinear": nonlinear,
        }


@dataclass
class LocalSensitivity:
    """Results from local sensitivity analysis.
    
    Attributes
    ----------
    gradients : np.ndarray
        Gradient of objective w.r.t. each parameter at the point.
    normalized_gradients : np.ndarray
        Normalized sensitivity (elasticity).
    point : np.ndarray
        Point at which sensitivity was computed.
    parameter_names : list of str, optional
        Names of parameters.
    """
    gradients: np.ndarray
    normalized_gradients: np.ndarray
    point: np.ndarray
    parameter_names: Optional[List[str]] = None
    
    def get_ranking(self) -> List[Tuple[int, float]]:
        """Get parameter ranking by absolute normalized gradient."""
        abs_sens = np.abs(self.normalized_gradients)
        ranking = sorted(enumerate(abs_sens), key=lambda x: x[1], reverse=True)
        return ranking


def sobol_sequence(n_samples: int, n_dims: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate Sobol sequence for quasi-Monte Carlo sampling.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_dims : int
        Number of dimensions.
    seed : int, optional
        Random seed for scrambling.
        
    Returns
    -------
    np.ndarray of shape (n_samples, n_dims)
        Sobol sequence in [0, 1]^n_dims.
    """
    from scipy.stats import qmc
    
    sampler = qmc.Sobol(d=n_dims, scramble=seed is not None, seed=seed)
    samples = sampler.random(n=n_samples)
    
    return samples


def saltelli_sample(
    n_samples: int,
    n_dims: int,
    bounds: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate Saltelli sampling matrices for Sobol analysis.
    
    Parameters
    ----------
    n_samples : int
        Base number of samples (total will be n_samples * (2*n_dims + 2)).
    n_dims : int
        Number of dimensions.
    bounds : np.ndarray of shape (n_dims, 2), optional
        Lower and upper bounds for each dimension.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    A : np.ndarray
        First sample matrix.
    B : np.ndarray
        Second sample matrix.
    AB_list : list of np.ndarray
        Matrices where each column i is from B, rest from A.
    """
    # Generate two independent Sobol sequences
    n_total = n_samples * 2
    base = sobol_sequence(n_total, n_dims, seed=seed)
    
    A = base[:n_samples]
    B = base[n_samples:]
    
    # Scale to bounds if provided
    if bounds is not None:
        bounds = np.asarray(bounds)
        A = bounds[:, 0] + A * (bounds[:, 1] - bounds[:, 0])
        B = bounds[:, 0] + B * (bounds[:, 1] - bounds[:, 0])
    
    # Create AB matrices
    AB_list = []
    for i in range(n_dims):
        AB = A.copy()
        AB[:, i] = B[:, i]
        AB_list.append(AB)
    
    return A, B, AB_list


def compute_sobol_indices(
    func: Callable[[np.ndarray], float],
    n_dims: int,
    bounds: Optional[np.ndarray] = None,
    n_samples: int = 1024,
    parameter_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
    compute_second_order: bool = False,
) -> SobolIndices:
    """Compute Sobol sensitivity indices.
    
    Uses the Saltelli estimator for efficient computation.
    
    Parameters
    ----------
    func : callable
        Objective function that takes an array of shape (n_dims,) and returns a scalar.
    n_dims : int
        Number of input dimensions.
    bounds : np.ndarray of shape (n_dims, 2), optional
        Bounds for each dimension. Defaults to [0, 1] for all.
    n_samples : int, default=1024
        Base number of samples.
    parameter_names : list of str, optional
        Names of parameters.
    seed : int, optional
        Random seed.
    compute_second_order : bool, default=False
        Whether to compute second-order interaction indices.
        
    Returns
    -------
    SobolIndices
        Computed sensitivity indices.
    """
    if bounds is None:
        bounds = np.array([[0, 1]] * n_dims)
    
    # Generate Saltelli samples
    A, B, AB_list = saltelli_sample(n_samples, n_dims, bounds, seed)
    
    # Evaluate function
    y_A = np.array([func(x) for x in A])
    y_B = np.array([func(x) for x in B])
    y_AB = [np.array([func(x) for x in AB]) for AB in AB_list]
    
    # Compute indices using Saltelli estimator
    f0 = np.mean(y_A)
    var_total = np.var(y_A)
    
    if var_total < 1e-10:
        # No variance, all indices are 0
        return SobolIndices(
            first_order=np.zeros(n_dims),
            total=np.zeros(n_dims),
            parameter_names=parameter_names,
        )
    
    # First-order indices
    first_order = np.zeros(n_dims)
    for i in range(n_dims):
        # S_i = V_i / V(Y)
        # V_i estimated by E[y_B * (y_AB_i - y_A)]
        first_order[i] = np.mean(y_B * (y_AB[i] - y_A)) / var_total
    
    # Total indices
    total = np.zeros(n_dims)
    for i in range(n_dims):
        # S_Ti = 1 - V_{~i} / V(Y)
        # V_{~i} estimated by 0.5 * E[(y_A - y_AB_i)^2]
        total[i] = 0.5 * np.mean((y_A - y_AB[i]) ** 2) / var_total
    
    # Clamp to valid range
    first_order = np.clip(first_order, 0, 1)
    total = np.clip(total, 0, 1)
    
    # Second-order indices (optional)
    second_order = None
    if compute_second_order and n_dims > 1:
        second_order = {}
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                # S_ij = S_{ij} - S_i - S_j
                # Approximate using AB matrices
                AB_ij = A.copy()
                AB_ij[:, i] = B[:, i]
                AB_ij[:, j] = B[:, j]
                y_AB_ij = np.array([func(x) for x in AB_ij])
                
                V_ij = np.mean(y_B * (y_AB_ij - y_A)) / var_total
                S_ij = V_ij - first_order[i] - first_order[j]
                second_order[(i, j)] = max(0, S_ij)
    
    return SobolIndices(
        first_order=first_order,
        total=total,
        second_order=second_order,
        parameter_names=parameter_names,
    )


def compute_sobol_from_surrogate(
    surrogate,
    space,
    n_samples: int = 1024,
    seed: Optional[int] = None,
    compute_second_order: bool = False,
) -> SobolIndices:
    """Compute Sobol indices using a fitted surrogate model.
    
    This is useful when function evaluations are expensive and you have
    a fitted Gaussian Process or other surrogate.
    
    Parameters
    ----------
    surrogate : object
        Fitted surrogate model with predict method.
    space : Space
        The parameter space.
    n_samples : int, default=1024
        Number of samples for Monte Carlo estimation.
    seed : int, optional
        Random seed.
    compute_second_order : bool, default=False
        Whether to compute second-order indices.
        
    Returns
    -------
    SobolIndices
        Computed sensitivity indices.
    """
    n_dims = len(space.dimensions)
    bounds = np.array([[d.low, d.high] for d in space.dimensions])
    names = [d.name for d in space.dimensions]
    
    def func(x):
        x_2d = x.reshape(1, -1)
        return surrogate.predict(x_2d)[0]
    
    return compute_sobol_indices(
        func=func,
        n_dims=n_dims,
        bounds=bounds,
        n_samples=n_samples,
        parameter_names=names,
        seed=seed,
        compute_second_order=compute_second_order,
    )


def morris_trajectories(
    n_dims: int,
    n_trajectories: int,
    levels: int = 4,
    bounds: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Generate trajectories for Morris screening.
    
    Parameters
    ----------
    n_dims : int
        Number of dimensions.
    n_trajectories : int
        Number of trajectories to generate.
    levels : int, default=4
        Number of levels for discretization.
    bounds : np.ndarray of shape (n_dims, 2), optional
        Bounds for each dimension.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    list of np.ndarray
        Each trajectory is an array of shape (n_dims + 1, n_dims).
    """
    rng = np.random.RandomState(seed)
    
    if bounds is None:
        bounds = np.array([[0, 1]] * n_dims)
    
    # Grid step
    delta = 1.0 / (levels - 1)
    
    trajectories = []
    
    for _ in range(n_trajectories):
        # Random starting point (on grid)
        start = rng.randint(0, levels - 1, size=n_dims) * delta
        
        # Random order of dimensions
        order = rng.permutation(n_dims)
        
        # Build trajectory
        trajectory = np.zeros((n_dims + 1, n_dims))
        trajectory[0] = start
        
        for i, dim in enumerate(order):
            trajectory[i + 1] = trajectory[i].copy()
            # Move in this dimension
            if trajectory[i, dim] + delta <= 1.0:
                trajectory[i + 1, dim] += delta
            else:
                trajectory[i + 1, dim] -= delta
        
        # Scale to bounds
        scaled = bounds[:, 0] + trajectory * (bounds[:, 1] - bounds[:, 0])
        trajectories.append(scaled)
    
    return trajectories


def compute_morris(
    func: Callable[[np.ndarray], float],
    n_dims: int,
    bounds: Optional[np.ndarray] = None,
    n_trajectories: int = 10,
    levels: int = 4,
    parameter_names: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> MorrisResult:
    """Compute Morris screening sensitivity measures.
    
    Morris method is a one-at-a-time (OAT) method that efficiently
    screens parameters to identify which are most influential.
    
    Parameters
    ----------
    func : callable
        Objective function.
    n_dims : int
        Number of input dimensions.
    bounds : np.ndarray of shape (n_dims, 2), optional
        Bounds for each dimension.
    n_trajectories : int, default=10
        Number of trajectories.
    levels : int, default=4
        Number of levels for discretization.
    parameter_names : list of str, optional
        Names of parameters.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    MorrisResult
        Morris screening results.
    """
    if bounds is None:
        bounds = np.array([[0, 1]] * n_dims)
    
    # Generate trajectories
    trajectories = morris_trajectories(n_dims, n_trajectories, levels, bounds, seed)
    
    # Compute elementary effects
    all_effects = [[] for _ in range(n_dims)]
    
    for trajectory in trajectories:
        # Evaluate function at each point
        y = np.array([func(x) for x in trajectory])
        
        # Compute elementary effects
        for i in range(n_dims):
            # Find where dimension i changed
            for j in range(len(trajectory) - 1):
                if not np.isclose(trajectory[j, i], trajectory[j + 1, i]):
                    dx = trajectory[j + 1, i] - trajectory[j, i]
                    dy = y[j + 1] - y[j]
                    # Normalize by delta in scaled space
                    delta = (bounds[i, 1] - bounds[i, 0]) / (levels - 1)
                    ee = dy / delta
                    all_effects[i].append(ee)
                    break
    
    # Compute statistics
    mu = np.zeros(n_dims)
    mu_star = np.zeros(n_dims)
    sigma = np.zeros(n_dims)
    
    for i in range(n_dims):
        effects = np.array(all_effects[i])
        if len(effects) > 0:
            mu[i] = np.mean(effects)
            mu_star[i] = np.mean(np.abs(effects))
            sigma[i] = np.std(effects)
    
    return MorrisResult(
        mu=mu,
        mu_star=mu_star,
        sigma=sigma,
        elementary_effects=all_effects,
        parameter_names=parameter_names,
    )


def compute_morris_from_surrogate(
    surrogate,
    space,
    n_trajectories: int = 10,
    levels: int = 4,
    seed: Optional[int] = None,
) -> MorrisResult:
    """Compute Morris screening using a fitted surrogate model.
    
    Parameters
    ----------
    surrogate : object
        Fitted surrogate model with predict method.
    space : Space
        The parameter space.
    n_trajectories : int, default=10
        Number of trajectories.
    levels : int, default=4
        Number of levels.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    MorrisResult
        Morris screening results.
    """
    n_dims = len(space.dimensions)
    bounds = np.array([[d.low, d.high] for d in space.dimensions])
    names = [d.name for d in space.dimensions]
    
    def func(x):
        x_2d = x.reshape(1, -1)
        return surrogate.predict(x_2d)[0]
    
    return compute_morris(
        func=func,
        n_dims=n_dims,
        bounds=bounds,
        n_trajectories=n_trajectories,
        levels=levels,
        parameter_names=names,
        seed=seed,
    )


def compute_local_sensitivity(
    func: Callable[[np.ndarray], float],
    point: np.ndarray,
    bounds: Optional[np.ndarray] = None,
    eps: float = 1e-5,
    parameter_names: Optional[List[str]] = None,
) -> LocalSensitivity:
    """Compute local sensitivity at a specific point.
    
    Uses finite differences to estimate gradients.
    
    Parameters
    ----------
    func : callable
        Objective function.
    point : np.ndarray
        Point at which to compute sensitivity.
    bounds : np.ndarray of shape (n_dims, 2), optional
        Bounds for normalization.
    eps : float, default=1e-5
        Step size for finite differences.
    parameter_names : list of str, optional
        Names of parameters.
        
    Returns
    -------
    LocalSensitivity
        Local sensitivity results.
    """
    point = np.asarray(point)
    n_dims = len(point)
    
    y0 = func(point)
    gradients = np.zeros(n_dims)
    
    for i in range(n_dims):
        # Forward difference
        point_plus = point.copy()
        point_plus[i] += eps
        y_plus = func(point_plus)
        
        gradients[i] = (y_plus - y0) / eps
    
    # Normalize gradients (elasticity)
    if bounds is not None:
        bounds = np.asarray(bounds)
        scale = bounds[:, 1] - bounds[:, 0]
    else:
        scale = np.ones(n_dims)
    
    normalized = gradients * scale * point / (np.abs(y0) + 1e-10)
    
    return LocalSensitivity(
        gradients=gradients,
        normalized_gradients=normalized,
        point=point,
        parameter_names=parameter_names,
    )


def compute_local_sensitivity_from_surrogate(
    surrogate,
    space,
    point: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> LocalSensitivity:
    """Compute local sensitivity using a surrogate model.
    
    Parameters
    ----------
    surrogate : object
        Fitted surrogate model with predict method.
    space : Space
        The parameter space.
    point : np.ndarray, optional
        Point at which to compute. If None, uses center of space.
    eps : float, default=1e-5
        Step size for finite differences.
        
    Returns
    -------
    LocalSensitivity
        Local sensitivity results.
    """
    n_dims = len(space.dimensions)
    bounds = np.array([[d.low, d.high] for d in space.dimensions])
    names = [d.name for d in space.dimensions]
    
    if point is None:
        point = (bounds[:, 0] + bounds[:, 1]) / 2
    
    def func(x):
        x_2d = x.reshape(1, -1)
        return surrogate.predict(x_2d)[0]
    
    return compute_local_sensitivity(
        func=func,
        point=point,
        bounds=bounds,
        eps=eps,
        parameter_names=names,
    )


def correlation_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    parameter_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compute correlation-based sensitivity measures.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input samples.
    y : np.ndarray of shape (n_samples,)
        Output values.
    parameter_names : list of str, optional
        Names of parameters.
        
    Returns
    -------
    dict
        Dictionary with Pearson and Spearman correlations.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_dims = X.shape[1]
    
    pearson = np.zeros(n_dims)
    spearman = np.zeros(n_dims)
    
    for i in range(n_dims):
        # Pearson correlation
        if np.std(X[:, i]) > 0 and np.std(y) > 0:
            pearson[i] = np.corrcoef(X[:, i], y)[0, 1]
        
        # Spearman correlation
        spearman[i], _ = spearmanr(X[:, i], y)
    
    return {
        "pearson": pearson,
        "spearman": spearman,
        "pearson_abs": np.abs(pearson),
        "spearman_abs": np.abs(spearman),
        "parameter_names": parameter_names,
    }


def main_effect_indices(
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    parameter_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute main effect indices from data.
    
    Estimates how much of the variance in y is explained by each input.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input samples.
    y : np.ndarray of shape (n_samples,)
        Output values.
    n_bins : int, default=10
        Number of bins for discretization.
    parameter_names : list of str, optional
        Names of parameters.
        
    Returns
    -------
    dict
        Main effect indices and means for each bin.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_dims = X.shape
    
    total_var = np.var(y)
    if total_var < 1e-10:
        return {
            "indices": np.zeros(n_dims),
            "parameter_names": parameter_names,
        }
    
    indices = np.zeros(n_dims)
    bin_means = []
    
    for i in range(n_dims):
        # Discretize this dimension
        bins = np.linspace(X[:, i].min(), X[:, i].max(), n_bins + 1)
        bin_indices = np.digitize(X[:, i], bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute mean for each bin
        means = []
        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                means.append(np.mean(y[mask]))
            else:
                means.append(np.nan)
        
        bin_means.append(means)
        
        # Compute variance of conditional means
        valid_means = [m for m in means if not np.isnan(m)]
        if len(valid_means) > 1:
            var_means = np.var(valid_means)
            indices[i] = var_means / total_var
    
    return {
        "indices": indices,
        "bin_means": bin_means,
        "parameter_names": parameter_names,
    }


class SensitivityAnalyzer:
    """Comprehensive sensitivity analysis for optimization.
    
    Combines multiple sensitivity methods for thorough analysis.
    
    Parameters
    ----------
    surrogate : object, optional
        Fitted surrogate model. If provided, uses for efficient analysis.
    space : Space, optional
        The parameter space.
    random_state : int, optional
        Random state for reproducibility.
        
    Examples
    --------
    >>> analyzer = SensitivityAnalyzer(surrogate=optimizer.surrogate, space=space)
    >>> report = analyzer.full_analysis()
    >>> print(report['sobol'].get_ranking())
    """
    
    def __init__(
        self,
        surrogate=None,
        space=None,
        random_state: Optional[int] = None,
    ) -> None:
        self.surrogate = surrogate
        self.space = space
        self.random_state = random_state
        
        self._results: Dict[str, Any] = {}
    
    def compute_sobol(
        self,
        n_samples: int = 1024,
        compute_second_order: bool = False,
    ) -> SobolIndices:
        """Compute Sobol sensitivity indices.
        
        Parameters
        ----------
        n_samples : int, default=1024
            Number of samples.
        compute_second_order : bool, default=False
            Whether to compute second-order indices.
            
        Returns
        -------
        SobolIndices
            Sobol sensitivity indices.
        """
        if self.surrogate is None or self.space is None:
            raise ValueError("Surrogate and space required for Sobol analysis")
        
        result = compute_sobol_from_surrogate(
            self.surrogate,
            self.space,
            n_samples=n_samples,
            seed=self.random_state,
            compute_second_order=compute_second_order,
        )
        self._results['sobol'] = result
        return result
    
    def compute_morris(
        self,
        n_trajectories: int = 10,
        levels: int = 4,
    ) -> MorrisResult:
        """Compute Morris screening.
        
        Parameters
        ----------
        n_trajectories : int, default=10
            Number of trajectories.
        levels : int, default=4
            Number of levels.
            
        Returns
        -------
        MorrisResult
            Morris screening results.
        """
        if self.surrogate is None or self.space is None:
            raise ValueError("Surrogate and space required for Morris analysis")
        
        result = compute_morris_from_surrogate(
            self.surrogate,
            self.space,
            n_trajectories=n_trajectories,
            levels=levels,
            seed=self.random_state,
        )
        self._results['morris'] = result
        return result
    
    def compute_local(
        self,
        point: Optional[np.ndarray] = None,
        eps: float = 1e-5,
    ) -> LocalSensitivity:
        """Compute local sensitivity.
        
        Parameters
        ----------
        point : np.ndarray, optional
            Point at which to compute.
        eps : float, default=1e-5
            Step size.
            
        Returns
        -------
        LocalSensitivity
            Local sensitivity results.
        """
        if self.surrogate is None or self.space is None:
            raise ValueError("Surrogate and space required")
        
        result = compute_local_sensitivity_from_surrogate(
            self.surrogate,
            self.space,
            point=point,
            eps=eps,
        )
        self._results['local'] = result
        return result
    
    def full_analysis(
        self,
        n_sobol_samples: int = 512,
        n_morris_trajectories: int = 10,
    ) -> Dict[str, Any]:
        """Run full sensitivity analysis.
        
        Parameters
        ----------
        n_sobol_samples : int, default=512
            Samples for Sobol analysis.
        n_morris_trajectories : int, default=10
            Trajectories for Morris screening.
            
        Returns
        -------
        dict
            Dictionary with all analysis results.
        """
        results = {}
        
        try:
            results['sobol'] = self.compute_sobol(n_samples=n_sobol_samples)
        except Exception as e:
            warnings.warn(f"Sobol analysis failed: {e}")
        
        try:
            results['morris'] = self.compute_morris(n_trajectories=n_morris_trajectories)
        except Exception as e:
            warnings.warn(f"Morris analysis failed: {e}")
        
        try:
            results['local'] = self.compute_local()
        except Exception as e:
            warnings.warn(f"Local analysis failed: {e}")
        
        return results
    
    def get_parameter_ranking(self, method: str = "sobol_total") -> List[Tuple[str, float]]:
        """Get overall parameter ranking.
        
        Parameters
        ----------
        method : str, default="sobol_total"
            Method to use: "sobol_total", "sobol_first", "morris", or "local".
            
        Returns
        -------
        list of (name, value) tuples
            Ranking from most to least important.
        """
        if method == "sobol_total" and 'sobol' in self._results:
            ranking = self._results['sobol'].get_ranking(by="total")
        elif method == "sobol_first" and 'sobol' in self._results:
            ranking = self._results['sobol'].get_ranking(by="first_order")
        elif method == "morris" and 'morris' in self._results:
            ranking = self._results['morris'].get_ranking()
        elif method == "local" and 'local' in self._results:
            ranking = self._results['local'].get_ranking()
        else:
            raise ValueError(f"Method {method} not available or not computed")
        
        # Get parameter names
        names = None
        if self.space:
            names = [d.name for d in self.space.dimensions]
        
        if names:
            return [(names[idx], val) for idx, val in ranking]
        else:
            return [(f"x{idx}", val) for idx, val in ranking]
    
    def generate_report(self) -> str:
        """Generate a text report of all analyses.
        
        Returns
        -------
        str
            Formatted report.
        """
        lines = ["=" * 60, "SENSITIVITY ANALYSIS REPORT", "=" * 60, ""]
        
        if 'sobol' in self._results:
            sobol = self._results['sobol']
            lines.append("SOBOL SENSITIVITY INDICES")
            lines.append("-" * 40)
            for idx, (i, val) in enumerate(sobol.get_ranking()):
                name = sobol.parameter_names[i] if sobol.parameter_names else f"x{i}"
                lines.append(f"  {idx+1}. {name}: S_T={sobol.total[i]:.4f}, S_1={sobol.first_order[i]:.4f}")
            lines.append("")
        
        if 'morris' in self._results:
            morris = self._results['morris']
            lines.append("MORRIS SCREENING")
            lines.append("-" * 40)
            for idx, (i, val) in enumerate(morris.get_ranking()):
                name = morris.parameter_names[i] if morris.parameter_names else f"x{i}"
                lines.append(f"  {idx+1}. {name}: μ*={morris.mu_star[i]:.4f}, σ={morris.sigma[i]:.4f}")
            
            # Classification
            classes = morris.classify_parameters()
            lines.append("")
            lines.append("  Classifications:")
            for cls, indices in classes.items():
                names = [morris.parameter_names[i] if morris.parameter_names else f"x{i}" for i in indices]
                lines.append(f"    {cls}: {', '.join(names) if names else 'none'}")
            lines.append("")
        
        if 'local' in self._results:
            local = self._results['local']
            lines.append("LOCAL SENSITIVITY")
            lines.append("-" * 40)
            for idx, (i, val) in enumerate(local.get_ranking()):
                name = local.parameter_names[i] if local.parameter_names else f"x{i}"
                lines.append(f"  {idx+1}. {name}: |∂y/∂x|={np.abs(local.gradients[i]):.4f}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
