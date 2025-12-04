"""Robust optimization under uncertainty.

This module provides tools for optimization when the objective function
has uncertainty or noise, enabling worst-case analysis and risk-aware
optimization.

Features:
- Robust optimization with uncertainty sets
- Risk measures (CVaR, VaR, mean-variance)
- Distributionally robust optimization
- Worst-case analysis
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class RiskMeasure(Enum):
    """Risk measures for robust optimization."""
    MEAN = "mean"
    WORST_CASE = "worst_case"
    CVAR = "cvar"  # Conditional Value at Risk
    VAR = "var"  # Value at Risk
    MEAN_VARIANCE = "mean_variance"
    ENTROPIC = "entropic"


@dataclass
class UncertaintySet:
    """Definition of parameter uncertainty.
    
    Attributes
    ----------
    type : str
        Type of uncertainty: "box", "ellipsoidal", "polytope", or "probabilistic".
    center : np.ndarray
        Center of the uncertainty set.
    radius : float or np.ndarray
        Size of uncertainty (interpretation depends on type).
    covariance : np.ndarray, optional
        Covariance matrix for ellipsoidal or probabilistic uncertainty.
    samples : np.ndarray, optional
        Explicit samples for sample-based uncertainty.
    """
    type: str
    center: np.ndarray
    radius: Union[float, np.ndarray] = 0.0
    covariance: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the uncertainty set.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        random_state : int, optional
            Random seed.
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_dims)
            Samples from the uncertainty set.
        """
        rng = np.random.RandomState(random_state)
        center = np.asarray(self.center)
        n_dims = len(center)
        
        if self.type == "box":
            # Uniform within box
            radius = self.radius if np.isscalar(self.radius) else np.asarray(self.radius)
            samples = center + rng.uniform(-radius, radius, size=(n_samples, n_dims))
        
        elif self.type == "ellipsoidal":
            # Uniform within ellipsoid
            if self.covariance is not None:
                # Sample from unit sphere and transform
                u = rng.randn(n_samples, n_dims)
                u = u / np.linalg.norm(u, axis=1, keepdims=True)
                r = rng.uniform(0, 1, size=(n_samples, 1)) ** (1.0 / n_dims)
                L = np.linalg.cholesky(self.covariance)
                samples = center + self.radius * r * (u @ L.T)
            else:
                # Spherical
                u = rng.randn(n_samples, n_dims)
                u = u / np.linalg.norm(u, axis=1, keepdims=True)
                r = rng.uniform(0, 1, size=(n_samples, 1)) ** (1.0 / n_dims)
                samples = center + self.radius * r * u
        
        elif self.type == "probabilistic":
            # Multivariate normal
            if self.covariance is not None:
                samples = rng.multivariate_normal(center, self.covariance, size=n_samples)
            else:
                samples = rng.multivariate_normal(
                    center, self.radius ** 2 * np.eye(n_dims), size=n_samples
                )
        
        elif self.type == "samples" and self.samples is not None:
            # Resample from provided samples
            indices = rng.choice(len(self.samples), size=n_samples, replace=True)
            samples = self.samples[indices]
        
        else:
            raise ValueError(f"Unknown uncertainty type: {self.type}")
        
        return samples


@dataclass
class RobustResult:
    """Result of robust optimization.
    
    Attributes
    ----------
    optimal_x : np.ndarray
        Optimal solution.
    robust_value : float
        Robust objective value (under chosen risk measure).
    mean_value : float
        Mean objective value.
    worst_case_value : float
        Worst-case objective value.
    confidence_interval : tuple
        Confidence interval for the objective.
    n_evaluations : int
        Number of function evaluations.
    """
    optimal_x: np.ndarray
    robust_value: float
    mean_value: float
    worst_case_value: float
    confidence_interval: Tuple[float, float]
    n_evaluations: int = 0


def compute_cvar(
    values: np.ndarray,
    alpha: float = 0.05,
    minimize: bool = True,
) -> float:
    """Compute Conditional Value at Risk (CVaR).
    
    Parameters
    ----------
    values : np.ndarray
        Sample of objective values.
    alpha : float, default=0.05
        Risk level (0 < alpha < 1). Lower alpha = more conservative.
    minimize : bool, default=True
        Whether we're minimizing (CVaR of high values) or maximizing.
        
    Returns
    -------
    float
        CVaR value.
    """
    values = np.asarray(values)
    
    if minimize:
        # CVaR is mean of worst alpha fraction (highest values for minimization)
        threshold = np.percentile(values, 100 * (1 - alpha))
        tail = values[values >= threshold]
    else:
        # For maximization, worst is lowest values
        threshold = np.percentile(values, 100 * alpha)
        tail = values[values <= threshold]
    
    if len(tail) == 0:
        return np.mean(values)
    
    return np.mean(tail)


def compute_var(
    values: np.ndarray,
    alpha: float = 0.05,
    minimize: bool = True,
) -> float:
    """Compute Value at Risk (VaR).
    
    Parameters
    ----------
    values : np.ndarray
        Sample of objective values.
    alpha : float, default=0.05
        Risk level.
    minimize : bool, default=True
        Whether we're minimizing.
        
    Returns
    -------
    float
        VaR value.
    """
    if minimize:
        return np.percentile(values, 100 * (1 - alpha))
    else:
        return np.percentile(values, 100 * alpha)


def compute_mean_variance(
    values: np.ndarray,
    lambda_var: float = 0.5,
) -> float:
    """Compute mean-variance objective.
    
    Parameters
    ----------
    values : np.ndarray
        Sample of objective values.
    lambda_var : float, default=0.5
        Weight on variance term.
        
    Returns
    -------
    float
        mean + lambda_var * variance
    """
    return np.mean(values) + lambda_var * np.var(values)


def compute_entropic_risk(
    values: np.ndarray,
    theta: float = 1.0,
    minimize: bool = True,
) -> float:
    """Compute entropic risk measure.
    
    Parameters
    ----------
    values : np.ndarray
        Sample of objective values.
    theta : float, default=1.0
        Risk aversion parameter (higher = more conservative).
    minimize : bool, default=True
        Whether we're minimizing.
        
    Returns
    -------
    float
        Entropic risk value.
    """
    if minimize:
        return (1 / theta) * np.log(np.mean(np.exp(theta * values)))
    else:
        return -(1 / theta) * np.log(np.mean(np.exp(-theta * values)))


class RobustAcquisition(ABC):
    """Base class for robust acquisition functions."""
    
    @abstractmethod
    def __call__(
        self,
        x: np.ndarray,
        surrogate,
        uncertainty_set: UncertaintySet,
        n_samples: int = 100,
    ) -> float:
        """Evaluate robust acquisition at a point.
        
        Parameters
        ----------
        x : np.ndarray
            Point to evaluate.
        surrogate : object
            Fitted surrogate model.
        uncertainty_set : UncertaintySet
            Uncertainty specification.
        n_samples : int
            Number of samples for Monte Carlo estimation.
            
        Returns
        -------
        float
            Robust acquisition value.
        """
        pass


class RobustExpectedImprovement(RobustAcquisition):
    """Robust Expected Improvement under uncertainty.
    
    Parameters
    ----------
    best_f : float
        Best observed value.
    risk_measure : RiskMeasure, default=RiskMeasure.CVAR
        Risk measure to use.
    alpha : float, default=0.1
        Risk level for CVaR/VaR.
    minimize : bool, default=True
        Whether minimizing the objective.
    """
    
    def __init__(
        self,
        best_f: float,
        risk_measure: RiskMeasure = RiskMeasure.CVAR,
        alpha: float = 0.1,
        minimize: bool = True,
    ) -> None:
        self.best_f = best_f
        self.risk_measure = risk_measure
        self.alpha = alpha
        self.minimize = minimize
    
    def __call__(
        self,
        x: np.ndarray,
        surrogate,
        uncertainty_set: UncertaintySet,
        n_samples: int = 100,
    ) -> float:
        """Compute robust expected improvement."""
        # Sample perturbations
        perturbations = uncertainty_set.sample(n_samples)
        
        # Evaluate surrogate at perturbed points
        n_dims = len(x)
        X_perturbed = x + perturbations[:, :n_dims] if perturbations.shape[1] >= n_dims else \
                      np.tile(x, (n_samples, 1)) + perturbations[:, :n_dims]
        
        # Handle both sklearn-style and custom surrogate APIs
        try:
            mu, sigma = surrogate.predict(X_perturbed, return_std=True)
        except TypeError:
            # Our custom surrogate always returns (mean, std)
            mu, sigma = surrogate.predict(X_perturbed)
        
        # Compute EI for each sample
        if self.minimize:
            improvement = self.best_f - mu
        else:
            improvement = mu - self.best_f
        
        z = improvement / (sigma + 1e-10)
        ei_samples = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Apply risk measure
        if self.risk_measure == RiskMeasure.MEAN:
            return np.mean(ei_samples)
        elif self.risk_measure == RiskMeasure.WORST_CASE:
            return np.min(ei_samples)
        elif self.risk_measure == RiskMeasure.CVAR:
            return compute_cvar(ei_samples, self.alpha, minimize=True)
        elif self.risk_measure == RiskMeasure.VAR:
            return compute_var(ei_samples, self.alpha, minimize=True)
        else:
            return np.mean(ei_samples)


class WorstCaseAcquisition(RobustAcquisition):
    """Worst-case acquisition under parameter uncertainty.
    
    Optimizes the worst-case (minimax) objective over the uncertainty set.
    
    Parameters
    ----------
    minimize : bool, default=True
        Whether minimizing the objective.
    """
    
    def __init__(self, minimize: bool = True) -> None:
        self.minimize = minimize
    
    def __call__(
        self,
        x: np.ndarray,
        surrogate,
        uncertainty_set: UncertaintySet,
        n_samples: int = 100,
    ) -> float:
        """Compute worst-case objective value."""
        perturbations = uncertainty_set.sample(n_samples)
        
        n_dims = len(x)
        if perturbations.shape[1] >= n_dims:
            X_perturbed = x + perturbations[:, :n_dims]
        else:
            X_perturbed = np.tile(x, (n_samples, 1)) + perturbations
        
        mu = surrogate.predict(X_perturbed)
        
        if self.minimize:
            # Worst case for minimization is maximum
            return np.max(mu)
        else:
            # Worst case for maximization is minimum
            return np.min(mu)


class CVaRAcquisition(RobustAcquisition):
    """CVaR-based acquisition for risk-averse optimization.
    
    Parameters
    ----------
    alpha : float, default=0.1
        Risk level (smaller = more conservative).
    minimize : bool, default=True
        Whether minimizing the objective.
    """
    
    def __init__(self, alpha: float = 0.1, minimize: bool = True) -> None:
        self.alpha = alpha
        self.minimize = minimize
    
    def __call__(
        self,
        x: np.ndarray,
        surrogate,
        uncertainty_set: UncertaintySet,
        n_samples: int = 100,
    ) -> float:
        """Compute CVaR of predicted values."""
        perturbations = uncertainty_set.sample(n_samples)
        
        n_dims = len(x)
        if perturbations.shape[1] >= n_dims:
            X_perturbed = x + perturbations[:, :n_dims]
        else:
            X_perturbed = np.tile(x, (n_samples, 1)) + perturbations
        
        mu = surrogate.predict(X_perturbed)
        
        return compute_cvar(mu, self.alpha, self.minimize)


class RobustOptimizer:
    """Bayesian optimizer with robustness considerations.
    
    Parameters
    ----------
    space : Space
        The parameter space.
    uncertainty : UncertaintySet or dict
        Uncertainty specification. If dict, used to create UncertaintySet.
    risk_measure : RiskMeasure, default=RiskMeasure.CVAR
        Risk measure for robust optimization.
    alpha : float, default=0.1
        Risk level for CVaR/VaR.
    n_samples : int, default=100
        Number of samples for Monte Carlo estimation.
    minimize : bool, default=True
        Whether minimizing the objective.
    random_state : int, optional
        Random state.
        
    Examples
    --------
    >>> from optiml import Space, Real
    >>> space = Space([Real(0, 1, name="x")])
    >>> uncertainty = UncertaintySet(type="box", center=np.zeros(1), radius=0.1)
    >>> optimizer = RobustOptimizer(space, uncertainty, risk_measure=RiskMeasure.CVAR)
    """
    
    def __init__(
        self,
        space,
        uncertainty: Union[UncertaintySet, Dict],
        risk_measure: RiskMeasure = RiskMeasure.CVAR,
        alpha: float = 0.1,
        n_samples: int = 100,
        minimize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        from optiml import BayesianOptimizer
        
        self.space = space
        self.risk_measure = risk_measure
        self.alpha = alpha
        self.n_samples = n_samples
        self.minimize = minimize
        self.random_state = random_state
        
        # Create uncertainty set if dict provided
        if isinstance(uncertainty, dict):
            self.uncertainty = UncertaintySet(**uncertainty)
        else:
            self.uncertainty = uncertainty
        
        # Create base optimizer
        self._base_optimizer = BayesianOptimizer(
            space=space,
            random_state=random_state,
        )
        
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._y_robust: List[float] = []
    
    def suggest(self) -> np.ndarray:
        """Suggest next point to evaluate.
        
        Uses robust acquisition function considering uncertainty.
        
        Returns
        -------
        np.ndarray
            Suggested point.
        """
        if len(self._X) < 5:
            # Initial random exploration
            return self._base_optimizer.suggest()
        
        # Fit surrogate
        X = np.array(self._X)
        y = np.array(self._y)
        self._base_optimizer.surrogate.fit(X, y)
        
        # Create robust acquisition
        best_f = min(y) if self.minimize else max(y)
        
        if self.risk_measure == RiskMeasure.WORST_CASE:
            acquisition = WorstCaseAcquisition(minimize=self.minimize)
        elif self.risk_measure in [RiskMeasure.CVAR, RiskMeasure.VAR]:
            acquisition = CVaRAcquisition(alpha=self.alpha, minimize=self.minimize)
        else:
            acquisition = RobustExpectedImprovement(
                best_f=best_f,
                risk_measure=self.risk_measure,
                alpha=self.alpha,
                minimize=self.minimize,
            )
        
        # Optimize acquisition
        best_x = None
        best_acq = float('inf') if self.minimize else float('-inf')
        
        # Random restarts
        rng = np.random.default_rng(self.random_state)
        n_dims = len(self.space.dimensions)
        
        for _ in range(10):
            x0 = self.space.sample(1, rng=rng)[0]
            
            acq_value = acquisition(
                x0, self._base_optimizer.surrogate, 
                self.uncertainty, self.n_samples
            )
            
            if self.minimize:
                if acq_value < best_acq:
                    best_acq = acq_value
                    best_x = x0
            else:
                if acq_value > best_acq:
                    best_acq = acq_value
                    best_x = x0
        
        return best_x if best_x is not None else self.space.sample(1)[0]
    
    def tell(
        self,
        x: np.ndarray,
        y: float,
        y_samples: Optional[np.ndarray] = None,
    ) -> None:
        """Record an observation.
        
        Parameters
        ----------
        x : np.ndarray
            Point that was evaluated.
        y : float
            Observed value (can be mean if multiple samples).
        y_samples : np.ndarray, optional
            Multiple samples of the objective at x.
        """
        self._X.append(np.asarray(x))
        self._y.append(y)
        
        # Compute robust value from samples if provided
        if y_samples is not None:
            if self.risk_measure == RiskMeasure.CVAR:
                robust_y = compute_cvar(y_samples, self.alpha, self.minimize)
            elif self.risk_measure == RiskMeasure.VAR:
                robust_y = compute_var(y_samples, self.alpha, self.minimize)
            elif self.risk_measure == RiskMeasure.WORST_CASE:
                robust_y = np.max(y_samples) if self.minimize else np.min(y_samples)
            elif self.risk_measure == RiskMeasure.MEAN_VARIANCE:
                robust_y = compute_mean_variance(y_samples)
            else:
                robust_y = np.mean(y_samples)
            self._y_robust.append(robust_y)
        else:
            self._y_robust.append(y)
        
        # Update base optimizer
        self._base_optimizer.tell([x], [y])
    
    def get_best(self) -> Tuple[np.ndarray, float, float]:
        """Get the best robust solution found.
        
        Returns
        -------
        x : np.ndarray
            Best point.
        y : float
            Mean objective at best point.
        y_robust : float
            Robust objective at best point.
        """
        if not self._X:
            raise ValueError("No observations recorded")
        
        if self.minimize:
            best_idx = np.argmin(self._y_robust)
        else:
            best_idx = np.argmax(self._y_robust)
        
        return self._X[best_idx], self._y[best_idx], self._y_robust[best_idx]
    
    def optimize(
        self,
        func: Callable,
        n_iter: int,
        n_samples_per_eval: int = 1,
    ) -> RobustResult:
        """Run robust optimization.
        
        Parameters
        ----------
        func : callable
            Objective function.
        n_iter : int
            Number of iterations.
        n_samples_per_eval : int, default=1
            Number of samples per evaluation for noisy objectives.
            
        Returns
        -------
        RobustResult
            Optimization result.
        """
        n_evals = 0
        
        for i in range(n_iter):
            x = self.suggest()
            
            # Evaluate (possibly with noise sampling)
            if n_samples_per_eval > 1:
                y_samples = np.array([func(x) for _ in range(n_samples_per_eval)])
                y = np.mean(y_samples)
                self.tell(x, y, y_samples)
            else:
                y = func(x)
                self.tell(x, y)
            
            n_evals += n_samples_per_eval
        
        # Get best solution
        best_x, mean_y, robust_y = self.get_best()
        
        # Compute statistics
        y_array = np.array(self._y)
        
        if self.minimize:
            worst_case = np.max(y_array)
        else:
            worst_case = np.min(y_array)
        
        ci = (np.percentile(y_array, 5), np.percentile(y_array, 95))
        
        return RobustResult(
            optimal_x=best_x,
            robust_value=robust_y,
            mean_value=mean_y,
            worst_case_value=worst_case,
            confidence_interval=ci,
            n_evaluations=n_evals,
        )


def robust_evaluation(
    func: Callable,
    x: np.ndarray,
    uncertainty: UncertaintySet,
    n_samples: int = 100,
    risk_measure: RiskMeasure = RiskMeasure.MEAN,
    alpha: float = 0.1,
    minimize: bool = True,
) -> Dict[str, float]:
    """Evaluate a function robustly under uncertainty.
    
    Parameters
    ----------
    func : callable
        Objective function.
    x : np.ndarray
        Point to evaluate.
    uncertainty : UncertaintySet
        Uncertainty specification.
    n_samples : int, default=100
        Number of samples.
    risk_measure : RiskMeasure, default=RiskMeasure.MEAN
        Risk measure.
    alpha : float, default=0.1
        Risk level.
    minimize : bool, default=True
        Whether minimizing.
        
    Returns
    -------
    dict
        Dictionary with mean, std, cvar, var, worst_case values.
    """
    perturbations = uncertainty.sample(n_samples)
    
    values = []
    for pert in perturbations:
        x_pert = x + pert[:len(x)] if len(pert) >= len(x) else x + pert
        values.append(func(x_pert))
    
    values = np.array(values)
    
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "cvar": compute_cvar(values, alpha, minimize),
        "var": compute_var(values, alpha, minimize),
        "worst_case": np.max(values) if minimize else np.min(values),
        "best_case": np.min(values) if minimize else np.max(values),
    }


class DistributionallyRobustOptimizer:
    """Distributionally robust optimization.
    
    Optimizes against the worst-case distribution in an ambiguity set
    centered around a reference distribution.
    
    Parameters
    ----------
    space : Space
        Parameter space.
    reference_distribution : callable
        Function that samples from reference distribution.
    wasserstein_radius : float
        Radius of Wasserstein ball (ambiguity set size).
    minimize : bool, default=True
        Whether minimizing.
    random_state : int, optional
        Random state.
    """
    
    def __init__(
        self,
        space,
        reference_distribution: Callable,
        wasserstein_radius: float = 0.1,
        minimize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        from optiml import BayesianOptimizer
        
        self.space = space
        self.reference_distribution = reference_distribution
        self.wasserstein_radius = wasserstein_radius
        self.minimize = minimize
        self.random_state = random_state
        
        self._base_optimizer = BayesianOptimizer(
            space=space,
            random_state=random_state,
        )
        
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
    
    def _compute_dro_value(
        self,
        x: np.ndarray,
        n_samples: int = 100,
    ) -> float:
        """Compute DRO objective using sample average approximation."""
        # Sample from reference distribution
        samples = np.array([self.reference_distribution() for _ in range(n_samples)])
        
        # Get predictions at perturbed points
        if len(self._X) >= 3:
            X_perturbed = x + samples[:, :len(x)] if samples.shape[1] >= len(x) else x + samples
            
            try:
                mu = self._base_optimizer.surrogate.predict(X_perturbed)
            except:
                # Fall back to evaluating at x
                mu, _ = self._base_optimizer.surrogate.predict(x.reshape(1, -1), return_std=True)
                mu = np.full(n_samples, mu[0])
        else:
            # Not enough data, return uniform estimate
            return 0.0
        
        # Worst-case over Wasserstein ball (approximate)
        # For small radius, this is approximately mean + radius * Lipschitz constant
        if self.minimize:
            return np.mean(mu) + self.wasserstein_radius * np.std(mu)
        else:
            return np.mean(mu) - self.wasserstein_radius * np.std(mu)
    
    def suggest(self) -> np.ndarray:
        """Suggest next point."""
        if len(self._X) < 5:
            return self._base_optimizer.suggest()
        
        # Simple random search for DRO-optimal point
        best_x = None
        best_val = float('inf') if self.minimize else float('-inf')
        
        rng = np.random.default_rng(self.random_state)
        
        for _ in range(20):
            x = self.space.sample(1, rng=rng)[0]
            val = self._compute_dro_value(x)
            
            if self.minimize:
                if val < best_val:
                    best_val = val
                    best_x = x
            else:
                if val > best_val:
                    best_val = val
                    best_x = x
        
        return best_x if best_x is not None else self.space.sample(1)[0]
    
    def tell(self, x: np.ndarray, y: float) -> None:
        """Record observation."""
        self._X.append(np.asarray(x))
        self._y.append(y)
        self._base_optimizer.tell([x], [y])
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get best solution."""
        if not self._X:
            raise ValueError("No observations")
        
        idx = np.argmin(self._y) if self.minimize else np.argmax(self._y)
        return self._X[idx], self._y[idx]


def create_robust_optimizer(
    space,
    uncertainty_type: str = "box",
    uncertainty_radius: float = 0.1,
    risk_measure: str = "cvar",
    alpha: float = 0.1,
    minimize: bool = True,
    random_state: Optional[int] = None,
) -> RobustOptimizer:
    """Factory function for creating robust optimizers.
    
    Parameters
    ----------
    space : Space
        Parameter space.
    uncertainty_type : str, default="box"
        Type of uncertainty: "box", "ellipsoidal", or "probabilistic".
    uncertainty_radius : float, default=0.1
        Size of uncertainty set.
    risk_measure : str, default="cvar"
        Risk measure: "mean", "cvar", "var", "worst_case", "mean_variance".
    alpha : float, default=0.1
        Risk level.
    minimize : bool, default=True
        Whether minimizing.
    random_state : int, optional
        Random state.
        
    Returns
    -------
    RobustOptimizer
        Configured robust optimizer.
        
    Examples
    --------
    >>> optimizer = create_robust_optimizer(
    ...     space, 
    ...     uncertainty_type="box",
    ...     uncertainty_radius=0.1,
    ...     risk_measure="cvar",
    ... )
    """
    n_dims = len(space.dimensions)
    
    uncertainty = UncertaintySet(
        type=uncertainty_type,
        center=np.zeros(n_dims),
        radius=uncertainty_radius,
    )
    
    risk_map = {
        "mean": RiskMeasure.MEAN,
        "cvar": RiskMeasure.CVAR,
        "var": RiskMeasure.VAR,
        "worst_case": RiskMeasure.WORST_CASE,
        "mean_variance": RiskMeasure.MEAN_VARIANCE,
        "entropic": RiskMeasure.ENTROPIC,
    }
    
    return RobustOptimizer(
        space=space,
        uncertainty=uncertainty,
        risk_measure=risk_map.get(risk_measure.lower(), RiskMeasure.CVAR),
        alpha=alpha,
        minimize=minimize,
        random_state=random_state,
    )
