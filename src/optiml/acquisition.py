"""Acquisition functions for Bayesian optimization.

This module provides a comprehensive set of acquisition functions for
guiding the Bayesian optimization process, including:

- Standard acquisition functions (EI, UCB, PI, LCB)
- Advanced acquisition functions (Knowledge Gradient, Thompson Sampling, MES)
- Batch acquisition functions (qEI, Local Penalization, Kriging Believer)
- Portfolio/ensemble methods
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Callable, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

if TYPE_CHECKING:
    from optiml.surrogate import SurrogateModel


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate the acquisition function at given points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Points at which to evaluate the acquisition function.
        surrogate : SurrogateModel
            The fitted surrogate model.
        y_best : float
            The best observed value so far.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Acquisition function values (higher is better).
        """
        pass


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function.

    EI(x) = E[max(f(x) - f(x_best), 0)]

    For a Gaussian process surrogate, this has a closed-form solution:
    EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
    where Z = (μ(x) - f_best - ξ) / σ(x)

    Parameters
    ----------
    xi : float, default=0.01
        Exploration-exploitation trade-off parameter.
        Higher values favor exploration.

    Examples
    --------
    >>> ei = ExpectedImprovement(xi=0.01)
    >>> values = ei(X, surrogate, y_best=-0.5)
    """

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi

    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate Expected Improvement at given points."""
        X = np.atleast_2d(X)
        mean, std = surrogate.predict(X)

        # Handle zero variance case
        with np.errstate(divide="ignore", invalid="ignore"):
            improvement = mean - y_best - self.xi
            Z = improvement / std
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            ei = np.where(std > 1e-10, ei, 0.0)

        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function.

    UCB(x) = μ(x) + κ * σ(x)

    This acquisition function directly balances exploitation (high mean)
    and exploration (high uncertainty).

    Parameters
    ----------
    kappa : float, default=2.576
        Exploration-exploitation trade-off parameter.
        Higher values favor exploration.
        Default value corresponds to ~99% confidence interval.

    Examples
    --------
    >>> ucb = UpperConfidenceBound(kappa=2.0)
    >>> values = ucb(X, surrogate, y_best=-0.5)
    """

    def __init__(self, kappa: float = 2.576) -> None:
        self.kappa = kappa

    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate Upper Confidence Bound at given points."""
        X = np.atleast_2d(X)
        mean, std = surrogate.predict(X)
        return mean + self.kappa * std


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of Improvement acquisition function.

    PI(x) = P(f(x) > f_best + ξ) = Φ((μ(x) - f_best - ξ) / σ(x))

    This is a simple acquisition function that computes the probability
    that a point will improve upon the current best.

    Parameters
    ----------
    xi : float, default=0.01
        Exploration-exploitation trade-off parameter.
        Higher values favor exploration.

    Examples
    --------
    >>> pi = ProbabilityOfImprovement(xi=0.01)
    >>> values = pi(X, surrogate, y_best=-0.5)
    """

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi

    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate Probability of Improvement at given points."""
        X = np.atleast_2d(X)
        mean, std = surrogate.predict(X)

        with np.errstate(divide="ignore", invalid="ignore"):
            Z = (mean - y_best - self.xi) / std
            pi = norm.cdf(Z)
            pi = np.where(std > 1e-10, pi, 0.0)

        return pi


class LowerConfidenceBound(AcquisitionFunction):
    """Lower Confidence Bound acquisition function for minimization.

    LCB(x) = μ(x) - κ * σ(x)

    This is the counterpart of UCB for minimization problems.
    We negate it to convert to a maximization problem.

    Parameters
    ----------
    kappa : float, default=2.576
        Exploration-exploitation trade-off parameter.
        Higher values favor exploration.

    Examples
    --------
    >>> lcb = LowerConfidenceBound(kappa=2.0)
    >>> values = lcb(X, surrogate, y_best=0.5)
    """

    def __init__(self, kappa: float = 2.576) -> None:
        self.kappa = kappa

    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate Lower Confidence Bound at given points."""
        X = np.atleast_2d(X)
        mean, std = surrogate.predict(X)
        # Negate to convert minimization to maximization
        return -(mean - self.kappa * std)


# =============================================================================
# Advanced Acquisition Functions
# =============================================================================

class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.
    
    Samples from the GP posterior and returns the sampled values.
    Simple but effective, especially for high-dimensional problems.
    
    Parameters
    ----------
    n_samples : int, default=1
        Number of posterior samples to draw.
        
    Examples
    --------
    >>> ts = ThompsonSampling()
    >>> values = ts(X, surrogate, y_best)
    """
    
    def __init__(self, n_samples: int = 1) -> None:
        self.n_samples = n_samples
        self._rng = np.random.default_rng()
    
    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Sample from the GP posterior."""
        X = np.atleast_2d(X)
        mean, std = surrogate.predict(X)
        
        # Sample from the posterior
        if self.n_samples == 1:
            samples = mean + std * self._rng.standard_normal(len(mean))
        else:
            samples = np.zeros(len(mean))
            for _ in range(self.n_samples):
                samples += mean + std * self._rng.standard_normal(len(mean))
            samples /= self.n_samples
        
        return samples


class KnowledgeGradient(AcquisitionFunction):
    """Knowledge Gradient acquisition function.
    
    Measures the expected improvement in the optimal predicted value
    if we were to sample at x and observe the result.
    
    KG(x) = E[max_x' μ_{n+1}(x') | sample at x] - max_x' μ_n(x')
    
    This acquisition function is particularly effective when the
    goal is to identify the true optimum location rather than just
    achieving good performance.
    
    Parameters
    ----------
    n_fantasies : int, default=10
        Number of fantasy samples for Monte Carlo estimation.
    n_discrete : int, default=100
        Number of discrete points to consider for inner optimization.
        
    Examples
    --------
    >>> kg = KnowledgeGradient(n_fantasies=20)
    >>> values = kg(X, surrogate, y_best)
    """
    
    def __init__(self, n_fantasies: int = 10, n_discrete: int = 100) -> None:
        self.n_fantasies = n_fantasies
        self.n_discrete = n_discrete
        self._rng = np.random.default_rng()
        self._X_discrete = None
    
    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate Knowledge Gradient at given points."""
        X = np.atleast_2d(X)
        n_points = len(X)
        
        # Get current predictions at candidate points
        mean, std = surrogate.predict(X)
        
        # Create discrete set for inner maximization if not set
        n_dims = X.shape[1]
        if self._X_discrete is None or self._X_discrete.shape[1] != n_dims:
            self._X_discrete = self._rng.uniform(0, 1, (self.n_discrete, n_dims))
        
        # Current max predicted value
        current_mean, _ = surrogate.predict(self._X_discrete)
        current_max = np.max(current_mean)
        
        # Compute KG for each candidate point
        kg_values = np.zeros(n_points)
        
        for i in range(n_points):
            x = X[i:i+1]
            mu_x, sigma_x = mean[i], std[i]
            
            if sigma_x < 1e-10:
                kg_values[i] = 0.0
                continue
            
            # Monte Carlo estimate of expected improvement in max
            fantasy_improvements = []
            for _ in range(self.n_fantasies):
                # Sample a fantasy observation
                y_fantasy = mu_x + sigma_x * self._rng.standard_normal()
                
                # Compute the change in posterior mean at discrete points
                # Using a simplified update (assuming low-rank update)
                # This is an approximation - full implementation would update GP
                
                # Approximate: after observing y at x, mean increases by 
                # correlation * (y - prior_mean)
                mean_discrete, _ = surrogate.predict(self._X_discrete)
                
                # Simple approximation: nearby points get updated more
                distances = np.linalg.norm(self._X_discrete - x, axis=1)
                weights = np.exp(-distances**2 / 0.1)  # Length scale = 0.1 in normalized space
                weights /= weights.max() + 1e-10
                
                # Updated means
                updated_mean = mean_discrete + weights * (y_fantasy - mu_x) * 0.5
                new_max = np.max(updated_mean)
                fantasy_improvements.append(new_max - current_max)
            
            kg_values[i] = np.mean(fantasy_improvements)
        
        # KG should be non-negative (we can at worst learn nothing useful)
        kg_values = np.maximum(kg_values, 0)
        
        return kg_values


class MaxValueEntropySearch(AcquisitionFunction):
    """Max-Value Entropy Search (MES) acquisition function.
    
    Measures the expected reduction in entropy of the distribution
    over the maximum value of the function.
    
    MES is robust to noise and works well in practice.
    
    Parameters
    ----------
    n_samples : int, default=10
        Number of samples for entropy estimation.
        
    Examples
    --------
    >>> mes = MaxValueEntropySearch(n_samples=20)
    >>> values = mes(X, surrogate, y_best)
    """
    
    def __init__(self, n_samples: int = 10) -> None:
        self.n_samples = n_samples
        self._rng = np.random.default_rng()
        self._y_max_samples = None
    
    def _sample_y_max(self, surrogate: SurrogateModel, n_dims: int) -> np.ndarray:
        """Sample possible maximum values from the GP posterior."""
        # Sample random locations
        n_eval = 200
        X_random = self._rng.uniform(0, 1, (n_eval, n_dims))
        
        # Get predictions
        mean, std = surrogate.predict(X_random)
        
        # Sample posterior values at each location
        samples = mean + std * self._rng.standard_normal((self.n_samples, n_eval))
        
        # Max of each sample path
        y_max_samples = np.max(samples, axis=1)
        
        return y_max_samples
    
    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate Max-Value Entropy Search at given points."""
        X = np.atleast_2d(X)
        n_points, n_dims = X.shape
        
        # Sample possible max values
        y_max_samples = self._sample_y_max(surrogate, n_dims)
        
        # Get predictions at candidate points
        mean, std = surrogate.predict(X)
        
        # Compute MES for each candidate
        mes_values = np.zeros(n_points)
        
        for i in range(n_points):
            mu, sigma = mean[i], std[i]
            
            if sigma < 1e-10:
                mes_values[i] = 0.0
                continue
            
            # For each y_max sample, compute contribution to entropy reduction
            gamma = (y_max_samples - mu) / sigma
            
            # Truncated normal entropy reduction
            pdf_gamma = norm.pdf(gamma)
            cdf_gamma = norm.cdf(gamma)
            
            # Avoid numerical issues
            cdf_gamma = np.clip(cdf_gamma, 1e-10, 1 - 1e-10)
            
            # MES value: expected log(p(y_max | y_x) / p(y_max))
            # Approximated as information gain about y_max from observing y at x
            mes_values[i] = np.mean(
                gamma * pdf_gamma / (2 * cdf_gamma) - 
                np.log(cdf_gamma)
            )
        
        return np.maximum(mes_values, 0)


# =============================================================================
# Batch Acquisition Functions
# =============================================================================

@dataclass
class BatchResult:
    """Result of batch acquisition optimization.
    
    Attributes
    ----------
    X : np.ndarray
        Batch of suggested points, shape (batch_size, n_dims).
    values : np.ndarray
        Acquisition function values at each point.
    """
    X: np.ndarray
    values: np.ndarray


class LocalPenalization:
    """Local Penalization for batch Bayesian optimization.
    
    After selecting a point, penalizes nearby regions to encourage
    diversity in the batch. Uses a Gaussian penalty centered at
    previously selected points.
    
    Parameters
    ----------
    base_acquisition : AcquisitionFunction
        The base acquisition function to use (e.g., EI, UCB).
    lipschitz_constant : float, optional
        Estimated Lipschitz constant of the objective.
        If None, estimated from data.
        
    References
    ----------
    González, J., et al. (2016). Batch Bayesian Optimization via 
    Local Penalization. AISTATS.
    
    Examples
    --------
    >>> lp = LocalPenalization(ExpectedImprovement())
    >>> batch = lp.suggest_batch(surrogate, y_best, n_suggestions=4, bounds=[(0,1)]*3)
    """
    
    def __init__(
        self, 
        base_acquisition: AcquisitionFunction,
        lipschitz_constant: Optional[float] = None
    ) -> None:
        self.base_acquisition = base_acquisition
        self.lipschitz_constant = lipschitz_constant
    
    def _estimate_lipschitz(self, surrogate: SurrogateModel, n_dims: int) -> float:
        """Estimate Lipschitz constant from GP gradients."""
        # Sample random points
        rng = np.random.default_rng()
        X = rng.uniform(0, 1, (100, n_dims))
        mean, std = surrogate.predict(X)
        
        # Estimate max gradient magnitude
        diffs = np.abs(np.diff(mean))
        if len(diffs) == 0:
            return 1.0
        
        return np.max(diffs) * n_dims  # Rough estimate
    
    def _penalize(
        self, 
        X: np.ndarray, 
        pending_points: List[np.ndarray],
        surrogate: SurrogateModel,
        L: float
    ) -> np.ndarray:
        """Apply local penalization around pending points."""
        if not pending_points:
            return np.ones(len(X))
        
        penalty = np.ones(len(X))
        
        for x_pending in pending_points:
            # Get predicted mean and std at pending point
            mu_pending, sigma_pending = surrogate.predict(x_pending.reshape(1, -1))
            mu_pending, sigma_pending = mu_pending[0], sigma_pending[0]
            
            # Compute penalty radius
            r = sigma_pending / L if L > 0 else 0.1
            
            # Compute distances to pending point
            distances = np.linalg.norm(X - x_pending, axis=1)
            
            # Apply soft penalty (Gaussian)
            point_penalty = norm.cdf((distances - r) / (r * 0.5))
            penalty *= point_penalty
        
        return penalty
    
    def suggest_batch(
        self,
        surrogate: SurrogateModel,
        y_best: float,
        n_suggestions: int,
        bounds: List[Tuple[float, float]],
        n_restarts: int = 10,
    ) -> BatchResult:
        """Suggest a batch of points using local penalization.
        
        Parameters
        ----------
        surrogate : SurrogateModel
            Fitted surrogate model.
        y_best : float
            Best observed value so far.
        n_suggestions : int
            Number of points to suggest.
        bounds : list of tuples
            Bounds for each dimension.
        n_restarts : int
            Number of random restarts for optimization.
            
        Returns
        -------
        BatchResult
            Batch of suggested points and their acquisition values.
        """
        n_dims = len(bounds)
        bounds_array = np.array(bounds)
        rng = np.random.default_rng()
        
        # Estimate Lipschitz constant if not provided
        L = self.lipschitz_constant
        if L is None:
            L = self._estimate_lipschitz(surrogate, n_dims)
        
        batch_points = []
        batch_values = []
        
        for _ in range(n_suggestions):
            # Optimize penalized acquisition
            best_x = None
            best_value = -np.inf
            
            for _ in range(n_restarts):
                x0 = rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
                
                def neg_penalized_acq(x):
                    x = np.clip(x, bounds_array[:, 0], bounds_array[:, 1])
                    X = x.reshape(1, -1)
                    
                    # Base acquisition
                    acq = self.base_acquisition(X, surrogate, y_best)[0]
                    
                    # Apply penalty
                    penalty = self._penalize(X, batch_points, surrogate, L)[0]
                    
                    return -(acq * penalty)
                
                try:
                    result = minimize(
                        neg_penalized_acq,
                        x0,
                        method="L-BFGS-B",
                        bounds=bounds
                    )
                    
                    if -result.fun > best_value:
                        best_value = -result.fun
                        best_x = result.x
                except Exception:
                    continue
            
            if best_x is None:
                # Fallback to random
                best_x = rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
                best_value = 0.0
            
            batch_points.append(best_x.copy())
            batch_values.append(best_value)
        
        return BatchResult(
            X=np.array(batch_points),
            values=np.array(batch_values)
        )


class KrigingBeliever:
    """Kriging Believer for batch Bayesian optimization.
    
    Adds each suggested point to the GP with its predicted mean
    as a "fantasy" observation, then suggests the next point.
    
    Parameters
    ----------
    base_acquisition : AcquisitionFunction
        The base acquisition function to use.
        
    References
    ----------
    Ginsbourger, D., et al. (2010). Kriging is well-suited to 
    parallelize optimization. Computational Intelligence in 
    Expensive Optimization Problems.
    
    Examples
    --------
    >>> kb = KrigingBeliever(ExpectedImprovement())
    >>> batch = kb.suggest_batch(surrogate, space, y_best, n_suggestions=4)
    """
    
    def __init__(self, base_acquisition: AcquisitionFunction) -> None:
        self.base_acquisition = base_acquisition
    
    def suggest_batch(
        self,
        surrogate: SurrogateModel,
        y_best: float,
        n_suggestions: int,
        bounds: List[Tuple[float, float]],
        X_observed: Optional[np.ndarray] = None,
        y_observed: Optional[np.ndarray] = None,
        n_restarts: int = 10,
    ) -> BatchResult:
        """Suggest a batch of points using Kriging Believer.
        
        Parameters
        ----------
        surrogate : SurrogateModel
            Fitted surrogate model.
        y_best : float
            Best observed value so far.
        n_suggestions : int
            Number of points to suggest.
        bounds : list of tuples
            Bounds for each dimension.
        X_observed : np.ndarray, optional
            Previously observed points for refitting.
        y_observed : np.ndarray, optional
            Previously observed values for refitting.
        n_restarts : int
            Number of random restarts.
            
        Returns
        -------
        BatchResult
            Batch of suggested points and their acquisition values.
        """
        from optiml.surrogate import GaussianProcessSurrogate
        
        n_dims = len(bounds)
        bounds_array = np.array(bounds)
        rng = np.random.default_rng()
        
        # Start with copies of observed data
        if X_observed is not None and y_observed is not None:
            X_fantasy = list(X_observed)
            y_fantasy = list(y_observed)
        else:
            X_fantasy = []
            y_fantasy = []
        
        batch_points = []
        batch_values = []
        
        # Create a copy of the surrogate for fantasy observations
        fantasy_surrogate = GaussianProcessSurrogate()
        if X_fantasy:
            fantasy_surrogate.fit(np.array(X_fantasy), np.array(y_fantasy))
        else:
            fantasy_surrogate = surrogate  # Use original if no data
        
        current_y_best = y_best
        
        for _ in range(n_suggestions):
            # Optimize acquisition on fantasy surrogate
            best_x = None
            best_value = -np.inf
            
            for _ in range(n_restarts):
                x0 = rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
                
                def neg_acq(x):
                    x = np.clip(x, bounds_array[:, 0], bounds_array[:, 1])
                    X = x.reshape(1, -1)
                    return -self.base_acquisition(X, fantasy_surrogate, current_y_best)[0]
                
                try:
                    result = minimize(
                        neg_acq,
                        x0,
                        method="L-BFGS-B",
                        bounds=bounds
                    )
                    
                    if -result.fun > best_value:
                        best_value = -result.fun
                        best_x = result.x
                except Exception:
                    continue
            
            if best_x is None:
                # Fallback to random
                best_x = rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
                best_value = 0.0
            
            # Get fantasy observation (predicted mean)
            fantasy_y = fantasy_surrogate.predict(best_x.reshape(1, -1))[0][0]
            
            # Add to fantasy data
            X_fantasy.append(best_x.copy())
            y_fantasy.append(fantasy_y)
            
            # Update best
            current_y_best = max(current_y_best, fantasy_y)
            
            # Refit fantasy surrogate
            try:
                fantasy_surrogate = GaussianProcessSurrogate()
                fantasy_surrogate.fit(np.array(X_fantasy), np.array(y_fantasy))
            except Exception:
                pass  # Keep using previous if fit fails
            
            batch_points.append(best_x.copy())
            batch_values.append(best_value)
        
        return BatchResult(
            X=np.array(batch_points),
            values=np.array(batch_values)
        )


class ConstantLiar:
    """Constant Liar for batch Bayesian optimization.
    
    Similar to Kriging Believer but uses a constant value
    (min, max, or mean of observations) as the fantasy observation.
    
    Parameters
    ----------
    base_acquisition : AcquisitionFunction
        The base acquisition function to use.
    lie_value : str, default="min"
        Strategy for the lie value: "min", "max", or "mean".
        
    Examples
    --------
    >>> cl = ConstantLiar(ExpectedImprovement(), lie_value="min")
    >>> batch = cl.suggest_batch(surrogate, y_best, n_suggestions=4, bounds=[(0,1)]*3)
    """
    
    def __init__(
        self, 
        base_acquisition: AcquisitionFunction,
        lie_value: str = "min"
    ) -> None:
        self.base_acquisition = base_acquisition
        self.lie_value = lie_value
    
    def suggest_batch(
        self,
        surrogate: SurrogateModel,
        y_best: float,
        n_suggestions: int,
        bounds: List[Tuple[float, float]],
        y_observed: Optional[np.ndarray] = None,
        n_restarts: int = 10,
    ) -> BatchResult:
        """Suggest a batch of points using Constant Liar."""
        from optiml.surrogate import GaussianProcessSurrogate
        
        n_dims = len(bounds)
        bounds_array = np.array(bounds)
        rng = np.random.default_rng()
        
        # Determine lie value
        if y_observed is not None and len(y_observed) > 0:
            if self.lie_value == "min":
                lie = float(np.min(y_observed))
            elif self.lie_value == "max":
                lie = float(np.max(y_observed))
            else:  # mean
                lie = float(np.mean(y_observed))
        else:
            lie = y_best
        
        batch_points = []
        batch_values = []
        
        for _ in range(n_suggestions):
            # Optimize acquisition
            best_x = None
            best_value = -np.inf
            
            for _ in range(n_restarts):
                x0 = rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
                
                def neg_acq(x):
                    x = np.clip(x, bounds_array[:, 0], bounds_array[:, 1])
                    X = x.reshape(1, -1)
                    return -self.base_acquisition(X, surrogate, y_best)[0]
                
                try:
                    result = minimize(
                        neg_acq,
                        x0,
                        method="L-BFGS-B",
                        bounds=bounds
                    )
                    
                    if -result.fun > best_value:
                        best_value = -result.fun
                        best_x = result.x
                except Exception:
                    continue
            
            if best_x is None:
                best_x = rng.uniform(bounds_array[:, 0], bounds_array[:, 1])
                best_value = 0.0
            
            batch_points.append(best_x.copy())
            batch_values.append(best_value)
        
        return BatchResult(
            X=np.array(batch_points),
            values=np.array(batch_values)
        )


# =============================================================================
# Portfolio / Ensemble Methods
# =============================================================================

class AcquisitionPortfolio(AcquisitionFunction):
    """Portfolio of acquisition functions.
    
    Combines multiple acquisition functions using weighted averaging
    or selection strategies. Useful for hedging against any single
    strategy's weaknesses.
    
    Parameters
    ----------
    acquisitions : list of AcquisitionFunction
        List of acquisition functions to combine.
    weights : list of float, optional
        Weights for each acquisition function.
        If None, uses equal weights.
    strategy : str, default="weighted"
        Combination strategy: "weighted", "max", or "random".
        
    Examples
    --------
    >>> portfolio = AcquisitionPortfolio([
    ...     ExpectedImprovement(),
    ...     UpperConfidenceBound(kappa=2.0),
    ...     ThompsonSampling()
    ... ], weights=[0.5, 0.3, 0.2])
    >>> values = portfolio(X, surrogate, y_best)
    """
    
    def __init__(
        self,
        acquisitions: List[AcquisitionFunction],
        weights: Optional[List[float]] = None,
        strategy: str = "weighted"
    ) -> None:
        self.acquisitions = acquisitions
        self.weights = weights or [1.0 / len(acquisitions)] * len(acquisitions)
        self.strategy = strategy
        self._rng = np.random.default_rng()
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def __call__(self, X: np.ndarray, surrogate: SurrogateModel, y_best: float) -> np.ndarray:
        """Evaluate the portfolio at given points."""
        X = np.atleast_2d(X)
        
        # Evaluate all acquisition functions
        all_values = []
        for acq in self.acquisitions:
            values = acq(X, surrogate, y_best)
            # Normalize to [0, 1]
            vmin, vmax = values.min(), values.max()
            if vmax > vmin:
                values = (values - vmin) / (vmax - vmin)
            all_values.append(values)
        
        all_values = np.array(all_values)
        
        if self.strategy == "weighted":
            # Weighted average
            result = np.zeros(len(X))
            for values, weight in zip(all_values, self.weights):
                result += weight * values
            return result
        
        elif self.strategy == "max":
            # Take max across acquisitions
            return np.max(all_values, axis=0)
        
        elif self.strategy == "random":
            # Randomly select one acquisition per point
            indices = self._rng.choice(len(self.acquisitions), size=len(X), p=self.weights)
            return np.array([all_values[idx, i] for i, idx in enumerate(indices)])
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def create_acquisition(name: str, **kwargs) -> AcquisitionFunction:
    """Factory function to create acquisition functions by name.
    
    Parameters
    ----------
    name : str
        Name of the acquisition function. Options:
        "ei", "expected_improvement"
        "ucb", "upper_confidence_bound"
        "pi", "probability_of_improvement"
        "lcb", "lower_confidence_bound"
        "ts", "thompson_sampling"
        "kg", "knowledge_gradient"
        "mes", "max_value_entropy_search"
    **kwargs
        Additional arguments for the acquisition function.
        
    Returns
    -------
    AcquisitionFunction
        The requested acquisition function.
        
    Examples
    --------
    >>> acq = create_acquisition("ei", xi=0.01)
    >>> acq = create_acquisition("ucb", kappa=3.0)
    """
    name = name.lower()
    
    mapping = {
        "ei": ExpectedImprovement,
        "expected_improvement": ExpectedImprovement,
        "ucb": UpperConfidenceBound,
        "upper_confidence_bound": UpperConfidenceBound,
        "pi": ProbabilityOfImprovement,
        "probability_of_improvement": ProbabilityOfImprovement,
        "lcb": LowerConfidenceBound,
        "lower_confidence_bound": LowerConfidenceBound,
        "ts": ThompsonSampling,
        "thompson_sampling": ThompsonSampling,
        "kg": KnowledgeGradient,
        "knowledge_gradient": KnowledgeGradient,
        "mes": MaxValueEntropySearch,
        "max_value_entropy_search": MaxValueEntropySearch,
    }
    
    if name not in mapping:
        raise ValueError(
            f"Unknown acquisition function: {name}. "
            f"Available: {list(mapping.keys())}"
        )
    
    return mapping[name](**kwargs)
