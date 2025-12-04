"""Acquisition functions for Bayesian optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

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
