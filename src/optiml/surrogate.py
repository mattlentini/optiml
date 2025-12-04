"""Surrogate models for Bayesian optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from optiml.space import Space


class SurrogateModel(ABC):
    """Base class for surrogate models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurrogateModel":
        """Fit the surrogate model to observed data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at given points."""
        pass


class GaussianProcessSurrogate(SurrogateModel):
    """Gaussian Process surrogate model with RBF kernel.

    This implementation uses a Radial Basis Function (RBF) kernel with
    automatic relevance determination (ARD) and optimizes hyperparameters
    using marginal likelihood maximization.

    Parameters
    ----------
    length_scale : float or np.ndarray, default=1.0
        Length scale parameter(s) for the RBF kernel.
    noise : float, default=1e-6
        Noise level added to the diagonal for numerical stability.
    normalize_y : bool, default=True
        Whether to normalize the target values.
    n_restarts : int, default=5
        Number of random restarts for hyperparameter optimization.

    Examples
    --------
    >>> gp = GaussianProcessSurrogate()
    >>> X = np.array([[0.1], [0.4], [0.7]])
    >>> y = np.array([0.5, 0.8, 0.3])
    >>> gp.fit(X, y)
    >>> mean, std = gp.predict(np.array([[0.2], [0.5]]))
    """

    def __init__(
        self,
        length_scale: float | np.ndarray = 1.0,
        noise: float = 1e-6,
        normalize_y: bool = True,
        n_restarts: int = 5,
    ) -> None:
        self.length_scale = np.atleast_1d(length_scale)
        self.noise = noise
        self.normalize_y = normalize_y
        self.n_restarts = n_restarts

        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the RBF kernel between two sets of points.

        K(x, x') = exp(-0.5 * sum_d ((x_d - x'_d) / l_d)^2)
        """
        # Scale by length scale
        X1_scaled = X1 / self.length_scale
        X2_scaled = X2 / self.length_scale

        # Compute squared Euclidean distances
        dists = cdist(X1_scaled, X2_scaled, metric="sqeuclidean")

        return np.exp(-0.5 * dists)

    def _negative_log_marginal_likelihood(
        self, theta: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Compute the negative log marginal likelihood."""
        # Unpack hyperparameters
        length_scale = np.exp(theta[:-1])
        noise = np.exp(theta[-1])

        # Store original and set new
        original_length_scale = self.length_scale
        original_noise = self.noise
        self.length_scale = length_scale
        self.noise = noise

        try:
            K = self._rbf_kernel(X, X) + noise * np.eye(X.shape[0])
            L = cholesky(K, lower=True)
            alpha = cho_solve((L, True), y)

            # Log marginal likelihood
            log_likelihood = -0.5 * np.dot(y, alpha)
            log_likelihood -= np.sum(np.log(np.diag(L)))
            log_likelihood -= 0.5 * X.shape[0] * np.log(2 * np.pi)

            return -log_likelihood

        except np.linalg.LinAlgError:
            return 1e10  # Return large value if Cholesky fails

        finally:
            # Restore original hyperparameters
            self.length_scale = original_length_scale
            self.noise = original_noise

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize hyperparameters using marginal likelihood maximization."""
        n_dims = X.shape[1]

        best_theta = None
        best_nll = np.inf

        # Initial guess and bounds
        theta_initial = np.zeros(n_dims + 1)
        bounds = [(-5, 5)] * n_dims + [(-10, 0)]  # length scales + noise

        rng = np.random.default_rng(42)

        for _ in range(self.n_restarts):
            # Random starting point
            theta_start = rng.uniform(-2, 2, n_dims + 1)

            try:
                result = minimize(
                    self._negative_log_marginal_likelihood,
                    theta_start,
                    args=(X, y),
                    method="L-BFGS-B",
                    bounds=bounds,
                )

                if result.fun < best_nll:
                    best_nll = result.fun
                    best_theta = result.x

            except Exception:
                continue

        if best_theta is not None:
            self.length_scale = np.exp(best_theta[:-1])
            self.noise = np.exp(best_theta[-1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        """Fit the Gaussian Process to observed data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input samples in normalized [0, 1]^n space.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GaussianProcessSurrogate
            Returns self.
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).ravel()

        # Normalize y if requested
        if self.normalize_y:
            self._y_mean = np.mean(y)
            self._y_std = np.std(y) if np.std(y) > 0 else 1.0
            y_normalized = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_normalized = y

        self._X_train = X
        self._y_train = y_normalized

        # Initialize length scale if needed
        if self.length_scale.shape[0] != X.shape[1]:
            self.length_scale = np.ones(X.shape[1])

        # Optimize hyperparameters
        self._optimize_hyperparameters(X, y_normalized)

        # Compute kernel matrix and Cholesky decomposition
        K = self._rbf_kernel(X, X) + self.noise * np.eye(X.shape[0])
        self._L = cholesky(K, lower=True)
        self._alpha = cho_solve((self._L, True), y_normalized)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation at given points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Points at which to predict.

        Returns
        -------
        mean : np.ndarray of shape (n_samples,)
            Predicted mean values.
        std : np.ndarray of shape (n_samples,)
            Predicted standard deviations.
        """
        if self._X_train is None or self._alpha is None or self._L is None:
            raise RuntimeError("Model must be fitted before making predictions.")

        X = np.atleast_2d(X)

        # Compute kernel between test and training points
        K_star = self._rbf_kernel(X, self._X_train)

        # Predictive mean
        mean = K_star @ self._alpha

        # Predictive variance
        v = solve_triangular(self._L, K_star.T, lower=True)
        K_star_star = self._rbf_kernel(X, X)
        var = np.diag(K_star_star) - np.sum(v**2, axis=0)

        # Ensure non-negative variance
        var = np.maximum(var, 1e-10)
        std = np.sqrt(var)

        # Denormalize
        mean = mean * self._y_std + self._y_mean
        std = std * self._y_std

        return mean, std

    def sample_y(
        self, X: np.ndarray, n_samples: int = 1, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample from the posterior distribution.

        Parameters
        ----------
        X : np.ndarray of shape (n_points, n_features)
            Points at which to sample.
        n_samples : int, default=1
            Number of samples to draw.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        samples : np.ndarray of shape (n_samples, n_points)
            Samples from the posterior.
        """
        if rng is None:
            rng = np.random.default_rng()

        mean, std = self.predict(X)

        # Sample from multivariate normal
        samples = rng.normal(mean, std, size=(n_samples, len(mean)))

        return samples
