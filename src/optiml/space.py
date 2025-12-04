"""Search space definitions for Bayesian optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class Dimension(ABC):
    """Base class for search space dimensions."""

    @abstractmethod
    def sample(self, n_samples: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample random points from this dimension."""
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform values to normalized [0, 1] space."""
        pass

    @abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transform values from normalized [0, 1] space back to original."""
        pass

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float]:
        """Return the bounds of this dimension in normalized space."""
        pass


@dataclass
class Real(Dimension):
    """A continuous real-valued dimension.

    Parameters
    ----------
    low : float
        Lower bound of the dimension.
    high : float
        Upper bound of the dimension.
    name : str, optional
        Name of the dimension.
    log_scale : bool, default=False
        If True, the dimension is sampled in log space.

    Examples
    --------
    >>> learning_rate = Real(1e-5, 1e-1, name="learning_rate", log_scale=True)
    >>> samples = learning_rate.sample(5)
    """

    low: float
    high: float
    name: str | None = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.log_scale and self.low <= 0:
            raise ValueError("low must be positive for log_scale=True")

    def sample(self, n_samples: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample random points from this dimension."""
        if rng is None:
            rng = np.random.default_rng()

        if self.log_scale:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return np.exp(rng.uniform(log_low, log_high, n_samples))
        return rng.uniform(self.low, self.high, n_samples)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform values to normalized [0, 1] space."""
        x = np.asarray(x)
        if self.log_scale:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return (np.log(x) - log_low) / (log_high - log_low)
        return (x - self.low) / (self.high - self.low)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transform values from normalized [0, 1] space back to original."""
        x = np.asarray(x)
        if self.log_scale:
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return np.exp(x * (log_high - log_low) + log_low)
        return x * (self.high - self.low) + self.low

    @property
    def bounds(self) -> tuple[float, float]:
        """Return the bounds in normalized space."""
        return (0.0, 1.0)


@dataclass
class Integer(Dimension):
    """A discrete integer-valued dimension.

    Parameters
    ----------
    low : int
        Lower bound of the dimension (inclusive).
    high : int
        Upper bound of the dimension (inclusive).
    name : str, optional
        Name of the dimension.

    Examples
    --------
    >>> n_layers = Integer(1, 10, name="n_layers")
    >>> samples = n_layers.sample(5)
    """

    low: int
    high: int
    name: str | None = None

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")

    def sample(self, n_samples: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample random points from this dimension."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(self.low, self.high + 1, n_samples)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform values to normalized [0, 1] space."""
        x = np.asarray(x)
        return (x - self.low) / (self.high - self.low)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transform values from normalized [0, 1] space back to original."""
        x = np.asarray(x)
        return np.round(x * (self.high - self.low) + self.low).astype(int)

    @property
    def bounds(self) -> tuple[float, float]:
        """Return the bounds in normalized space."""
        return (0.0, 1.0)


@dataclass
class Categorical(Dimension):
    """A categorical dimension with discrete choices.

    Parameters
    ----------
    categories : Sequence[Any]
        List of possible categorical values.
    name : str, optional
        Name of the dimension.

    Examples
    --------
    >>> activation = Categorical(["relu", "tanh", "sigmoid"], name="activation")
    >>> samples = activation.sample(5)
    """

    categories: Sequence[Any]
    name: str | None = None

    def __post_init__(self) -> None:
        if len(self.categories) < 2:
            raise ValueError("categories must have at least 2 elements")
        self._categories = list(self.categories)

    def sample(self, n_samples: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample random points from this dimension."""
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.integers(0, len(self._categories), n_samples)
        return np.array([self._categories[i] for i in indices])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform categorical values to normalized indices."""
        x = np.asarray(x)
        result = np.zeros(x.shape)
        for i, val in enumerate(x.flat):
            result.flat[i] = self._categories.index(val) / (len(self._categories) - 1)
        return result

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transform normalized indices back to categorical values."""
        x = np.asarray(x)
        indices = np.round(x * (len(self._categories) - 1)).astype(int)
        indices = np.clip(indices, 0, len(self._categories) - 1)
        return np.array([self._categories[i] for i in indices.flat]).reshape(x.shape)

    @property
    def bounds(self) -> tuple[float, float]:
        """Return the bounds in normalized space."""
        return (0.0, 1.0)


class Space:
    """A search space composed of multiple dimensions.

    Parameters
    ----------
    dimensions : list[Dimension]
        List of dimension objects defining the search space.

    Examples
    --------
    >>> space = Space([
    ...     Real(1e-5, 1e-1, name="learning_rate", log_scale=True),
    ...     Integer(1, 10, name="n_layers"),
    ...     Categorical(["relu", "tanh"], name="activation"),
    ... ])
    >>> samples = space.sample(10)
    """

    def __init__(self, dimensions: list[Dimension]) -> None:
        self.dimensions = dimensions
        self._n_dims = len(dimensions)

    def __len__(self) -> int:
        return self._n_dims

    def __getitem__(self, index: int) -> Dimension:
        return self.dimensions[index]

    @property
    def dimension_names(self) -> list[str | None]:
        """Return the names of all dimensions."""
        return [d.name for d in self.dimensions]

    def sample(self, n_samples: int = 1, rng: np.random.Generator | None = None) -> list[list[Any]]:
        """Sample random points from the search space.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        list[list[Any]]
            List of sampled points, where each point is a list of values.
        """
        if rng is None:
            rng = np.random.default_rng()

        samples = []
        for _ in range(n_samples):
            point = [dim.sample(1, rng)[0] for dim in self.dimensions]
            samples.append(point)
        return samples

    def transform(self, X: list[list[Any]]) -> np.ndarray:
        """Transform points to normalized [0, 1]^n space.

        Parameters
        ----------
        X : list[list[Any]]
            List of points in original space.

        Returns
        -------
        np.ndarray
            Transformed points in normalized space.
        """
        X_transformed = np.zeros((len(X), self._n_dims))
        for i, point in enumerate(X):
            for j, (val, dim) in enumerate(zip(point, self.dimensions)):
                X_transformed[i, j] = dim.transform(np.array([val]))[0]
        return X_transformed

    def inverse_transform(self, X: np.ndarray) -> list[list[Any]]:
        """Transform points from normalized space back to original.

        Parameters
        ----------
        X : np.ndarray
            Points in normalized [0, 1]^n space.

        Returns
        -------
        list[list[Any]]
            Points in original space.
        """
        result = []
        for i in range(X.shape[0]):
            point = []
            for j, dim in enumerate(self.dimensions):
                val = dim.inverse_transform(np.array([X[i, j]]))[0]
                point.append(val)
            result.append(point)
        return result
