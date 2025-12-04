"""
Kernel Functions for Gaussian Process Models.

This module provides a variety of kernel (covariance) functions for
Gaussian Process surrogate models, including:
- Radial Basis Function (RBF) / Squared Exponential
- Matérn family (1/2, 3/2, 5/2)
- Rational Quadratic
- Periodic kernels
- Composite kernels (Sum, Product)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.spatial.distance import cdist


class Kernel(ABC):
    """Abstract base class for kernel functions.

    All kernels must implement __call__ to compute the covariance matrix
    and have hyperparameters that can be optimized.
    """

    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        """Compute the kernel matrix.

        Parameters
        ----------
        X1 : np.ndarray
            First set of points, shape (n1, n_features).
        X2 : np.ndarray, optional
            Second set of points, shape (n2, n_features).
            If None, computes K(X1, X1).

        Returns
        -------
        np.ndarray
            Kernel matrix, shape (n1, n2) or (n1, n1).
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        """Get current hyperparameter values."""
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """Set hyperparameter values."""
        pass

    @abstractmethod
    def clone(self) -> "Kernel":
        """Create a copy of this kernel with the same parameters."""
        pass

    @property
    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for hyperparameter optimization (in log space)."""
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of hyperparameters."""
        pass

    def get_params_array(self) -> np.ndarray:
        """Get hyperparameters as a numpy array (log-transformed)."""
        params = self.get_params()
        return np.log(np.array(list(params.values())))

    def set_params_array(self, params: np.ndarray) -> None:
        """Set hyperparameters from numpy array (log-transformed)."""
        param_names = list(self.get_params().keys())
        param_dict = {name: np.exp(val) for name, val in zip(param_names, params)}
        self.set_params(**param_dict)

    def __add__(self, other: "Kernel") -> "SumKernel":
        """Add two kernels."""
        return SumKernel(self, other)

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        """Multiply two kernels."""
        return ProductKernel(self, other)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params_str})"


class RBF(Kernel):
    """Radial Basis Function (Squared Exponential) kernel.

    k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter (l).
    variance : float, default=1.0
        Signal variance (σ²).
    length_scale_bounds : tuple, default=(1e-5, 1e5)
        Bounds for length scale optimization.
    variance_bounds : tuple, default=(1e-5, 1e5)
        Bounds for variance optimization.
    ard : bool, default=False
        If True, use Automatic Relevance Determination (separate
        length scale per dimension).
    n_dims : int, optional
        Number of dimensions (required if ard=True).

    Examples
    --------
    >>> kernel = RBF(length_scale=1.0, variance=1.0)
    >>> X = np.random.randn(10, 2)
    >>> K = kernel(X)
    >>> print(K.shape)  # (10, 10)
    """

    def __init__(
        self,
        length_scale: float | np.ndarray = 1.0,
        variance: float = 1.0,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
        variance_bounds: Tuple[float, float] = (1e-5, 1e5),
        ard: bool = False,
        n_dims: int | None = None,
    ):
        self.ard = ard
        self.n_dims = n_dims
        
        if ard and n_dims is not None:
            if isinstance(length_scale, (int, float)):
                self.length_scale = np.full(n_dims, float(length_scale))
            else:
                self.length_scale = np.asarray(length_scale)
        else:
            self.length_scale = float(length_scale) if not isinstance(length_scale, np.ndarray) else length_scale
        
        self.variance = float(variance)
        self.length_scale_bounds = length_scale_bounds
        self.variance_bounds = variance_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = np.atleast_2d(X2)
        
        # Scale by length scale
        if self.ard and isinstance(self.length_scale, np.ndarray):
            X1_scaled = X1 / self.length_scale
            X2_scaled = X2 / self.length_scale
        else:
            X1_scaled = X1 / self.length_scale
            X2_scaled = X2 / self.length_scale
        
        # Squared Euclidean distance
        dist_sq = cdist(X1_scaled, X2_scaled, metric='sqeuclidean')
        
        return self.variance * np.exp(-0.5 * dist_sq)

    def get_params(self) -> Dict[str, float]:
        if self.ard and isinstance(self.length_scale, np.ndarray):
            params = {f"length_scale_{i}": ls for i, ls in enumerate(self.length_scale)}
        else:
            params = {"length_scale": float(self.length_scale)}
        params["variance"] = self.variance
        return params

    def set_params(self, **params) -> None:
        if "variance" in params:
            self.variance = params.pop("variance")
        
        if self.ard and self.n_dims is not None:
            for i in range(self.n_dims):
                key = f"length_scale_{i}"
                if key in params:
                    self.length_scale[i] = params[key]
        elif "length_scale" in params:
            self.length_scale = params["length_scale"]

    def clone(self) -> "RBF":
        return RBF(
            length_scale=self.length_scale.copy() if isinstance(self.length_scale, np.ndarray) else self.length_scale,
            variance=self.variance,
            length_scale_bounds=self.length_scale_bounds,
            variance_bounds=self.variance_bounds,
            ard=self.ard,
            n_dims=self.n_dims,
        )

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        log_bounds = []
        if self.ard and self.n_dims is not None:
            for _ in range(self.n_dims):
                log_bounds.append((np.log(self.length_scale_bounds[0]), np.log(self.length_scale_bounds[1])))
        else:
            log_bounds.append((np.log(self.length_scale_bounds[0]), np.log(self.length_scale_bounds[1])))
        log_bounds.append((np.log(self.variance_bounds[0]), np.log(self.variance_bounds[1])))
        return log_bounds

    @property
    def n_params(self) -> int:
        if self.ard and self.n_dims is not None:
            return self.n_dims + 1
        return 2


class Matern(Kernel):
    """Matérn kernel family.

    The Matérn kernel interpolates between different levels of smoothness.
    Common choices for nu are 1/2, 3/2, and 5/2.

    k(x, x') = σ² * (2^(1-ν) / Γ(ν)) * (√(2ν) * d / l)^ν * K_ν(√(2ν) * d / l)

    Where K_ν is the modified Bessel function.

    For specific values of ν, we have closed forms:
    - ν = 1/2: Exponential kernel (rough, not differentiable)
    - ν = 3/2: Once differentiable
    - ν = 5/2: Twice differentiable (common default)
    - ν → ∞: RBF kernel (infinitely differentiable)

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter.
    variance : float, default=1.0
        Signal variance.
    nu : float, default=2.5
        Smoothness parameter. Common values: 0.5, 1.5, 2.5, 5.0

    Examples
    --------
    >>> kernel = Matern(length_scale=1.0, nu=2.5)
    >>> X = np.random.randn(10, 2)
    >>> K = kernel(X)
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        nu: float = 2.5,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
        variance_bounds: Tuple[float, float] = (1e-5, 1e5),
    ):
        if nu not in [0.5, 1.5, 2.5, 5.0]:
            raise ValueError(f"nu must be one of [0.5, 1.5, 2.5, 5.0], got {nu}")
        
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.nu = nu
        self.length_scale_bounds = length_scale_bounds
        self.variance_bounds = variance_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = np.atleast_2d(X2)
        
        # Euclidean distance
        dist = cdist(X1 / self.length_scale, X2 / self.length_scale, metric='euclidean')
        
        if self.nu == 0.5:
            # Exponential kernel
            K = np.exp(-dist)
        elif self.nu == 1.5:
            # Matérn 3/2
            sqrt3_dist = np.sqrt(3) * dist
            K = (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
        elif self.nu == 2.5:
            # Matérn 5/2
            sqrt5_dist = np.sqrt(5) * dist
            K = (1 + sqrt5_dist + sqrt5_dist**2 / 3) * np.exp(-sqrt5_dist)
        elif self.nu == 5.0:
            # Higher order Matérn (approximation closer to RBF)
            sqrt10_dist = np.sqrt(10) * dist
            K = (1 + sqrt10_dist + sqrt10_dist**2 / 3 + sqrt10_dist**3 / 15) * np.exp(-sqrt10_dist)
        
        return self.variance * K

    def get_params(self) -> Dict[str, float]:
        return {"length_scale": self.length_scale, "variance": self.variance}

    def set_params(self, **params) -> None:
        if "length_scale" in params:
            self.length_scale = params["length_scale"]
        if "variance" in params:
            self.variance = params["variance"]

    def clone(self) -> "Matern":
        return Matern(
            length_scale=self.length_scale,
            variance=self.variance,
            nu=self.nu,
            length_scale_bounds=self.length_scale_bounds,
            variance_bounds=self.variance_bounds,
        )

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [
            (np.log(self.length_scale_bounds[0]), np.log(self.length_scale_bounds[1])),
            (np.log(self.variance_bounds[0]), np.log(self.variance_bounds[1])),
        ]

    @property
    def n_params(self) -> int:
        return 2


class Matern12(Matern):
    """Matérn kernel with ν = 1/2 (Exponential kernel)."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, **kwargs):
        super().__init__(length_scale=length_scale, variance=variance, nu=0.5, **kwargs)


class Matern32(Matern):
    """Matérn kernel with ν = 3/2."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, **kwargs):
        super().__init__(length_scale=length_scale, variance=variance, nu=1.5, **kwargs)


class Matern52(Matern):
    """Matérn kernel with ν = 5/2 (recommended default)."""

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, **kwargs):
        super().__init__(length_scale=length_scale, variance=variance, nu=2.5, **kwargs)


class RationalQuadratic(Kernel):
    """Rational Quadratic kernel.

    A mixture of RBF kernels with different length scales.

    k(x, x') = σ² * (1 + ||x - x'||² / (2 * α * l²))^(-α)

    As α → ∞, this approaches the RBF kernel.

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter.
    variance : float, default=1.0
        Signal variance.
    alpha : float, default=1.0
        Scale mixture parameter.

    Examples
    --------
    >>> kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    >>> X = np.random.randn(10, 2)
    >>> K = kernel(X)
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        alpha: float = 1.0,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
        variance_bounds: Tuple[float, float] = (1e-5, 1e5),
        alpha_bounds: Tuple[float, float] = (1e-5, 1e5),
    ):
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.alpha = float(alpha)
        self.length_scale_bounds = length_scale_bounds
        self.variance_bounds = variance_bounds
        self.alpha_bounds = alpha_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = np.atleast_2d(X2)
        
        dist_sq = cdist(X1, X2, metric='sqeuclidean')
        
        K = (1 + dist_sq / (2 * self.alpha * self.length_scale**2)) ** (-self.alpha)
        
        return self.variance * K

    def get_params(self) -> Dict[str, float]:
        return {
            "length_scale": self.length_scale,
            "variance": self.variance,
            "alpha": self.alpha,
        }

    def set_params(self, **params) -> None:
        if "length_scale" in params:
            self.length_scale = params["length_scale"]
        if "variance" in params:
            self.variance = params["variance"]
        if "alpha" in params:
            self.alpha = params["alpha"]

    def clone(self) -> "RationalQuadratic":
        return RationalQuadratic(
            length_scale=self.length_scale,
            variance=self.variance,
            alpha=self.alpha,
            length_scale_bounds=self.length_scale_bounds,
            variance_bounds=self.variance_bounds,
            alpha_bounds=self.alpha_bounds,
        )

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [
            (np.log(self.length_scale_bounds[0]), np.log(self.length_scale_bounds[1])),
            (np.log(self.variance_bounds[0]), np.log(self.variance_bounds[1])),
            (np.log(self.alpha_bounds[0]), np.log(self.alpha_bounds[1])),
        ]

    @property
    def n_params(self) -> int:
        return 3


class Periodic(Kernel):
    """Periodic kernel for cyclic data.

    k(x, x') = σ² * exp(-2 * sin²(π * ||x - x'|| / p) / l²)

    Parameters
    ----------
    length_scale : float, default=1.0
        Length scale parameter.
    variance : float, default=1.0
        Signal variance.
    period : float, default=1.0
        Period of the repeating pattern.

    Examples
    --------
    >>> kernel = Periodic(period=24.0)  # For hourly data with daily cycle
    >>> t = np.linspace(0, 48, 100).reshape(-1, 1)
    >>> K = kernel(t)
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        variance: float = 1.0,
        period: float = 1.0,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
        variance_bounds: Tuple[float, float] = (1e-5, 1e5),
        period_bounds: Tuple[float, float] = (1e-5, 1e5),
    ):
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.period = float(period)
        self.length_scale_bounds = length_scale_bounds
        self.variance_bounds = variance_bounds
        self.period_bounds = period_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = np.atleast_2d(X2)
        
        dist = cdist(X1, X2, metric='euclidean')
        
        sin_term = np.sin(np.pi * dist / self.period)
        K = np.exp(-2 * sin_term**2 / self.length_scale**2)
        
        return self.variance * K

    def get_params(self) -> Dict[str, float]:
        return {
            "length_scale": self.length_scale,
            "variance": self.variance,
            "period": self.period,
        }

    def set_params(self, **params) -> None:
        if "length_scale" in params:
            self.length_scale = params["length_scale"]
        if "variance" in params:
            self.variance = params["variance"]
        if "period" in params:
            self.period = params["period"]

    def clone(self) -> "Periodic":
        return Periodic(
            length_scale=self.length_scale,
            variance=self.variance,
            period=self.period,
            length_scale_bounds=self.length_scale_bounds,
            variance_bounds=self.variance_bounds,
            period_bounds=self.period_bounds,
        )

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [
            (np.log(self.length_scale_bounds[0]), np.log(self.length_scale_bounds[1])),
            (np.log(self.variance_bounds[0]), np.log(self.variance_bounds[1])),
            (np.log(self.period_bounds[0]), np.log(self.period_bounds[1])),
        ]

    @property
    def n_params(self) -> int:
        return 3


class WhiteNoise(Kernel):
    """White noise kernel (adds noise variance to diagonal).

    k(x, x') = σ² * δ(x, x')

    Where δ is the Kronecker delta.

    Parameters
    ----------
    noise_variance : float, default=1.0
        Noise variance (σ²).

    Examples
    --------
    >>> kernel = RBF() + WhiteNoise(noise_variance=0.1)
    >>> X = np.random.randn(10, 2)
    >>> K = kernel(X)  # RBF + noise on diagonal
    """

    def __init__(
        self,
        noise_variance: float = 1.0,
        noise_variance_bounds: Tuple[float, float] = (1e-10, 1e5),
    ):
        self.noise_variance = float(noise_variance)
        self.noise_variance_bounds = noise_variance_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        if X2 is None:
            return self.noise_variance * np.eye(len(X1))
        else:
            X2 = np.atleast_2d(X2)
            # Only add noise when X1 == X2 (check if same array)
            if X1.shape == X2.shape and np.allclose(X1, X2):
                return self.noise_variance * np.eye(len(X1))
            return np.zeros((len(X1), len(X2)))

    def get_params(self) -> Dict[str, float]:
        return {"noise_variance": self.noise_variance}

    def set_params(self, **params) -> None:
        if "noise_variance" in params:
            self.noise_variance = params["noise_variance"]

    def clone(self) -> "WhiteNoise":
        return WhiteNoise(
            noise_variance=self.noise_variance,
            noise_variance_bounds=self.noise_variance_bounds,
        )

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(np.log(self.noise_variance_bounds[0]), np.log(self.noise_variance_bounds[1]))]

    @property
    def n_params(self) -> int:
        return 1


class ConstantKernel(Kernel):
    """Constant kernel (constant covariance).

    k(x, x') = c

    Parameters
    ----------
    constant : float, default=1.0
        Constant value.
    """

    def __init__(
        self,
        constant: float = 1.0,
        constant_bounds: Tuple[float, float] = (1e-5, 1e5),
    ):
        self.constant = float(constant)
        self.constant_bounds = constant_bounds

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = np.atleast_2d(X2)
        
        return self.constant * np.ones((len(X1), len(X2)))

    def get_params(self) -> Dict[str, float]:
        return {"constant": self.constant}

    def set_params(self, **params) -> None:
        if "constant" in params:
            self.constant = params["constant"]

    def clone(self) -> "ConstantKernel":
        return ConstantKernel(
            constant=self.constant,
            constant_bounds=self.constant_bounds,
        )

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(np.log(self.constant_bounds[0]), np.log(self.constant_bounds[1]))]

    @property
    def n_params(self) -> int:
        return 1


class SumKernel(Kernel):
    """Sum of two kernels.

    k(x, x') = k1(x, x') + k2(x, x')

    Typically used to combine different patterns, e.g., trend + periodic.
    """

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        return self.k1(X1, X2) + self.k2(X1, X2)

    def get_params(self) -> Dict[str, float]:
        params = {}
        for k, v in self.k1.get_params().items():
            params[f"k1_{k}"] = v
        for k, v in self.k2.get_params().items():
            params[f"k2_{k}"] = v
        return params

    def set_params(self, **params) -> None:
        k1_params = {k[3:]: v for k, v in params.items() if k.startswith("k1_")}
        k2_params = {k[3:]: v for k, v in params.items() if k.startswith("k2_")}
        if k1_params:
            self.k1.set_params(**k1_params)
        if k2_params:
            self.k2.set_params(**k2_params)

    def clone(self) -> "SumKernel":
        return SumKernel(self.k1.clone(), self.k2.clone())

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return self.k1.bounds + self.k2.bounds

    @property
    def n_params(self) -> int:
        return self.k1.n_params + self.k2.n_params

    def __repr__(self) -> str:
        return f"({self.k1} + {self.k2})"


class ProductKernel(Kernel):
    """Product of two kernels.

    k(x, x') = k1(x, x') * k2(x, x')

    Typically used to model interactions or combine local/global patterns.
    """

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        return self.k1(X1, X2) * self.k2(X1, X2)

    def get_params(self) -> Dict[str, float]:
        params = {}
        for k, v in self.k1.get_params().items():
            params[f"k1_{k}"] = v
        for k, v in self.k2.get_params().items():
            params[f"k2_{k}"] = v
        return params

    def set_params(self, **params) -> None:
        k1_params = {k[3:]: v for k, v in params.items() if k.startswith("k1_")}
        k2_params = {k[3:]: v for k, v in params.items() if k.startswith("k2_")}
        if k1_params:
            self.k1.set_params(**k1_params)
        if k2_params:
            self.k2.set_params(**k2_params)

    def clone(self) -> "ProductKernel":
        return ProductKernel(self.k1.clone(), self.k2.clone())

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return self.k1.bounds + self.k2.bounds

    @property
    def n_params(self) -> int:
        return self.k1.n_params + self.k2.n_params

    def __repr__(self) -> str:
        return f"({self.k1} * {self.k2})"


# Convenience aliases
SquaredExponential = RBF
Exponential = Matern12


def create_kernel(
    name: str,
    **kwargs
) -> Kernel:
    """Factory function to create kernels by name.

    Parameters
    ----------
    name : str
        Kernel name. One of: 'rbf', 'matern12', 'matern32', 'matern52',
        'rational_quadratic', 'periodic', 'white_noise', 'constant'.
    **kwargs
        Kernel-specific parameters.

    Returns
    -------
    Kernel
        The created kernel instance.

    Examples
    --------
    >>> kernel = create_kernel('matern52', length_scale=2.0)
    >>> kernel = create_kernel('rbf', ard=True, n_dims=3)
    """
    kernels = {
        'rbf': RBF,
        'squared_exponential': RBF,
        'matern12': Matern12,
        'matern32': Matern32,
        'matern52': Matern52,
        'matern': Matern,
        'exponential': Matern12,
        'rational_quadratic': RationalQuadratic,
        'rq': RationalQuadratic,
        'periodic': Periodic,
        'white_noise': WhiteNoise,
        'noise': WhiteNoise,
        'constant': ConstantKernel,
    }
    
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    
    if name_lower not in kernels:
        available = ', '.join(sorted(kernels.keys()))
        raise ValueError(f"Unknown kernel '{name}'. Available: {available}")
    
    return kernels[name_lower](**kwargs)


# Default kernel for Bayesian Optimization
def default_kernel(n_dims: int = 1, use_ard: bool = True) -> Kernel:
    """Create the default kernel for Bayesian optimization.

    The default is Matérn 5/2 + White Noise, which balances:
    - Smoothness (suitable for most response surfaces)
    - Noise handling (for experimental noise)
    - ARD (for feature relevance detection)

    Parameters
    ----------
    n_dims : int, default=1
        Number of input dimensions.
    use_ard : bool, default=True
        Whether to use Automatic Relevance Determination.

    Returns
    -------
    Kernel
        Default kernel for BO.
    """
    if use_ard and n_dims > 1:
        main_kernel = RBF(length_scale=1.0, variance=1.0, ard=True, n_dims=n_dims)
    else:
        main_kernel = Matern52(length_scale=1.0, variance=1.0)
    
    noise = WhiteNoise(noise_variance=1e-4)
    
    return main_kernel + noise
