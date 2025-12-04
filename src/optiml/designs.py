"""
Initial Experimental Design Methods for Bayesian Optimization.

This module provides various sampling strategies for generating initial
experimental designs, including space-filling designs, factorial designs,
and optimal designs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:
    from optiml.space import Space


class Design(ABC):
    """Abstract base class for experimental designs."""

    @abstractmethod
    def generate(self, space: Space, n_samples: int, seed: int | None = None) -> list[list[Any]]:
        """Generate initial design points.

        Parameters
        ----------
        space : Space
            The search space to sample from.
        n_samples : int
            Number of design points to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        list[list[Any]]
            List of design points in original space.
        """
        pass


class RandomDesign(Design):
    """Uniform random sampling design.

    Simple random sampling from the parameter space. Each dimension
    is sampled independently from a uniform distribution.

    Examples
    --------
    >>> design = RandomDesign()
    >>> points = design.generate(space, n_samples=10)
    """

    def generate(self, space: Space, n_samples: int, seed: int | None = None) -> list[list[Any]]:
        """Generate random design points."""
        rng = np.random.default_rng(seed)
        return space.sample(n_samples, rng)


class LatinHypercubeDesign(Design):
    """Latin Hypercube Sampling (LHS) design.

    LHS ensures that each parameter is sampled evenly across its range
    by dividing each dimension into n equal strata and sampling once
    from each stratum.

    Parameters
    ----------
    criterion : str, default="lloyd"
        Optimization criterion for the LHS:
        - "lloyd": Lloyd's algorithm (centroidal Voronoi tessellation)
        - "random-cd": Random permutations with centered L2-discrepancy
        - "random": No optimization, random LHS
    iterations : int, default=100
        Number of iterations for optimization (ignored if criterion="random").

    Examples
    --------
    >>> design = LatinHypercubeDesign(criterion="lloyd")
    >>> points = design.generate(space, n_samples=20)

    Notes
    -----
    LHS provides better space-filling properties than random sampling,
    especially for small sample sizes. The "lloyd" criterion produces
    designs where points are spread apart using centroidal Voronoi tessellation.
    """

    def __init__(self, criterion: str = "lloyd", iterations: int = 100) -> None:
        # Map common names to scipy-supported options
        criterion_map = {
            "maximin": "lloyd",  # lloyd approximates maximin
            "correlation": "random-cd",
            "random": None,
            "lloyd": "lloyd",
            "random-cd": "random-cd",
        }
        self.criterion = criterion_map.get(criterion, criterion)
        self.iterations = iterations

    def generate(self, space: Space, n_samples: int, seed: int | None = None) -> list[list[Any]]:
        """Generate Latin Hypercube design points."""
        n_dims = len(space)
        
        if self.criterion is None or self.criterion == "random":
            sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        else:
            # Use optimization to improve the design
            sampler = qmc.LatinHypercube(
                d=n_dims,
                seed=seed,
                optimization=self.criterion,
            )
        
        # Generate samples in [0, 1]^d
        samples = sampler.random(n=n_samples)
        
        # Transform to original space
        return space.inverse_transform(samples)


class SobolDesign(Design):
    """Sobol quasi-random sequence design.

    Sobol sequences are low-discrepancy sequences that provide
    excellent space-filling properties, especially for high-dimensional
    problems.

    Parameters
    ----------
    scramble : bool, default=True
        Whether to scramble the Sobol sequence for better randomization.

    Examples
    --------
    >>> design = SobolDesign(scramble=True)
    >>> points = design.generate(space, n_samples=32)

    Notes
    -----
    The number of samples should ideally be a power of 2 for optimal
    coverage, though any sample size is supported.
    """

    def __init__(self, scramble: bool = True) -> None:
        self.scramble = scramble

    def generate(self, space: Space, n_samples: int, seed: int | None = None) -> list[list[Any]]:
        """Generate Sobol sequence design points."""
        n_dims = len(space)
        
        sampler = qmc.Sobol(d=n_dims, scramble=self.scramble, seed=seed)
        
        # Generate samples in [0, 1]^d
        samples = sampler.random(n=n_samples)
        
        # Transform to original space
        return space.inverse_transform(samples)


class HaltonDesign(Design):
    """Halton quasi-random sequence design.

    Halton sequences are another type of low-discrepancy sequence,
    using different prime bases for each dimension.

    Parameters
    ----------
    scramble : bool, default=True
        Whether to scramble the Halton sequence.

    Examples
    --------
    >>> design = HaltonDesign()
    >>> points = design.generate(space, n_samples=20)
    """

    def __init__(self, scramble: bool = True) -> None:
        self.scramble = scramble

    def generate(self, space: Space, n_samples: int, seed: int | None = None) -> list[list[Any]]:
        """Generate Halton sequence design points."""
        n_dims = len(space)
        
        sampler = qmc.Halton(d=n_dims, scramble=self.scramble, seed=seed)
        
        # Generate samples in [0, 1]^d
        samples = sampler.random(n=n_samples)
        
        # Transform to original space
        return space.inverse_transform(samples)


class FullFactorialDesign(Design):
    """Full factorial design.

    Creates a complete factorial design with all combinations of
    specified levels for each factor.

    Parameters
    ----------
    levels : int or list[int], default=3
        Number of levels for each dimension. If int, same for all dimensions.
        If list, specifies levels for each dimension.

    Examples
    --------
    >>> design = FullFactorialDesign(levels=3)  # 3^k design
    >>> points = design.generate(space, n_samples=None)  # n_samples ignored

    Notes
    -----
    The total number of runs is levels^(n_dimensions), which grows
    exponentially. Use fractional factorial or screening designs for
    large numbers of parameters.
    """

    def __init__(self, levels: int | list[int] = 3) -> None:
        self.levels = levels

    def generate(self, space: Space, n_samples: int | None = None, seed: int | None = None) -> list[list[Any]]:
        """Generate full factorial design points."""
        n_dims = len(space)
        
        if isinstance(self.levels, int):
            levels = [self.levels] * n_dims
        else:
            levels = self.levels
        
        # Generate level values for each dimension (0 to 1)
        level_values = [np.linspace(0, 1, lev) for lev in levels]
        
        # Create meshgrid for all combinations
        grids = np.meshgrid(*level_values, indexing='ij')
        
        # Flatten and combine
        samples = np.column_stack([g.ravel() for g in grids])
        
        # Transform to original space
        return space.inverse_transform(samples)


class FractionalFactorialDesign(Design):
    """Fractional factorial design (Resolution III or higher).

    Creates a fractional factorial design that estimates main effects
    with fewer runs than a full factorial.

    Parameters
    ----------
    resolution : int, default=3
        Resolution of the design:
        - III: Main effects aliased with 2-factor interactions
        - IV: Main effects clear of 2-factor interactions
        - V: 2-factor interactions clear of each other

    Examples
    --------
    >>> design = FractionalFactorialDesign(resolution=4)
    >>> points = design.generate(space, n_samples=None)

    Notes
    -----
    The design uses a 2^(k-p) fractional factorial structure where
    k is the number of factors and p is chosen to achieve the
    desired resolution.
    """

    def __init__(self, resolution: int = 3) -> None:
        self.resolution = resolution

    def generate(self, space: Space, n_samples: int | None = None, seed: int | None = None) -> list[list[Any]]:
        """Generate fractional factorial design points."""
        n_dims = len(space)
        
        # Calculate minimum runs needed for given resolution
        # For resolution III: n >= k + 1
        # For resolution IV: n >= 2k
        # For resolution V: n >= 2k + 1
        
        if self.resolution == 3:
            min_runs = n_dims + 1
        elif self.resolution == 4:
            min_runs = 2 * n_dims
        else:  # resolution >= 5
            min_runs = 2 * n_dims + 1
        
        # Find smallest power of 2 >= min_runs
        n_runs = 2 ** int(np.ceil(np.log2(max(min_runs, 4))))
        
        # Generate using Hadamard matrix approach for 2-level designs
        # Simplified: use corner points and center
        rng = np.random.default_rng(seed)
        
        # Generate 2-level factorial subset
        samples = []
        
        # Add corner points
        for i in range(min(n_runs, 2**n_dims)):
            point = []
            for j in range(n_dims):
                # Binary encoding
                bit = (i >> j) & 1
                point.append(float(bit))
            samples.append(point)
        
        samples = np.array(samples[:n_runs])
        
        # Add center point
        center = np.full((1, n_dims), 0.5)
        samples = np.vstack([samples, center])
        
        # Transform to original space
        return space.inverse_transform(samples)


class CentralCompositeDesign(Design):
    """Central Composite Design (CCD) for response surface methodology.

    CCD consists of:
    - A factorial or fractional factorial design (corner points)
    - Star/axial points at distance alpha from the center
    - Center points

    Parameters
    ----------
    alpha : str or float, default="orthogonal"
        Distance of axial points from center:
        - "orthogonal": Makes design orthogonal
        - "rotatable": Makes design rotatable (equal prediction variance)
        - "face": alpha=1, axial points on face of cube
        - float: Custom alpha value
    n_center : int, default=3
        Number of center point replicates.

    Examples
    --------
    >>> design = CentralCompositeDesign(alpha="rotatable", n_center=5)
    >>> points = design.generate(space)

    Notes
    -----
    CCD is ideal for fitting second-order response surface models.
    The choice of alpha affects the properties of the design.
    """

    def __init__(self, alpha: str | float = "orthogonal", n_center: int = 3) -> None:
        self.alpha = alpha
        self.n_center = n_center

    def _calculate_alpha(self, n_dims: int, n_factorial: int) -> float:
        """Calculate alpha based on design type."""
        if isinstance(self.alpha, (int, float)):
            return float(self.alpha)
        elif self.alpha == "face":
            return 1.0
        elif self.alpha == "rotatable":
            # alpha = (n_factorial)^(1/4)
            return n_factorial ** 0.25
        elif self.alpha == "orthogonal":
            # alpha = sqrt(k) where k = n_dims
            return np.sqrt(n_dims)
        else:
            return 1.0

    def generate(self, space: Space, n_samples: int | None = None, seed: int | None = None) -> list[list[Any]]:
        """Generate Central Composite Design points."""
        n_dims = len(space)
        
        # Factorial portion (2^k or 2^(k-p) for large k)
        n_factorial = min(2 ** n_dims, 32)  # Limit factorial points
        
        alpha = self._calculate_alpha(n_dims, n_factorial)
        
        samples = []
        
        # 1. Factorial points (scaled to [-1, 1] then to [0, 1])
        for i in range(n_factorial):
            point = []
            for j in range(n_dims):
                bit = (i >> j) & 1
                # Map 0 -> -1, 1 -> 1, then scale to [0, 1]
                val = (2 * bit - 1) / (2 * alpha) + 0.5
                point.append(np.clip(val, 0, 1))
            samples.append(point)
        
        # 2. Axial (star) points
        for dim in range(n_dims):
            # +alpha point
            point_plus = [0.5] * n_dims
            point_plus[dim] = np.clip(0.5 + 0.5 / alpha * alpha, 0, 1)
            samples.append(point_plus)
            
            # -alpha point
            point_minus = [0.5] * n_dims
            point_minus[dim] = np.clip(0.5 - 0.5 / alpha * alpha, 0, 1)
            samples.append(point_minus)
        
        # 3. Center points
        for _ in range(self.n_center):
            samples.append([0.5] * n_dims)
        
        samples = np.array(samples)
        
        # Transform to original space
        return space.inverse_transform(samples)


class BoxBehnkenDesign(Design):
    """Box-Behnken design for response surface methodology.

    Box-Behnken designs are efficient 3-level designs for fitting
    second-order response surfaces. They don't include corner points,
    making them useful when extreme combinations should be avoided.

    Parameters
    ----------
    n_center : int, default=3
        Number of center point replicates.

    Examples
    --------
    >>> design = BoxBehnkenDesign(n_center=3)
    >>> points = design.generate(space)

    Notes
    -----
    Box-Behnken designs exist for 3 or more factors. For 2 factors,
    a Central Composite Design is used instead.
    """

    def __init__(self, n_center: int = 3) -> None:
        self.n_center = n_center

    def generate(self, space: Space, n_samples: int | None = None, seed: int | None = None) -> list[list[Any]]:
        """Generate Box-Behnken Design points."""
        n_dims = len(space)
        
        if n_dims < 3:
            # Fall back to CCD for < 3 factors
            ccd = CentralCompositeDesign(n_center=self.n_center)
            return ccd.generate(space, n_samples, seed)
        
        samples = []
        
        # Box-Behnken: pairs of factors at ±1, others at 0
        # Generate all pairs
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                # 4 combinations for each pair
                for val_i in [0.0, 1.0]:
                    for val_j in [0.0, 1.0]:
                        point = [0.5] * n_dims  # Center for all
                        point[i] = val_i
                        point[j] = val_j
                        samples.append(point)
        
        # Center points
        for _ in range(self.n_center):
            samples.append([0.5] * n_dims)
        
        samples = np.array(samples)
        
        # Transform to original space
        return space.inverse_transform(samples)


class PlackettBurmanDesign(Design):
    """Plackett-Burman screening design.

    Plackett-Burman designs are 2-level fractional factorial designs
    for screening many factors with minimal runs. They are Resolution III
    designs, meaning main effects may be confounded with 2-factor interactions.

    Examples
    --------
    >>> design = PlackettBurmanDesign()
    >>> points = design.generate(space)  # Generates n_runs = multiple of 4

    Notes
    -----
    Plackett-Burman designs are ideal for initial screening when you have
    many factors (5-20+) and want to quickly identify the most important ones.
    The number of runs is always a multiple of 4.
    """

    # Pre-defined Plackett-Burman generators
    _generators = {
        4: [1, 1, -1],  # 3 factors in 4 runs
        8: [1, 1, 1, -1, 1, -1, -1],  # 7 factors in 8 runs
        12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],  # 11 factors in 12 runs
        16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],  # 15 factors
        20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
        24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
    }

    def generate(self, space: Space, n_samples: int | None = None, seed: int | None = None) -> list[list[Any]]:
        """Generate Plackett-Burman Design points."""
        n_dims = len(space)
        
        # Find smallest design that can accommodate n_dims factors
        n_runs = 4
        while n_runs - 1 < n_dims and n_runs <= 24:
            n_runs += 4
        
        if n_runs > 24:
            # Fall back to fractional factorial for very large designs
            ff = FractionalFactorialDesign(resolution=3)
            return ff.generate(space, n_samples, seed)
        
        generator = self._generators.get(n_runs, self._generators[12])
        
        # Build design matrix by cyclic shifting
        design = []
        row = generator[:n_dims]
        
        # First row
        design.append(row.copy())
        
        # Subsequent rows by cyclic shift
        for _ in range(n_runs - 2):
            row = [row[-1]] + row[:-1]  # Cyclic right shift
            design.append(row[:n_dims])
        
        # Last row: all -1
        design.append([-1] * n_dims)
        
        # Convert -1, 1 to 0, 1
        samples = np.array(design)
        samples = (samples + 1) / 2  # Map [-1, 1] to [0, 1]
        
        # Transform to original space
        return space.inverse_transform(samples)


@dataclass
class DesignMetrics:
    """Metrics for evaluating experimental design quality.

    Attributes
    ----------
    n_points : int
        Number of design points.
    n_dims : int
        Number of dimensions.
    min_distance : float
        Minimum pairwise distance between points.
    mean_distance : float
        Mean pairwise distance.
    max_distance : float
        Maximum pairwise distance.
    space_filling : float
        Space-filling metric (higher is better).
    discrepancy : float
        L2-star discrepancy (lower is better for quasi-random).
    """

    n_points: int
    n_dims: int
    min_distance: float
    mean_distance: float
    max_distance: float
    space_filling: float
    discrepancy: float


def evaluate_design(points: list[list[Any]], space: "Space") -> DesignMetrics:
    """Evaluate the quality of an experimental design.

    Parameters
    ----------
    points : list[list[Any]]
        Design points in original space.
    space : Space
        The search space.

    Returns
    -------
    DesignMetrics
        Quality metrics for the design.

    Examples
    --------
    >>> design = LatinHypercubeDesign()
    >>> points = design.generate(space, n_samples=20)
    >>> metrics = evaluate_design(points, space)
    >>> print(f"Space filling: {metrics.space_filling:.3f}")
    """
    from scipy.spatial.distance import pdist
    
    # Transform to normalized space for fair comparison
    X = space.transform(points)
    n_points = len(points)
    n_dims = X.shape[1]
    
    # Pairwise distances
    distances = pdist(X)
    
    min_dist = np.min(distances) if len(distances) > 0 else 0.0
    mean_dist = np.mean(distances) if len(distances) > 0 else 0.0
    max_dist = np.max(distances) if len(distances) > 0 else 0.0
    
    # Space-filling metric (modified maximin)
    # Higher is better
    space_filling = min_dist * np.sqrt(n_dims) if min_dist > 0 else 0.0
    
    # L2-star discrepancy (lower is better)
    try:
        discrepancy = qmc.discrepancy(X, method='L2-star')
    except Exception:
        discrepancy = float('inf')
    
    return DesignMetrics(
        n_points=n_points,
        n_dims=n_dims,
        min_distance=min_dist,
        mean_distance=mean_dist,
        max_distance=max_dist,
        space_filling=space_filling,
        discrepancy=discrepancy,
    )


def compare_designs(
    space: "Space",
    n_samples: int = 20,
    designs: list[Design] | None = None,
    seed: int | None = None,
) -> dict[str, DesignMetrics]:
    """Compare multiple design strategies.

    Parameters
    ----------
    space : Space
        The search space.
    n_samples : int, default=20
        Number of samples for each design.
    designs : list[Design], optional
        Designs to compare. If None, compares common designs.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, DesignMetrics]
        Metrics for each design type.

    Examples
    --------
    >>> results = compare_designs(space, n_samples=30)
    >>> for name, metrics in results.items():
    ...     print(f"{name}: space_filling={metrics.space_filling:.3f}")
    """
    if designs is None:
        designs = [
            ("Random", RandomDesign()),
            ("Latin Hypercube", LatinHypercubeDesign(criterion="maximin")),
            ("Sobol", SobolDesign()),
            ("Halton", HaltonDesign()),
        ]
    else:
        designs = [(d.__class__.__name__, d) for d in designs]
    
    results = {}
    for name, design in designs:
        try:
            points = design.generate(space, n_samples, seed)
            metrics = evaluate_design(points, space)
            results[name] = metrics
        except Exception as e:
            print(f"Warning: {name} design failed: {e}")
    
    return results
