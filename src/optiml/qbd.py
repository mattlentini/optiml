"""Quality by Design (QbD) Design Space module.

This module provides tools for defining, visualizing, and validating design spaces
according to ICH Q8/Q9/Q10/Q14 guidelines for analytical method development.

Features:
- Design space calculation based on probability of meeting specifications
- Method Operable Design Region (MODR) determination
- Edge of Failure analysis
- Monte Carlo robustness simulation
- Control strategy recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.ndimage import binary_dilation, binary_erosion

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SpecificationType(Enum):
    """Types of specification limits."""
    MINIMUM = "minimum"      # >= value
    MAXIMUM = "maximum"      # <= value
    TARGET = "target"        # = value ± tolerance
    RANGE = "range"          # between min and max


@dataclass
class Specification:
    """A specification limit for a response variable.
    
    Attributes
    ----------
    name : str
        Response variable name.
    spec_type : SpecificationType
        Type of specification.
    value : float
        Target or limit value.
    tolerance : float, optional
        Tolerance for target specifications.
    lower : float, optional
        Lower limit for range specifications.
    upper : float, optional
        Upper limit for range specifications.
    weight : float
        Importance weight (0-1).
    """
    name: str
    spec_type: SpecificationType
    value: Optional[float] = None
    tolerance: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    weight: float = 1.0
    
    def check(self, response_value: float) -> bool:
        """Check if a response value meets this specification."""
        if self.spec_type == SpecificationType.MINIMUM:
            return response_value >= self.value
        elif self.spec_type == SpecificationType.MAXIMUM:
            return response_value <= self.value
        elif self.spec_type == SpecificationType.TARGET:
            return abs(response_value - self.value) <= self.tolerance
        elif self.spec_type == SpecificationType.RANGE:
            return self.lower <= response_value <= self.upper
        return False
    
    def probability(
        self,
        mean: float,
        std: float,
    ) -> float:
        """Calculate probability of meeting specification.
        
        Parameters
        ----------
        mean : float
            Predicted mean response.
        std : float
            Predicted standard deviation.
            
        Returns
        -------
        float
            Probability of meeting specification (0-1).
        """
        if std == 0:
            return 1.0 if self.check(mean) else 0.0
        
        if self.spec_type == SpecificationType.MINIMUM:
            # P(X >= value) = 1 - CDF(value)
            return 1 - stats.norm.cdf(self.value, mean, std)
        
        elif self.spec_type == SpecificationType.MAXIMUM:
            # P(X <= value) = CDF(value)
            return stats.norm.cdf(self.value, mean, std)
        
        elif self.spec_type == SpecificationType.TARGET:
            lower = self.value - self.tolerance
            upper = self.value + self.tolerance
            return (
                stats.norm.cdf(upper, mean, std) -
                stats.norm.cdf(lower, mean, std)
            )
        
        elif self.spec_type == SpecificationType.RANGE:
            return (
                stats.norm.cdf(self.upper, mean, std) -
                stats.norm.cdf(self.lower, mean, std)
            )
        
        return 0.0


def spec_minimum(name: str, value: float, weight: float = 1.0) -> Specification:
    """Create a minimum specification (response >= value)."""
    return Specification(name=name, spec_type=SpecificationType.MINIMUM, 
                        value=value, weight=weight)


def spec_maximum(name: str, value: float, weight: float = 1.0) -> Specification:
    """Create a maximum specification (response <= value)."""
    return Specification(name=name, spec_type=SpecificationType.MAXIMUM,
                        value=value, weight=weight)


def spec_target(
    name: str,
    target: float,
    tolerance: float,
    weight: float = 1.0,
) -> Specification:
    """Create a target specification (response = target ± tolerance)."""
    return Specification(name=name, spec_type=SpecificationType.TARGET,
                        value=target, tolerance=tolerance, weight=weight)


def spec_range(
    name: str,
    lower: float,
    upper: float,
    weight: float = 1.0,
) -> Specification:
    """Create a range specification (lower <= response <= upper)."""
    return Specification(name=name, spec_type=SpecificationType.RANGE,
                        lower=lower, upper=upper, weight=weight)


@dataclass
class DesignSpacePoint:
    """A point in the design space with probability of success.
    
    Attributes
    ----------
    parameters : Dict[str, float]
        Parameter values at this point.
    probability : float
        Joint probability of meeting all specifications.
    individual_probs : Dict[str, float]
        Probability for each individual specification.
    is_in_design_space : bool
        Whether point is in the design space (prob >= threshold).
    is_in_modr : bool
        Whether point is in the MODR.
    """
    parameters: Dict[str, float]
    probability: float
    individual_probs: Dict[str, float] = field(default_factory=dict)
    is_in_design_space: bool = False
    is_in_modr: bool = False


@dataclass
class DesignSpaceResult:
    """Result of design space calculation.
    
    Attributes
    ----------
    grid_points : np.ndarray
        Grid of parameter values (n_points x n_params).
    probabilities : np.ndarray
        Probability surface over the grid.
    design_space_mask : np.ndarray
        Boolean mask indicating design space region.
    modr_mask : np.ndarray
        Boolean mask indicating MODR.
    edge_of_failure : np.ndarray
        Boolean mask indicating edge of failure region.
    parameter_names : List[str]
        Names of parameters.
    specifications : List[Specification]
        Specifications used.
    confidence_level : float
        Confidence level used for design space.
    volume_fraction : float
        Fraction of parameter space in design space.
    modr_volume_fraction : float
        Fraction in MODR.
    """
    grid_points: np.ndarray
    probabilities: np.ndarray
    design_space_mask: np.ndarray
    modr_mask: np.ndarray
    edge_of_failure: np.ndarray
    parameter_names: List[str]
    specifications: List[Specification]
    confidence_level: float
    volume_fraction: float = 0.0
    modr_volume_fraction: float = 0.0
    grid_shape: Tuple[int, ...] = field(default_factory=tuple)


class DesignSpace:
    """Quality by Design (QbD) Design Space calculator.
    
    Calculates the design space as the region where there is a high
    probability of meeting all specifications simultaneously.
    
    Parameters
    ----------
    surrogates : Dict[str, Any]
        Fitted surrogate models for each response variable.
        Keys are response names, values are fitted GP models.
    specifications : List[Specification]
        List of specifications to meet.
    confidence_level : float, default=0.95
        Probability threshold for design space membership.
    modr_margin : float, default=0.05
        Additional margin for MODR (probability must exceed
        confidence_level + modr_margin).
    
    Examples
    --------
    >>> from optiml.qbd import DesignSpace, spec_minimum, spec_maximum
    >>> 
    >>> # Assume we have fitted GP models for Resolution and RunTime
    >>> ds = DesignSpace(
    ...     surrogates={"Resolution": gp_resolution, "RunTime": gp_runtime},
    ...     specifications=[
    ...         spec_minimum("Resolution", 2.0),
    ...         spec_maximum("RunTime", 30.0),
    ...     ],
    ...     confidence_level=0.95,
    ... )
    >>> 
    >>> # Calculate design space over parameter grid
    >>> result = ds.calculate(
    ...     parameter_ranges={"pH": (5.0, 8.0), "Temperature": (20, 40)},
    ...     n_points=50,
    ... )
    """
    
    def __init__(
        self,
        surrogates: Dict[str, Any],
        specifications: List[Specification],
        confidence_level: float = 0.95,
        modr_margin: float = 0.05,
    ):
        self.surrogates = surrogates
        self.specifications = specifications
        self.confidence_level = confidence_level
        self.modr_margin = modr_margin
        
        # Validate specifications match surrogates
        for spec in specifications:
            if spec.name not in surrogates:
                raise ValueError(
                    f"No surrogate model for specification '{spec.name}'. "
                    f"Available: {list(surrogates.keys())}"
                )
    
    def calculate(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        n_points: int = 50,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> DesignSpaceResult:
        """Calculate the design space over a parameter grid.
        
        Parameters
        ----------
        parameter_ranges : Dict[str, Tuple[float, float]]
            Range (min, max) for each parameter.
        n_points : int, default=50
            Number of grid points per dimension.
        correlation_matrix : np.ndarray, optional
            Correlation matrix between response variables for joint
            probability calculation. If None, assumes independence.
            
        Returns
        -------
        DesignSpaceResult
            Design space calculation results.
        """
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)
        
        # Create grid
        grids = []
        for name in param_names:
            low, high = parameter_ranges[name]
            grids.append(np.linspace(low, high, n_points))
        
        mesh = np.meshgrid(*grids, indexing='ij')
        grid_shape = mesh[0].shape
        grid_points = np.column_stack([m.flatten() for m in mesh])
        n_total = grid_points.shape[0]
        
        # Calculate probabilities for each specification
        spec_probs = np.zeros((n_total, len(self.specifications)))
        
        for i, spec in enumerate(self.specifications):
            surrogate = self.surrogates[spec.name]
            
            # Get predictions with uncertainty
            mean, std = self._predict_with_uncertainty(surrogate, grid_points)
            
            # Calculate probability of meeting this specification
            for j in range(n_total):
                spec_probs[j, i] = spec.probability(mean[j], std[j])
        
        # Calculate joint probability (weighted)
        weights = np.array([spec.weight for spec in self.specifications])
        weights = weights / weights.sum()
        
        if correlation_matrix is None:
            # Assume independence: P(all) = product of individual
            joint_probs = np.prod(spec_probs, axis=1)
        else:
            # Use correlation for joint probability (more complex)
            # For now, use weighted geometric mean as approximation
            joint_probs = np.prod(np.power(spec_probs, weights), axis=1)
        
        # Reshape probabilities
        prob_surface = joint_probs.reshape(grid_shape)
        
        # Determine design space (probability >= confidence level)
        design_space_mask = prob_surface >= self.confidence_level
        
        # Determine MODR (more conservative - higher probability)
        modr_threshold = min(self.confidence_level + self.modr_margin, 1.0)
        modr_mask = prob_surface >= modr_threshold
        
        # Find edge of failure (boundary of design space)
        edge_of_failure = self._find_edge_of_failure(design_space_mask)
        
        # Calculate volume fractions
        volume_fraction = np.mean(design_space_mask)
        modr_volume_fraction = np.mean(modr_mask)
        
        return DesignSpaceResult(
            grid_points=grid_points,
            probabilities=prob_surface,
            design_space_mask=design_space_mask,
            modr_mask=modr_mask,
            edge_of_failure=edge_of_failure,
            parameter_names=param_names,
            specifications=self.specifications,
            confidence_level=self.confidence_level,
            volume_fraction=volume_fraction,
            modr_volume_fraction=modr_volume_fraction,
            grid_shape=grid_shape,
        )
    
    def _predict_with_uncertainty(
        self,
        surrogate: Any,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions with uncertainty from surrogate model."""
        # Handle sklearn GP
        if hasattr(surrogate, 'predict'):
            try:
                mean, std = surrogate.predict(X, return_std=True)
            except TypeError:
                # Some surrogates don't support return_std
                result = surrogate.predict(X)
                if isinstance(result, tuple):
                    mean, std = result
                else:
                    mean = result
                    std = np.zeros_like(mean)
        else:
            raise ValueError(f"Unknown surrogate type: {type(surrogate)}")
        
        return mean, std
    
    def _find_edge_of_failure(
        self,
        design_space_mask: np.ndarray,
    ) -> np.ndarray:
        """Find the edge of failure region (boundary of design space)."""
        # Dilate then XOR with original to get boundary
        dilated = binary_dilation(design_space_mask)
        edge = dilated ^ design_space_mask
        return edge
    
    def evaluate_point(
        self,
        parameters: Dict[str, float],
    ) -> DesignSpacePoint:
        """Evaluate a single point in the design space.
        
        Parameters
        ----------
        parameters : Dict[str, float]
            Parameter values to evaluate.
            
        Returns
        -------
        DesignSpacePoint
            Evaluation result with probabilities.
        """
        X = np.array([[parameters[name] for name in sorted(parameters.keys())]])
        
        individual_probs = {}
        for spec in self.specifications:
            surrogate = self.surrogates[spec.name]
            mean, std = self._predict_with_uncertainty(surrogate, X)
            prob = spec.probability(mean[0], std[0])
            individual_probs[spec.name] = prob
        
        # Joint probability
        joint_prob = np.prod(list(individual_probs.values()))
        
        return DesignSpacePoint(
            parameters=parameters,
            probability=joint_prob,
            individual_probs=individual_probs,
            is_in_design_space=joint_prob >= self.confidence_level,
            is_in_modr=joint_prob >= (self.confidence_level + self.modr_margin),
        )


@dataclass
class RobustnessResult:
    """Result of Monte Carlo robustness analysis.
    
    Attributes
    ----------
    nominal_point : Dict[str, float]
        Nominal parameter values tested.
    probability_of_success : float
        Probability of meeting specifications under variation.
    n_simulations : int
        Number of Monte Carlo simulations.
    variation_level : float
        Variation level used (as fraction).
    critical_parameters : List[str]
        Parameters most affecting probability.
    individual_probabilities : Dict[str, float]
        Probability for each specification.
    """
    nominal_point: Dict[str, float]
    probability_of_success: float
    n_simulations: int
    variation_level: float
    critical_parameters: List[str]
    individual_probabilities: Dict[str, float]


def monte_carlo_robustness(
    design_space: DesignSpace,
    nominal_point: Dict[str, float],
    parameter_ranges: Dict[str, Tuple[float, float]],
    variation_level: float = 0.05,
    n_simulations: int = 1000,
    variation_type: str = "uniform",
    random_state: Optional[int] = None,
) -> RobustnessResult:
    """Perform Monte Carlo robustness analysis.
    
    Simulates random variations around a nominal operating point to
    estimate the probability of meeting specifications under real-world
    variability.
    
    Parameters
    ----------
    design_space : DesignSpace
        Design space with specifications.
    nominal_point : Dict[str, float]
        Nominal parameter values.
    parameter_ranges : Dict[str, Tuple[float, float]]
        Valid ranges for each parameter (for clipping).
    variation_level : float, default=0.05
        Variation as fraction of range (5% = 0.05).
    n_simulations : int, default=1000
        Number of Monte Carlo iterations.
    variation_type : str, default="uniform"
        Type of variation: "uniform" or "normal".
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    RobustnessResult
        Robustness analysis results.
    """
    rng = np.random.default_rng(random_state)
    
    param_names = list(nominal_point.keys())
    n_params = len(param_names)
    
    # Calculate variation amounts for each parameter
    variations = {}
    for name in param_names:
        low, high = parameter_ranges[name]
        range_size = high - low
        variations[name] = variation_level * range_size
    
    # Generate random samples
    successes = np.zeros(n_simulations)
    spec_successes = {spec.name: np.zeros(n_simulations) 
                      for spec in design_space.specifications}
    
    for i in range(n_simulations):
        # Generate varied point
        varied = {}
        for name in param_names:
            low, high = parameter_ranges[name]
            if variation_type == "uniform":
                delta = rng.uniform(-variations[name], variations[name])
            else:  # normal
                delta = rng.normal(0, variations[name] / 3)  # 3σ = variation
            
            varied[name] = np.clip(nominal_point[name] + delta, low, high)
        
        # Evaluate at varied point
        result = design_space.evaluate_point(varied)
        
        successes[i] = 1 if result.is_in_design_space else 0
        for spec_name, prob in result.individual_probs.items():
            # Consider success if probability > 0.5 (more than likely)
            spec_successes[spec_name][i] = 1 if prob > 0.5 else 0
    
    # Calculate probabilities
    prob_success = np.mean(successes)
    individual_probs = {name: np.mean(s) for name, s in spec_successes.items()}
    
    # Identify critical parameters (using sensitivity analysis)
    critical = _identify_critical_parameters(
        design_space,
        nominal_point,
        parameter_ranges,
        variations,
        rng,
    )
    
    return RobustnessResult(
        nominal_point=nominal_point,
        probability_of_success=prob_success,
        n_simulations=n_simulations,
        variation_level=variation_level,
        critical_parameters=critical,
        individual_probabilities=individual_probs,
    )


def _identify_critical_parameters(
    design_space: DesignSpace,
    nominal_point: Dict[str, float],
    parameter_ranges: Dict[str, Tuple[float, float]],
    variations: Dict[str, float],
    rng: np.random.Generator,
    n_samples: int = 100,
) -> List[str]:
    """Identify parameters that most affect probability of success."""
    param_names = list(nominal_point.keys())
    sensitivities = {}
    
    for param in param_names:
        # Vary only this parameter
        probs = []
        for _ in range(n_samples):
            varied = nominal_point.copy()
            low, high = parameter_ranges[param]
            delta = rng.uniform(-variations[param], variations[param])
            varied[param] = np.clip(nominal_point[param] + delta, low, high)
            
            result = design_space.evaluate_point(varied)
            probs.append(result.probability)
        
        # Sensitivity = variance of probability when varying this parameter
        sensitivities[param] = np.var(probs)
    
    # Sort by sensitivity (highest first)
    sorted_params = sorted(sensitivities.keys(), 
                          key=lambda x: sensitivities[x], 
                          reverse=True)
    
    # Return top parameters that account for majority of sensitivity
    total_sensitivity = sum(sensitivities.values())
    if total_sensitivity == 0:
        return []
    
    critical = []
    cumulative = 0
    for param in sorted_params:
        critical.append(param)
        cumulative += sensitivities[param]
        if cumulative / total_sensitivity >= 0.8:  # 80% of total
            break
    
    return critical


@dataclass
class ControlStrategy:
    """Recommended control strategy for design space operation.
    
    Attributes
    ----------
    parameter : str
        Parameter name.
    nominal : float
        Recommended nominal value.
    control_limit_lower : float
        Lower control limit.
    control_limit_upper : float
        Upper control limit.
    criticality : str
        Criticality level: "critical", "key", "non-critical".
    control_type : str
        Type of control: "in-process", "specification", "normal".
    monitoring_frequency : str
        Recommended monitoring frequency.
    """
    parameter: str
    nominal: float
    control_limit_lower: float
    control_limit_upper: float
    criticality: str
    control_type: str
    monitoring_frequency: str


def recommend_control_strategy(
    design_space: DesignSpace,
    result: DesignSpaceResult,
    optimal_point: Dict[str, float],
    parameter_ranges: Dict[str, Tuple[float, float]],
) -> List[ControlStrategy]:
    """Generate control strategy recommendations.
    
    Parameters
    ----------
    design_space : DesignSpace
        Design space object.
    result : DesignSpaceResult
        Design space calculation result.
    optimal_point : Dict[str, float]
        Optimal operating point.
    parameter_ranges : Dict[str, Tuple[float, float]]
        Valid ranges for each parameter.
        
    Returns
    -------
    List[ControlStrategy]
        Recommended control strategies for each parameter.
    """
    strategies = []
    
    # Perform robustness analysis to find critical parameters
    robustness = monte_carlo_robustness(
        design_space,
        optimal_point,
        parameter_ranges,
        variation_level=0.03,
        n_simulations=500,
        random_state=42,
    )
    
    critical_params_list = robustness.critical_parameters
    critical_params_set = set(critical_params_list)
    
    for param in result.parameter_names:
        low, high = parameter_ranges[param]
        range_size = high - low
        nominal = optimal_point.get(param, (low + high) / 2)
        
        # Determine criticality
        most_critical = critical_params_list[0] if critical_params_list else None
        if param == most_critical:  # Most critical
            criticality = "critical"
            margin = 0.01 * range_size
            control_type = "in-process"
            monitoring = "continuous"
        elif param in critical_params_set:  # Other critical
            criticality = "key"
            margin = 0.02 * range_size
            control_type = "specification"
            monitoring = "per batch"
        else:
            criticality = "non-critical"
            margin = 0.05 * range_size
            control_type = "normal"
            monitoring = "periodic"
        
        # Find control limits within MODR if possible
        control_lower = max(low, nominal - margin)
        control_upper = min(high, nominal + margin)
        
        strategies.append(ControlStrategy(
            parameter=param,
            nominal=nominal,
            control_limit_lower=control_lower,
            control_limit_upper=control_upper,
            criticality=criticality,
            control_type=control_type,
            monitoring_frequency=monitoring,
        ))
    
    return strategies


def plot_design_space_2d(
    result: DesignSpaceResult,
    param_x: str,
    param_y: str,
    ax: Optional[Any] = None,
    show_modr: bool = True,
    show_edge: bool = True,
    cmap: str = "RdYlGn",
) -> Any:
    """Plot 2D design space.
    
    Parameters
    ----------
    result : DesignSpaceResult
        Design space calculation result.
    param_x : str
        Parameter for x-axis.
    param_y : str
        Parameter for y-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_modr : bool, default=True
        Whether to show MODR overlay.
    show_edge : bool, default=True
        Whether to show edge of failure.
    cmap : str, default="RdYlGn"
        Colormap for probability surface.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    import matplotlib.pyplot as plt
    
    if len(result.grid_shape) != 2:
        raise ValueError("This function only supports 2D design spaces")
    
    # Get axis indices
    ix = result.parameter_names.index(param_x)
    iy = result.parameter_names.index(param_y)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot probability surface
    x_vals = np.unique(result.grid_points[:, ix])
    y_vals = np.unique(result.grid_points[:, iy])
    
    im = ax.contourf(
        x_vals, y_vals,
        result.probabilities.T if ix < iy else result.probabilities,
        levels=20,
        cmap=cmap,
        vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax, label="Probability of Success")
    
    # Design space boundary
    ax.contour(
        x_vals, y_vals,
        result.design_space_mask.T if ix < iy else result.design_space_mask,
        levels=[0.5],
        colors='blue',
        linewidths=2,
        linestyles='solid',
    )
    
    # MODR boundary
    if show_modr and np.any(result.modr_mask):
        ax.contour(
            x_vals, y_vals,
            result.modr_mask.T if ix < iy else result.modr_mask,
            levels=[0.5],
            colors='green',
            linewidths=2,
            linestyles='dashed',
        )
    
    # Edge of failure
    if show_edge and np.any(result.edge_of_failure):
        edge_x = result.grid_points[result.edge_of_failure.flatten(), ix]
        edge_y = result.grid_points[result.edge_of_failure.flatten(), iy]
        ax.scatter(edge_x, edge_y, c='red', s=10, alpha=0.5, 
                  label='Edge of Failure')
    
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Design Space (p ≥ {result.confidence_level:.0%})")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='Design Space'),
    ]
    if show_modr and np.any(result.modr_mask):
        legend_elements.append(
            Line2D([0], [0], color='green', linewidth=2, linestyle='dashed',
                   label='MODR')
        )
    ax.legend(handles=legend_elements, loc='best')
    
    return ax


def export_design_space_report(
    result: DesignSpaceResult,
    robustness: Optional[RobustnessResult] = None,
    control_strategy: Optional[List[ControlStrategy]] = None,
    filepath: str = "design_space_report.html",
) -> str:
    """Export design space report as HTML.
    
    Parameters
    ----------
    result : DesignSpaceResult
        Design space calculation result.
    robustness : RobustnessResult, optional
        Robustness analysis result.
    control_strategy : List[ControlStrategy], optional
        Control strategy recommendations.
    filepath : str
        Output file path.
        
    Returns
    -------
    str
        Path to generated report.
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Design Space Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2951AA; }}
        h2 {{ color: #1a3a6e; border-bottom: 2px solid #8AAAE9; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #2951AA; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #2951AA; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .warning {{ color: orange; }}
    </style>
</head>
<body>
    <h1>QbD Design Space Report</h1>
    
    <h2>Summary</h2>
    <table>
        <tr>
            <td>Confidence Level</td>
            <td class="metric">{result.confidence_level:.1%}</td>
        </tr>
        <tr>
            <td>Design Space Volume</td>
            <td class="metric">{result.volume_fraction:.1%}</td>
        </tr>
        <tr>
            <td>MODR Volume</td>
            <td class="metric">{result.modr_volume_fraction:.1%}</td>
        </tr>
    </table>
    
    <h2>Specifications</h2>
    <table>
        <tr>
            <th>Response</th>
            <th>Type</th>
            <th>Limit</th>
            <th>Weight</th>
        </tr>
"""
    
    for spec in result.specifications:
        if spec.spec_type == SpecificationType.MINIMUM:
            limit_str = f"≥ {spec.value}"
        elif spec.spec_type == SpecificationType.MAXIMUM:
            limit_str = f"≤ {spec.value}"
        elif spec.spec_type == SpecificationType.TARGET:
            limit_str = f"= {spec.value} ± {spec.tolerance}"
        else:
            limit_str = f"{spec.lower} - {spec.upper}"
        
        html += f"""        <tr>
            <td>{spec.name}</td>
            <td>{spec.spec_type.value}</td>
            <td>{limit_str}</td>
            <td>{spec.weight:.1f}</td>
        </tr>
"""
    
    html += """    </table>
"""
    
    if robustness:
        status_class = "pass" if robustness.probability_of_success >= 0.95 else (
            "warning" if robustness.probability_of_success >= 0.9 else "fail"
        )
        html += f"""
    <h2>Robustness Analysis</h2>
    <table>
        <tr>
            <td>Variation Level</td>
            <td>{robustness.variation_level:.1%}</td>
        </tr>
        <tr>
            <td>Monte Carlo Simulations</td>
            <td>{robustness.n_simulations}</td>
        </tr>
        <tr>
            <td>Probability of Success</td>
            <td class="metric {status_class}">{robustness.probability_of_success:.1%}</td>
        </tr>
        <tr>
            <td>Critical Parameters</td>
            <td>{', '.join(robustness.critical_parameters) or 'None identified'}</td>
        </tr>
    </table>
"""
    
    if control_strategy:
        html += """
    <h2>Control Strategy Recommendations</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Nominal</th>
            <th>Control Limits</th>
            <th>Criticality</th>
            <th>Control Type</th>
            <th>Monitoring</th>
        </tr>
"""
        for cs in control_strategy:
            crit_class = (
                "fail" if cs.criticality == "critical" else
                "warning" if cs.criticality == "key" else ""
            )
            html += f"""        <tr>
            <td>{cs.parameter}</td>
            <td>{cs.nominal:.3g}</td>
            <td>{cs.control_limit_lower:.3g} - {cs.control_limit_upper:.3g}</td>
            <td class="{crit_class}">{cs.criticality}</td>
            <td>{cs.control_type}</td>
            <td>{cs.monitoring_frequency}</td>
        </tr>
"""
        html += """    </table>
"""
    
    html += """
    <h2>Regulatory Notes</h2>
    <p>This design space was calculated according to ICH Q8(R2) guidelines.
    The design space represents the multidimensional combination and 
    interaction of input variables that have been demonstrated to provide
    assurance of quality.</p>
    
    <p>Working within the design space is not considered a change and
    movement out of the design space is considered to be a change and
    would normally initiate a regulatory post-approval change process.</p>
    
    <footer>
        <p><small>Generated by OptiML QbD Module</small></p>
    </footer>
</body>
</html>
"""
    
    with open(filepath, 'w') as f:
        f.write(html)
    
    return filepath
