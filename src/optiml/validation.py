"""Validation module for analytical method assessment.

This module provides tools for validating analytical methods according to
ICH Q2 guidelines and other industry standards.

Features:
- ICH Q2 validation parameters (accuracy, precision, linearity, etc.)
- Bioassay quality metrics (Z-factor, EC50, signal-to-noise)
- Process capability analysis (Cp, Cpk, Pp, Ppk)
- Specification limit tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


class ValidationLevel(Enum):
    """Validation level according to ICH Q2."""
    FULL = "full"
    PARTIAL = "partial"
    VERIFICATION = "verification"


@dataclass
class AccuracyResult:
    """Accuracy validation result.
    
    Attributes
    ----------
    recovery_mean : float
        Mean percent recovery.
    recovery_std : float
        Standard deviation of recovery.
    recovery_values : np.ndarray
        Individual recovery values.
    bias : float
        Systematic bias (mean recovery - 100).
    acceptable : bool
        Whether accuracy meets criteria.
    criteria : tuple
        (min, max) acceptable recovery range.
    """
    recovery_mean: float
    recovery_std: float
    recovery_values: np.ndarray
    bias: float
    acceptable: bool
    criteria: Tuple[float, float] = (97.0, 103.0)


@dataclass
class PrecisionResult:
    """Precision validation result.
    
    Attributes
    ----------
    repeatability_rsd : float
        Repeatability RSD (same day, same analyst).
    intermediate_precision_rsd : float
        Intermediate precision RSD (different days/analysts).
    reproducibility_rsd : float
        Reproducibility RSD (different labs).
    pooled_rsd : float
        Overall pooled RSD.
    acceptable : bool
        Whether precision meets criteria.
    criteria : float
        Maximum acceptable RSD.
    """
    repeatability_rsd: float
    intermediate_precision_rsd: Optional[float] = None
    reproducibility_rsd: Optional[float] = None
    pooled_rsd: Optional[float] = None
    acceptable: bool = True
    criteria: float = 2.0


@dataclass
class LinearityResult:
    """Linearity validation result.
    
    Attributes
    ----------
    slope : float
        Regression slope.
    intercept : float
        Y-intercept.
    r_squared : float
        Coefficient of determination (R²).
    standard_error : float
        Standard error of regression.
    residuals : np.ndarray
        Regression residuals.
    concentrations : np.ndarray
        Concentration values (x).
    responses : np.ndarray
        Response values (y).
    acceptable : bool
        Whether linearity meets criteria.
    criteria : float
        Minimum acceptable R².
    """
    slope: float
    intercept: float
    r_squared: float
    standard_error: float
    residuals: np.ndarray
    concentrations: np.ndarray
    responses: np.ndarray
    acceptable: bool = True
    criteria: float = 0.999


@dataclass
class LODResult:
    """Limit of Detection (LOD) result.
    
    Attributes
    ----------
    lod_value : float
        Calculated LOD.
    method : str
        Method used: "signal_noise", "blank_std", "regression".
    signal_to_noise : float, optional
        S/N ratio at LOD (if S/N method used).
    """
    lod_value: float
    method: str
    signal_to_noise: Optional[float] = None


@dataclass
class LOQResult:
    """Limit of Quantitation (LOQ) result.
    
    Attributes
    ----------
    loq_value : float
        Calculated LOQ.
    method : str
        Method used: "signal_noise", "blank_std", "regression".
    signal_to_noise : float, optional
        S/N ratio at LOQ (if S/N method used).
    accuracy_at_loq : float, optional
        Accuracy at LOQ concentration.
    precision_at_loq : float, optional
        RSD at LOQ concentration.
    """
    loq_value: float
    method: str
    signal_to_noise: Optional[float] = None
    accuracy_at_loq: Optional[float] = None
    precision_at_loq: Optional[float] = None


@dataclass
class RobustnessResult:
    """Robustness study result.
    
    Attributes
    ----------
    critical_parameters : List[str]
        Parameters with significant effects.
    effects : Dict[str, float]
        Effect magnitude for each parameter.
    acceptable : bool
        Whether method is acceptably robust.
    """
    critical_parameters: List[str]
    effects: Dict[str, float]
    acceptable: bool = True


@dataclass
class SpecificityResult:
    """Specificity validation result.
    
    Attributes
    ----------
    is_specific : bool
        Whether method is specific for the analyte.
    peak_purity : float, optional
        Peak purity value (0-1).
    resolution : float, optional
        Resolution from nearest peak.
    interference_detected : bool
        Whether interference was detected.
    """
    is_specific: bool
    peak_purity: Optional[float] = None
    resolution: Optional[float] = None
    interference_detected: bool = False


@dataclass
class RangeResult:
    """Range validation result.
    
    Attributes
    ----------
    lower_limit : float
        Lower validated range.
    upper_limit : float
        Upper validated range.
    range_units : str
        Units for range.
    linear_in_range : bool
        Whether linearity holds throughout range.
    accuracy_in_range : bool
        Whether accuracy is acceptable throughout range.
    """
    lower_limit: float
    upper_limit: float
    range_units: str = ""
    linear_in_range: bool = True
    accuracy_in_range: bool = True


@dataclass 
class ICH_Q2_Report:
    """Complete ICH Q2 validation report.
    
    Attributes
    ----------
    method_name : str
        Name of the analytical method.
    analyte : str
        Name of the analyte.
    accuracy : AccuracyResult, optional
        Accuracy validation result.
    precision : PrecisionResult, optional
        Precision validation result.
    linearity : LinearityResult, optional
        Linearity validation result.
    lod : LODResult, optional
        Limit of detection result.
    loq : LOQResult, optional
        Limit of quantitation result.
    specificity : SpecificityResult, optional
        Specificity result.
    range : RangeResult, optional
        Validated range.
    robustness : RobustnessResult, optional
        Robustness study result.
    overall_pass : bool
        Whether all criteria are met.
    """
    method_name: str
    analyte: str
    accuracy: Optional[AccuracyResult] = None
    precision: Optional[PrecisionResult] = None
    linearity: Optional[LinearityResult] = None
    lod: Optional[LODResult] = None
    loq: Optional[LOQResult] = None
    specificity: Optional[SpecificityResult] = None
    range: Optional[RangeResult] = None
    robustness: Optional[RobustnessResult] = None
    overall_pass: bool = True


def calculate_accuracy(
    measured: np.ndarray,
    expected: np.ndarray,
    criteria: Tuple[float, float] = (97.0, 103.0),
) -> AccuracyResult:
    """Calculate accuracy (percent recovery).
    
    Parameters
    ----------
    measured : np.ndarray
        Measured values.
    expected : np.ndarray
        True/expected values.
    criteria : tuple, default=(97.0, 103.0)
        Acceptable recovery range.
        
    Returns
    -------
    AccuracyResult
        Accuracy validation result.
    """
    measured = np.asarray(measured)
    expected = np.asarray(expected)
    
    recovery = (measured / expected) * 100
    recovery_mean = np.mean(recovery)
    recovery_std = np.std(recovery, ddof=1)
    bias = recovery_mean - 100.0
    
    acceptable = criteria[0] <= recovery_mean <= criteria[1]
    
    return AccuracyResult(
        recovery_mean=recovery_mean,
        recovery_std=recovery_std,
        recovery_values=recovery,
        bias=bias,
        acceptable=acceptable,
        criteria=criteria,
    )


def calculate_precision(
    data: np.ndarray,
    groups: Optional[np.ndarray] = None,
    criteria: float = 2.0,
) -> PrecisionResult:
    """Calculate precision (RSD).
    
    Parameters
    ----------
    data : np.ndarray
        Measured values. If 2D, rows are replicates, columns are groups.
    groups : np.ndarray, optional
        Group labels for intermediate precision calculation.
    criteria : float, default=2.0
        Maximum acceptable RSD (%).
        
    Returns
    -------
    PrecisionResult
        Precision validation result.
    """
    data = np.asarray(data)
    
    if data.ndim == 1:
        # Single group - repeatability only
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        repeatability_rsd = (std / mean) * 100 if mean != 0 else 0
        
        return PrecisionResult(
            repeatability_rsd=repeatability_rsd,
            acceptable=repeatability_rsd <= criteria,
            criteria=criteria,
        )
    
    # Multiple groups
    group_means = np.mean(data, axis=0)
    group_stds = np.std(data, axis=0, ddof=1)
    
    # Within-group RSD (repeatability)
    repeatability_rsds = (group_stds / group_means) * 100
    repeatability_rsd = np.mean(repeatability_rsds[~np.isnan(repeatability_rsds)])
    
    # Between-group (intermediate precision)
    overall_mean = np.mean(data)
    overall_std = np.std(data, ddof=1)
    intermediate_precision_rsd = (np.std(group_means, ddof=1) / overall_mean) * 100
    
    # Pooled
    pooled_rsd = (overall_std / overall_mean) * 100
    
    acceptable = max(repeatability_rsd, pooled_rsd) <= criteria
    
    return PrecisionResult(
        repeatability_rsd=repeatability_rsd,
        intermediate_precision_rsd=intermediate_precision_rsd,
        pooled_rsd=pooled_rsd,
        acceptable=acceptable,
        criteria=criteria,
    )


def calculate_linearity(
    concentrations: np.ndarray,
    responses: np.ndarray,
    criteria: float = 0.999,
) -> LinearityResult:
    """Calculate linearity (regression).
    
    Parameters
    ----------
    concentrations : np.ndarray
        Concentration values (x).
    responses : np.ndarray
        Response values (y).
    criteria : float, default=0.999
        Minimum acceptable R².
        
    Returns
    -------
    LinearityResult
        Linearity validation result.
    """
    concentrations = np.asarray(concentrations).flatten()
    responses = np.asarray(responses).flatten()
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(concentrations, responses)
    r_squared = r_value ** 2
    
    # Residuals
    predicted = slope * concentrations + intercept
    residuals = responses - predicted
    
    # Standard error of regression
    n = len(concentrations)
    se_regression = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    return LinearityResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        standard_error=se_regression,
        residuals=residuals,
        concentrations=concentrations,
        responses=responses,
        acceptable=r_squared >= criteria,
        criteria=criteria,
    )


def calculate_lod(
    blank_responses: Optional[np.ndarray] = None,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
    residual_std: Optional[float] = None,
    signal_to_noise: float = 3.0,
    method: str = "auto",
) -> LODResult:
    """Calculate Limit of Detection (LOD).
    
    Parameters
    ----------
    blank_responses : np.ndarray, optional
        Response values from blank samples.
    slope : float, optional
        Calibration curve slope.
    intercept : float, optional
        Calibration curve intercept.
    residual_std : float, optional
        Standard deviation of residuals from calibration.
    signal_to_noise : float, default=3.0
        S/N ratio for LOD calculation.
    method : str, default="auto"
        Method: "blank_std", "regression", or "auto".
        
    Returns
    -------
    LODResult
        LOD result.
    """
    if method == "auto":
        if blank_responses is not None:
            method = "blank_std"
        elif slope is not None and residual_std is not None:
            method = "regression"
        else:
            raise ValueError("Insufficient data for LOD calculation")
    
    if method == "blank_std":
        if blank_responses is None:
            raise ValueError("blank_responses required for blank_std method")
        blank_std = np.std(blank_responses, ddof=1)
        blank_mean = np.mean(blank_responses)
        
        if slope is not None:
            lod_value = (blank_mean + signal_to_noise * blank_std - intercept) / slope if slope != 0 else 0
        else:
            lod_value = blank_mean + signal_to_noise * blank_std
            
    elif method == "regression":
        if slope is None or residual_std is None:
            raise ValueError("slope and residual_std required for regression method")
        lod_value = (3.3 * residual_std) / slope if slope != 0 else 0
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return LODResult(
        lod_value=abs(lod_value),
        method=method,
        signal_to_noise=signal_to_noise,
    )


def calculate_loq(
    blank_responses: Optional[np.ndarray] = None,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
    residual_std: Optional[float] = None,
    signal_to_noise: float = 10.0,
    method: str = "auto",
) -> LOQResult:
    """Calculate Limit of Quantitation (LOQ).
    
    Parameters
    ----------
    blank_responses : np.ndarray, optional
        Response values from blank samples.
    slope : float, optional
        Calibration curve slope.
    intercept : float, optional
        Calibration curve intercept.
    residual_std : float, optional
        Standard deviation of residuals from calibration.
    signal_to_noise : float, default=10.0
        S/N ratio for LOQ calculation.
    method : str, default="auto"
        Method: "blank_std", "regression", or "auto".
        
    Returns
    -------
    LOQResult
        LOQ result.
    """
    if method == "auto":
        if blank_responses is not None:
            method = "blank_std"
        elif slope is not None and residual_std is not None:
            method = "regression"
        else:
            raise ValueError("Insufficient data for LOQ calculation")
    
    if method == "blank_std":
        if blank_responses is None:
            raise ValueError("blank_responses required for blank_std method")
        blank_std = np.std(blank_responses, ddof=1)
        blank_mean = np.mean(blank_responses)
        
        if slope is not None:
            loq_value = (blank_mean + signal_to_noise * blank_std - intercept) / slope if slope != 0 else 0
        else:
            loq_value = blank_mean + signal_to_noise * blank_std
            
    elif method == "regression":
        if slope is None or residual_std is None:
            raise ValueError("slope and residual_std required for regression method")
        loq_value = (10.0 * residual_std) / slope if slope != 0 else 0
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return LOQResult(
        loq_value=abs(loq_value),
        method=method,
        signal_to_noise=signal_to_noise,
    )


# Bioassay quality metrics

@dataclass
class ZFactorResult:
    """Z-factor calculation result.
    
    Attributes
    ----------
    z_factor : float
        Z-factor value (-inf to 1).
    z_prime : float
        Z'-factor (without sample variation).
    positive_mean : float
        Mean of positive controls.
    negative_mean : float
        Mean of negative controls.
    positive_std : float
        Std of positive controls.
    negative_std : float
        Std of negative controls.
    quality : str
        Assay quality: "excellent", "good", "marginal", "poor".
    """
    z_factor: float
    z_prime: float
    positive_mean: float
    negative_mean: float
    positive_std: float
    negative_std: float
    quality: str


def calculate_z_factor(
    positive_controls: np.ndarray,
    negative_controls: np.ndarray,
    sample_data: Optional[np.ndarray] = None,
) -> ZFactorResult:
    """Calculate Z-factor for assay quality assessment.
    
    Parameters
    ----------
    positive_controls : np.ndarray
        Positive control values.
    negative_controls : np.ndarray
        Negative control values.
    sample_data : np.ndarray, optional
        Sample data for Z-factor (vs Z'-factor) calculation.
        
    Returns
    -------
    ZFactorResult
        Z-factor result with quality assessment.
    """
    pos = np.asarray(positive_controls)
    neg = np.asarray(negative_controls)
    
    pos_mean = np.mean(pos)
    neg_mean = np.mean(neg)
    pos_std = np.std(pos, ddof=1)
    neg_std = np.std(neg, ddof=1)
    
    # Z'-factor (controls only)
    dynamic_range = abs(pos_mean - neg_mean)
    if dynamic_range > 0:
        z_prime = 1 - (3 * (pos_std + neg_std) / dynamic_range)
    else:
        z_prime = -np.inf
    
    # Z-factor (with sample variation if provided)
    if sample_data is not None:
        sample = np.asarray(sample_data)
        sample_std = np.std(sample, ddof=1)
        if dynamic_range > 0:
            z_factor = 1 - (3 * (sample_std + neg_std) / dynamic_range)
        else:
            z_factor = -np.inf
    else:
        z_factor = z_prime
    
    # Quality assessment
    if z_factor >= 0.5:
        quality = "excellent"
    elif z_factor >= 0.0:
        quality = "good"
    elif z_factor >= -0.5:
        quality = "marginal"
    else:
        quality = "poor"
    
    return ZFactorResult(
        z_factor=z_factor,
        z_prime=z_prime,
        positive_mean=pos_mean,
        negative_mean=neg_mean,
        positive_std=pos_std,
        negative_std=neg_std,
        quality=quality,
    )


@dataclass
class SignalMetrics:
    """Signal quality metrics.
    
    Attributes
    ----------
    signal_to_background : float
        Signal-to-background ratio.
    signal_to_noise : float
        Signal-to-noise ratio.
    dynamic_range : float
        Dynamic range (max/min signal).
    window : float
        Assay window (positive - negative).
    """
    signal_to_background: float
    signal_to_noise: float
    dynamic_range: float
    window: float


def calculate_signal_metrics(
    signal: np.ndarray,
    background: np.ndarray,
) -> SignalMetrics:
    """Calculate signal quality metrics.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal values (positive samples).
    background : np.ndarray
        Background values (blanks/negative).
        
    Returns
    -------
    SignalMetrics
        Signal quality metrics.
    """
    signal = np.asarray(signal)
    background = np.asarray(background)
    
    signal_mean = np.mean(signal)
    background_mean = np.mean(background)
    background_std = np.std(background, ddof=1)
    
    # S/B
    s_b = signal_mean / background_mean if background_mean != 0 else np.inf
    
    # S/N
    s_n = (signal_mean - background_mean) / background_std if background_std != 0 else np.inf
    
    # Dynamic range
    dynamic_range = np.max(signal) / np.min(background) if np.min(background) != 0 else np.inf
    
    # Window
    window = signal_mean - background_mean
    
    return SignalMetrics(
        signal_to_background=s_b,
        signal_to_noise=s_n,
        dynamic_range=dynamic_range,
        window=window,
    )


@dataclass
class EC50Result:
    """EC50/IC50 fitting result.
    
    Attributes
    ----------
    ec50 : float
        EC50 or IC50 value.
    hill_slope : float
        Hill coefficient (slope).
    top : float
        Maximum response.
    bottom : float
        Minimum response.
    r_squared : float
        Goodness of fit.
    concentrations : np.ndarray
        Concentration values.
    responses : np.ndarray
        Response values.
    fitted_curve : np.ndarray
        Fitted response values.
    """
    ec50: float
    hill_slope: float
    top: float
    bottom: float
    r_squared: float
    concentrations: np.ndarray
    responses: np.ndarray
    fitted_curve: np.ndarray


def _four_param_logistic(x, bottom, top, ec50, hill):
    """Four-parameter logistic function."""
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hill)


def calculate_ec50(
    concentrations: np.ndarray,
    responses: np.ndarray,
    is_inhibition: bool = False,
) -> EC50Result:
    """Fit 4-parameter logistic and calculate EC50/IC50.
    
    Parameters
    ----------
    concentrations : np.ndarray
        Concentration values (must be positive).
    responses : np.ndarray
        Response values.
    is_inhibition : bool, default=False
        If True, fit for IC50 (decreasing curve).
        
    Returns
    -------
    EC50Result
        Fitted curve and EC50 value.
    """
    concentrations = np.asarray(concentrations)
    responses = np.asarray(responses)
    
    # Initial parameter estimates
    bottom_init = np.min(responses)
    top_init = np.max(responses)
    ec50_init = np.median(concentrations)
    hill_init = 1.0 if not is_inhibition else -1.0
    
    # Bounds
    bounds = (
        [0, 0, 1e-12, -10],  # Lower bounds
        [np.inf, np.inf, np.inf, 10]  # Upper bounds
    )
    
    try:
        popt, _ = curve_fit(
            _four_param_logistic,
            concentrations,
            responses,
            p0=[bottom_init, top_init, ec50_init, hill_init],
            bounds=bounds,
            maxfev=5000,
        )
        bottom, top, ec50, hill = popt
    except Exception:
        # Fall back to simple estimate
        bottom = bottom_init
        top = top_init
        ec50 = ec50_init
        hill = hill_init
    
    # Calculate fitted curve
    fitted = _four_param_logistic(concentrations, bottom, top, ec50, hill)
    
    # R-squared
    ss_res = np.sum((responses - fitted) ** 2)
    ss_tot = np.sum((responses - np.mean(responses)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return EC50Result(
        ec50=ec50,
        hill_slope=hill,
        top=top,
        bottom=bottom,
        r_squared=r_squared,
        concentrations=concentrations,
        responses=responses,
        fitted_curve=fitted,
    )


# Process capability

@dataclass
class CapabilityResult:
    """Process capability analysis result.
    
    Attributes
    ----------
    cp : float
        Potential capability index (Cp).
    cpk : float
        Actual capability index (Cpk).
    pp : float
        Process performance (Pp).
    ppk : float
        Process performance index (Ppk).
    sigma_level : float
        Sigma level.
    ppm_defective : float
        Parts per million defective.
    within_spec_pct : float
        Percentage within specification.
    """
    cp: float
    cpk: float
    pp: float
    ppk: float
    sigma_level: float
    ppm_defective: float
    within_spec_pct: float


def calculate_capability(
    data: np.ndarray,
    lsl: float,
    usl: float,
    subgroup_size: int = 1,
) -> CapabilityResult:
    """Calculate process capability indices.
    
    Parameters
    ----------
    data : np.ndarray
        Process data.
    lsl : float
        Lower specification limit.
    usl : float
        Upper specification limit.
    subgroup_size : int, default=1
        Size of rational subgroups for Cp/Cpk.
        
    Returns
    -------
    CapabilityResult
        Capability analysis result.
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    
    # Overall standard deviation (for Pp/Ppk)
    sigma_overall = np.std(data, ddof=1)
    
    # Within-subgroup standard deviation (for Cp/Cpk)
    if subgroup_size > 1 and n >= subgroup_size * 2:
        # Estimate from subgroup ranges
        n_subgroups = n // subgroup_size
        subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
        ranges = np.ptp(subgroups, axis=1)
        d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d2_value = d2.get(subgroup_size, 2.326)
        sigma_within = np.mean(ranges) / d2_value
    else:
        sigma_within = sigma_overall
    
    # Capability indices
    spec_range = usl - lsl
    
    # Cp and Cpk (short-term)
    cp = spec_range / (6 * sigma_within) if sigma_within > 0 else np.inf
    cpu = (usl - mean) / (3 * sigma_within) if sigma_within > 0 else np.inf
    cpl = (mean - lsl) / (3 * sigma_within) if sigma_within > 0 else np.inf
    cpk = min(cpu, cpl)
    
    # Pp and Ppk (long-term)
    pp = spec_range / (6 * sigma_overall) if sigma_overall > 0 else np.inf
    ppu = (usl - mean) / (3 * sigma_overall) if sigma_overall > 0 else np.inf
    ppl = (mean - lsl) / (3 * sigma_overall) if sigma_overall > 0 else np.inf
    ppk = min(ppu, ppl)
    
    # Sigma level
    sigma_level = 3 * cpk
    
    # PPM defective
    z_upper = (usl - mean) / sigma_overall if sigma_overall > 0 else np.inf
    z_lower = (mean - lsl) / sigma_overall if sigma_overall > 0 else np.inf
    
    ppm_upper = stats.norm.sf(z_upper) * 1e6
    ppm_lower = stats.norm.cdf(-z_lower) * 1e6
    ppm_defective = ppm_upper + ppm_lower
    
    # Within spec percentage
    within_spec_pct = (1 - ppm_defective / 1e6) * 100
    
    return CapabilityResult(
        cp=cp,
        cpk=cpk,
        pp=pp,
        ppk=ppk,
        sigma_level=sigma_level,
        ppm_defective=ppm_defective,
        within_spec_pct=within_spec_pct,
    )


def validate_method(
    method_name: str,
    analyte: str,
    accuracy_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    precision_data: Optional[np.ndarray] = None,
    linearity_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    blank_data: Optional[np.ndarray] = None,
    level: ValidationLevel = ValidationLevel.FULL,
) -> ICH_Q2_Report:
    """Perform complete ICH Q2 method validation.
    
    Parameters
    ----------
    method_name : str
        Name of the analytical method.
    analyte : str
        Name of the analyte.
    accuracy_data : tuple, optional
        (measured, expected) arrays for accuracy.
    precision_data : np.ndarray, optional
        Data for precision calculation.
    linearity_data : tuple, optional
        (concentrations, responses) for linearity.
    blank_data : np.ndarray, optional
        Blank responses for LOD/LOQ.
    level : ValidationLevel, default=FULL
        Validation level to perform.
        
    Returns
    -------
    ICH_Q2_Report
        Complete validation report.
    """
    report = ICH_Q2_Report(method_name=method_name, analyte=analyte)
    overall_pass = True
    
    # Accuracy
    if accuracy_data is not None:
        measured, expected = accuracy_data
        report.accuracy = calculate_accuracy(measured, expected)
        if not report.accuracy.acceptable:
            overall_pass = False
    
    # Precision
    if precision_data is not None:
        report.precision = calculate_precision(precision_data)
        if not report.precision.acceptable:
            overall_pass = False
    
    # Linearity
    if linearity_data is not None:
        conc, resp = linearity_data
        report.linearity = calculate_linearity(conc, resp)
        if not report.linearity.acceptable:
            overall_pass = False
        
        # LOD and LOQ from linearity
        if blank_data is not None:
            report.lod = calculate_lod(
                blank_responses=blank_data,
                slope=report.linearity.slope,
                intercept=report.linearity.intercept,
            )
            report.loq = calculate_loq(
                blank_responses=blank_data,
                slope=report.linearity.slope,
                intercept=report.linearity.intercept,
            )
        else:
            report.lod = calculate_lod(
                slope=report.linearity.slope,
                residual_std=report.linearity.standard_error,
                method="regression",
            )
            report.loq = calculate_loq(
                slope=report.linearity.slope,
                residual_std=report.linearity.standard_error,
                method="regression",
            )
    
    report.overall_pass = overall_pass
    
    return report
