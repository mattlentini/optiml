"""
Statistical Analysis Module for Bayesian Optimization Results.

This module provides comprehensive statistical analysis tools for
analyzing optimization experiments, including:
- Summary statistics
- ANOVA and effects analysis
- Model diagnostics
- Confidence intervals
- Parameter importance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize

if TYPE_CHECKING:
    from optiml.space import Space


@dataclass
class SummaryStatistics:
    """Summary statistics for optimization results.

    Attributes
    ----------
    n_trials : int
        Number of completed trials.
    mean : float
        Mean objective value.
    std : float
        Standard deviation.
    min : float
        Minimum value.
    max : float
        Maximum value.
    median : float
        Median value.
    q25 : float
        25th percentile.
    q75 : float
        75th percentile.
    iqr : float
        Interquartile range.
    cv : float
        Coefficient of variation (std/mean * 100).
    skewness : float
        Skewness of the distribution.
    kurtosis : float
        Excess kurtosis.
    """

    n_trials: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    iqr: float
    cv: float
    skewness: float
    kurtosis: float

    def __str__(self) -> str:
        return (
            f"Summary Statistics (n={self.n_trials}):\n"
            f"  Mean:     {self.mean:.4g}\n"
            f"  Std Dev:  {self.std:.4g}\n"
            f"  Min:      {self.min:.4g}\n"
            f"  Max:      {self.max:.4g}\n"
            f"  Median:   {self.median:.4g}\n"
            f"  IQR:      {self.iqr:.4g}\n"
            f"  CV:       {self.cv:.2f}%\n"
            f"  Skewness: {self.skewness:.3f}\n"
            f"  Kurtosis: {self.kurtosis:.3f}"
        )


def calculate_summary_statistics(y: np.ndarray) -> SummaryStatistics:
    """Calculate comprehensive summary statistics.

    Parameters
    ----------
    y : np.ndarray
        Array of objective values.

    Returns
    -------
    SummaryStatistics
        Computed statistics.

    Examples
    --------
    >>> y = np.array([1.2, 2.3, 1.8, 2.1, 1.5])
    >>> stats = calculate_summary_statistics(y)
    >>> print(stats)
    """
    y = np.asarray(y)
    n = len(y)
    
    if n == 0:
        return SummaryStatistics(
            n_trials=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
            median=np.nan, q25=np.nan, q75=np.nan, iqr=np.nan,
            cv=np.nan, skewness=np.nan, kurtosis=np.nan
        )
    
    mean = np.mean(y)
    std = np.std(y, ddof=1) if n > 1 else 0.0
    q25, median, q75 = np.percentile(y, [25, 50, 75])
    
    return SummaryStatistics(
        n_trials=n,
        mean=mean,
        std=std,
        min=np.min(y),
        max=np.max(y),
        median=median,
        q25=q25,
        q75=q75,
        iqr=q75 - q25,
        cv=(std / abs(mean) * 100) if mean != 0 else np.inf,
        skewness=stats.skew(y) if n > 2 else 0.0,
        kurtosis=stats.kurtosis(y) if n > 3 else 0.0,
    )


@dataclass
class ParameterEffect:
    """Effect of a single parameter on the response.

    Attributes
    ----------
    name : str
        Parameter name.
    main_effect : float
        Main effect (average effect of changing from low to high).
    linear_coef : float
        Linear regression coefficient.
    correlation : float
        Pearson correlation with response.
    p_value : float
        P-value for significance of linear relationship.
    importance : float
        Relative importance (0-1 scale).
    effect_direction : str
        "positive", "negative", or "nonlinear"
    """

    name: str
    main_effect: float
    linear_coef: float
    correlation: float
    p_value: float
    importance: float
    effect_direction: str


@dataclass
class InteractionEffect:
    """Interaction effect between two parameters.

    Attributes
    ----------
    param1 : str
        First parameter name.
    param2 : str
        Second parameter name.
    interaction_strength : float
        Strength of interaction effect.
    p_value : float
        P-value for significance.
    """

    param1: str
    param2: str
    interaction_strength: float
    p_value: float


@dataclass
class EffectsAnalysis:
    """Complete effects analysis results.

    Attributes
    ----------
    parameter_effects : list[ParameterEffect]
        Main effects for each parameter.
    interaction_effects : list[InteractionEffect]
        Two-way interaction effects.
    total_r_squared : float
        R² of full linear model.
    adjusted_r_squared : float
        Adjusted R².
    model_p_value : float
        P-value for overall model significance.
    """

    parameter_effects: List[ParameterEffect]
    interaction_effects: List[InteractionEffect]
    total_r_squared: float
    adjusted_r_squared: float
    model_p_value: float

    def get_sorted_effects(self, by: str = "importance") -> List[ParameterEffect]:
        """Get parameter effects sorted by a metric.

        Parameters
        ----------
        by : str, default="importance"
            Metric to sort by: "importance", "main_effect", "correlation", "p_value"

        Returns
        -------
        list[ParameterEffect]
            Sorted effects list.
        """
        key_funcs = {
            "importance": lambda e: -e.importance,
            "main_effect": lambda e: -abs(e.main_effect),
            "correlation": lambda e: -abs(e.correlation),
            "p_value": lambda e: e.p_value,
        }
        return sorted(self.parameter_effects, key=key_funcs.get(by, key_funcs["importance"]))


def analyze_effects(
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str] | None = None,
) -> EffectsAnalysis:
    """Perform comprehensive effects analysis.

    Parameters
    ----------
    X : np.ndarray
        Parameter values, shape (n_samples, n_params).
    y : np.ndarray
        Response values, shape (n_samples,).
    param_names : list[str], optional
        Names for each parameter.

    Returns
    -------
    EffectsAnalysis
        Complete effects analysis results.

    Examples
    --------
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([1.2, 2.3, 3.1, 4.2])
    >>> effects = analyze_effects(X, y, param_names=["pH", "Temperature"])
    >>> for eff in effects.get_sorted_effects():
    ...     print(f"{eff.name}: importance={eff.importance:.3f}")
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_params = X.shape
    
    if param_names is None:
        param_names = [f"X{i+1}" for i in range(n_params)]
    
    # Normalize X to [0, 1] for comparable effects
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid division by zero
    X_norm = (X - X_min) / X_range
    
    # Center y
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    parameter_effects = []
    total_importance = 0.0
    
    # Analyze each parameter
    for i in range(n_params):
        x_i = X_norm[:, i]
        
        # Linear correlation
        if np.std(x_i) > 0:
            corr, p_value = stats.pearsonr(x_i, y)
        else:
            corr, p_value = 0.0, 1.0
        
        # Linear coefficient (simple regression)
        if np.std(x_i) > 0:
            slope = np.cov(x_i, y)[0, 1] / np.var(x_i)
        else:
            slope = 0.0
        
        # Main effect: difference between high and low
        low_mask = x_i < 0.5
        high_mask = x_i >= 0.5
        
        if np.sum(low_mask) > 0 and np.sum(high_mask) > 0:
            main_effect = np.mean(y[high_mask]) - np.mean(y[low_mask])
        else:
            main_effect = 0.0
        
        # Determine effect direction
        if abs(corr) < 0.3:
            direction = "weak/nonlinear"
        elif corr > 0:
            direction = "positive"
        else:
            direction = "negative"
        
        # Importance (based on correlation magnitude)
        importance = abs(corr) ** 2  # R² for this parameter
        total_importance += importance
        
        parameter_effects.append(ParameterEffect(
            name=param_names[i],
            main_effect=main_effect,
            linear_coef=slope,
            correlation=corr,
            p_value=p_value,
            importance=importance,  # Will be normalized later
            effect_direction=direction,
        ))
    
    # Normalize importance to sum to 1
    if total_importance > 0:
        for eff in parameter_effects:
            eff.importance = eff.importance / total_importance
    
    # Analyze interactions
    interaction_effects = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            # Create interaction term
            interaction = X_norm[:, i] * X_norm[:, j]
            
            if np.std(interaction) > 0:
                corr_int, p_int = stats.pearsonr(interaction, y_centered)
            else:
                corr_int, p_int = 0.0, 1.0
            
            interaction_effects.append(InteractionEffect(
                param1=param_names[i],
                param2=param_names[j],
                interaction_strength=abs(corr_int),
                p_value=p_int,
            ))
    
    # Sort interactions by strength
    interaction_effects.sort(key=lambda x: -x.interaction_strength)
    
    # Overall model fit (multiple linear regression)
    if n_samples > n_params + 1:
        X_with_intercept = np.column_stack([np.ones(n_samples), X_norm])
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            y_pred = X_with_intercept @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - ss_res / ss_tot
            else:
                r_squared = 0.0
            
            # Adjusted R²
            adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_params - 1)
            
            # F-test for model significance
            if ss_res > 0 and n_samples > n_params + 1:
                f_stat = (ss_tot - ss_res) / n_params / (ss_res / (n_samples - n_params - 1))
                model_p_value = 1 - stats.f.cdf(f_stat, n_params, n_samples - n_params - 1)
            else:
                model_p_value = 1.0
        except np.linalg.LinAlgError:
            r_squared = 0.0
            adj_r_squared = 0.0
            model_p_value = 1.0
    else:
        r_squared = 0.0
        adj_r_squared = 0.0
        model_p_value = 1.0
    
    return EffectsAnalysis(
        parameter_effects=parameter_effects,
        interaction_effects=interaction_effects,
        total_r_squared=r_squared,
        adjusted_r_squared=adj_r_squared,
        model_p_value=model_p_value,
    )


@dataclass
class ANOVAResult:
    """ANOVA (Analysis of Variance) results.

    Attributes
    ----------
    source : str
        Source of variation (parameter name or "Residual").
    df : int
        Degrees of freedom.
    sum_squares : float
        Sum of squares.
    mean_square : float
        Mean square (SS / df).
    f_value : float
        F-statistic.
    p_value : float
        P-value.
    pct_contribution : float
        Percentage contribution to total variation.
    """

    source: str
    df: int
    sum_squares: float
    mean_square: float
    f_value: float
    p_value: float
    pct_contribution: float


@dataclass
class ANOVATable:
    """Complete ANOVA table.

    Attributes
    ----------
    rows : list[ANOVAResult]
        ANOVA results for each source.
    total_df : int
        Total degrees of freedom.
    total_ss : float
        Total sum of squares.
    r_squared : float
        R-squared (model fit).
    adj_r_squared : float
        Adjusted R-squared.
    """

    rows: List[ANOVAResult]
    total_df: int
    total_ss: float
    r_squared: float
    adj_r_squared: float

    def __str__(self) -> str:
        lines = [
            "ANOVA Table",
            "=" * 80,
            f"{'Source':<20} {'DF':>6} {'SS':>12} {'MS':>12} {'F':>10} {'p-value':>10} {'%Contrib':>10}",
            "-" * 80,
        ]
        
        for row in self.rows:
            lines.append(
                f"{row.source:<20} {row.df:>6} {row.sum_squares:>12.4f} "
                f"{row.mean_square:>12.4f} {row.f_value:>10.3f} {row.p_value:>10.4f} "
                f"{row.pct_contribution:>9.2f}%"
            )
        
        lines.extend([
            "-" * 80,
            f"{'Total':<20} {self.total_df:>6} {self.total_ss:>12.4f}",
            "",
            f"R² = {self.r_squared:.4f}, Adjusted R² = {self.adj_r_squared:.4f}",
        ])
        
        return "\n".join(lines)

    def significant_factors(self, alpha: float = 0.05) -> List[str]:
        """Get list of statistically significant factors.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        list[str]
            Names of significant factors.
        """
        return [row.source for row in self.rows if row.source != "Residual" and row.p_value < alpha]


def perform_anova(
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str] | None = None,
    include_interactions: bool = False,
) -> ANOVATable:
    """Perform Analysis of Variance on experimental data.

    Parameters
    ----------
    X : np.ndarray
        Parameter values, shape (n_samples, n_params).
    y : np.ndarray
        Response values, shape (n_samples,).
    param_names : list[str], optional
        Names for each parameter.
    include_interactions : bool, default=False
        Whether to include 2-way interactions in the model.

    Returns
    -------
    ANOVATable
        Complete ANOVA table.

    Examples
    --------
    >>> X = np.random.rand(30, 3)
    >>> y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(30)*0.5
    >>> anova = perform_anova(X, y, ["pH", "Temp", "Flow"])
    >>> print(anova)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_params = X.shape
    
    if param_names is None:
        param_names = [f"X{i+1}" for i in range(n_params)]
    
    # Normalize X
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    X_norm = (X - X_min) / X_range
    
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    
    anova_rows = []
    
    # Build design matrix
    design_columns = [np.ones(n_samples)]  # Intercept
    column_names = ["Intercept"]
    
    # Main effects
    for i in range(n_params):
        design_columns.append(X_norm[:, i])
        column_names.append(param_names[i])
    
    # Interactions
    if include_interactions:
        for i in range(n_params):
            for j in range(i + 1, n_params):
                design_columns.append(X_norm[:, i] * X_norm[:, j])
                column_names.append(f"{param_names[i]}×{param_names[j]}")
    
    design_matrix = np.column_stack(design_columns)
    
    # Full model fit
    try:
        beta_full, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
        y_pred_full = design_matrix @ beta_full
        ss_residual = np.sum((y - y_pred_full) ** 2)
        ss_model = ss_total - ss_residual
    except np.linalg.LinAlgError:
        ss_residual = ss_total
        ss_model = 0.0
    
    # Sequential SS for each term
    current_ss = ss_total
    
    for idx, name in enumerate(column_names[1:], 1):  # Skip intercept
        # Reduced model without this term
        reduced_cols = [j for j in range(len(column_names)) if j != idx]
        reduced_matrix = design_matrix[:, reduced_cols]
        
        try:
            beta_reduced, _, _, _ = np.linalg.lstsq(reduced_matrix, y, rcond=None)
            y_pred_reduced = reduced_matrix @ beta_reduced
            ss_reduced_residual = np.sum((y - y_pred_reduced) ** 2)
            
            # SS for this term
            ss_term = ss_reduced_residual - ss_residual
            ss_term = max(0, ss_term)  # Ensure non-negative
            
        except np.linalg.LinAlgError:
            ss_term = 0.0
        
        df = 1  # 1 df per continuous term
        ms = ss_term / df if df > 0 else 0.0
        
        # F-test
        df_residual = n_samples - len(column_names)
        ms_residual = ss_residual / df_residual if df_residual > 0 else 1.0
        
        if ms_residual > 0:
            f_value = ms / ms_residual
            p_value = 1 - stats.f.cdf(f_value, df, df_residual)
        else:
            f_value = 0.0
            p_value = 1.0
        
        pct_contribution = (ss_term / ss_total * 100) if ss_total > 0 else 0.0
        
        anova_rows.append(ANOVAResult(
            source=name,
            df=df,
            sum_squares=ss_term,
            mean_square=ms,
            f_value=f_value,
            p_value=p_value,
            pct_contribution=pct_contribution,
        ))
    
    # Residual row
    df_residual = max(1, n_samples - len(column_names))
    ms_residual = ss_residual / df_residual
    pct_residual = (ss_residual / ss_total * 100) if ss_total > 0 else 0.0
    
    anova_rows.append(ANOVAResult(
        source="Residual",
        df=df_residual,
        sum_squares=ss_residual,
        mean_square=ms_residual,
        f_value=0.0,
        p_value=1.0,
        pct_contribution=pct_residual,
    ))
    
    # Calculate R²
    r_squared = 1 - ss_residual / ss_total if ss_total > 0 else 0.0
    n_predictors = len(column_names) - 1
    adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_predictors - 1)
    
    return ANOVATable(
        rows=anova_rows,
        total_df=n_samples - 1,
        total_ss=ss_total,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
    )


@dataclass
class Residuals:
    """Residual analysis results.

    Attributes
    ----------
    raw : np.ndarray
        Raw residuals (y - y_pred).
    standardized : np.ndarray
        Standardized residuals.
    studentized : np.ndarray
        Studentized residuals.
    leverage : np.ndarray
        Leverage values (hat matrix diagonal).
    cooks_distance : np.ndarray
        Cook's distance for each point.
    """

    raw: np.ndarray
    standardized: np.ndarray
    studentized: np.ndarray
    leverage: np.ndarray
    cooks_distance: np.ndarray

    def outliers(self, threshold: float = 2.0) -> np.ndarray:
        """Identify potential outliers based on studentized residuals.

        Parameters
        ----------
        threshold : float, default=2.0
            Threshold for studentized residuals.

        Returns
        -------
        np.ndarray
            Boolean mask of outliers.
        """
        return np.abs(self.studentized) > threshold

    def influential_points(self, threshold: float | None = None) -> np.ndarray:
        """Identify influential points based on Cook's distance.

        Parameters
        ----------
        threshold : float, optional
            Threshold for Cook's distance. Default is 4/n.

        Returns
        -------
        np.ndarray
            Boolean mask of influential points.
        """
        if threshold is None:
            threshold = 4.0 / len(self.cooks_distance)
        return self.cooks_distance > threshold


def calculate_residuals(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> Residuals:
    """Calculate comprehensive residual diagnostics.

    Parameters
    ----------
    X : np.ndarray
        Parameter values, shape (n_samples, n_params).
    y : np.ndarray
        Observed response values.
    y_pred : np.ndarray, optional
        Predicted values. If None, fits a linear model.

    Returns
    -------
    Residuals
        Comprehensive residual analysis.

    Examples
    --------
    >>> X = np.random.rand(30, 3)
    >>> y = 2*X[:, 0] + np.random.randn(30)*0.5
    >>> residuals = calculate_residuals(X, y)
    >>> print(f"Outliers: {np.sum(residuals.outliers())}")
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_params = X.shape
    
    # Fit linear model if predictions not provided
    if y_pred is None:
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            y_pred = X_with_intercept @ beta
        except np.linalg.LinAlgError:
            y_pred = np.full(n_samples, np.mean(y))
    else:
        X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Raw residuals
    raw_residuals = y - y_pred
    
    # Hat matrix (leverage)
    try:
        H = X_with_intercept @ np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
        leverage = np.diag(H)
    except np.linalg.LinAlgError:
        leverage = np.full(n_samples, 1.0 / n_samples)
    
    # MSE
    df_residual = max(1, n_samples - n_params - 1)
    mse = np.sum(raw_residuals ** 2) / df_residual
    
    # Standardized residuals
    std_residuals = raw_residuals / np.sqrt(mse) if mse > 0 else raw_residuals
    
    # Studentized residuals
    denominator = np.sqrt(mse * (1 - leverage))
    denominator[denominator == 0] = 1e-10
    studentized = raw_residuals / denominator
    
    # Cook's distance
    p = n_params + 1  # Including intercept
    cooks_d = (studentized ** 2 / p) * (leverage / (1 - leverage))
    cooks_d = np.nan_to_num(cooks_d, nan=0.0, posinf=0.0, neginf=0.0)
    
    return Residuals(
        raw=raw_residuals,
        standardized=std_residuals,
        studentized=studentized,
        leverage=leverage,
        cooks_distance=cooks_d,
    )


@dataclass
class ConfidenceInterval:
    """Confidence interval for a parameter or prediction.

    Attributes
    ----------
    point_estimate : float
        Point estimate.
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    confidence_level : float
        Confidence level (e.g., 0.95).
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float

    def contains(self, value: float) -> bool:
        """Check if a value is within the confidence interval."""
        return self.lower <= value <= self.upper

    def width(self) -> float:
        """Return the width of the confidence interval."""
        return self.upper - self.lower

    def __str__(self) -> str:
        return f"{self.point_estimate:.4g} [{self.lower:.4g}, {self.upper:.4g}] ({self.confidence_level*100:.0f}% CI)"


def confidence_interval_mean(
    y: np.ndarray,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Calculate confidence interval for the mean.

    Parameters
    ----------
    y : np.ndarray
        Sample values.
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    ConfidenceInterval
        Confidence interval for the population mean.
    """
    y = np.asarray(y)
    n = len(y)
    mean = np.mean(y)
    
    if n < 2:
        return ConfidenceInterval(mean, mean, mean, confidence)
    
    se = stats.sem(y)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * se
    
    return ConfidenceInterval(
        point_estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
    )


def prediction_interval(
    y: np.ndarray,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Calculate prediction interval for a new observation.

    Parameters
    ----------
    y : np.ndarray
        Sample values.
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    ConfidenceInterval
        Prediction interval for a new observation.
    """
    y = np.asarray(y)
    n = len(y)
    mean = np.mean(y)
    
    if n < 2:
        return ConfidenceInterval(mean, mean, mean, confidence)
    
    std = np.std(y, ddof=1)
    se_pred = std * np.sqrt(1 + 1/n)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * se_pred
    
    return ConfidenceInterval(
        point_estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
    )


@dataclass
class NormalityTest:
    """Results of normality test.

    Attributes
    ----------
    test_name : str
        Name of the test.
    statistic : float
        Test statistic.
    p_value : float
        P-value.
    is_normal : bool
        Whether data appears normally distributed (at alpha=0.05).
    """

    test_name: str
    statistic: float
    p_value: float
    is_normal: bool


def check_normality(y: np.ndarray, alpha: float = 0.05) -> NormalityTest:
    """Check if data is normally distributed.

    Uses Shapiro-Wilk test for n < 5000, Anderson-Darling otherwise.

    Parameters
    ----------
    y : np.ndarray
        Sample values.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    NormalityTest
        Test results.

    Examples
    --------
    >>> y = np.random.randn(50)
    >>> result = check_normality(y)
    >>> print(f"Normal: {result.is_normal}, p={result.p_value:.4f}")
    """
    y = np.asarray(y)
    n = len(y)
    
    if n < 3:
        return NormalityTest("N/A", 0.0, 1.0, True)
    
    if n < 5000:
        stat, p_value = stats.shapiro(y)
        test_name = "Shapiro-Wilk"
    else:
        result = stats.anderson(y, dist='norm')
        stat = result.statistic
        # Convert to approximate p-value
        critical_values = result.critical_values
        sig_levels = np.array([15, 10, 5, 2.5, 1]) / 100
        if stat < critical_values[0]:
            p_value = 0.15
        elif stat > critical_values[-1]:
            p_value = 0.01
        else:
            # Interpolate
            idx = np.searchsorted(critical_values, stat)
            p_value = sig_levels[min(idx, len(sig_levels)-1)]
        test_name = "Anderson-Darling"
    
    return NormalityTest(
        test_name=test_name,
        statistic=stat,
        p_value=p_value,
        is_normal=p_value >= alpha,
    )


def calculate_partial_dependence(
    X: np.ndarray,
    y: np.ndarray,
    param_index: int,
    n_grid: int = 50,
    percentile_range: Tuple[float, float] = (5, 95),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate partial dependence for a single parameter.

    Shows the marginal effect of a parameter on the response,
    averaging over the other parameters.

    Parameters
    ----------
    X : np.ndarray
        Parameter values, shape (n_samples, n_params).
    y : np.ndarray
        Response values.
    param_index : int
        Index of the parameter to analyze.
    n_grid : int, default=50
        Number of grid points.
    percentile_range : tuple, default=(5, 95)
        Percentile range for the grid.

    Returns
    -------
    grid : np.ndarray
        Parameter values for the grid.
    pd_values : np.ndarray
        Mean partial dependence values.
    pd_std : np.ndarray
        Standard deviation of partial dependence.

    Examples
    --------
    >>> X = np.random.rand(100, 3)
    >>> y = 2*X[:, 0] + np.random.randn(100)*0.5
    >>> grid, pd_mean, pd_std = calculate_partial_dependence(X, y, param_index=0)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Get grid range
    param_values = X[:, param_index]
    low, high = np.percentile(param_values, percentile_range)
    grid = np.linspace(low, high, n_grid)
    
    # Fit a simple model
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    except np.linalg.LinAlgError:
        return grid, np.full(n_grid, np.mean(y)), np.zeros(n_grid)
    
    pd_values = []
    pd_std_values = []
    
    for val in grid:
        # Replace param_index column with constant value
        X_modified = X.copy()
        X_modified[:, param_index] = val
        X_mod_intercept = np.column_stack([np.ones(len(X)), X_modified])
        
        # Predict
        y_pred = X_mod_intercept @ beta
        pd_values.append(np.mean(y_pred))
        pd_std_values.append(np.std(y_pred))
    
    return grid, np.array(pd_values), np.array(pd_std_values)


def calculate_all_partial_dependence(
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str] | None = None,
    n_grid: int = 50,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculate partial dependence for all parameters.

    Parameters
    ----------
    X : np.ndarray
        Parameter values.
    y : np.ndarray
        Response values.
    param_names : list[str], optional
        Parameter names.
    n_grid : int, default=50
        Number of grid points.

    Returns
    -------
    dict
        Dictionary mapping parameter name to (grid, mean, std).
    """
    n_params = X.shape[1]
    if param_names is None:
        param_names = [f"X{i+1}" for i in range(n_params)]
    
    results = {}
    for i, name in enumerate(param_names):
        grid, pd_mean, pd_std = calculate_partial_dependence(X, y, i, n_grid)
        results[name] = (grid, pd_mean, pd_std)
    
    return results
