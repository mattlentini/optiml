"""
Visualization Module for Bayesian Optimization.

This module provides comprehensive visualization functions for
analyzing optimization results, including:
- Convergence plots
- Parameter importance
- Partial dependence plots
- Contour plots
- Pareto front visualization
- Acquisition function plots
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from optiml.space import Space


def _import_matplotlib():
    """Import matplotlib with proper backend handling."""
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


def plot_convergence(
    y_values: np.ndarray,
    minimize: bool = True,
    ax: Optional["Axes"] = None,
    show_trials: bool = True,
    show_best: bool = True,
    title: str = "Optimization Convergence",
    xlabel: str = "Trial",
    ylabel: str = "Objective Value",
    trial_color: str = "#8AAAE9",
    best_color: str = "#2951AA",
    figsize: Tuple[int, int] = (10, 6),
) -> "Figure":
    """Plot optimization convergence over trials.

    Parameters
    ----------
    y_values : np.ndarray
        Objective values for each trial.
    minimize : bool, default=True
        Whether optimization was minimizing.
    ax : Axes, optional
        Matplotlib axes to plot on.
    show_trials : bool, default=True
        Whether to show individual trial points.
    show_best : bool, default=True
        Whether to show the running best line.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    trial_color : str
        Color for trial points.
    best_color : str
        Color for best line.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.

    Examples
    --------
    >>> y = np.random.randn(50).cumsum()
    >>> fig = plot_convergence(y, minimize=True)
    >>> fig.savefig("convergence.png")
    """
    _, plt = _import_matplotlib()
    
    y_values = np.asarray(y_values)
    n_trials = len(y_values)
    trials = np.arange(1, n_trials + 1)
    
    # Calculate running best
    if minimize:
        running_best = np.minimum.accumulate(y_values)
    else:
        running_best = np.maximum.accumulate(y_values)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    if show_trials:
        ax.scatter(trials, y_values, c=trial_color, alpha=0.6, s=50, 
                  label="Trial Results", zorder=2)
    
    if show_best:
        ax.plot(trials, running_best, c=best_color, linewidth=2, 
               label="Best So Far", zorder=3)
        ax.fill_between(trials, running_best, y_values, 
                       alpha=0.1, color=best_color)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_trials)
    
    plt.tight_layout()
    return fig


def plot_parameter_importance(
    param_names: List[str],
    importance_values: np.ndarray | List[float],
    ax: Optional["Axes"] = None,
    title: str = "Parameter Importance",
    color: str = "#2951AA",
    figsize: Tuple[int, int] = (10, 6),
    sort: bool = True,
) -> "Figure":
    """Plot parameter importance as a horizontal bar chart.

    Parameters
    ----------
    param_names : list[str]
        Names of parameters.
    importance_values : array-like
        Importance values for each parameter.
    ax : Axes, optional
        Matplotlib axes to plot on.
    title : str
        Plot title.
    color : str
        Bar color.
    figsize : tuple
        Figure size.
    sort : bool, default=True
        Whether to sort by importance.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    
    importance_values = np.asarray(importance_values)
    
    if sort:
        sorted_idx = np.argsort(importance_values)
        param_names = [param_names[i] for i in sorted_idx]
        importance_values = importance_values[sorted_idx]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    y_pos = np.arange(len(param_names))
    
    bars = ax.barh(y_pos, importance_values, color=color, alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, importance_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(importance_values) * 1.15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_partial_dependence(
    grid: np.ndarray,
    pd_values: np.ndarray,
    pd_std: np.ndarray | None = None,
    param_name: str = "Parameter",
    ax: Optional["Axes"] = None,
    title: str | None = None,
    color: str = "#2951AA",
    fill_color: str = "#8AAAE9",
    figsize: Tuple[int, int] = (8, 5),
    show_rug: bool = False,
    rug_data: np.ndarray | None = None,
) -> "Figure":
    """Plot partial dependence for a single parameter.

    Parameters
    ----------
    grid : np.ndarray
        Parameter values.
    pd_values : np.ndarray
        Partial dependence values.
    pd_std : np.ndarray, optional
        Standard deviation for confidence band.
    param_name : str
        Parameter name for axis label.
    ax : Axes, optional
        Matplotlib axes.
    title : str, optional
        Plot title.
    color : str
        Line color.
    fill_color : str
        Confidence band color.
    figsize : tuple
        Figure size.
    show_rug : bool, default=False
        Whether to show rug plot of data points.
    rug_data : np.ndarray, optional
        Data points for rug plot.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ax.plot(grid, pd_values, color=color, linewidth=2, label="Partial Dependence")
    
    if pd_std is not None:
        ax.fill_between(grid, pd_values - pd_std, pd_values + pd_std,
                       color=fill_color, alpha=0.3, label="±1 Std")
    
    if show_rug and rug_data is not None:
        ax.plot(rug_data, np.full_like(rug_data, ax.get_ylim()[0]), 
               '|', color='gray', alpha=0.5, markersize=10)
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Response", fontsize=12)
    
    if title is None:
        title = f"Partial Dependence: {param_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_partial_dependence_grid(
    pd_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ncols: int = 3,
    figsize_per_plot: Tuple[int, int] = (4, 3),
    suptitle: str = "Partial Dependence Plots",
) -> "Figure":
    """Plot partial dependence for multiple parameters in a grid.

    Parameters
    ----------
    pd_results : dict
        Dictionary mapping parameter name to (grid, mean, std) tuples.
    ncols : int, default=3
        Number of columns in the grid.
    figsize_per_plot : tuple
        Figure size per subplot.
    suptitle : str
        Main title.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    
    n_params = len(pd_results)
    nrows = (n_params + ncols - 1) // ncols
    
    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes)
    
    for idx, (param_name, (grid, pd_mean, pd_std)) in enumerate(pd_results.items()):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        ax.plot(grid, pd_mean, color='#2951AA', linewidth=2)
        ax.fill_between(grid, pd_mean - pd_std, pd_mean + pd_std,
                       color='#8AAAE9', alpha=0.3)
        
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("Response", fontsize=10)
        ax.set_title(param_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_params, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)
    
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_contour(
    X: np.ndarray,
    y: np.ndarray,
    param1_idx: int,
    param2_idx: int,
    param1_name: str = "Parameter 1",
    param2_name: str = "Parameter 2",
    resolution: int = 50,
    ax: Optional["Axes"] = None,
    title: str | None = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (8, 6),
    show_points: bool = True,
    point_color: str = "white",
    show_best: bool = True,
    minimize: bool = True,
) -> "Figure":
    """Plot 2D contour of response surface.

    Parameters
    ----------
    X : np.ndarray
        Parameter values, shape (n_samples, n_params).
    y : np.ndarray
        Response values.
    param1_idx : int
        Index of first parameter.
    param2_idx : int
        Index of second parameter.
    param1_name : str
        Name of first parameter.
    param2_name : str
        Name of second parameter.
    resolution : int, default=50
        Grid resolution.
    ax : Axes, optional
        Matplotlib axes.
    title : str, optional
        Plot title.
    cmap : str, default="viridis"
        Colormap name.
    figsize : tuple
        Figure size.
    show_points : bool, default=True
        Whether to show data points.
    point_color : str
        Color for data points.
    show_best : bool, default=True
        Whether to highlight the best point.
    minimize : bool, default=True
        Whether optimization is minimizing.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    from scipy.interpolate import griddata
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    x1 = X[:, param1_idx]
    x2 = X[:, param2_idx]
    
    # Create grid
    x1_grid = np.linspace(x1.min(), x1.max(), resolution)
    x2_grid = np.linspace(x2.min(), x2.max(), resolution)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    
    # Interpolate
    Z = griddata((x1, x2), y, (X1_grid, X2_grid), method='cubic')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Contour plot
    contour = ax.contourf(X1_grid, X2_grid, Z, levels=20, cmap=cmap, alpha=0.8)
    ax.contour(X1_grid, X2_grid, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    plt.colorbar(contour, ax=ax, label="Response")
    
    if show_points:
        ax.scatter(x1, x2, c=point_color, edgecolors='black', s=50, 
                  alpha=0.7, zorder=5, label="Trials")
    
    if show_best:
        best_idx = np.argmin(y) if minimize else np.argmax(y)
        ax.scatter(x1[best_idx], x2[best_idx], c='red', s=200, 
                  marker='*', edgecolors='white', linewidths=2,
                  zorder=10, label="Best")
    
    ax.set_xlabel(param1_name, fontsize=12)
    ax.set_ylabel(param2_name, fontsize=12)
    
    if title is None:
        title = f"Response Surface: {param1_name} vs {param2_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig


def plot_slice(
    predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    space: "Space",
    fixed_point: np.ndarray,
    param_idx: int,
    param_name: str | None = None,
    n_points: int = 100,
    ax: Optional["Axes"] = None,
    title: str | None = None,
    show_uncertainty: bool = True,
    figsize: Tuple[int, int] = (8, 5),
) -> "Figure":
    """Plot a 1D slice through the surrogate model.

    Parameters
    ----------
    predict_fn : callable
        Function that takes X and returns (mean, std).
    space : Space
        Parameter space.
    fixed_point : np.ndarray
        Point to slice through.
    param_idx : int
        Index of parameter to vary.
    param_name : str, optional
        Name of parameter.
    n_points : int, default=100
        Number of points in slice.
    ax : Axes, optional
        Matplotlib axes.
    title : str, optional
        Plot title.
    show_uncertainty : bool, default=True
        Whether to show uncertainty bands.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get parameter bounds
    dim = space.dimensions[param_idx]
    low, high = dim.low, dim.high
    
    # Create slice
    param_values = np.linspace(low, high, n_points)
    X_slice = np.tile(fixed_point, (n_points, 1))
    X_slice[:, param_idx] = param_values
    
    # Predict
    mean, std = predict_fn(X_slice)
    
    ax.plot(param_values, mean, color='#2951AA', linewidth=2, label='Mean Prediction')
    
    if show_uncertainty:
        for n_std, alpha in [(1, 0.3), (2, 0.15)]:
            ax.fill_between(param_values, mean - n_std * std, mean + n_std * std,
                           color='#8AAAE9', alpha=alpha, label=f'±{n_std} Std')
    
    ax.axvline(fixed_point[param_idx], color='red', linestyle='--', alpha=0.5,
              label='Current Value')
    
    if param_name is None:
        param_name = f"Parameter {param_idx + 1}"
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Predicted Response", fontsize=12)
    
    if title is None:
        title = f"Model Slice: {param_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pareto_front(
    objectives: np.ndarray,
    pareto_mask: np.ndarray | None = None,
    obj1_name: str = "Objective 1",
    obj2_name: str = "Objective 2",
    ax: Optional["Axes"] = None,
    title: str = "Pareto Front",
    dominated_color: str = "#CCCCCC",
    pareto_color: str = "#2951AA",
    figsize: Tuple[int, int] = (8, 6),
    show_front_line: bool = True,
) -> "Figure":
    """Plot 2D Pareto front for multi-objective optimization.

    Parameters
    ----------
    objectives : np.ndarray
        Objective values, shape (n_samples, 2).
    pareto_mask : np.ndarray, optional
        Boolean mask indicating Pareto-optimal points.
    obj1_name : str
        Name of first objective.
    obj2_name : str
        Name of second objective.
    ax : Axes, optional
        Matplotlib axes.
    title : str
        Plot title.
    dominated_color : str
        Color for dominated points.
    pareto_color : str
        Color for Pareto-optimal points.
    figsize : tuple
        Figure size.
    show_front_line : bool, default=True
        Whether to connect Pareto front points.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    
    objectives = np.asarray(objectives)
    
    if pareto_mask is None:
        pareto_mask = compute_pareto_mask(objectives)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot dominated points
    dominated = ~pareto_mask
    ax.scatter(objectives[dominated, 0], objectives[dominated, 1],
              c=dominated_color, s=50, alpha=0.5, label="Dominated")
    
    # Plot Pareto points
    pareto_points = objectives[pareto_mask]
    ax.scatter(pareto_points[:, 0], pareto_points[:, 1],
              c=pareto_color, s=100, edgecolors='white', linewidths=2,
              label="Pareto Optimal", zorder=5)
    
    if show_front_line and len(pareto_points) > 1:
        # Sort by first objective for line plot
        sorted_idx = np.argsort(pareto_points[:, 0])
        sorted_pareto = pareto_points[sorted_idx]
        ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1],
               c=pareto_color, linewidth=2, alpha=0.7, linestyle='--')
    
    ax.set_xlabel(obj1_name, fontsize=12)
    ax.set_ylabel(obj2_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_pareto_mask(objectives: np.ndarray, minimize: bool = True) -> np.ndarray:
    """Compute Pareto optimality mask.

    Parameters
    ----------
    objectives : np.ndarray
        Objective values, shape (n_samples, n_objectives).
    minimize : bool, default=True
        Whether objectives are to be minimized.

    Returns
    -------
    np.ndarray
        Boolean mask of Pareto-optimal points.
    """
    n_samples = len(objectives)
    is_pareto = np.ones(n_samples, dtype=bool)
    
    for i in range(n_samples):
        if is_pareto[i]:
            for j in range(n_samples):
                if i != j and is_pareto[j]:
                    if minimize:
                        # j dominates i if j <= i for all objectives and j < i for at least one
                        dominates = np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
                    else:
                        dominates = np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i])
                    
                    if dominates:
                        is_pareto[i] = False
                        break
    
    return is_pareto


def plot_acquisition(
    acquisition_values: np.ndarray,
    param_values: np.ndarray,
    param_name: str = "Parameter",
    ax: Optional["Axes"] = None,
    title: str = "Acquisition Function",
    color: str = "#E9967A",
    figsize: Tuple[int, int] = (8, 5),
    show_max: bool = True,
) -> "Figure":
    """Plot 1D acquisition function.

    Parameters
    ----------
    acquisition_values : np.ndarray
        Acquisition function values.
    param_values : np.ndarray
        Parameter values.
    param_name : str
        Parameter name.
    ax : Axes, optional
        Matplotlib axes.
    title : str
        Plot title.
    color : str
        Line color.
    figsize : tuple
        Figure size.
    show_max : bool, default=True
        Whether to highlight maximum.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    ax.fill_between(param_values, 0, acquisition_values, color=color, alpha=0.3)
    ax.plot(param_values, acquisition_values, color=color, linewidth=2)
    
    if show_max:
        max_idx = np.argmax(acquisition_values)
        ax.axvline(param_values[max_idx], color='red', linestyle='--', 
                  alpha=0.7, label=f'Max at {param_values[max_idx]:.3f}')
        ax.scatter([param_values[max_idx]], [acquisition_values[max_idx]],
                  c='red', s=100, zorder=10)
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Acquisition Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_optimization_summary(
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str],
    minimize: bool = True,
    figsize: Tuple[int, int] = (16, 10),
) -> "Figure":
    """Create a comprehensive optimization summary visualization.

    Includes:
    - Convergence plot
    - Parameter importance
    - Best parameters as bar chart
    - Partial dependence for top parameters

    Parameters
    ----------
    X : np.ndarray
        Parameter values.
    y : np.ndarray
        Objective values.
    param_names : list[str]
        Parameter names.
    minimize : bool, default=True
        Whether optimization was minimizing.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    from .statistics import analyze_effects, calculate_all_partial_dependence
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Convergence plot (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    n_trials = len(y)
    trials = np.arange(1, n_trials + 1)
    running_best = np.minimum.accumulate(y) if minimize else np.maximum.accumulate(y)
    
    ax1.scatter(trials, y, c='#8AAAE9', alpha=0.6, s=50, label="Trials")
    ax1.plot(trials, running_best, c='#2951AA', linewidth=2, label="Best So Far")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Objective Value")
    ax1.set_title("Optimization Convergence", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter importance (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    effects = analyze_effects(X, y, param_names)
    importance = [eff.importance for eff in effects.parameter_effects]
    sorted_idx = np.argsort(importance)
    
    ax2.barh(range(len(param_names)), [importance[i] for i in sorted_idx],
            color='#2951AA', alpha=0.8)
    ax2.set_yticks(range(len(param_names)))
    ax2.set_yticklabels([param_names[i] for i in sorted_idx])
    ax2.set_xlabel("Importance")
    ax2.set_title("Parameter Importance", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Best parameters (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    best_idx = np.argmin(y) if minimize else np.argmax(y)
    best_params = X[best_idx]
    
    # Normalize for visualization
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1
    best_normalized = (best_params - X_min) / X_range
    
    bars = ax3.barh(range(len(param_names)), best_normalized, color='#2951AA', alpha=0.8)
    ax3.set_yticks(range(len(param_names)))
    ax3.set_yticklabels(param_names)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Normalized Value")
    ax3.set_title(f"Best Parameters (y={y[best_idx]:.4g})", fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add actual values as text
    for bar, val in zip(bars, best_params):
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3g}', va='center', fontsize=9)
    
    # 4. Partial dependence for top 2 parameters (bottom middle and right)
    pd_results = calculate_all_partial_dependence(X, y, param_names)
    
    # Get top 2 parameters by importance
    top_params = [param_names[i] for i in sorted_idx[-2:]][::-1]
    
    for plot_idx, param in enumerate(top_params[:2]):
        ax = fig.add_subplot(gs[1, 1 + plot_idx])
        grid, pd_mean, pd_std = pd_results[param]
        
        ax.plot(grid, pd_mean, color='#2951AA', linewidth=2)
        ax.fill_between(grid, pd_mean - pd_std, pd_mean + pd_std,
                       color='#8AAAE9', alpha=0.3)
        ax.set_xlabel(param)
        ax.set_ylabel("Response")
        ax.set_title(f"Partial Dependence: {param}", fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Optimization Summary", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_residuals_diagnostic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (12, 10),
) -> "Figure":
    """Create residual diagnostic plots.

    Includes:
    - Actual vs Predicted
    - Residuals vs Predicted
    - Residual histogram
    - Q-Q plot

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    figsize : tuple
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    _, plt = _import_matplotlib()
    from scipy import stats
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, y_true, c='#2951AA', alpha=0.6, s=50)
    lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='y = x')
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Actual vs Predicted", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, c='#2951AA', alpha=0.6, s=50)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predicted", fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual histogram
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, color='#2951AA', alpha=0.7, edgecolor='white')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Residual Distribution", fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    ax4 = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title("Normal Q-Q Plot", fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
