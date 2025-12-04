"""Enhanced visualization module with interactive Plotly charts.

This module provides publication-quality, interactive visualizations for
Bayesian optimization results using Plotly.

Features:
- 3D response surfaces with rotation
- Parallel coordinates for high-dimensional data
- Interactive contour plots with design space overlay
- Pareto front visualization
- Convergence animations
- Acquisition function landscapes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

# Try to import plotly, provide fallback
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None
    px = None


def _check_plotly():
    """Check if plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )


@dataclass
class PlotlyTheme:
    """Theme configuration for Plotly plots.
    
    Attributes
    ----------
    colorscale : str
        Colorscale name for heatmaps and surfaces.
    paper_bgcolor : str
        Background color of the paper.
    plot_bgcolor : str
        Background color of the plot area.
    font_color : str
        Color for text and labels.
    gridcolor : str
        Color for grid lines.
    title_font_size : int
        Font size for titles.
    axis_font_size : int
        Font size for axis labels.
    """
    colorscale: str = "Viridis"
    paper_bgcolor: str = "white"
    plot_bgcolor: str = "white"
    font_color: str = "black"
    gridcolor: str = "#E5E5E5"
    title_font_size: int = 16
    axis_font_size: int = 12


# Predefined themes
LIGHT_THEME = PlotlyTheme()
DARK_THEME = PlotlyTheme(
    colorscale="Plasma",
    paper_bgcolor="#1E1E1E",
    plot_bgcolor="#1E1E1E",
    font_color="white",
    gridcolor="#3E3E3E"
)


def surface_3d(
    X: np.ndarray,
    y: np.ndarray,
    x_name: str = "X1",
    y_name: str = "X2",
    z_name: str = "Response",
    resolution: int = 50,
    surrogate=None,
    title: str = "Response Surface",
    theme: PlotlyTheme = LIGHT_THEME,
    show_points: bool = True,
    opacity: float = 0.8,
) -> "go.Figure":
    """Create an interactive 3D response surface plot.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix with exactly 2 features.
    y : np.ndarray of shape (n_samples,)
        Response values.
    x_name : str, default="X1"
        Name for x-axis.
    y_name : str, default="X2"
        Name for y-axis.
    z_name : str, default="Response"
        Name for z-axis.
    resolution : int, default=50
        Grid resolution for the surface.
    surrogate : object, optional
        Fitted surrogate model for prediction. If None, interpolates data.
    title : str, default="Response Surface"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme configuration.
    show_points : bool, default=True
        Whether to show observed data points.
    opacity : float, default=0.8
        Surface opacity (0-1).
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    _check_plotly()
    
    X = np.atleast_2d(X)
    if X.shape[1] != 2:
        raise ValueError("surface_3d requires exactly 2 features")
    
    # Create grid
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Predict on grid
    if surrogate is not None:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        try:
            z_pred, _ = surrogate.predict(grid_points)
        except:
            z_pred = surrogate.predict(grid_points)
        zz = z_pred.reshape(xx.shape)
    else:
        # Use scipy interpolation
        from scipy.interpolate import griddata
        zz = griddata(X, y, (xx, yy), method='cubic')
    
    # Create surface
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale=theme.colorscale,
        opacity=opacity,
        name="Surface",
        colorbar=dict(title=z_name),
    ))
    
    # Add observed points
    if show_points:
        fig.add_trace(go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=y,
            mode="markers",
            marker=dict(
                size=5,
                color=y,
                colorscale=theme.colorscale,
                line=dict(width=1, color="black"),
            ),
            name="Observations",
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        scene=dict(
            xaxis_title=x_name,
            yaxis_title=y_name,
            zaxis_title=z_name,
            xaxis=dict(
                backgroundcolor=theme.plot_bgcolor,
                gridcolor=theme.gridcolor,
            ),
            yaxis=dict(
                backgroundcolor=theme.plot_bgcolor,
                gridcolor=theme.gridcolor,
            ),
            zaxis=dict(
                backgroundcolor=theme.plot_bgcolor,
                gridcolor=theme.gridcolor,
            ),
        ),
        paper_bgcolor=theme.paper_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def contour_plot(
    X: np.ndarray,
    y: np.ndarray,
    x_name: str = "X1",
    y_name: str = "X2",
    z_name: str = "Response",
    resolution: int = 100,
    surrogate=None,
    title: str = "Contour Plot",
    theme: PlotlyTheme = LIGHT_THEME,
    show_points: bool = True,
    n_contours: int = 15,
    design_space: Optional[Dict[str, Tuple[str, float]]] = None,
) -> "go.Figure":
    """Create an interactive contour plot.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix with exactly 2 features.
    y : np.ndarray of shape (n_samples,)
        Response values.
    x_name : str, default="X1"
        Name for x-axis.
    y_name : str, default="X2"
        Name for y-axis.
    z_name : str, default="Response"
        Name for z-axis/colorbar.
    resolution : int, default=100
        Grid resolution.
    surrogate : object, optional
        Fitted surrogate model.
    title : str, default="Contour Plot"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
    show_points : bool, default=True
        Whether to show observed points.
    n_contours : int, default=15
        Number of contour levels.
    design_space : dict, optional
        Specification for design space overlay.
        Keys are response names, values are (operator, threshold) tuples.
        E.g., {"Response": (">=", 2.0)}
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    _check_plotly()
    
    X = np.atleast_2d(X)
    if X.shape[1] != 2:
        raise ValueError("contour_plot requires exactly 2 features")
    
    # Create grid
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Predict on grid
    if surrogate is not None:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        try:
            z_pred, _ = surrogate.predict(grid_points)
        except:
            z_pred = surrogate.predict(grid_points)
        zz = z_pred.reshape(xx.shape)
    else:
        from scipy.interpolate import griddata
        zz = griddata(X, y, (xx, yy), method='cubic')
    
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=zz,
        colorscale=theme.colorscale,
        ncontours=n_contours,
        colorbar=dict(title=z_name),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color=theme.font_color),
        ),
    ))
    
    # Add design space overlay if specified
    if design_space is not None:
        mask = np.ones_like(zz, dtype=bool)
        for response_name, (op, threshold) in design_space.items():
            if op == ">=":
                mask &= zz >= threshold
            elif op == "<=":
                mask &= zz <= threshold
            elif op == ">":
                mask &= zz > threshold
            elif op == "<":
                mask &= zz < threshold
        
        # Add design space boundary
        fig.add_trace(go.Contour(
            x=x_range,
            y=y_range,
            z=mask.astype(float),
            showscale=False,
            contours=dict(
                start=0.5,
                end=0.5,
                coloring="lines",
            ),
            line=dict(color="red", width=3),
            name="Design Space",
        ))
    
    # Add observed points
    if show_points:
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                size=10,
                color=y,
                colorscale=theme.colorscale,
                line=dict(width=1, color="black"),
            ),
            name="Observations",
            hovertemplate=f"{x_name}: %{{x:.3f}}<br>{y_name}: %{{y:.3f}}<br>{z_name}: %{{marker.color:.3f}}<extra></extra>",
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        xaxis_title=x_name,
        yaxis_title=y_name,
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def parallel_coordinates(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    response_name: str = "Response",
    title: str = "Parallel Coordinates",
    theme: PlotlyTheme = LIGHT_THEME,
    highlight_top_n: Optional[int] = None,
) -> "go.Figure":
    """Create a parallel coordinates plot for high-dimensional data.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Response values.
    feature_names : list of str, optional
        Names for each feature.
    response_name : str, default="Response"
        Name for the response variable.
    title : str, default="Parallel Coordinates"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
    highlight_top_n : int, optional
        Highlight top N best points.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    _check_plotly()
    
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(n_features)]
    
    # Build dimensions
    dimensions = []
    for i, name in enumerate(feature_names):
        dimensions.append(dict(
            range=[X[:, i].min(), X[:, i].max()],
            label=name,
            values=X[:, i],
        ))
    
    # Add response as last dimension
    dimensions.append(dict(
        range=[y.min(), y.max()],
        label=response_name,
        values=y,
    ))
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=y,
            colorscale=theme.colorscale,
            showscale=True,
            colorbar=dict(title=response_name),
        ),
        dimensions=dimensions,
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def pareto_front(
    objectives: np.ndarray,
    objective_names: Optional[List[str]] = None,
    title: str = "Pareto Front",
    theme: PlotlyTheme = LIGHT_THEME,
    minimize: Optional[List[bool]] = None,
) -> "go.Figure":
    """Visualize Pareto front for multi-objective optimization.
    
    Parameters
    ----------
    objectives : np.ndarray of shape (n_samples, n_objectives)
        Objective values. Supports 2 or 3 objectives.
    objective_names : list of str, optional
        Names for each objective.
    title : str, default="Pareto Front"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
    minimize : list of bool, optional
        Whether each objective is minimized. Default all True.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    _check_plotly()
    from optiml.multi_objective import is_pareto_optimal
    
    objectives = np.atleast_2d(objectives)
    n_samples, n_objectives = objectives.shape
    
    if n_objectives not in [2, 3]:
        raise ValueError("pareto_front supports 2 or 3 objectives")
    
    if objective_names is None:
        objective_names = [f"Objective {i+1}" for i in range(n_objectives)]
    
    if minimize is None:
        minimize = [True] * n_objectives
    
    # Compute Pareto mask
    # For maximization, negate to use minimization-based Pareto
    obj_for_pareto = objectives.copy()
    for i, is_min in enumerate(minimize):
        if not is_min:
            obj_for_pareto[:, i] = -obj_for_pareto[:, i]
    
    pareto_mask = is_pareto_optimal(obj_for_pareto)
    
    fig = go.Figure()
    
    if n_objectives == 2:
        # 2D Pareto plot
        # Dominated points
        fig.add_trace(go.Scatter(
            x=objectives[~pareto_mask, 0],
            y=objectives[~pareto_mask, 1],
            mode="markers",
            marker=dict(size=8, color="lightgray", line=dict(width=1, color="gray")),
            name="Dominated",
        ))
        
        # Pareto optimal points
        pareto_obj = objectives[pareto_mask]
        # Sort by first objective for connected line
        sort_idx = np.argsort(pareto_obj[:, 0])
        pareto_sorted = pareto_obj[sort_idx]
        
        fig.add_trace(go.Scatter(
            x=pareto_sorted[:, 0],
            y=pareto_sorted[:, 1],
            mode="lines+markers",
            marker=dict(size=12, color="red", line=dict(width=2, color="darkred")),
            line=dict(width=2, color="red"),
            name="Pareto Optimal",
        ))
        
        fig.update_layout(
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
        )
    else:
        # 3D Pareto plot
        fig.add_trace(go.Scatter3d(
            x=objectives[~pareto_mask, 0],
            y=objectives[~pareto_mask, 1],
            z=objectives[~pareto_mask, 2],
            mode="markers",
            marker=dict(size=5, color="lightgray"),
            name="Dominated",
        ))
        
        fig.add_trace(go.Scatter3d(
            x=objectives[pareto_mask, 0],
            y=objectives[pareto_mask, 1],
            z=objectives[pareto_mask, 2],
            mode="markers",
            marker=dict(size=8, color="red", line=dict(width=1, color="darkred")),
            name="Pareto Optimal",
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title=objective_names[0],
                yaxis_title=objective_names[1],
                zaxis_title=objective_names[2],
            )
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def convergence_animation(
    X_history: List[np.ndarray],
    y_history: List[float],
    best_history: Optional[List[float]] = None,
    title: str = "Optimization Progress",
    theme: PlotlyTheme = LIGHT_THEME,
    minimize: bool = True,
) -> "go.Figure":
    """Create an animated convergence plot.
    
    Parameters
    ----------
    X_history : list of np.ndarray
        Parameter values at each iteration.
    y_history : list of float
        Objective values at each iteration.
    best_history : list of float, optional
        Best found value at each iteration.
    title : str, default="Optimization Progress"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
    minimize : bool, default=True
        Whether objective is minimized.
        
    Returns
    -------
    go.Figure
        Plotly figure with animation.
    """
    _check_plotly()
    
    n_iter = len(y_history)
    iterations = list(range(1, n_iter + 1))
    
    # Compute best if not provided
    if best_history is None:
        best_history = []
        best_so_far = float('inf') if minimize else float('-inf')
        for val in y_history:
            if minimize:
                best_so_far = min(best_so_far, val)
            else:
                best_so_far = max(best_so_far, val)
            best_history.append(best_so_far)
    
    # Create frames for animation
    frames = []
    for i in range(1, n_iter + 1):
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=iterations[:i],
                    y=y_history[:i],
                    mode="markers",
                    marker=dict(size=8, color="blue"),
                    name="Observations",
                ),
                go.Scatter(
                    x=iterations[:i],
                    y=best_history[:i],
                    mode="lines",
                    line=dict(width=3, color="red"),
                    name="Best Found",
                ),
            ],
            name=str(i),
        ))
    
    # Initial frame
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[iterations[0]],
                y=[y_history[0]],
                mode="markers",
                marker=dict(size=8, color="blue"),
                name="Observations",
            ),
            go.Scatter(
                x=[iterations[0]],
                y=[best_history[0]],
                mode="lines",
                line=dict(width=3, color="red"),
                name="Best Found",
            ),
        ],
        frames=frames,
    )
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 200, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}]),
                ],
            ),
        ],
        sliders=[dict(
            active=0,
            steps=[dict(args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}],
                       label=str(i+1), method="animate")
                  for i, f in enumerate(frames)],
            transition=dict(duration=0),
            x=0.1,
            y=0,
            len=0.8,
        )],
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        xaxis_title="Iteration",
        yaxis_title="Objective Value",
        xaxis=dict(range=[0, n_iter + 1]),
        yaxis=dict(range=[min(y_history) * 0.9, max(y_history) * 1.1]),
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def acquisition_landscape(
    surrogate,
    acquisition,
    bounds: List[Tuple[float, float]],
    current_best: float,
    resolution: int = 100,
    x_name: str = "X1",
    y_name: str = "X2",
    title: str = "Acquisition Function",
    theme: PlotlyTheme = LIGHT_THEME,
    observations: Optional[np.ndarray] = None,
) -> "go.Figure":
    """Visualize acquisition function landscape in 2D.
    
    Parameters
    ----------
    surrogate : object
        Fitted surrogate model.
    acquisition : object
        Acquisition function with evaluate() method.
    bounds : list of (float, float)
        Bounds for each dimension.
    current_best : float
        Current best observed value.
    resolution : int, default=100
        Grid resolution.
    x_name : str, default="X1"
        X-axis label.
    y_name : str, default="X2"
        Y-axis label.
    title : str, default="Acquisition Function"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
    observations : np.ndarray, optional
        Observed points to overlay.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    _check_plotly()
    
    if len(bounds) != 2:
        raise ValueError("acquisition_landscape requires exactly 2 dimensions")
    
    # Create grid
    x_range = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y_range = np.linspace(bounds[1][0], bounds[1][1], resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Evaluate acquisition
    acq_values = acquisition.evaluate(grid)
    zz = acq_values.reshape(xx.shape)
    
    # Find suggested point
    best_idx = np.argmax(acq_values)
    best_x = grid[best_idx]
    
    fig = go.Figure()
    
    # Add acquisition heatmap
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=zz,
        colorscale="Hot",
        colorbar=dict(title="Acquisition"),
    ))
    
    # Add observations
    if observations is not None:
        fig.add_trace(go.Scatter(
            x=observations[:, 0],
            y=observations[:, 1],
            mode="markers",
            marker=dict(size=10, color="blue", line=dict(width=1, color="white")),
            name="Observations",
        ))
    
    # Add suggested point
    fig.add_trace(go.Scatter(
        x=[best_x[0]],
        y=[best_x[1]],
        mode="markers",
        marker=dict(size=15, color="green", symbol="star", 
                    line=dict(width=2, color="white")),
        name="Suggested Next",
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        xaxis_title=x_name,
        yaxis_title=y_name,
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def uncertainty_plot(
    X: np.ndarray,
    surrogate,
    bounds: List[Tuple[float, float]],
    resolution: int = 100,
    title: str = "Prediction Uncertainty",
    theme: PlotlyTheme = LIGHT_THEME,
) -> "go.Figure":
    """Visualize GP prediction uncertainty in 2D.
    
    Parameters
    ----------
    X : np.ndarray
        Observed points.
    surrogate : object
        Fitted surrogate model with predict returning (mean, std).
    bounds : list of (float, float)
        Bounds for each dimension.
    resolution : int, default=100
        Grid resolution.
    title : str, default="Prediction Uncertainty"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
        
    Returns
    -------
    go.Figure
        Plotly figure with mean and uncertainty subplots.
    """
    _check_plotly()
    
    if len(bounds) != 2:
        raise ValueError("uncertainty_plot requires exactly 2 dimensions")
    
    # Create grid
    x_range = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y_range = np.linspace(bounds[1][0], bounds[1][1], resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions
    try:
        # Try sklearn-style API first
        mean, std = surrogate.predict(grid, return_std=True)
    except TypeError:
        # Fall back to our custom surrogate
        mean, std = surrogate.predict(grid)
    mean_grid = mean.reshape(xx.shape)
    std_grid = std.reshape(xx.shape)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Mean Prediction", "Prediction Uncertainty (Std)")
    )
    
    # Mean plot
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=mean_grid,
        colorscale=theme.colorscale,
        colorbar=dict(title="Mean", x=0.45),
    ), row=1, col=1)
    
    # Std plot
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=std_grid,
        colorscale="Reds",
        colorbar=dict(title="Std", x=1.0),
    ), row=1, col=2)
    
    # Add observations to both
    for col in [1, 2]:
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(size=8, color="black", line=dict(width=1, color="white")),
            showlegend=False,
        ), row=1, col=col)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        paper_bgcolor=theme.paper_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def effects_plot(
    effects: Dict[str, float],
    title: str = "Parameter Effects",
    theme: PlotlyTheme = LIGHT_THEME,
    show_error_bars: bool = False,
    errors: Optional[Dict[str, float]] = None,
) -> "go.Figure":
    """Create a Pareto chart of parameter effects.
    
    Parameters
    ----------
    effects : dict
        Dictionary mapping parameter names to effect magnitudes.
    title : str, default="Parameter Effects"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
    show_error_bars : bool, default=False
        Whether to show error bars.
    errors : dict, optional
        Standard errors for each effect.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    _check_plotly()
    
    # Sort by absolute effect
    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [x[0] for x in sorted_effects]
    values = [x[1] for x in sorted_effects]
    
    fig = go.Figure()
    
    colors = ["green" if v > 0 else "red" for v in values]
    
    error_y = None
    if show_error_bars and errors is not None:
        error_y = dict(
            type="data",
            array=[errors.get(n, 0) for n in names],
            visible=True,
        )
    
    fig.add_trace(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        error_y=error_y,
    ))
    
    # Add reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        xaxis_title="Parameter",
        yaxis_title="Effect",
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


def slice_plot(
    surrogate,
    fixed_point: np.ndarray,
    param_idx: int,
    bounds: Tuple[float, float],
    param_name: str = "Parameter",
    response_name: str = "Response",
    n_points: int = 100,
    title: str = "Slice Plot",
    theme: PlotlyTheme = LIGHT_THEME,
    show_uncertainty: bool = True,
) -> "go.Figure":
    """Create a 1D slice through the response surface.
    
    Parameters
    ----------
    surrogate : object
        Fitted surrogate model.
    fixed_point : np.ndarray
        Point at which to slice (values for other parameters).
    param_idx : int
        Index of parameter to vary.
    bounds : tuple of float
        (low, high) bounds for the varying parameter.
    param_name : str, default="Parameter"
        Name of the varying parameter.
    response_name : str, default="Response"
        Name of the response.
    n_points : int, default=100
        Number of points to evaluate.
    title : str, default="Slice Plot"
        Plot title.
    theme : PlotlyTheme, default=LIGHT_THEME
        Visual theme.
    show_uncertainty : bool, default=True
        Whether to show confidence bands.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    _check_plotly()
    
    # Create evaluation points
    x_values = np.linspace(bounds[0], bounds[1], n_points)
    X_eval = np.tile(fixed_point, (n_points, 1))
    X_eval[:, param_idx] = x_values
    
    # Get predictions
    try:
        # Try sklearn-style API first
        mean, std = surrogate.predict(X_eval, return_std=True)
    except TypeError:
        # Fall back to our custom surrogate
        mean, std = surrogate.predict(X_eval)
    
    fig = go.Figure()
    
    if show_uncertainty:
        # Add confidence bands
        upper = mean + 2 * std
        lower = mean - 2 * std
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
        ))
    
    # Add mean
    fig.add_trace(go.Scatter(
        x=x_values,
        y=mean,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Mean Prediction',
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=theme.title_font_size)),
        xaxis_title=param_name,
        yaxis_title=response_name,
        paper_bgcolor=theme.paper_bgcolor,
        plot_bgcolor=theme.plot_bgcolor,
        font=dict(color=theme.font_color, size=theme.axis_font_size),
    )
    
    return fig


# Convenience function to save figures
def save_figure(
    fig: "go.Figure",
    filename: str,
    format: str = "html",
    width: int = 1200,
    height: int = 800,
) -> None:
    """Save a Plotly figure to file.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to save.
    filename : str
        Output filename.
    format : str, default="html"
        Output format: "html", "png", "svg", "pdf", "json".
    width : int, default=1200
        Image width in pixels.
    height : int, default=800
        Image height in pixels.
    """
    _check_plotly()
    
    if format == "html":
        fig.write_html(filename)
    elif format == "json":
        fig.write_json(filename)
    elif format in ["png", "svg", "pdf"]:
        try:
            fig.write_image(filename, width=width, height=height)
        except Exception as e:
            warnings.warn(
                f"Could not save as {format}. Install kaleido: pip install kaleido. "
                f"Error: {e}"
            )
    else:
        raise ValueError(f"Unknown format: {format}")
