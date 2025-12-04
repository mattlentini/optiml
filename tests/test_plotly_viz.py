"""Tests for the Plotly visualization module."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Check if plotly is available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

if PLOTLY_AVAILABLE:
    from optiml.plotly_viz import (
        PlotlyTheme,
        LIGHT_THEME,
        DARK_THEME,
        surface_3d,
        contour_plot,
        parallel_coordinates,
        pareto_front,
        convergence_animation,
        uncertainty_plot,
        effects_plot,
        slice_plot,
    )


# Skip all tests if plotly not available
pytestmark = pytest.mark.skipif(
    not PLOTLY_AVAILABLE,
    reason="Plotly not installed"
)


@pytest.fixture
def sample_2d_data():
    """Create sample 2D data for testing."""
    np.random.seed(42)
    X = np.random.uniform(0, 1, (30, 2))
    y = np.sin(X[:, 0] * np.pi) + np.cos(X[:, 1] * np.pi) + np.random.normal(0, 0.1, 30)
    return X, y


@pytest.fixture
def sample_high_dim_data():
    """Create sample high-dimensional data."""
    np.random.seed(42)
    X = np.random.uniform(0, 1, (50, 5))
    y = X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 50)
    return X, y


@pytest.fixture
def trained_gp(sample_2d_data):
    """Create a trained GP model."""
    X, y = sample_2d_data
    gp = GaussianProcessRegressor(kernel=RBF(), random_state=42)
    gp.fit(X, y)
    return gp


class TestPlotlyTheme:
    """Tests for PlotlyTheme."""
    
    def test_light_theme_defaults(self):
        """Test light theme has expected defaults."""
        assert LIGHT_THEME.paper_bgcolor == "white"
        assert LIGHT_THEME.font_color == "black"
    
    def test_dark_theme_defaults(self):
        """Test dark theme has expected defaults."""
        assert DARK_THEME.paper_bgcolor == "#1E1E1E"
        assert DARK_THEME.font_color == "white"
    
    def test_custom_theme(self):
        """Test creating custom theme."""
        theme = PlotlyTheme(colorscale="Blues", title_font_size=20)
        assert theme.colorscale == "Blues"
        assert theme.title_font_size == 20


class TestSurface3D:
    """Tests for surface_3d function."""
    
    def test_basic_surface(self, sample_2d_data):
        """Test basic 3D surface creation."""
        X, y = sample_2d_data
        fig = surface_3d(X, y)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # At least surface
    
    def test_surface_with_surrogate(self, sample_2d_data, trained_gp):
        """Test surface with surrogate model."""
        X, y = sample_2d_data
        fig = surface_3d(X, y, surrogate=trained_gp)
        
        assert isinstance(fig, go.Figure)
    
    def test_surface_custom_names(self, sample_2d_data):
        """Test surface with custom axis names."""
        X, y = sample_2d_data
        fig = surface_3d(
            X, y,
            x_name="pH",
            y_name="Temperature",
            z_name="Yield",
            title="Process Optimization"
        )
        
        assert fig.layout.title.text == "Process Optimization"
    
    def test_surface_no_points(self, sample_2d_data):
        """Test surface without observed points."""
        X, y = sample_2d_data
        fig = surface_3d(X, y, show_points=False)
        
        # Should only have surface, not scatter
        assert len(fig.data) == 1
    
    def test_surface_dark_theme(self, sample_2d_data):
        """Test surface with dark theme."""
        X, y = sample_2d_data
        fig = surface_3d(X, y, theme=DARK_THEME)
        
        assert fig.layout.paper_bgcolor == "#1E1E1E"
    
    def test_surface_wrong_dimensions(self, sample_high_dim_data):
        """Test surface raises error for wrong dimensions."""
        X, y = sample_high_dim_data
        
        with pytest.raises(ValueError, match="exactly 2 features"):
            surface_3d(X, y)


class TestContourPlot:
    """Tests for contour_plot function."""
    
    def test_basic_contour(self, sample_2d_data):
        """Test basic contour creation."""
        X, y = sample_2d_data
        fig = contour_plot(X, y)
        
        assert isinstance(fig, go.Figure)
    
    def test_contour_with_surrogate(self, sample_2d_data, trained_gp):
        """Test contour with surrogate model."""
        X, y = sample_2d_data
        fig = contour_plot(X, y, surrogate=trained_gp)
        
        assert isinstance(fig, go.Figure)
    
    def test_contour_design_space(self, sample_2d_data):
        """Test contour with design space overlay."""
        X, y = sample_2d_data
        fig = contour_plot(
            X, y,
            design_space={"Response": (">=", 0.5)}
        )
        
        # Should have contour + design space boundary + points
        assert len(fig.data) >= 2
    
    def test_contour_no_points(self, sample_2d_data):
        """Test contour without points."""
        X, y = sample_2d_data
        fig = contour_plot(X, y, show_points=False)
        
        assert len(fig.data) == 1


class TestParallelCoordinates:
    """Tests for parallel_coordinates function."""
    
    def test_basic_parallel(self, sample_high_dim_data):
        """Test basic parallel coordinates."""
        X, y = sample_high_dim_data
        fig = parallel_coordinates(X, y)
        
        assert isinstance(fig, go.Figure)
    
    def test_parallel_with_names(self, sample_high_dim_data):
        """Test parallel with custom feature names."""
        X, y = sample_high_dim_data
        names = ["pH", "Temp", "Time", "Conc", "Flow"]
        fig = parallel_coordinates(X, y, feature_names=names)
        
        assert isinstance(fig, go.Figure)
    
    def test_parallel_2d_data(self, sample_2d_data):
        """Test parallel with 2D data."""
        X, y = sample_2d_data
        fig = parallel_coordinates(X, y)
        
        assert isinstance(fig, go.Figure)


class TestParetoFront:
    """Tests for pareto_front function."""
    
    def test_2d_pareto(self):
        """Test 2D Pareto front."""
        # Create some objectives with clear Pareto front
        objectives = np.array([
            [1.0, 5.0],  # Pareto
            [2.0, 3.0],  # Pareto
            [3.0, 2.0],  # Pareto
            [5.0, 1.0],  # Pareto
            [3.0, 4.0],  # Dominated
            [4.0, 3.0],  # Dominated
        ])
        
        fig = pareto_front(objectives)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Dominated + Pareto
    
    def test_3d_pareto(self):
        """Test 3D Pareto front."""
        objectives = np.random.uniform(0, 1, (20, 3))
        
        fig = pareto_front(objectives)
        
        assert isinstance(fig, go.Figure)
    
    def test_pareto_custom_names(self):
        """Test Pareto with custom objective names."""
        objectives = np.random.uniform(0, 1, (20, 2))
        
        fig = pareto_front(
            objectives,
            objective_names=["Resolution", "Run Time"]
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_pareto_wrong_dimensions(self):
        """Test Pareto with wrong number of objectives."""
        objectives = np.random.uniform(0, 1, (20, 5))
        
        with pytest.raises(ValueError, match="2 or 3 objectives"):
            pareto_front(objectives)


class TestConvergenceAnimation:
    """Tests for convergence_animation function."""
    
    def test_basic_animation(self):
        """Test basic convergence animation."""
        X_history = [np.random.uniform(0, 1, 2) for _ in range(20)]
        y_history = list(np.cumsum(np.random.uniform(-0.1, 0.1, 20)) + 1)
        
        fig = convergence_animation(X_history, y_history)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == 20
    
    def test_animation_with_best(self):
        """Test animation with explicit best history."""
        X_history = [np.random.uniform(0, 1, 2) for _ in range(10)]
        y_history = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        best_history = [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        
        fig = convergence_animation(X_history, y_history, best_history)
        
        assert isinstance(fig, go.Figure)


class TestUncertaintyPlot:
    """Tests for uncertainty_plot function."""
    
    def test_basic_uncertainty(self, sample_2d_data, trained_gp):
        """Test basic uncertainty plot."""
        X, y = sample_2d_data
        bounds = [(0, 1), (0, 1)]
        
        fig = uncertainty_plot(X, trained_gp, bounds)
        
        assert isinstance(fig, go.Figure)
        # Should have 2 subplots (mean and std)
        assert len(fig.data) >= 2


class TestEffectsPlot:
    """Tests for effects_plot function."""
    
    def test_basic_effects(self):
        """Test basic effects plot."""
        effects = {"pH": 0.5, "Temperature": -0.3, "Time": 0.2}
        
        fig = effects_plot(effects)
        
        assert isinstance(fig, go.Figure)
    
    def test_effects_with_errors(self):
        """Test effects with error bars."""
        effects = {"pH": 0.5, "Temperature": -0.3, "Time": 0.2}
        errors = {"pH": 0.1, "Temperature": 0.08, "Time": 0.05}
        
        fig = effects_plot(effects, show_error_bars=True, errors=errors)
        
        assert isinstance(fig, go.Figure)


class TestSlicePlot:
    """Tests for slice_plot function."""
    
    def test_basic_slice(self, trained_gp):
        """Test basic slice plot."""
        fixed_point = np.array([0.5, 0.5])
        
        fig = slice_plot(
            trained_gp,
            fixed_point,
            param_idx=0,
            bounds=(0, 1)
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_slice_no_uncertainty(self, trained_gp):
        """Test slice without uncertainty bands."""
        fixed_point = np.array([0.5, 0.5])
        
        fig = slice_plot(
            trained_gp,
            fixed_point,
            param_idx=0,
            bounds=(0, 1),
            show_uncertainty=False
        )
        
        assert isinstance(fig, go.Figure)
        # Only mean line
        assert len(fig.data) == 1


class TestIntegration:
    """Integration tests for visualization module."""
    
    def test_full_workflow(self, sample_2d_data):
        """Test creating multiple plots from same data."""
        X, y = sample_2d_data
        
        # Create multiple visualizations
        surface = surface_3d(X, y, title="Surface")
        contour = contour_plot(X, y, title="Contour")
        
        assert isinstance(surface, go.Figure)
        assert isinstance(contour, go.Figure)
    
    def test_theme_consistency(self, sample_2d_data):
        """Test theme is applied consistently."""
        X, y = sample_2d_data
        
        fig = surface_3d(X, y, theme=DARK_THEME)
        
        # Check theme properties
        assert fig.layout.paper_bgcolor == DARK_THEME.paper_bgcolor
        assert fig.layout.font.color == DARK_THEME.font_color


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
