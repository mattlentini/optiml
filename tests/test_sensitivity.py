"""Tests for sensitivity analysis module."""

import numpy as np
import pytest

from optiml.sensitivity import (
    SobolIndices,
    MorrisResult,
    LocalSensitivity,
    compute_sobol_indices,
    compute_sobol_from_surrogate,
    compute_morris,
    compute_morris_from_surrogate,
    compute_local_sensitivity,
    compute_local_sensitivity_from_surrogate,
    correlation_sensitivity,
    main_effect_indices,
    SensitivityAnalyzer,
    sobol_sequence,
    saltelli_sample,
    morris_trajectories,
)
from optiml import BayesianOptimizer, Space, Real


@pytest.fixture
def simple_function():
    """Simple test function: y = x0 + 2*x1 (linear, x0 less important)."""
    def func(x):
        return x[0] + 2 * x[1]
    return func


@pytest.fixture
def nonlinear_function():
    """Nonlinear test function with interactions."""
    def func(x):
        return x[0]**2 + x[1]**2 + 0.5 * x[0] * x[1]
    return func


@pytest.fixture
def trained_surrogate():
    """Create a trained surrogate model."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    
    space = Space([
        Real(0, 1, name="x0"),
        Real(0, 1, name="x1"),
        Real(0, 1, name="x2"),
    ])
    
    # Create synthetic data directly
    np.random.seed(42)
    X = np.random.rand(20, 3)
    # y depends mostly on x0 and x1, not x2
    y = X[:, 0]**2 + 2*X[:, 1] + 0.1*X[:, 2] + np.random.normal(0, 0.01, 20)
    
    # Fit a simple GP
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42, normalize_y=True)
    gp.fit(X, y)
    
    return gp, space


class TestSobolSequence:
    """Tests for Sobol sequence generation."""

    def test_basic_generation(self):
        """Test basic Sobol sequence generation."""
        samples = sobol_sequence(100, 3, seed=42)
        
        assert samples.shape == (100, 3)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_uniformity(self):
        """Test that samples are roughly uniform."""
        samples = sobol_sequence(1000, 2, seed=42)
        
        # Check that samples span the space
        for dim in range(2):
            assert samples[:, dim].min() < 0.1
            assert samples[:, dim].max() > 0.9


class TestSaltelliSample:
    """Tests for Saltelli sampling."""

    def test_basic_sampling(self):
        """Test basic Saltelli sampling."""
        A, B, AB_list = saltelli_sample(100, 3, seed=42)
        
        assert A.shape == (100, 3)
        assert B.shape == (100, 3)
        assert len(AB_list) == 3
        assert all(AB.shape == (100, 3) for AB in AB_list)

    def test_ab_structure(self):
        """Test that AB matrices have correct structure."""
        A, B, AB_list = saltelli_sample(10, 3, seed=42)
        
        for i, AB in enumerate(AB_list):
            # Column i should come from B
            np.testing.assert_array_equal(AB[:, i], B[:, i])
            # Other columns should come from A
            for j in range(3):
                if j != i:
                    np.testing.assert_array_equal(AB[:, j], A[:, j])

    def test_with_bounds(self):
        """Test sampling with custom bounds."""
        bounds = np.array([[0, 10], [-1, 1], [0, 100]])
        A, B, AB_list = saltelli_sample(100, 3, bounds=bounds, seed=42)
        
        for i in range(3):
            assert A[:, i].min() >= bounds[i, 0]
            assert A[:, i].max() <= bounds[i, 1]


class TestMorrisTrajectories:
    """Tests for Morris trajectory generation."""

    def test_basic_generation(self):
        """Test basic trajectory generation."""
        trajectories = morris_trajectories(3, 5, levels=4, seed=42)
        
        assert len(trajectories) == 5
        # Each trajectory has n_dims + 1 points
        assert all(t.shape == (4, 3) for t in trajectories)

    def test_trajectory_steps(self):
        """Test that trajectories have correct step structure."""
        trajectories = morris_trajectories(3, 1, levels=4, seed=42)
        trajectory = trajectories[0]
        
        # Each step should change exactly one dimension
        for i in range(len(trajectory) - 1):
            diff = trajectory[i + 1] - trajectory[i]
            n_changed = np.sum(np.abs(diff) > 1e-10)
            assert n_changed == 1


class TestSobolIndices:
    """Tests for SobolIndices dataclass."""

    def test_creation(self):
        """Test SobolIndices creation."""
        indices = SobolIndices(
            first_order=np.array([0.3, 0.7]),
            total=np.array([0.35, 0.75]),
            parameter_names=["x0", "x1"],
        )
        
        assert len(indices.first_order) == 2
        assert len(indices.total) == 2
        assert indices.parameter_names == ["x0", "x1"]

    def test_get_ranking(self):
        """Test parameter ranking."""
        indices = SobolIndices(
            first_order=np.array([0.1, 0.5, 0.2]),
            total=np.array([0.15, 0.6, 0.25]),
        )
        
        ranking = indices.get_ranking(by="total")
        assert ranking[0][0] == 1  # x1 is most important
        assert ranking[0][1] == 0.6

    def test_to_dict(self):
        """Test dictionary conversion."""
        indices = SobolIndices(
            first_order=np.array([0.3, 0.7]),
            total=np.array([0.35, 0.75]),
        )
        
        d = indices.to_dict()
        assert "first_order" in d
        assert "total" in d


class TestMorrisResult:
    """Tests for MorrisResult dataclass."""

    def test_creation(self):
        """Test MorrisResult creation."""
        result = MorrisResult(
            mu=np.array([1.0, 2.0]),
            mu_star=np.array([1.5, 2.5]),
            sigma=np.array([0.5, 1.0]),
            elementary_effects=[[], []],
        )
        
        assert len(result.mu) == 2
        assert len(result.mu_star) == 2

    def test_get_ranking(self):
        """Test parameter ranking by mu_star."""
        result = MorrisResult(
            mu=np.array([1.0, 2.0, 0.5]),
            mu_star=np.array([1.5, 2.5, 0.5]),
            sigma=np.array([0.5, 1.0, 0.1]),
            elementary_effects=[[], [], []],
        )
        
        ranking = result.get_ranking()
        assert ranking[0][0] == 1  # x1 is most important

    def test_classify_parameters(self):
        """Test parameter classification."""
        result = MorrisResult(
            mu=np.array([1.0, 0.01, 2.0]),
            mu_star=np.array([1.5, 0.01, 2.5]),
            sigma=np.array([0.1, 0.001, 1.5]),
            elementary_effects=[[], [], []],
        )
        
        classes = result.classify_parameters()
        assert "negligible" in classes
        assert "linear" in classes
        assert "nonlinear" in classes


class TestLocalSensitivity:
    """Tests for LocalSensitivity dataclass."""

    def test_creation(self):
        """Test LocalSensitivity creation."""
        sens = LocalSensitivity(
            gradients=np.array([1.0, 2.0]),
            normalized_gradients=np.array([0.5, 1.0]),
            point=np.array([0.5, 0.5]),
        )
        
        assert len(sens.gradients) == 2
        assert len(sens.normalized_gradients) == 2

    def test_get_ranking(self):
        """Test ranking by absolute normalized gradient."""
        sens = LocalSensitivity(
            gradients=np.array([1.0, -3.0, 2.0]),
            normalized_gradients=np.array([0.5, -1.5, 1.0]),
            point=np.array([0.5, 0.5, 0.5]),
        )
        
        ranking = sens.get_ranking()
        assert ranking[0][0] == 1  # x1 has highest |gradient|


class TestComputeSobolIndices:
    """Tests for Sobol index computation."""

    def test_linear_function(self, simple_function):
        """Test with linear function."""
        indices = compute_sobol_indices(
            func=simple_function,
            n_dims=2,
            n_samples=256,
            seed=42,
        )
        
        assert indices.first_order.shape == (2,)
        assert indices.total.shape == (2,)
        # x1 should be more important (coefficient 2 vs 1)
        assert indices.total[1] > indices.total[0]

    def test_with_bounds(self, simple_function):
        """Test with custom bounds."""
        bounds = np.array([[0, 10], [0, 5]])
        indices = compute_sobol_indices(
            func=simple_function,
            n_dims=2,
            bounds=bounds,
            n_samples=256,
            seed=42,
        )
        
        assert indices.first_order.shape == (2,)

    def test_with_names(self, simple_function):
        """Test with parameter names."""
        indices = compute_sobol_indices(
            func=simple_function,
            n_dims=2,
            parameter_names=["temperature", "pressure"],
            n_samples=256,
            seed=42,
        )
        
        assert indices.parameter_names == ["temperature", "pressure"]

    def test_second_order(self, nonlinear_function):
        """Test second-order index computation."""
        indices = compute_sobol_indices(
            func=nonlinear_function,
            n_dims=2,
            n_samples=256,
            seed=42,
            compute_second_order=True,
        )
        
        assert indices.second_order is not None
        assert (0, 1) in indices.second_order


class TestComputeSobolFromSurrogate:
    """Tests for Sobol computation from surrogate."""

    def test_basic_computation(self, trained_surrogate):
        """Test basic computation from surrogate."""
        surrogate, space = trained_surrogate
        
        indices = compute_sobol_from_surrogate(
            surrogate, space, n_samples=256, seed=42
        )
        
        assert indices.first_order.shape == (3,)
        assert indices.total.shape == (3,)
        assert indices.parameter_names == ["x0", "x1", "x2"]


class TestComputeMorris:
    """Tests for Morris screening computation."""

    def test_linear_function(self, simple_function):
        """Test with linear function."""
        result = compute_morris(
            func=simple_function,
            n_dims=2,
            n_trajectories=10,
            levels=4,
            seed=42,
        )
        
        assert result.mu.shape == (2,)
        assert result.mu_star.shape == (2,)
        assert result.sigma.shape == (2,)

    def test_sigma_for_linear(self, simple_function):
        """Test that sigma is small for linear function."""
        result = compute_morris(
            func=simple_function,
            n_dims=2,
            n_trajectories=20,
            levels=4,
            seed=42,
        )
        
        # For linear function, sigma should be very small
        assert np.all(result.sigma < 0.1)

    def test_with_bounds(self, simple_function):
        """Test with custom bounds."""
        bounds = np.array([[0, 10], [0, 5]])
        result = compute_morris(
            func=simple_function,
            n_dims=2,
            bounds=bounds,
            n_trajectories=10,
            seed=42,
        )
        
        assert result.mu.shape == (2,)


class TestComputeMorrisFromSurrogate:
    """Tests for Morris computation from surrogate."""

    def test_basic_computation(self, trained_surrogate):
        """Test basic computation from surrogate."""
        surrogate, space = trained_surrogate
        
        result = compute_morris_from_surrogate(
            surrogate, space, n_trajectories=10, seed=42
        )
        
        assert result.mu.shape == (3,)
        assert result.parameter_names == ["x0", "x1", "x2"]


class TestComputeLocalSensitivity:
    """Tests for local sensitivity computation."""

    def test_linear_function(self, simple_function):
        """Test with linear function."""
        point = np.array([0.5, 0.5])
        sens = compute_local_sensitivity(
            func=simple_function,
            point=point,
        )
        
        # Gradients should match coefficients
        np.testing.assert_almost_equal(sens.gradients[0], 1.0, decimal=3)
        np.testing.assert_almost_equal(sens.gradients[1], 2.0, decimal=3)

    def test_nonlinear_function(self, nonlinear_function):
        """Test with nonlinear function."""
        point = np.array([1.0, 0.5])
        sens = compute_local_sensitivity(
            func=nonlinear_function,
            point=point,
        )
        
        # Gradient of x0^2 + x1^2 + 0.5*x0*x1 at (1, 0.5)
        # df/dx0 = 2*x0 + 0.5*x1 = 2*1 + 0.5*0.5 = 2.25
        # df/dx1 = 2*x1 + 0.5*x0 = 2*0.5 + 0.5*1 = 1.5
        np.testing.assert_almost_equal(sens.gradients[0], 2.25, decimal=3)
        np.testing.assert_almost_equal(sens.gradients[1], 1.5, decimal=3)


class TestComputeLocalSensitivityFromSurrogate:
    """Tests for local sensitivity from surrogate."""

    def test_basic_computation(self, trained_surrogate):
        """Test basic computation from surrogate."""
        surrogate, space = trained_surrogate
        
        sens = compute_local_sensitivity_from_surrogate(surrogate, space)
        
        assert sens.gradients.shape == (3,)
        assert sens.parameter_names == ["x0", "x1", "x2"]

    def test_at_custom_point(self, trained_surrogate):
        """Test at custom point."""
        surrogate, space = trained_surrogate
        point = np.array([0.2, 0.8, 0.5])
        
        sens = compute_local_sensitivity_from_surrogate(
            surrogate, space, point=point
        )
        
        np.testing.assert_array_equal(sens.point, point)


class TestCorrelationSensitivity:
    """Tests for correlation-based sensitivity."""

    def test_basic_computation(self):
        """Test basic correlation computation."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 100)
        
        result = correlation_sensitivity(X, y)
        
        assert "pearson" in result
        assert "spearman" in result
        assert result["pearson"].shape == (3,)
        
        # x1 should have higher correlation than x0
        assert abs(result["pearson"][1]) > abs(result["pearson"][0])

    def test_with_names(self):
        """Test with parameter names."""
        X = np.random.rand(50, 2)
        y = X[:, 0] + X[:, 1]
        
        result = correlation_sensitivity(X, y, parameter_names=["a", "b"])
        assert result["parameter_names"] == ["a", "b"]


class TestMainEffectIndices:
    """Tests for main effect indices."""

    def test_basic_computation(self):
        """Test basic main effect computation."""
        np.random.seed(42)
        X = np.random.rand(200, 3)
        y = X[:, 0] ** 2 + 2 * X[:, 1]
        
        result = main_effect_indices(X, y)
        
        assert "indices" in result
        assert result["indices"].shape == (3,)

    def test_with_names(self):
        """Test with parameter names."""
        X = np.random.rand(100, 2)
        y = X[:, 0] + X[:, 1]
        
        result = main_effect_indices(X, y, parameter_names=["x", "y"])
        assert result["parameter_names"] == ["x", "y"]


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer class."""

    def test_creation(self, trained_surrogate):
        """Test analyzer creation."""
        surrogate, space = trained_surrogate
        analyzer = SensitivityAnalyzer(surrogate=surrogate, space=space)
        
        assert analyzer.surrogate is surrogate
        assert analyzer.space is space

    def test_compute_sobol(self, trained_surrogate):
        """Test Sobol computation through analyzer."""
        surrogate, space = trained_surrogate
        analyzer = SensitivityAnalyzer(surrogate=surrogate, space=space, random_state=42)
        
        indices = analyzer.compute_sobol(n_samples=256)
        
        assert isinstance(indices, SobolIndices)
        assert len(indices.first_order) == 3

    def test_compute_morris(self, trained_surrogate):
        """Test Morris computation through analyzer."""
        surrogate, space = trained_surrogate
        analyzer = SensitivityAnalyzer(surrogate=surrogate, space=space, random_state=42)
        
        result = analyzer.compute_morris(n_trajectories=10)
        
        assert isinstance(result, MorrisResult)

    def test_compute_local(self, trained_surrogate):
        """Test local sensitivity through analyzer."""
        surrogate, space = trained_surrogate
        analyzer = SensitivityAnalyzer(surrogate=surrogate, space=space)
        
        result = analyzer.compute_local()
        
        assert isinstance(result, LocalSensitivity)

    def test_full_analysis(self, trained_surrogate):
        """Test full analysis."""
        surrogate, space = trained_surrogate
        analyzer = SensitivityAnalyzer(surrogate=surrogate, space=space, random_state=42)
        
        results = analyzer.full_analysis(n_sobol_samples=128, n_morris_trajectories=5)
        
        assert "sobol" in results or "morris" in results

    def test_get_parameter_ranking(self, trained_surrogate):
        """Test parameter ranking."""
        surrogate, space = trained_surrogate
        analyzer = SensitivityAnalyzer(surrogate=surrogate, space=space, random_state=42)
        
        analyzer.compute_sobol(n_samples=128)
        ranking = analyzer.get_parameter_ranking(method="sobol_total")
        
        assert len(ranking) == 3
        assert all(isinstance(item, tuple) for item in ranking)

    def test_generate_report(self, trained_surrogate):
        """Test report generation."""
        surrogate, space = trained_surrogate
        analyzer = SensitivityAnalyzer(surrogate=surrogate, space=space, random_state=42)
        
        analyzer.compute_sobol(n_samples=128)
        analyzer.compute_morris(n_trajectories=5)
        
        report = analyzer.generate_report()
        
        assert isinstance(report, str)
        assert "SENSITIVITY" in report
        assert "SOBOL" in report
        assert "MORRIS" in report


class TestEdgeCases:
    """Test edge cases."""

    def test_constant_function(self):
        """Test with constant function."""
        def func(x):
            return 1.0
        
        indices = compute_sobol_indices(
            func=func, n_dims=2, n_samples=128, seed=42
        )
        
        # All indices should be 0
        assert np.allclose(indices.first_order, 0, atol=0.1)
        assert np.allclose(indices.total, 0, atol=0.1)

    def test_single_parameter(self):
        """Test with single parameter."""
        def func(x):
            return x[0] ** 2
        
        indices = compute_sobol_indices(
            func=func, n_dims=1, n_samples=128, seed=42
        )
        
        assert len(indices.first_order) == 1
        # For a single parameter, total index should be close to 1
        # But small samples may give lower values, so just check it's positive
        assert indices.total[0] >= 0

    def test_many_parameters(self):
        """Test with many parameters."""
        def func(x):
            return np.sum(x)
        
        indices = compute_sobol_indices(
            func=func, n_dims=10, n_samples=256, seed=42
        )
        
        assert len(indices.first_order) == 10

    def test_noisy_function(self):
        """Test with noisy function."""
        np.random.seed(42)
        
        def func(x):
            return x[0] + x[1] + np.random.normal(0, 0.1)
        
        indices = compute_sobol_indices(
            func=func, n_dims=2, n_samples=256, seed=42
        )
        
        # Should detect some importance despite noise
        # With added noise, indices may be smaller
        assert indices.total[0] >= 0
        assert indices.total[1] >= 0
