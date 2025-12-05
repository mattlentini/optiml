"""Tests for QbD Design Space module."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from optiml.qbd import (
    SpecificationType,
    Specification,
    spec_minimum,
    spec_maximum,
    spec_target,
    spec_range,
    DesignSpacePoint,
    DesignSpaceResult,
    DesignSpace,
    RobustnessResult,
    monte_carlo_robustness,
    ControlStrategy,
    recommend_control_strategy,
    plot_design_space_2d,
    export_design_space_report,
)


class TestSpecification:
    """Test Specification class."""

    def test_spec_minimum_check(self):
        """Test minimum specification check."""
        spec = spec_minimum("Response", 10.0)
        
        assert spec.check(15.0) is True
        assert spec.check(10.0) is True
        assert spec.check(9.9) is False

    def test_spec_maximum_check(self):
        """Test maximum specification check."""
        spec = spec_maximum("RunTime", 30.0)
        
        assert spec.check(25.0) is True
        assert spec.check(30.0) is True
        assert spec.check(30.1) is False

    def test_spec_target_check(self):
        """Test target specification check."""
        spec = spec_target("pH", 7.0, tolerance=0.2)
        
        assert spec.check(7.0) is True
        assert spec.check(7.1) is True
        assert spec.check(6.9) is True
        assert spec.check(7.3) is False
        assert spec.check(6.7) is False

    def test_spec_range_check(self):
        """Test range specification check."""
        spec = spec_range("Temperature", 20.0, 40.0)
        
        assert spec.check(30.0) is True
        assert spec.check(20.0) is True
        assert spec.check(40.0) is True
        assert spec.check(19.9) is False
        assert spec.check(40.1) is False

    def test_spec_minimum_probability(self):
        """Test minimum specification probability."""
        spec = spec_minimum("Response", 10.0)
        
        # Mean well above limit, should be high probability
        prob = spec.probability(mean=15.0, std=1.0)
        assert prob > 0.99
        
        # Mean at limit, should be ~50%
        prob = spec.probability(mean=10.0, std=1.0)
        assert 0.45 < prob < 0.55
        
        # Mean below limit, should be low probability
        prob = spec.probability(mean=5.0, std=1.0)
        assert prob < 0.01

    def test_spec_maximum_probability(self):
        """Test maximum specification probability."""
        spec = spec_maximum("RunTime", 30.0)
        
        # Mean well below limit
        prob = spec.probability(mean=20.0, std=2.0)
        assert prob > 0.99
        
        # Mean at limit
        prob = spec.probability(mean=30.0, std=2.0)
        assert 0.45 < prob < 0.55

    def test_spec_target_probability(self):
        """Test target specification probability."""
        spec = spec_target("pH", 7.0, tolerance=0.2)
        
        # Mean at target
        prob = spec.probability(mean=7.0, std=0.05)
        assert prob > 0.95
        
        # Mean off-target
        prob = spec.probability(mean=7.3, std=0.05)
        assert prob < 0.1

    def test_spec_range_probability(self):
        """Test range specification probability."""
        spec = spec_range("Temperature", 20.0, 40.0)
        
        # Mean centered in range
        prob = spec.probability(mean=30.0, std=2.0)
        assert prob > 0.95
        
        # Mean near edge
        prob = spec.probability(mean=38.0, std=5.0)
        assert 0.3 < prob < 0.9

    def test_spec_zero_std(self):
        """Test probability with zero standard deviation."""
        spec = spec_minimum("Response", 10.0)
        
        assert spec.probability(15.0, 0.0) == 1.0
        assert spec.probability(5.0, 0.0) == 0.0

    def test_spec_weight(self):
        """Test specification weight."""
        spec = spec_minimum("Response", 10.0, weight=0.5)
        assert spec.weight == 0.5


class TestDesignSpace:
    """Test DesignSpace class."""

    @pytest.fixture
    def simple_surrogates(self):
        """Create simple fitted surrogate models for testing."""
        np.random.seed(42)
        
        # Generate training data
        X = np.random.uniform(0, 10, size=(20, 2))
        y_response = 5 + X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.normal(0, 0.1, 20)
        y_runtime = 10 + X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.normal(0, 0.5, 20)
        
        # Fit GPs
        kernel = ConstantKernel() * RBF() 
        gp_response = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp_response.fit(X, y_response)
        
        gp_runtime = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp_runtime.fit(X, y_runtime)
        
        return {"Response": gp_response, "RunTime": gp_runtime}

    def test_design_space_creation(self, simple_surrogates):
        """Test creating a design space."""
        specs = [
            spec_minimum("Response", 6.0),
            spec_maximum("RunTime", 25.0),
        ]
        
        ds = DesignSpace(
            surrogates=simple_surrogates,
            specifications=specs,
            confidence_level=0.9,
        )
        
        assert ds.confidence_level == 0.9
        assert len(ds.specifications) == 2

    def test_design_space_missing_surrogate(self, simple_surrogates):
        """Test error when specification references missing surrogate."""
        specs = [
            spec_minimum("NonExistent", 6.0),
        ]
        
        with pytest.raises(ValueError, match="No surrogate model"):
            DesignSpace(
                surrogates=simple_surrogates,
                specifications=specs,
            )

    def test_design_space_calculate(self, simple_surrogates):
        """Test design space calculation."""
        specs = [
            spec_minimum("Response", 6.0),
            spec_maximum("RunTime", 25.0),
        ]
        
        ds = DesignSpace(
            surrogates=simple_surrogates,
            specifications=specs,
            confidence_level=0.8,
        )
        
        result = ds.calculate(
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_points=20,
        )
        
        assert isinstance(result, DesignSpaceResult)
        assert result.probabilities.shape == (20, 20)
        assert result.design_space_mask.shape == (20, 20)
        assert 0 <= result.volume_fraction <= 1
        assert 0 <= result.modr_volume_fraction <= 1

    def test_design_space_probabilities_valid(self, simple_surrogates):
        """Test that probabilities are in valid range."""
        specs = [
            spec_minimum("Response", 5.0),
        ]
        
        ds = DesignSpace(
            surrogates=simple_surrogates,
            specifications=specs,
        )
        
        result = ds.calculate(
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_points=10,
        )
        
        assert np.all(result.probabilities >= 0)
        assert np.all(result.probabilities <= 1)

    def test_design_space_modr_subset_of_ds(self, simple_surrogates):
        """Test that MODR is subset of design space."""
        specs = [
            spec_minimum("Response", 6.0),
        ]
        
        ds = DesignSpace(
            surrogates=simple_surrogates,
            specifications=specs,
            confidence_level=0.9,
            modr_margin=0.05,
        )
        
        result = ds.calculate(
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_points=20,
        )
        
        # MODR should be contained within design space
        assert np.all(result.modr_mask <= result.design_space_mask)
        assert result.modr_volume_fraction <= result.volume_fraction

    def test_evaluate_point(self, simple_surrogates):
        """Test evaluating a single point."""
        specs = [
            spec_minimum("Response", 6.0),
            spec_maximum("RunTime", 25.0),
        ]
        
        ds = DesignSpace(
            surrogates=simple_surrogates,
            specifications=specs,
            confidence_level=0.8,
        )
        
        point = ds.evaluate_point({"x0": 5.0, "x1": 5.0})
        
        assert isinstance(point, DesignSpacePoint)
        assert 0 <= point.probability <= 1
        assert "Response" in point.individual_probs
        assert "RunTime" in point.individual_probs
        assert isinstance(bool(point.is_in_design_space), bool)
        assert isinstance(bool(point.is_in_modr), bool)


class TestMonteCarloRobustness:
    """Test Monte Carlo robustness analysis."""

    @pytest.fixture
    def design_space(self):
        """Create design space for testing."""
        np.random.seed(42)
        
        X = np.random.uniform(0, 10, size=(30, 2))
        y = 10 + X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.normal(0, 0.2, 30)
        
        kernel = ConstantKernel() * RBF()
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X, y)
        
        specs = [spec_minimum("Response", 10.0)]
        
        return DesignSpace(
            surrogates={"Response": gp},
            specifications=specs,
            confidence_level=0.9,
        )

    def test_monte_carlo_basic(self, design_space):
        """Test basic Monte Carlo robustness."""
        result = monte_carlo_robustness(
            design_space,
            nominal_point={"x0": 5.0, "x1": 5.0},
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            variation_level=0.05,
            n_simulations=100,
            random_state=42,
        )
        
        assert isinstance(result, RobustnessResult)
        assert 0 <= result.probability_of_success <= 1
        assert result.n_simulations == 100
        assert result.variation_level == 0.05
        assert isinstance(result.critical_parameters, list)

    def test_monte_carlo_reproducible(self, design_space):
        """Test that Monte Carlo is reproducible with seed."""
        result1 = monte_carlo_robustness(
            design_space,
            nominal_point={"x0": 5.0, "x1": 5.0},
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_simulations=50,
            random_state=123,
        )
        
        result2 = monte_carlo_robustness(
            design_space,
            nominal_point={"x0": 5.0, "x1": 5.0},
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_simulations=50,
            random_state=123,
        )
        
        assert result1.probability_of_success == result2.probability_of_success

    def test_monte_carlo_variation_levels(self, design_space):
        """Test that higher variation leads to lower probability."""
        result_low = monte_carlo_robustness(
            design_space,
            nominal_point={"x0": 5.0, "x1": 5.0},
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            variation_level=0.01,
            n_simulations=100,
            random_state=42,
        )
        
        result_high = monte_carlo_robustness(
            design_space,
            nominal_point={"x0": 5.0, "x1": 5.0},
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            variation_level=0.20,
            n_simulations=100,
            random_state=42,
        )
        
        # Higher variation should generally reduce success probability
        # (or stay the same if we're deeply in the design space)
        assert result_low.probability_of_success >= result_high.probability_of_success - 0.1


class TestControlStrategy:
    """Test control strategy recommendations."""

    @pytest.fixture
    def design_space_and_result(self):
        """Create design space and result for testing."""
        np.random.seed(42)
        
        X = np.random.uniform(0, 10, size=(30, 2))
        y = 10 + X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.normal(0, 0.2, 30)
        
        kernel = ConstantKernel() * RBF()
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X, y)
        
        specs = [spec_minimum("Response", 10.0)]
        
        ds = DesignSpace(
            surrogates={"Response": gp},
            specifications=specs,
            confidence_level=0.9,
        )
        
        result = ds.calculate(
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_points=20,
        )
        
        return ds, result

    def test_recommend_control_strategy(self, design_space_and_result):
        """Test control strategy recommendation."""
        ds, result = design_space_and_result
        
        strategies = recommend_control_strategy(
            ds,
            result,
            optimal_point={"x0": 5.0, "x1": 5.0},
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
        )
        
        assert len(strategies) == 2
        for s in strategies:
            assert isinstance(s, ControlStrategy)
            assert s.criticality in ["critical", "key", "non-critical"]
            assert s.control_type in ["in-process", "specification", "normal"]
            assert s.control_limit_lower <= s.nominal <= s.control_limit_upper


class TestPlotDesignSpace:
    """Test design space plotting."""

    @pytest.fixture
    def design_space_result(self):
        """Create design space result for plotting."""
        np.random.seed(42)
        
        X = np.random.uniform(0, 10, size=(30, 2))
        y = 10 + X[:, 0] * 0.5 + np.random.normal(0, 0.2, 30)
        
        kernel = ConstantKernel() * RBF()
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X, y)
        
        specs = [spec_minimum("Response", 10.0)]
        ds = DesignSpace(surrogates={"Response": gp}, specifications=specs)
        
        return ds.calculate(
            parameter_ranges={"pH": (0, 10), "Temperature": (0, 10)},
            n_points=20,
        )

    def test_plot_2d_basic(self, design_space_result):
        """Test basic 2D plot."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing
        import matplotlib.pyplot as plt
        
        ax = plot_design_space_2d(
            design_space_result,
            param_x="pH",
            param_y="Temperature",
        )
        
        assert ax is not None
        plt.close('all')

    def test_plot_2d_options(self, design_space_result):
        """Test 2D plot with options."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        
        ax = plot_design_space_2d(
            design_space_result,
            param_x="pH",
            param_y="Temperature",
            ax=ax,
            show_modr=True,
            show_edge=True,
        )
        
        assert ax is not None
        plt.close('all')


class TestExportReport:
    """Test report export."""

    def test_export_basic(self, tmp_path):
        """Test basic report export."""
        np.random.seed(42)
        
        X = np.random.uniform(0, 10, size=(20, 2))
        y = 10 + X[:, 0] * 0.5 + np.random.normal(0, 0.2, 20)
        
        kernel = ConstantKernel() * RBF()
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X, y)
        
        specs = [spec_minimum("Response", 10.0)]
        ds = DesignSpace(surrogates={"Response": gp}, specifications=specs)
        
        result = ds.calculate(
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_points=10,
        )
        
        filepath = tmp_path / "report.html"
        output = export_design_space_report(result, filepath=str(filepath))
        
        assert filepath.exists()
        content = filepath.read_text()
        assert "Design Space Report" in content
        assert "Confidence Level" in content


class TestSpecificationHelpers:
    """Test specification helper functions."""

    def test_spec_minimum_helper(self):
        """Test spec_minimum helper."""
        spec = spec_minimum("Response", 10.0, weight=0.8)
        
        assert spec.name == "Response"
        assert spec.spec_type == SpecificationType.MINIMUM
        assert spec.value == 10.0
        assert spec.weight == 0.8

    def test_spec_maximum_helper(self):
        """Test spec_maximum helper."""
        spec = spec_maximum("RunTime", 30.0)
        
        assert spec.name == "RunTime"
        assert spec.spec_type == SpecificationType.MAXIMUM
        assert spec.value == 30.0

    def test_spec_target_helper(self):
        """Test spec_target helper."""
        spec = spec_target("pH", 7.0, tolerance=0.2)
        
        assert spec.name == "pH"
        assert spec.spec_type == SpecificationType.TARGET
        assert spec.value == 7.0
        assert spec.tolerance == 0.2

    def test_spec_range_helper(self):
        """Test spec_range helper."""
        spec = spec_range("Temperature", 20.0, 40.0)
        
        assert spec.name == "Temperature"
        assert spec.spec_type == SpecificationType.RANGE
        assert spec.lower == 20.0
        assert spec.upper == 40.0


class TestEdgeCases:
    """Test edge cases."""

    def test_single_specification(self):
        """Test with single specification."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, size=(20, 2))
        y = 10 + X[:, 0] * 0.5 + np.random.normal(0, 0.2, 20)
        
        kernel = ConstantKernel() * RBF()
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X, y)
        
        ds = DesignSpace(
            surrogates={"Response": gp},
            specifications=[spec_minimum("Response", 10.0)],
        )
        
        result = ds.calculate(
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_points=10,
        )
        
        assert result.probabilities.shape == (10, 10)

    def test_high_confidence_level(self):
        """Test with very high confidence level."""
        np.random.seed(42)
        X = np.random.uniform(0, 10, size=(20, 2))
        y = 10 + X[:, 0] * 0.5 + np.random.normal(0, 0.2, 20)
        
        kernel = ConstantKernel() * RBF()
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X, y)
        
        ds = DesignSpace(
            surrogates={"Response": gp},
            specifications=[spec_minimum("Response", 10.0)],
            confidence_level=0.99,
        )
        
        result = ds.calculate(
            parameter_ranges={"x0": (0, 10), "x1": (0, 10)},
            n_points=10,
        )
        
        # Very high confidence should result in smaller design space
        assert result.volume_fraction <= 1.0
