"""Tests for the statistical analysis module."""

import numpy as np
import pytest
from optiml.statistics import (
    calculate_summary_statistics,
    analyze_effects,
    perform_anova,
    calculate_residuals,
    confidence_interval_mean,
    prediction_interval,
    check_normality,
    calculate_partial_dependence,
    calculate_all_partial_dependence,
)


class TestSummaryStatistics:
    """Tests for summary statistics calculation."""

    def test_basic_statistics(self):
        """Test basic summary statistics."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = calculate_summary_statistics(y)
        
        assert stats.n_trials == 5
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(5.0)

    def test_empty_array(self):
        """Test with empty array."""
        y = np.array([])
        stats = calculate_summary_statistics(y)
        
        assert stats.n_trials == 0
        assert np.isnan(stats.mean)

    def test_single_value(self):
        """Test with single value."""
        y = np.array([42.0])
        stats = calculate_summary_statistics(y)
        
        assert stats.n_trials == 1
        assert stats.mean == pytest.approx(42.0)
        assert stats.std == pytest.approx(0.0)


class TestEffectsAnalysis:
    """Tests for parameter effects analysis."""

    def test_analyze_effects_linear(self):
        """Test effects analysis with linear relationship."""
        np.random.seed(42)
        X = np.random.rand(50, 3)
        # Strong effect on first parameter, weak on others
        y = 5 * X[:, 0] + 0.1 * X[:, 1] + np.random.randn(50) * 0.1
        
        effects = analyze_effects(X, y, ["P1", "P2", "P3"])
        
        # First parameter should have highest importance
        sorted_effects = effects.get_sorted_effects()
        assert sorted_effects[0].name == "P1"
        assert sorted_effects[0].importance > 0.5

    def test_analyze_effects_with_names(self):
        """Test effects with custom parameter names."""
        X = np.random.rand(20, 2)
        y = X[:, 0] + X[:, 1]
        
        effects = analyze_effects(X, y, ["pH", "Temperature"])
        
        assert effects.parameter_effects[0].name == "pH"
        assert effects.parameter_effects[1].name == "Temperature"


class TestANOVA:
    """Tests for ANOVA."""

    def test_anova_basic(self):
        """Test basic ANOVA computation."""
        np.random.seed(42)
        X = np.random.rand(30, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(30) * 0.5
        
        anova = perform_anova(X, y, ["X1", "X2", "X3"])
        
        assert len(anova.rows) == 4  # 3 parameters + residual
        assert anova.rows[-1].source == "Residual"
        assert anova.r_squared > 0
        assert anova.r_squared <= 1

    def test_anova_significant_factors(self):
        """Test finding significant factors."""
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = 10 * X[:, 0] + np.random.randn(50) * 0.1  # Only X1 matters
        
        anova = perform_anova(X, y, ["X1", "X2"])
        significant = anova.significant_factors(alpha=0.05)
        
        assert "X1" in significant


class TestResiduals:
    """Tests for residual analysis."""

    def test_residuals_basic(self):
        """Test basic residual calculation."""
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = 2 * X[:, 0] + X[:, 1] + np.random.randn(30) * 0.1
        
        residuals = calculate_residuals(X, y)
        
        assert len(residuals.raw) == 30
        assert len(residuals.standardized) == 30
        assert len(residuals.studentized) == 30
        assert len(residuals.leverage) == 30
        assert len(residuals.cooks_distance) == 30

    def test_outliers_detection(self):
        """Test outlier detection."""
        np.random.seed(42)
        X = np.random.rand(30, 2)
        y = X[:, 0] + np.random.randn(30) * 0.1
        y[0] = 100  # Add outlier
        
        residuals = calculate_residuals(X, y)
        outliers = residuals.outliers(threshold=2.0)
        
        assert outliers[0] == True  # First point should be outlier


class TestConfidenceIntervals:
    """Tests for confidence intervals."""

    def test_confidence_interval_mean(self):
        """Test confidence interval for mean."""
        np.random.seed(42)
        y = np.random.randn(100) + 5  # Mean around 5
        
        ci = confidence_interval_mean(y, confidence=0.95)
        
        assert ci.lower < ci.point_estimate < ci.upper
        assert ci.contains(5.0) or abs(ci.point_estimate - 5.0) < 0.5
        assert ci.confidence_level == 0.95

    def test_prediction_interval(self):
        """Test prediction interval."""
        np.random.seed(42)
        y = np.random.randn(50) + 10
        
        pi = prediction_interval(y, confidence=0.95)
        
        # Prediction interval should be wider than confidence interval
        ci = confidence_interval_mean(y, confidence=0.95)
        assert pi.width() > ci.width()


class TestNormalityTest:
    """Tests for normality testing."""

    def test_normal_data(self):
        """Test with normal data."""
        np.random.seed(42)
        y = np.random.randn(100)
        
        result = check_normality(y)
        
        assert result.test_name == "Shapiro-Wilk"
        assert result.is_normal == True
        assert result.p_value > 0.05

    def test_non_normal_data(self):
        """Test with non-normal data."""
        np.random.seed(42)
        y = np.random.exponential(1.0, 100)
        
        result = check_normality(y)
        
        # Exponential data should not be normal
        assert result.p_value < 0.1 or not result.is_normal


class TestPartialDependence:
    """Tests for partial dependence calculation."""

    def test_partial_dependence_single(self):
        """Test partial dependence for single parameter."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = 2 * X[:, 0] + 0.5 * X[:, 1]
        
        grid, pd_mean, pd_std = calculate_partial_dependence(X, y, param_index=0)
        
        assert len(grid) == 50  # Default n_grid
        assert len(pd_mean) == 50
        assert len(pd_std) == 50
        # PD should increase with increasing parameter (positive effect)
        assert pd_mean[-1] > pd_mean[0]

    def test_all_partial_dependence(self):
        """Test partial dependence for all parameters."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = X[:, 0] + X[:, 1] + X[:, 2]
        
        results = calculate_all_partial_dependence(X, y, ["A", "B", "C"])
        
        assert len(results) == 3
        assert "A" in results
        assert "B" in results
        assert "C" in results
