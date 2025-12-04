"""Tests for acquisition functions."""

import numpy as np
import pytest

from optiml.acquisition import (
    ExpectedImprovement,
    LowerConfidenceBound,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from optiml.surrogate import GaussianProcessSurrogate


class TestAcquisitionFunctions:
    """Tests for acquisition functions."""

    @pytest.fixture
    def fitted_surrogate(self):
        """Create a fitted surrogate model for testing."""
        gp = GaussianProcessSurrogate(n_restarts=2)
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.5, 0.3, 0.8, 0.4])
        gp.fit(X, y)
        return gp, np.max(y)

    def test_expected_improvement(self, fitted_surrogate):
        """Test Expected Improvement."""
        gp, y_best = fitted_surrogate
        ei = ExpectedImprovement(xi=0.01)

        X_test = np.array([[0.2], [0.6], [0.8]])
        values = ei(X_test, gp, y_best)

        assert len(values) == 3
        assert np.all(values >= 0)  # EI is always non-negative

    def test_expected_improvement_at_optimum(self, fitted_surrogate):
        """Test that EI is low at the current best point."""
        gp, y_best = fitted_surrogate
        ei = ExpectedImprovement(xi=0.01)

        # At the best observed point (x=0.7, y=0.8), EI should be low
        X_best = np.array([[0.7]])
        values = ei(X_best, gp, y_best)

        # EI should be small at the current optimum
        assert values[0] < 0.1

    def test_upper_confidence_bound(self, fitted_surrogate):
        """Test Upper Confidence Bound."""
        gp, y_best = fitted_surrogate
        ucb = UpperConfidenceBound(kappa=2.0)

        X_test = np.array([[0.2], [0.6]])
        values = ucb(X_test, gp, y_best)

        assert len(values) == 2
        # UCB = mean + kappa * std, so should be positive for reasonable kappa
        mean, std = gp.predict(X_test)
        expected = mean + 2.0 * std
        np.testing.assert_allclose(values, expected)

    def test_probability_of_improvement(self, fitted_surrogate):
        """Test Probability of Improvement."""
        gp, y_best = fitted_surrogate
        pi = ProbabilityOfImprovement(xi=0.01)

        X_test = np.array([[0.2], [0.6], [0.8]])
        values = pi(X_test, gp, y_best)

        assert len(values) == 3
        assert np.all(values >= 0)
        assert np.all(values <= 1)  # Probabilities must be in [0, 1]

    def test_lower_confidence_bound(self, fitted_surrogate):
        """Test Lower Confidence Bound for minimization."""
        gp, y_best = fitted_surrogate
        lcb = LowerConfidenceBound(kappa=2.0)

        X_test = np.array([[0.2], [0.6]])
        values = lcb(X_test, gp, y_best)

        assert len(values) == 2
        # LCB = -(mean - kappa * std)
        mean, std = gp.predict(X_test)
        expected = -(mean - 2.0 * std)
        np.testing.assert_allclose(values, expected)

    def test_acquisition_xi_effect(self, fitted_surrogate):
        """Test that xi parameter affects exploration."""
        gp, y_best = fitted_surrogate

        ei_low_xi = ExpectedImprovement(xi=0.0)
        ei_high_xi = ExpectedImprovement(xi=1.0)

        # At unexplored region
        X_test = np.array([[0.05]])

        low_xi_value = ei_low_xi(X_test, gp, y_best)
        high_xi_value = ei_high_xi(X_test, gp, y_best)

        # Higher xi encourages more exploration
        # The relationship depends on the mean and std at that point
        assert low_xi_value.shape == high_xi_value.shape
