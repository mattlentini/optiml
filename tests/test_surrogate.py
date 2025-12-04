"""Tests for surrogate models."""

import numpy as np
import pytest

from optiml.surrogate import GaussianProcessSurrogate


class TestGaussianProcessSurrogate:
    """Tests for Gaussian Process surrogate model."""

    def test_fit_predict(self):
        """Test basic fit and predict functionality."""
        gp = GaussianProcessSurrogate(n_restarts=2)

        # Simple 1D data
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.sin(X.ravel() * 2 * np.pi)

        gp.fit(X, y)
        mean, std = gp.predict(X)

        # Predictions at training points should be close to actual values
        np.testing.assert_allclose(mean, y, atol=0.1)
        # Uncertainty at training points should be low
        assert np.all(std < 0.2)

    def test_uncertainty_increases_away_from_data(self):
        """Test that uncertainty increases away from training data."""
        gp = GaussianProcessSurrogate(n_restarts=2)

        X_train = np.array([[0.0], [1.0]])
        y_train = np.array([0.0, 1.0])

        gp.fit(X_train, y_train)

        X_test = np.array([[0.0], [0.5], [1.0]])
        mean, std = gp.predict(X_test)

        # Uncertainty should be highest at 0.5 (between training points)
        assert std[1] > std[0]
        assert std[1] > std[2]

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises an error."""
        gp = GaussianProcessSurrogate()

        with pytest.raises(RuntimeError):
            gp.predict(np.array([[0.5]]))

    def test_sample_y(self):
        """Test sampling from posterior."""
        gp = GaussianProcessSurrogate(n_restarts=2)

        X = np.array([[0.2], [0.4], [0.6], [0.8]])
        y = np.array([0.1, 0.3, 0.2, 0.4])

        gp.fit(X, y)

        X_test = np.array([[0.3], [0.5]])
        samples = gp.sample_y(X_test, n_samples=100, rng=np.random.default_rng(42))

        assert samples.shape == (100, 2)

    def test_normalize_y(self):
        """Test that y normalization works correctly."""
        gp = GaussianProcessSurrogate(normalize_y=True, n_restarts=2)

        X = np.array([[0.1], [0.5], [0.9]])
        y = np.array([100.0, 150.0, 200.0])  # Large values

        gp.fit(X, y)
        mean, std = gp.predict(X)

        # Predictions should still be close to training values
        np.testing.assert_allclose(mean, y, rtol=0.1)

    def test_multidimensional_input(self):
        """Test with multidimensional input."""
        gp = GaussianProcessSurrogate(n_restarts=2)

        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (20, 3))
        y = np.sum(X ** 2, axis=1)

        gp.fit(X, y)
        mean, std = gp.predict(X)

        # Should fit reasonably well
        correlation = np.corrcoef(mean, y)[0, 1]
        assert correlation > 0.9
