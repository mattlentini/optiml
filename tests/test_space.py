"""Tests for search space definitions."""

import numpy as np
import pytest

from optiml.space import Categorical, Integer, Real, Space


class TestReal:
    """Tests for Real dimension."""

    def test_sample(self):
        """Test sampling from real dimension."""
        dim = Real(0.0, 1.0, name="x")
        samples = dim.sample(100, rng=np.random.default_rng(42))
        assert len(samples) == 100
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_log_scale(self):
        """Test log-scale sampling."""
        dim = Real(1e-5, 1e-1, name="lr", log_scale=True)
        samples = dim.sample(1000, rng=np.random.default_rng(42))
        # Log-scale should produce more small values
        median = np.median(samples)
        assert median < 0.01  # Median should be closer to lower bound in log space

    def test_transform_inverse(self):
        """Test transform and inverse transform."""
        dim = Real(2.0, 10.0, name="x")
        original = np.array([2.0, 6.0, 10.0])
        transformed = dim.transform(original)
        recovered = dim.inverse_transform(transformed)
        np.testing.assert_allclose(original, recovered)

    def test_transform_log_scale(self):
        """Test transform with log scale."""
        dim = Real(1e-4, 1.0, name="x", log_scale=True)
        original = np.array([1e-4, 0.01, 1.0])
        transformed = dim.transform(original)
        recovered = dim.inverse_transform(transformed)
        np.testing.assert_allclose(original, recovered, rtol=1e-5)

    def test_invalid_bounds(self):
        """Test that invalid bounds raise an error."""
        with pytest.raises(ValueError):
            Real(1.0, 0.0, name="x")

    def test_log_scale_negative(self):
        """Test that log scale with non-positive lower bound raises error."""
        with pytest.raises(ValueError):
            Real(-1.0, 1.0, name="x", log_scale=True)


class TestInteger:
    """Tests for Integer dimension."""

    def test_sample(self):
        """Test sampling from integer dimension."""
        dim = Integer(1, 10, name="n")
        samples = dim.sample(100, rng=np.random.default_rng(42))
        assert len(samples) == 100
        assert all(1 <= s <= 10 for s in samples)
        assert all(isinstance(s, (int, np.integer)) for s in samples)

    def test_transform_inverse(self):
        """Test transform and inverse transform."""
        dim = Integer(0, 10, name="n")
        original = np.array([0, 5, 10])
        transformed = dim.transform(original)
        recovered = dim.inverse_transform(transformed)
        np.testing.assert_array_equal(original, recovered)


class TestCategorical:
    """Tests for Categorical dimension."""

    def test_sample(self):
        """Test sampling from categorical dimension."""
        dim = Categorical(["a", "b", "c"], name="cat")
        samples = dim.sample(100, rng=np.random.default_rng(42))
        assert len(samples) == 100
        assert all(s in ["a", "b", "c"] for s in samples)

    def test_transform_inverse(self):
        """Test transform and inverse transform."""
        dim = Categorical(["x", "y", "z"], name="cat")
        original = np.array(["x", "z", "y"])
        transformed = dim.transform(original)
        recovered = dim.inverse_transform(transformed)
        np.testing.assert_array_equal(original, recovered)

    def test_insufficient_categories(self):
        """Test that fewer than 2 categories raises error."""
        with pytest.raises(ValueError):
            Categorical(["only_one"], name="cat")


class TestSpace:
    """Tests for Space class."""

    def test_sample(self):
        """Test sampling from space."""
        space = Space([
            Real(0.0, 1.0, name="x"),
            Integer(1, 5, name="n"),
            Categorical(["a", "b"], name="cat"),
        ])
        samples = space.sample(10, rng=np.random.default_rng(42))
        assert len(samples) == 10
        assert all(len(s) == 3 for s in samples)

    def test_transform_inverse(self):
        """Test transform and inverse transform."""
        space = Space([
            Real(0.0, 10.0, name="x"),
            Integer(1, 5, name="n"),
        ])
        original = [[5.0, 3], [2.5, 1]]
        transformed = space.transform(original)
        recovered = space.inverse_transform(transformed)
        
        # Check values are close
        for orig, rec in zip(original, recovered):
            np.testing.assert_allclose(orig[0], rec[0], rtol=1e-5)
            assert orig[1] == rec[1]

    def test_dimension_names(self):
        """Test getting dimension names."""
        space = Space([
            Real(0.0, 1.0, name="x"),
            Integer(1, 5, name="n"),
        ])
        assert space.dimension_names == ["x", "n"]
