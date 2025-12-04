"""Tests for Bayesian optimizer."""

import numpy as np
import pytest

from optiml import BayesianOptimizer, Space, Real, Integer


class TestBayesianOptimizer:
    """Tests for BayesianOptimizer."""

    def test_optimize_simple_function(self):
        """Test optimization of a simple quadratic function."""
        def objective(params):
            x, y = params
            return -(x - 2) ** 2 - (y - 3) ** 2

        space = Space([
            Real(0, 5, name="x"),
            Real(0, 5, name="y"),
        ])

        optimizer = BayesianOptimizer(
            space,
            n_initial=5,
            maximize=True,
            random_state=42,
        )
        result = optimizer.optimize(objective, n_iterations=25)

        # Should find optimum near (2, 3)
        assert abs(result.x_best[0] - 2) < 0.5
        assert abs(result.x_best[1] - 3) < 0.5
        assert result.y_best > -0.5

    def test_optimize_minimize(self):
        """Test minimization mode."""
        def objective(params):
            x = params[0]
            return (x - 3) ** 2

        space = Space([Real(0, 6, name="x")])

        optimizer = BayesianOptimizer(
            space,
            n_initial=3,
            maximize=False,
            random_state=42,
        )
        result = optimizer.optimize(objective, n_iterations=15)

        # Should find minimum near x=3
        assert abs(result.x_best[0] - 3) < 0.5
        assert result.y_best < 0.5

    def test_suggest_tell_interface(self):
        """Test the suggest/tell interface."""
        def objective(params):
            return -params[0] ** 2

        space = Space([Real(-5, 5, name="x")])
        optimizer = BayesianOptimizer(space, n_initial=3, random_state=42)

        for _ in range(10):
            params = optimizer.suggest()
            value = objective(params)
            optimizer.tell(params, value)

        result = optimizer.get_result()
        assert len(result.x_history) == 10
        assert len(result.y_history) == 10

    def test_callback(self):
        """Test callback function."""
        call_count = [0]

        def callback(params, value, iteration):
            call_count[0] += 1

        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space, n_initial=2, random_state=42)
        optimizer.optimize(lambda x: x[0], n_iterations=5, callback=callback)

        assert call_count[0] == 5

    def test_reset(self):
        """Test reset functionality."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space, random_state=42)

        optimizer.tell([0.5], 1.0)
        optimizer.tell([0.3], 0.8)
        assert len(optimizer._X) == 2

        optimizer.reset()
        assert len(optimizer._X) == 0
        assert len(optimizer._y) == 0

    def test_with_integer_dimension(self):
        """Test optimization with integer dimension."""
        def objective(params):
            n = params[0]
            return -abs(n - 5)

        space = Space([Integer(1, 10, name="n")])
        optimizer = BayesianOptimizer(space, n_initial=3, maximize=True, random_state=42)
        result = optimizer.optimize(objective, n_iterations=15)

        # Should find optimum close to 5
        assert abs(result.x_best[0] - 5) <= 1

    def test_empty_result_raises(self):
        """Test that getting result without observations raises error."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space)

        with pytest.raises(RuntimeError):
            optimizer.get_result()
