"""Tests for the constraints module."""

import numpy as np
import pytest
from optiml.constraints import (
    LinearConstraint,
    NonlinearConstraint,
    BoundConstraint,
    SumConstraint,
    BlackBoxConstraint,
    ConstraintHandler,
    PenaltyMethod,
    ConstrainedExpectedImprovement,
    sample_feasible_points,
)


class TestLinearConstraint:
    """Tests for linear constraints."""

    def test_linear_constraint_basic(self):
        """Test basic linear constraint evaluation."""
        # x1 + 2*x2 <= 10
        constraint = LinearConstraint(a=[1, 2], b=10)
        
        X = np.array([[1, 1], [5, 5], [2, 3]])
        values = constraint(X)
        
        # Values: 1+2=3-10=-7, 5+10=15-10=5, 2+6=8-10=-2
        np.testing.assert_array_almost_equal(values, [-7, 5, -2])

    def test_linear_constraint_feasibility(self):
        """Test linear constraint feasibility check."""
        constraint = LinearConstraint(a=[1, 1], b=5)
        
        X = np.array([[1, 1], [3, 3], [2, 2]])
        feasible = constraint.is_feasible(X)
        
        assert feasible[0] == True  # 2 <= 5
        assert feasible[1] == False  # 6 > 5
        assert feasible[2] == True  # 4 <= 5


class TestNonlinearConstraint:
    """Tests for nonlinear constraints."""

    def test_nonlinear_constraint_circle(self):
        """Test nonlinear constraint (unit circle)."""
        # x1^2 + x2^2 <= 1
        constraint = NonlinearConstraint(
            lambda x: x[:, 0]**2 + x[:, 1]**2 - 1,
            name="unit_circle"
        )
        
        X = np.array([[0, 0], [1, 0], [1, 1]])
        feasible = constraint.is_feasible(X)
        
        assert feasible[0] == True  # 0 <= 1
        assert feasible[1] == True  # 1 <= 1
        assert feasible[2] == False  # 2 > 1


class TestBoundConstraint:
    """Tests for bound constraints."""

    def test_bound_constraint(self):
        """Test bound constraint."""
        constraint = BoundConstraint(low=[0, 0], high=[10, 10])
        
        X = np.array([[5, 5], [-1, 5], [5, 15]])
        feasible = constraint.is_feasible(X)
        
        assert feasible[0] == True
        assert feasible[1] == False  # Below lower bound
        assert feasible[2] == False  # Above upper bound


class TestSumConstraint:
    """Tests for sum constraints."""

    def test_sum_constraint_leq(self):
        """Test sum constraint with <=."""
        constraint = SumConstraint(bound=10, constraint_type="<=")
        
        X = np.array([[3, 3, 3], [5, 5, 5], [2, 2, 2]])
        feasible = constraint.is_feasible(X)
        
        assert feasible[0] == True  # 9 <= 10
        assert feasible[1] == False  # 15 > 10
        assert feasible[2] == True  # 6 <= 10

    def test_sum_constraint_geq(self):
        """Test sum constraint with >=."""
        constraint = SumConstraint(bound=5, constraint_type=">=")
        
        X = np.array([[1, 1, 1], [3, 3, 3]])
        feasible = constraint.is_feasible(X)
        
        assert feasible[0] == False  # 3 < 5
        assert feasible[1] == True  # 9 >= 5

    def test_sum_constraint_indices(self):
        """Test sum constraint with specific indices."""
        constraint = SumConstraint(bound=5, indices=[0, 2], constraint_type="<=")
        
        X = np.array([[2, 100, 2], [5, 100, 5]])
        feasible = constraint.is_feasible(X)
        
        assert feasible[0] == True  # 2 + 2 = 4 <= 5
        assert feasible[1] == False  # 5 + 5 = 10 > 5


class TestBlackBoxConstraint:
    """Tests for black-box constraints."""

    def test_black_box_add_observation(self):
        """Test adding observations to black-box constraint."""
        bb = BlackBoxConstraint()
        
        bb.add_observation(np.array([1, 2]), 0.5)
        bb.add_observation(np.array([3, 4]), -0.5)
        
        assert len(bb.g_observed) == 2
        assert bb.X_observed.shape == (2, 2)

    def test_black_box_predict(self):
        """Test prediction with black-box constraint."""
        bb = BlackBoxConstraint()
        
        # Add some observations
        X = np.random.rand(10, 2)
        g = X[:, 0] - 0.5  # Simple linear constraint
        
        for x, g_val in zip(X, g):
            bb.add_observation(x, g_val)
        
        bb.fit()
        
        X_test = np.array([[0.2, 0.5], [0.8, 0.5]])
        mean, std = bb.predict(X_test)
        
        assert len(mean) == 2
        assert len(std) == 2

    def test_probability_of_feasibility(self):
        """Test probability of feasibility calculation."""
        bb = BlackBoxConstraint()
        
        # Add observations where lower x values are feasible
        for x in np.linspace(0, 1, 10):
            bb.add_observation(np.array([x]), x - 0.5)
        
        bb.fit()
        
        X_test = np.array([[0.1], [0.5], [0.9]])
        pof = bb.probability_of_feasibility(X_test)
        
        # Lower values should have higher probability of feasibility
        assert pof[0] > pof[2]


class TestConstraintHandler:
    """Tests for constraint handler."""

    def test_constraint_handler_multiple(self):
        """Test handler with multiple constraints."""
        handler = ConstraintHandler([
            LinearConstraint([1, 1], 10),
            BoundConstraint([0, 0], [10, 10]),
        ])
        
        X = np.array([[5, 4], [-1, 5], [8, 8]])
        feasible = handler.is_feasible(X)
        
        assert feasible[0] == True  # Both satisfied
        assert feasible[1] == False  # Bound violated
        assert feasible[2] == False  # Sum > 10

    def test_constraint_violation(self):
        """Test constraint violation calculation."""
        handler = ConstraintHandler([
            LinearConstraint([1, 0], 5),  # x1 <= 5
        ])
        
        X = np.array([[3, 0], [5, 0], [7, 0]])
        violation = handler.constraint_violation(X)
        
        assert violation[0] == pytest.approx(0)  # Feasible
        assert violation[1] == pytest.approx(0)  # On boundary
        assert violation[2] == pytest.approx(2)  # Violation of 2

    def test_filter_feasible(self):
        """Test filtering to feasible points."""
        handler = ConstraintHandler([
            LinearConstraint([1, 1], 10),
        ])
        
        X = np.array([[3, 3], [8, 8], [4, 4], [7, 7]])
        feasible_points = handler.filter_feasible(X)
        
        assert len(feasible_points) == 2  # Only first and third


class TestPenaltyMethod:
    """Tests for penalty method."""

    def test_penalty_quadratic(self):
        """Test quadratic penalty method."""
        handler = ConstraintHandler([
            LinearConstraint([1, 0], 5),
        ])
        
        penalty_method = PenaltyMethod(handler, penalty_weight=100, penalty_type="quadratic")
        
        y = np.array([1.0, 1.0, 1.0])
        X = np.array([[3, 0], [5, 0], [7, 0]])
        
        penalized = penalty_method.apply(y, X, minimize=True)
        
        assert penalized[0] == pytest.approx(1.0)  # No penalty
        assert penalized[1] == pytest.approx(1.0)  # No penalty (on boundary)
        assert penalized[2] > 1.0  # Penalty added

    def test_penalty_linear(self):
        """Test linear penalty method."""
        handler = ConstraintHandler([
            LinearConstraint([1, 0], 5),
        ])
        
        penalty_method = PenaltyMethod(handler, penalty_weight=10, penalty_type="linear")
        
        y = np.array([1.0])
        X = np.array([[7, 0]])
        
        penalized = penalty_method.apply(y, X, minimize=True)
        
        # Violation is 2, penalty is 10 * 2 = 20
        assert penalized[0] == pytest.approx(21.0)


class TestConstrainedExpectedImprovement:
    """Tests for constrained EI."""

    def test_constrained_ei(self):
        """Test constrained EI weighting."""
        handler = ConstraintHandler([
            LinearConstraint([1, 0], 0.5),  # x1 <= 0.5
        ])
        
        cei = ConstrainedExpectedImprovement(handler)
        
        ei_values = np.array([1.0, 1.0])
        X = np.array([[0.3, 0], [0.8, 0]])
        
        weighted_ei = cei.apply(ei_values, X)
        
        assert weighted_ei[0] == pytest.approx(1.0)  # Feasible: full EI
        assert weighted_ei[1] == pytest.approx(0.0)  # Infeasible: zero EI


class TestSampleFeasiblePoints:
    """Tests for feasible point sampling."""

    def test_sample_feasible_basic(self):
        """Test basic feasible point sampling."""
        handler = ConstraintHandler([
            LinearConstraint([1, 1], 1.5),  # x1 + x2 <= 1.5
        ])
        
        bounds = np.array([[0, 1], [0, 1]])
        
        points = sample_feasible_points(handler, bounds, n_samples=50)
        
        assert len(points) == 50
        # All points should satisfy constraint
        assert np.all(handler.is_feasible(points))

    def test_sample_feasible_tight_constraint(self):
        """Test sampling with tight constraint."""
        handler = ConstraintHandler([
            NonlinearConstraint(lambda x: x[:, 0]**2 + x[:, 1]**2 - 0.25),  # Very small feasible region
        ])
        
        bounds = np.array([[0, 1], [0, 1]])
        
        points = sample_feasible_points(handler, bounds, n_samples=10)
        
        assert len(points) == 10
        assert np.all(handler.is_feasible(points))
