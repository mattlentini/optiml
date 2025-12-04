"""Tests for the multi-objective optimization module."""

import numpy as np
import pytest
from optiml.multi_objective import (
    is_pareto_optimal,
    compute_pareto_front,
    compute_hypervolume,
    compute_crowding_distance,
    ParetoFront,
    WeightedSum,
    Chebyshev,
    AugmentedChebyshev,
    ParEGO,
    ExpectedHypervolumeImprovement,
    MultiObjectiveOptimizer,
    generate_weight_vectors,
)


class TestParetoOptimality:
    """Tests for Pareto optimality determination."""

    def test_is_pareto_optimal_2d(self):
        """Test Pareto optimality in 2D."""
        Y = np.array([
            [1, 4],
            [2, 3],
            [3, 2],
            [4, 1],
            [3, 3],  # Dominated by [2, 3] and [3, 2]
        ])
        
        mask = is_pareto_optimal(Y, minimize=True)
        
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == True
        assert mask[3] == True
        assert mask[4] == False  # Dominated

    def test_is_pareto_optimal_maximize(self):
        """Test Pareto optimality when maximizing."""
        Y = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        
        mask = is_pareto_optimal(Y, minimize=False)
        
        assert mask[0] == False  # Dominated by [2, 2]
        assert mask[1] == False  # Dominated by [3, 3]
        assert mask[2] == True  # Not dominated

    def test_compute_pareto_front(self):
        """Test extracting Pareto front."""
        X = np.random.rand(20, 3)
        Y = np.random.rand(20, 2)
        
        front = compute_pareto_front(X, Y, minimize=True)
        
        assert isinstance(front, ParetoFront)
        assert front.n_points <= 20
        assert front.n_objectives == 2
        
        # All points should be Pareto optimal
        mask = is_pareto_optimal(front.Y, minimize=True)
        assert np.all(mask)


class TestHypervolume:
    """Tests for hypervolume computation."""

    def test_hypervolume_2d_simple(self):
        """Test 2D hypervolume with simple case."""
        Y = np.array([[1, 1]])  # Single point
        reference = np.array([2, 2])
        
        hv = compute_hypervolume(Y, reference)
        
        # Area = 1 * 1 = 1
        assert hv == pytest.approx(1.0, abs=0.1)

    def test_hypervolume_2d_multiple(self):
        """Test 2D hypervolume with multiple points."""
        Y = np.array([
            [1, 3],
            [2, 2],
            [3, 1],
        ])
        reference = np.array([4, 4])
        
        hv = compute_hypervolume(Y, reference)
        
        # Should be positive
        assert hv > 0

    def test_hypervolume_empty(self):
        """Test hypervolume with empty front."""
        Y = np.array([]).reshape(0, 2)
        reference = np.array([1, 1])
        
        hv = compute_hypervolume(Y, reference)
        
        assert hv == 0.0


class TestCrowdingDistance:
    """Tests for crowding distance computation."""

    def test_crowding_distance_basic(self):
        """Test basic crowding distance."""
        Y = np.array([
            [1, 5],
            [2, 4],
            [3, 3],
            [4, 2],
            [5, 1],
        ])
        
        distances = compute_crowding_distance(Y)
        
        # Boundary points should have infinite distance
        assert distances[0] == np.inf
        assert distances[-1] == np.inf
        
        # Interior points should have finite distance
        assert np.isfinite(distances[1])
        assert np.isfinite(distances[2])
        assert np.isfinite(distances[3])

    def test_crowding_distance_two_points(self):
        """Test crowding distance with only two points."""
        Y = np.array([[1, 2], [2, 1]])
        
        distances = compute_crowding_distance(Y)
        
        assert np.all(distances == np.inf)


class TestScalarization:
    """Tests for scalarization methods."""

    def test_weighted_sum(self):
        """Test weighted sum scalarization."""
        scalarize = WeightedSum(weights=np.array([0.5, 0.5]))
        
        Y = np.array([[2, 4], [1, 3]])
        result = scalarize(Y)
        
        np.testing.assert_array_almost_equal(result, [3.0, 2.0])

    def test_chebyshev(self):
        """Test Chebyshev scalarization."""
        scalarize = Chebyshev(weights=[1.0, 1.0], ideal_point=[0, 0])
        
        Y = np.array([[2, 4], [3, 3]])
        result = scalarize(Y)
        
        # max(2, 4) = 4, max(3, 3) = 3
        np.testing.assert_array_almost_equal(result, [4.0, 3.0])

    def test_augmented_chebyshev(self):
        """Test augmented Chebyshev scalarization."""
        scalarize = AugmentedChebyshev(weights=[1.0, 1.0], ideal_point=[0, 0], rho=0.1)
        
        Y = np.array([[2, 4]])
        result = scalarize(Y)
        
        # max + rho * sum = 4 + 0.1 * (2 + 4) = 4.6
        assert result[0] == pytest.approx(4.6)


class TestParEGO:
    """Tests for ParEGO method."""

    def test_parego_weight_sampling(self):
        """Test ParEGO weight sampling."""
        parego = ParEGO(n_objectives=3, rng=np.random.default_rng(42))
        
        weights = parego.sample_weights()
        
        assert len(weights) == 3
        assert np.sum(weights) == pytest.approx(1.0)
        assert np.all(weights >= 0)

    def test_parego_scalarization(self):
        """Test ParEGO scalarization."""
        parego = ParEGO(n_objectives=2)
        
        Y = np.array([[1, 2], [2, 1], [1.5, 1.5]])
        
        scalarized = parego.scalarize(Y, weights=np.array([0.5, 0.5]))
        
        assert len(scalarized) == 3


class TestMultiObjectiveOptimizer:
    """Tests for multi-objective optimizer."""

    def test_mo_optimizer_fit(self):
        """Test fitting multi-objective optimizer."""
        optimizer = MultiObjectiveOptimizer(n_objectives=2)
        
        X = np.random.rand(20, 3)
        Y = np.column_stack([
            X[:, 0] + X[:, 1],
            X[:, 1] + X[:, 2],
        ])
        
        optimizer.fit(X, Y)
        
        assert optimizer._pareto_front is not None
        assert len(optimizer.surrogates) == 2

    def test_mo_optimizer_predict(self):
        """Test prediction with multi-objective optimizer."""
        optimizer = MultiObjectiveOptimizer(n_objectives=2)
        
        X_train = np.random.rand(30, 3)
        Y_train = np.column_stack([
            X_train[:, 0],
            X_train[:, 1],
        ])
        
        optimizer.fit(X_train, Y_train)
        
        X_test = np.random.rand(5, 3)
        mean, std = optimizer.predict(X_test)
        
        assert mean.shape == (5, 2)
        assert std.shape == (5, 2)

    def test_mo_optimizer_suggest(self):
        """Test suggestion from multi-objective optimizer."""
        optimizer = MultiObjectiveOptimizer(n_objectives=2, method="parego")
        
        X_train = np.random.rand(20, 2)
        Y_train = np.column_stack([
            X_train[:, 0]**2,
            X_train[:, 1]**2,
        ])
        
        optimizer.fit(X_train, Y_train)
        
        bounds = np.array([[0, 1], [0, 1]])
        next_x = optimizer.suggest(bounds)
        
        assert len(next_x) == 2
        assert np.all(next_x >= 0) and np.all(next_x <= 1)

    def test_mo_optimizer_pareto_results(self):
        """Test getting Pareto optimal results."""
        optimizer = MultiObjectiveOptimizer(n_objectives=2)
        
        X = np.random.rand(50, 2)
        Y = np.column_stack([
            X[:, 0],
            1 - X[:, 0] + 0.1 * np.random.rand(50),
        ])
        
        optimizer.fit(X, Y)
        
        X_pareto, Y_pareto = optimizer.get_pareto_optimal_results()
        
        assert len(X_pareto) > 0
        assert X_pareto.shape[1] == 2
        assert Y_pareto.shape[1] == 2


class TestParetoFront:
    """Tests for ParetoFront container."""

    def test_pareto_front_hypervolume(self):
        """Test hypervolume calculation from ParetoFront."""
        front = ParetoFront(
            X=np.array([[0.5, 0.5], [0.3, 0.7]]),
            Y=np.array([[1, 3], [2, 2]]),
            n_objectives=2,
        )
        
        hv = front.dominated_hypervolume(reference_point=np.array([5, 5]))
        
        assert hv > 0

    def test_pareto_front_crowding(self):
        """Test crowding distance from ParetoFront."""
        front = ParetoFront(
            X=np.random.rand(5, 2),
            Y=np.array([
                [1, 5],
                [2, 4],
                [3, 3],
                [4, 2],
                [5, 1],
            ]),
            n_objectives=2,
        )
        
        distances = front.crowding_distance()
        
        assert len(distances) == 5

    def test_pareto_front_select_by_crowding(self):
        """Test selection by crowding distance."""
        front = ParetoFront(
            X=np.arange(10).reshape(5, 2),
            Y=np.array([
                [1, 5],
                [2, 4],
                [3, 3],
                [4, 2],
                [5, 1],
            ]),
            n_objectives=2,
        )
        
        X_selected, Y_selected = front.select_by_crowding(n_select=3)
        
        assert len(X_selected) == 3
        assert len(Y_selected) == 3


class TestWeightVectorGeneration:
    """Tests for weight vector generation."""

    def test_generate_weight_vectors_2d(self):
        """Test weight vector generation for 2 objectives."""
        weights = generate_weight_vectors(n_objectives=2, n_divisions=5)
        
        # Should have n_divisions + 1 vectors
        assert len(weights) == 6
        
        # All should sum to 1
        for w in weights:
            assert np.sum(w) == pytest.approx(1.0)

    def test_generate_weight_vectors_3d(self):
        """Test weight vector generation for 3 objectives."""
        weights = generate_weight_vectors(n_objectives=3, n_divisions=4)
        
        # All should be valid weight vectors
        for w in weights:
            assert len(w) == 3
            assert np.sum(w) == pytest.approx(1.0)
            assert np.all(w >= 0)


class TestExpectedHypervolumeImprovement:
    """Tests for EHVI acquisition function."""

    def test_ehvi_basic(self):
        """Test basic EHVI computation."""
        front = ParetoFront(
            X=np.array([[0.5, 0.5]]),
            Y=np.array([[2, 2]]),
            n_objectives=2,
        )
        
        ehvi = ExpectedHypervolumeImprovement(
            pareto_front=front,
            reference_point=np.array([5, 5]),
            n_samples=32,
        )
        
        mean = np.array([[1.5, 1.5], [3, 3]])
        std = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        values = ehvi(mean, std, rng=np.random.default_rng(42))
        
        assert len(values) == 2
        # Point at [1.5, 1.5] should have higher EHVI (improves front)
        assert values[0] > values[1]
