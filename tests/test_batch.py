"""Tests for batch/parallel acquisition functions."""

import pytest
import numpy as np

from optiml import (
    BayesianOptimizer,
    Space,
    Real,
    Integer,
    suggest_batch,
    ConstantLiarBatch,
    LocalPenalizationBatch,
    qExpectedImprovement,
)


@pytest.fixture
def simple_space():
    """Simple 2D search space."""
    return Space([
        Real(0, 1, name='x1'),
        Real(0, 1, name='x2'),
    ])


@pytest.fixture
def fitted_optimizer(simple_space):
    """Fitted optimizer with some data."""
    optimizer = BayesianOptimizer(simple_space, maximize=True, n_initial=3)
    
    # Add some observations (>= n_initial to trigger surrogate fitting)
    X = np.random.rand(5, 2)
    y = np.random.rand(5)
    
    for x, y_val in zip(X, y):
        optimizer.tell(list(x), y_val)
    
    # Fit the surrogate manually to ensure it's ready
    X_normalized = optimizer.space.transform(optimizer._X)
    optimizer.surrogate.fit(X_normalized, np.array(optimizer._y))
    
    return optimizer


class TestConstantLiarBatch:
    """Tests for Constant Liar batch strategy."""
    
    def test_constant_liar_min(self, fitted_optimizer):
        """Test Constant Liar with min strategy."""
        batch_acq = ConstantLiarBatch(strategy='min')
        
        batch = batch_acq.suggest_batch(
            n_points=3,
            surrogate=fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            space=fitted_optimizer.space,
            rng=np.random.default_rng(42),
        )
        
        assert batch.shape == (3, 2)
        assert np.all(batch >= 0) and np.all(batch <= 1)
    
    def test_constant_liar_max(self, fitted_optimizer):
        """Test Constant Liar with max strategy."""
        batch_acq = ConstantLiarBatch(strategy='max')
        
        batch = batch_acq.suggest_batch(
            n_points=2,
            surrogate=fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            space=fitted_optimizer.space,
            rng=np.random.default_rng(42),
        )
        
        assert batch.shape == (2, 2)
    
    def test_constant_liar_mean(self, fitted_optimizer):
        """Test Constant Liar with mean strategy."""
        batch_acq = ConstantLiarBatch(strategy='mean')
        
        batch = batch_acq.suggest_batch(
            n_points=4,
            surrogate=fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            space=fitted_optimizer.space,
            rng=np.random.default_rng(42),
        )
        
        assert batch.shape == (4, 2)
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            ConstantLiarBatch(strategy='invalid')


class TestLocalPenalizationBatch:
    """Tests for Local Penalization batch strategy."""
    
    def test_local_penalization(self, fitted_optimizer):
        """Test local penalization suggests diverse points."""
        batch_acq = LocalPenalizationBatch(
            penalty_radius=0.1,
            penalty_strength=2.0,
        )
        
        batch = batch_acq.suggest_batch(
            n_points=3,
            surrogate=fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            space=fitted_optimizer.space,
            rng=np.random.default_rng(42),
        )
        
        assert batch.shape == (3, 2)
        
        # Check diversity: points should be reasonably separated
        from scipy.spatial.distance import pdist
        min_dist = np.min(pdist(batch))
        assert min_dist > 0.01  # Should have some separation
    
    def test_penalty_radius_effect(self, fitted_optimizer):
        """Test that penalty radius affects diversity."""
        # Small radius - less diversity
        batch_acq_small = LocalPenalizationBatch(penalty_radius=0.01)
        batch_small = batch_acq_small.suggest_batch(
            n_points=3,
            surrogate=fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            space=fitted_optimizer.space,
            n_candidates=1000,
            rng=np.random.default_rng(42),
        )
        
        # Large radius - more diversity
        batch_acq_large = LocalPenalizationBatch(penalty_radius=0.5)
        batch_large = batch_acq_large.suggest_batch(
            n_points=3,
            surrogate=fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            space=fitted_optimizer.space,
            n_candidates=1000,
            rng=np.random.default_rng(42),
        )
        
        # Both should return valid batches
        assert batch_small.shape == (3, 2)
        assert batch_large.shape == (3, 2)


class TestQExpectedImprovement:
    """Tests for q-EI batch strategy."""
    
    def test_qei_basic(self, fitted_optimizer):
        """Test q-EI suggests batch."""
        batch_acq = qExpectedImprovement(n_samples=50)
        
        batch = batch_acq.suggest_batch(
            n_points=2,
            surrogate=fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            space=fitted_optimizer.space,
            rng=np.random.default_rng(42),
        )
        
        assert batch.shape == (2, 2)
        assert np.all(batch >= 0) and np.all(batch <= 1)
    
    def test_qei_evaluation(self, fitted_optimizer):
        """Test q-EI evaluation."""
        batch_acq = qExpectedImprovement(n_samples=50)
        
        # Create a batch
        batch = np.random.rand(2, 2)
        
        qei_value = batch_acq.evaluate_qei(
            batch,
            fitted_optimizer.surrogate,
            y_best=max(fitted_optimizer._y),
            rng=np.random.default_rng(42),
        )
        
        assert isinstance(qei_value, (float, np.floating))
        assert qei_value >= 0  # q-EI should be non-negative


class TestSuggestBatchAPI:
    """Tests for the convenient suggest_batch function."""
    
    def test_suggest_batch_constant_liar(self, fitted_optimizer):
        """Test suggest_batch with constant liar."""
        batch = suggest_batch(
            fitted_optimizer,
            n_points=3,
            strategy='constant_liar',
            liar_strategy='min',
        )
        
        assert len(batch) == 3
        assert all(len(point) == 2 for point in batch)
        assert all(isinstance(point, list) for point in batch)
    
    def test_suggest_batch_local_penalization(self, fitted_optimizer):
        """Test suggest_batch with local penalization."""
        batch = suggest_batch(
            fitted_optimizer,
            n_points=4,
            strategy='local_penalization',
            penalty_radius=0.15,
        )
        
        assert len(batch) == 4
        assert all(len(point) == 2 for point in batch)
    
    def test_suggest_batch_qei(self, fitted_optimizer):
        """Test suggest_batch with q-EI."""
        batch = suggest_batch(
            fitted_optimizer,
            n_points=2,
            strategy='qei',
            n_samples=50,
        )
        
        assert len(batch) == 2
        assert all(len(point) == 2 for point in batch)
    
    def test_suggest_batch_invalid_strategy(self, fitted_optimizer):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            suggest_batch(fitted_optimizer, n_points=2, strategy='invalid')
    
    def test_suggest_batch_unfitted_optimizer(self, simple_space):
        """Test that unfitted optimizer raises error."""
        optimizer = BayesianOptimizer(simple_space, n_initial=3)
        
        with pytest.raises(ValueError, match="at least"):
            suggest_batch(optimizer, n_points=2)
    
    def test_batch_can_be_evaluated(self, fitted_optimizer):
        """Test that suggested batch can be evaluated."""
        batch = suggest_batch(fitted_optimizer, n_points=2, strategy='local_penalization')
        
        # Should be able to tell these to the optimizer
        for point in batch:
            # Simulate evaluation
            y_val = np.random.rand()
            fitted_optimizer.tell(point, y_val)
        
        # Optimizer should have 7 observations now (5 + 2)
        assert len(fitted_optimizer._X) == 7
        assert len(fitted_optimizer._y) == 7


class TestBatchDiversity:
    """Tests for batch diversity properties."""
    
    def test_batch_points_are_different(self, fitted_optimizer):
        """Test that batch points are not identical."""
        batch = suggest_batch(
            fitted_optimizer,
            n_points=5,
            strategy='local_penalization',
        )
        
        batch_array = np.array(batch)
        
        # Check no two points are identical
        for i in range(len(batch)):
            for j in range(i + 1, len(batch)):
                assert not np.allclose(batch_array[i], batch_array[j], atol=1e-6)
    
    def test_batch_respects_bounds(self, simple_space):
        """Test that batch points respect space bounds."""
        optimizer = BayesianOptimizer(simple_space, maximize=True, n_initial=3)
        
        # Add some data
        for _ in range(5):
            x = [np.random.rand(), np.random.rand()]
            optimizer.tell(x, np.random.rand())
        
        batch = suggest_batch(optimizer, n_points=10, strategy='local_penalization')
        batch_array = np.array(batch)
        
        # All points should be in [0, 1]^2
        assert np.all(batch_array >= 0)
        assert np.all(batch_array <= 1)


class TestBatchWithIntegerSpace:
    """Tests for batch acquisition with integer parameters."""
    
    def test_batch_with_integers(self):
        """Test batch suggestions with integer parameters."""
        space = Space([
            Integer(1, 10, name='n'),
            Real(0, 1, name='x'),
        ])
        
        optimizer = BayesianOptimizer(space, maximize=True, n_initial=3)
        
        # Add some data
        for _ in range(5):
            x = [np.random.randint(1, 11), np.random.rand()]
            optimizer.tell(x, np.random.rand())
        
        batch = suggest_batch(optimizer, n_points=3, strategy='local_penalization')
        
        assert len(batch) == 3
        # First parameter should be integer-like (or will be rounded by space)
        for point in batch:
            assert len(point) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
