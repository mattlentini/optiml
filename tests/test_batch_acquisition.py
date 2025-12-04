"""Tests for advanced acquisition functions and batch optimization."""

import numpy as np
import pytest
from optiml.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ThompsonSampling,
    KnowledgeGradient,
    MaxValueEntropySearch,
    AcquisitionPortfolio,
    LocalPenalization,
    KrigingBeliever,
    ConstantLiar,
    create_acquisition,
)
from optiml.surrogate import GaussianProcessSurrogate
from optiml import BayesianOptimizer, Space, Real


class MockSurrogate:
    """Mock surrogate for testing acquisition functions."""
    
    def __init__(self, mean_fn=None, std_fn=None):
        self.mean_fn = mean_fn or (lambda X: np.zeros(len(X)))
        self.std_fn = std_fn or (lambda X: np.ones(len(X)))
    
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.mean_fn(X), self.std_fn(X)
    
    def fit(self, X, y):
        pass


class TestThompsonSampling:
    """Tests for Thompson Sampling acquisition."""
    
    def test_basic_evaluation(self):
        """Test basic Thompson Sampling evaluation."""
        ts = ThompsonSampling()
        surrogate = MockSurrogate(
            mean_fn=lambda X: np.array([x[0] for x in X]),
            std_fn=lambda X: np.ones(len(X)) * 0.5
        )
        
        X = np.array([[0.2], [0.5], [0.8]])
        values = ts(X, surrogate, y_best=0.5)
        
        assert len(values) == 3
        # Values should be centered around mean with some randomness
    
    def test_multiple_samples(self):
        """Test Thompson Sampling with multiple samples."""
        ts = ThompsonSampling(n_samples=10)
        surrogate = MockSurrogate()
        
        X = np.array([[0.5]])
        values = ts(X, surrogate, y_best=0.0)
        
        assert len(values) == 1


class TestKnowledgeGradient:
    """Tests for Knowledge Gradient acquisition."""
    
    def test_basic_evaluation(self):
        """Test basic Knowledge Gradient evaluation."""
        kg = KnowledgeGradient(n_fantasies=5, n_discrete=50)
        surrogate = MockSurrogate(
            mean_fn=lambda X: np.sin(X[:, 0] * np.pi),
            std_fn=lambda X: np.ones(len(X)) * 0.3
        )
        
        X = np.array([[0.25], [0.5], [0.75]])
        values = kg(X, surrogate, y_best=0.5)
        
        assert len(values) == 3
        assert np.all(values >= 0)  # KG should be non-negative
    
    def test_zero_std_handling(self):
        """Test KG handles zero variance correctly."""
        kg = KnowledgeGradient()
        surrogate = MockSurrogate(std_fn=lambda X: np.zeros(len(X)))
        
        X = np.array([[0.5]])
        values = kg(X, surrogate, y_best=0.0)
        
        assert values[0] == 0.0  # Zero variance = zero information gain


class TestMaxValueEntropySearch:
    """Tests for MES acquisition."""
    
    def test_basic_evaluation(self):
        """Test basic MES evaluation."""
        mes = MaxValueEntropySearch(n_samples=5)
        surrogate = MockSurrogate(
            mean_fn=lambda X: X[:, 0],
            std_fn=lambda X: np.ones(len(X)) * 0.5
        )
        
        X = np.array([[0.3], [0.5], [0.7]])
        values = mes(X, surrogate, y_best=0.5)
        
        assert len(values) == 3
        assert np.all(values >= 0)  # MES should be non-negative


class TestAcquisitionPortfolio:
    """Tests for portfolio acquisition."""
    
    def test_weighted_combination(self):
        """Test weighted combination of acquisitions."""
        portfolio = AcquisitionPortfolio([
            ExpectedImprovement(xi=0.0),
            UpperConfidenceBound(kappa=1.0),
        ], weights=[0.5, 0.5], strategy="weighted")
        
        surrogate = MockSurrogate(
            mean_fn=lambda X: X[:, 0],
            std_fn=lambda X: np.ones(len(X)) * 0.2
        )
        
        X = np.array([[0.3], [0.5], [0.7]])
        values = portfolio(X, surrogate, y_best=0.4)
        
        assert len(values) == 3
    
    def test_max_strategy(self):
        """Test max strategy for portfolio."""
        portfolio = AcquisitionPortfolio([
            ExpectedImprovement(),
            UpperConfidenceBound(),
        ], strategy="max")
        
        surrogate = MockSurrogate()
        X = np.array([[0.5]])
        values = portfolio(X, surrogate, y_best=0.0)
        
        assert len(values) == 1
    
    def test_random_strategy(self):
        """Test random selection strategy."""
        portfolio = AcquisitionPortfolio([
            ExpectedImprovement(),
            UpperConfidenceBound(),
        ], strategy="random")
        
        surrogate = MockSurrogate()
        X = np.array([[0.3], [0.5], [0.7]])
        values = portfolio(X, surrogate, y_best=0.0)
        
        assert len(values) == 3


class TestLocalPenalization:
    """Tests for Local Penalization batch method."""
    
    def test_suggest_batch(self):
        """Test batch suggestion with local penalization."""
        lp = LocalPenalization(ExpectedImprovement())
        
        # Create and fit a real surrogate
        surrogate = GaussianProcessSurrogate()
        X_train = np.array([[0.2], [0.5], [0.8]])
        y_train = np.array([0.3, 0.8, 0.4])
        surrogate.fit(X_train, y_train)
        
        batch = lp.suggest_batch(
            surrogate,
            y_best=0.8,
            n_suggestions=3,
            bounds=[(0, 1)]
        )
        
        assert len(batch.X) == 3
        assert batch.X.shape == (3, 1)
        # Points should be somewhat spread out due to penalization
        assert not np.allclose(batch.X[0], batch.X[1])


class TestKrigingBeliever:
    """Tests for Kriging Believer batch method."""
    
    def test_suggest_batch(self):
        """Test batch suggestion with Kriging Believer."""
        kb = KrigingBeliever(ExpectedImprovement())
        
        # Create and fit a real surrogate
        surrogate = GaussianProcessSurrogate()
        X_train = np.array([[0.2], [0.5], [0.8]])
        y_train = np.array([0.3, 0.8, 0.4])
        surrogate.fit(X_train, y_train)
        
        batch = kb.suggest_batch(
            surrogate,
            y_best=0.8,
            n_suggestions=3,
            bounds=[(0, 1)],
            X_observed=X_train,
            y_observed=y_train
        )
        
        assert len(batch.X) == 3
        assert batch.X.shape == (3, 1)


class TestConstantLiar:
    """Tests for Constant Liar batch method."""
    
    def test_suggest_batch_min(self):
        """Test batch suggestion with min lie value."""
        cl = ConstantLiar(ExpectedImprovement(), lie_value="min")
        
        surrogate = GaussianProcessSurrogate()
        X_train = np.array([[0.2], [0.5], [0.8]])
        y_train = np.array([0.3, 0.8, 0.4])
        surrogate.fit(X_train, y_train)
        
        batch = cl.suggest_batch(
            surrogate,
            y_best=0.8,
            n_suggestions=3,
            bounds=[(0, 1)],
            y_observed=y_train
        )
        
        assert len(batch.X) == 3
    
    def test_suggest_batch_max(self):
        """Test batch suggestion with max lie value."""
        cl = ConstantLiar(ExpectedImprovement(), lie_value="max")
        
        surrogate = GaussianProcessSurrogate()
        X_train = np.array([[0.2], [0.5], [0.8]])
        y_train = np.array([0.3, 0.8, 0.4])
        surrogate.fit(X_train, y_train)
        
        batch = cl.suggest_batch(
            surrogate,
            y_best=0.8,
            n_suggestions=2,
            bounds=[(0, 1)],
            y_observed=y_train
        )
        
        assert len(batch.X) == 2


class TestCreateAcquisition:
    """Tests for the create_acquisition factory function."""
    
    def test_create_ei(self):
        """Test creating Expected Improvement."""
        acq = create_acquisition("ei", xi=0.02)
        assert isinstance(acq, ExpectedImprovement)
        assert acq.xi == 0.02
    
    def test_create_ucb(self):
        """Test creating UCB."""
        acq = create_acquisition("ucb", kappa=3.0)
        assert isinstance(acq, UpperConfidenceBound)
        assert acq.kappa == 3.0
    
    def test_create_ts(self):
        """Test creating Thompson Sampling."""
        acq = create_acquisition("thompson_sampling", n_samples=5)
        assert isinstance(acq, ThompsonSampling)
    
    def test_create_kg(self):
        """Test creating Knowledge Gradient."""
        acq = create_acquisition("knowledge_gradient")
        assert isinstance(acq, KnowledgeGradient)
    
    def test_create_mes(self):
        """Test creating MES."""
        acq = create_acquisition("mes")
        assert isinstance(acq, MaxValueEntropySearch)
    
    def test_unknown_acquisition(self):
        """Test error for unknown acquisition."""
        with pytest.raises(ValueError):
            create_acquisition("unknown_acquisition")


class TestBatchOptimization:
    """Integration tests for batch optimization in BayesianOptimizer."""
    
    def test_suggest_batch_local_penalization(self):
        """Test batch suggestion with local penalization."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space, n_initial=3)
        
        # Add some initial data
        for _ in range(5):
            x = optimizer.suggest()
            y = x[0] ** 2
            optimizer.tell(x, y)
        
        # Request batch
        batch = optimizer.suggest_batch(n_suggestions=3, strategy="local_penalization")
        
        assert len(batch) == 3
        for point in batch:
            assert len(point) == 1
            assert 0 <= point[0] <= 1
    
    def test_suggest_batch_kriging_believer(self):
        """Test batch suggestion with Kriging Believer."""
        space = Space([Real(0, 1, name="x"), Real(0, 1, name="y")])
        optimizer = BayesianOptimizer(space, n_initial=3)
        
        # Add initial data
        for _ in range(5):
            x = optimizer.suggest()
            y = sum(xi ** 2 for xi in x)
            optimizer.tell(x, y)
        
        batch = optimizer.suggest_batch(n_suggestions=4, strategy="kriging_believer")
        
        assert len(batch) == 4
    
    def test_suggest_batch_random(self):
        """Test random batch suggestion."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space, n_initial=3)
        
        batch = optimizer.suggest_batch(n_suggestions=5, strategy="random")
        
        assert len(batch) == 5
    
    def test_tell_batch(self):
        """Test recording batch observations."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space, n_initial=3)
        
        # Get batch
        batch = optimizer.suggest_batch(n_suggestions=3, strategy="random")
        results = [x[0] ** 2 for x in batch]
        
        # Record batch
        optimizer.tell_batch(batch, results)
        
        assert len(optimizer._X) == 3
        assert len(optimizer._y) == 3
    
    def test_invalid_batch_strategy(self):
        """Test error for invalid batch strategy."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space, n_initial=3)
        
        # Add some data first
        for _ in range(5):
            x = optimizer.suggest()
            optimizer.tell(x, x[0])
        
        with pytest.raises(ValueError):
            optimizer.suggest_batch(n_suggestions=3, strategy="invalid_strategy")


class TestAcquisitionFunctionBehavior:
    """Tests for acquisition function mathematical properties."""
    
    def test_ei_zero_std(self):
        """EI should be zero when uncertainty is zero."""
        ei = ExpectedImprovement()
        surrogate = MockSurrogate(
            mean_fn=lambda X: np.ones(len(X)),
            std_fn=lambda X: np.zeros(len(X))
        )
        
        X = np.array([[0.5]])
        values = ei(X, surrogate, y_best=0.0)
        
        assert values[0] == 0.0
    
    def test_ucb_higher_for_uncertain(self):
        """UCB should prefer points with higher uncertainty."""
        ucb = UpperConfidenceBound(kappa=2.0)
        
        # Create surrogate where std increases with x
        surrogate = MockSurrogate(
            mean_fn=lambda X: np.zeros(len(X)),
            std_fn=lambda X: X[:, 0]  # std = x
        )
        
        X = np.array([[0.2], [0.8]])
        values = ucb(X, surrogate, y_best=0.0)
        
        # Higher x should have higher UCB (more uncertainty)
        assert values[1] > values[0]
    
    def test_ei_positive_improvement(self):
        """EI should be positive when mean > y_best."""
        ei = ExpectedImprovement(xi=0.0)
        
        surrogate = MockSurrogate(
            mean_fn=lambda X: np.ones(len(X)) * 2.0,
            std_fn=lambda X: np.ones(len(X)) * 0.5
        )
        
        X = np.array([[0.5]])
        values = ei(X, surrogate, y_best=0.0)
        
        assert values[0] > 0
