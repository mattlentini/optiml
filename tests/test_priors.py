"""Tests for the prior knowledge module."""

import numpy as np
import pytest
from optiml.priors import (
    ParameterPrior,
    ExperimentPrior,
    PriorKnowledgeBuilder,
    PriorAwareBayesianOptimizer,
    get_prior_for_experiment,
    create_prior_aware_optimizer,
)
from optiml import Space, Real, Integer, Categorical


class TestParameterPrior:
    """Tests for ParameterPrior class."""

    def test_real_parameter_prior(self):
        """Test prior for real-valued parameter."""
        prior = ParameterPrior(
            name="temperature",
            param_type="real",
            mean_optimal=37.0,
            std_optimal=2.0,
            best_values=[35.0, 37.0, 38.0, 36.5],
            confidence=0.8,
            low=20.0,
            high=60.0,
        )
        
        assert prior.name == "temperature"
        assert prior.confidence == 0.8
        
        # Test distribution
        dist = prior.get_prior_distribution()
        assert dist is not None
        
        # Sample should be within bounds
        samples = prior.sample(100)
        assert samples is not None
        assert len(samples) == 100
        assert np.all(samples >= 20.0)
        assert np.all(samples <= 60.0)

    def test_categorical_parameter_prior(self):
        """Test prior for categorical parameter."""
        prior = ParameterPrior(
            name="solvent",
            param_type="categorical",
            value_counts={"methanol": 5, "acetonitrile": 3, "water": 2},
            best_values=["methanol", "acetonitrile", "water"],
            confidence=0.7,
        )
        
        # Test category probabilities
        probs = prior.get_category_probabilities()
        assert probs["methanol"] == 0.5
        assert probs["acetonitrile"] == 0.3
        assert probs["water"] == 0.2
        
        # Test sampling
        samples = prior.sample(100)
        assert samples is not None
        assert len(samples) == 100
        assert all(s in ["methanol", "acetonitrile", "water"] for s in samples)

    def test_empty_prior(self):
        """Test prior with no data."""
        prior = ParameterPrior(
            name="unknown",
            param_type="real",
            confidence=0.0,
        )
        
        assert prior.get_prior_distribution() is None
        assert prior.sample(10) is None


class TestExperimentPrior:
    """Tests for ExperimentPrior class."""

    def test_experiment_prior_with_data(self):
        """Test experiment prior with parameter priors."""
        param_priors = {
            "x": ParameterPrior(
                name="x",
                param_type="real",
                mean_optimal=5.0,
                std_optimal=1.0,
                confidence=0.8,
                low=0.0,
                high=10.0,
            ),
            "y": ParameterPrior(
                name="y",
                param_type="integer",
                mean_optimal=3.0,
                std_optimal=0.5,
                confidence=0.6,
                low=1,
                high=5,
            ),
        }
        
        prior = ExperimentPrior(
            parameter_priors=param_priors,
            n_experiments=5,
            n_trials=25,
            expected_best=0.95,
            objective_variance=0.01,
        )
        
        assert prior.n_experiments == 5
        assert prior.n_trials == 25
        assert prior.expected_best == 0.95

    def test_sample_warm_start(self):
        """Test warm-start point sampling."""
        param_priors = {
            "x": ParameterPrior(
                name="x",
                param_type="real",
                mean_optimal=5.0,
                std_optimal=1.0,
                confidence=0.8,
                low=0.0,
                high=10.0,
            ),
        }
        
        prior = ExperimentPrior(parameter_priors=param_priors)
        
        points = prior.sample_warm_start(n=5)
        assert len(points) == 5
        for point in points:
            assert "x" in point
            assert 0.0 <= point["x"] <= 10.0

    def test_prior_mean_function(self):
        """Test GP prior mean function."""
        prior = ExperimentPrior(
            expected_best=0.85,
            warm_start_points=[({'x': 5}, 0.85)],
        )
        
        mean_fn = prior.get_prior_mean_function()
        X_test = np.random.rand(10, 2)
        means = mean_fn(X_test)
        
        assert len(means) == 10
        assert np.allclose(means, 0.85)


class MockDatabase:
    """Mock database for testing prior knowledge builder."""
    
    def __init__(self, experiments=None):
        self.experiments = experiments or []
    
    def list_experiments(self, include_archived=False):
        return self.experiments


class TestPriorKnowledgeBuilder:
    """Tests for PriorKnowledgeBuilder class."""

    def test_parameter_similarity(self):
        """Test parameter similarity computation."""
        db = MockDatabase()
        builder = PriorKnowledgeBuilder(db)
        
        params1 = [
            {'name': 'temperature', 'param_type': 'real'},
            {'name': 'pH', 'param_type': 'real'},
            {'name': 'flow_rate', 'param_type': 'real'},
        ]
        
        params2 = [
            {'name': 'Temperature', 'param_type': 'real'},
            {'name': 'pH', 'param_type': 'real'},
            {'name': 'pressure', 'param_type': 'real'},
        ]
        
        similarity = builder.compute_parameter_similarity(params1, params2)
        
        # 2 out of 4 unique names match, types should all match for common
        assert 0.4 < similarity < 0.8

    def test_find_similar_experiments(self):
        """Test finding similar experiments."""
        experiments = [
            {
                'id': 1,
                'name': 'HPLC Method 1',
                'template_id': 'hplc',
                'parameters': [
                    {'name': 'temperature', 'param_type': 'real'},
                    {'name': 'flow_rate', 'param_type': 'real'},
                ],
                'trials': [
                    {'parameters': {'temperature': 35, 'flow_rate': 1.0}, 'objective_value': 0.9},
                    {'parameters': {'temperature': 40, 'flow_rate': 1.2}, 'objective_value': 0.95},
                    {'parameters': {'temperature': 38, 'flow_rate': 1.1}, 'objective_value': 0.92},
                ],
            },
            {
                'id': 2,
                'name': 'HPLC Method 2',
                'template_id': 'hplc',
                'parameters': [
                    {'name': 'temperature', 'param_type': 'real'},
                    {'name': 'flow_rate', 'param_type': 'real'},
                    {'name': 'pH', 'param_type': 'real'},
                ],
                'trials': [
                    {'parameters': {'temperature': 37, 'flow_rate': 1.0, 'pH': 7.0}, 'objective_value': 0.88},
                    {'parameters': {'temperature': 42, 'flow_rate': 1.3, 'pH': 6.5}, 'objective_value': 0.93},
                    {'parameters': {'temperature': 39, 'flow_rate': 1.1, 'pH': 7.2}, 'objective_value': 0.91},
                ],
            },
        ]
        
        db = MockDatabase(experiments)
        builder = PriorKnowledgeBuilder(db, similarity_threshold=0.3)
        
        target_params = [
            {'name': 'temperature', 'param_type': 'real'},
            {'name': 'flow_rate', 'param_type': 'real'},
        ]
        
        similar = builder.find_similar_experiments(target_params, template_id='hplc')
        
        assert len(similar) >= 1
        # First experiment should match best (exact params)
        assert similar[0][0]['id'] == 1 or similar[0][1] > 0.5

    def test_build_parameter_prior(self):
        """Test building prior from best trials."""
        db = MockDatabase()
        builder = PriorKnowledgeBuilder(db)
        
        best_trials = [
            ({'temperature': 35.0, 'pH': 7.0}, 0.90, True),
            ({'temperature': 38.0, 'pH': 7.2}, 0.95, True),
            ({'temperature': 37.0, 'pH': 7.1}, 0.92, True),
            ({'temperature': 36.5, 'pH': 7.0}, 0.91, True),
        ]
        
        prior = builder.build_parameter_prior(
            'temperature', 'real', best_trials, low=20.0, high=60.0
        )
        
        assert prior.name == 'temperature'
        assert len(prior.best_values) == 4
        assert prior.mean_optimal is not None
        assert 35.0 < prior.mean_optimal < 39.0
        assert prior.confidence > 0

    def test_build_experiment_prior(self):
        """Test building complete experiment prior."""
        experiments = [
            {
                'id': 1,
                'name': 'Experiment 1',
                'minimize': True,
                'parameters': [
                    {'name': 'x', 'param_type': 'real'},
                ],
                'trials': [
                    {'parameters': {'x': 1.0}, 'objective_value': 0.5},
                    {'parameters': {'x': 2.0}, 'objective_value': 0.3},
                    {'parameters': {'x': 1.5}, 'objective_value': 0.2},  # Best
                ],
            },
        ]
        
        db = MockDatabase(experiments)
        builder = PriorKnowledgeBuilder(db, similarity_threshold=0.3)
        
        target_params = [
            {'name': 'x', 'param_type': 'real', 'low': 0.0, 'high': 5.0},
        ]
        
        prior = builder.build_experiment_prior(target_params)
        
        assert prior.n_experiments >= 1
        assert 'x' in prior.parameter_priors


class TestPriorAwareBayesianOptimizer:
    """Tests for PriorAwareBayesianOptimizer class."""

    def test_optimizer_with_prior(self):
        """Test optimizer with prior knowledge."""
        space = Space([
            Real(0, 10, name="x"),
            Real(0, 10, name="y"),
        ])
        
        param_priors = {
            "x": ParameterPrior(
                name="x",
                param_type="real",
                mean_optimal=5.0,
                std_optimal=1.0,
                confidence=0.8,
                low=0.0,
                high=10.0,
            ),
            "y": ParameterPrior(
                name="y",
                param_type="real",
                mean_optimal=5.0,
                std_optimal=1.0,
                confidence=0.8,
                low=0.0,
                high=10.0,
            ),
        }
        
        prior = ExperimentPrior(parameter_priors=param_priors)
        
        optimizer = PriorAwareBayesianOptimizer(
            space, prior, prior_weight=0.8, maximize=True
        )
        
        # Initial suggestions should be biased toward prior
        suggestions = [optimizer.suggest() for _ in range(10)]
        
        # At least some should be near the prior mean
        x_values = [s[0] for s in suggestions]
        y_values = [s[1] for s in suggestions]
        
        # Mean should be closer to 5 than to 0 or 10
        assert np.mean(x_values) > 2.5
        assert np.mean(x_values) < 7.5

    def test_optimizer_with_warm_start(self):
        """Test optimizer with warm-start points."""
        space = Space([
            Real(0, 10, name="x"),
        ])
        
        warm_start = [
            ({'x': 5.0}, 0.9),
            ({'x': 6.0}, 0.85),
        ]
        
        prior = ExperimentPrior(
            warm_start_points=warm_start,
            parameter_priors={},
        )
        
        optimizer = PriorAwareBayesianOptimizer(
            space, prior, prior_weight=0.5, maximize=True
        )
        
        # Warm-start points should be added
        assert len(optimizer._optimizer._X) >= 2

    def test_optimizer_tell_and_result(self):
        """Test tell and get_result methods."""
        space = Space([Real(0, 10, name="x")])
        prior = ExperimentPrior()
        
        optimizer = PriorAwareBayesianOptimizer(
            space, prior, prior_weight=0.5, maximize=True
        )
        
        # Add some observations
        optimizer.tell([2.0], 0.5)
        optimizer.tell([5.0], 0.9)
        optimizer.tell([8.0], 0.7)
        
        result = optimizer.get_result()
        assert result.y_best == 0.9
        assert result.x_best == [5.0]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_prior_aware_optimizer(self):
        """Test create_prior_aware_optimizer function."""
        space = Space([Real(0, 10, name="x")])
        prior = ExperimentPrior()
        
        optimizer = create_prior_aware_optimizer(
            space, prior, prior_weight=0.6, maximize=True
        )
        
        assert isinstance(optimizer, PriorAwareBayesianOptimizer)
        assert optimizer.prior_weight == 0.6


class TestIntegration:
    """Integration tests for prior knowledge system."""

    def test_full_workflow(self):
        """Test complete prior knowledge workflow."""
        # Create mock historical data
        experiments = [
            {
                'id': 1,
                'name': 'Historical Exp 1',
                'minimize': True,
                'parameters': [
                    {'name': 'learning_rate', 'param_type': 'real'},
                    {'name': 'n_layers', 'param_type': 'integer'},
                ],
                'trials': [
                    {'parameters': {'learning_rate': 0.01, 'n_layers': 2}, 'objective_value': 0.15},
                    {'parameters': {'learning_rate': 0.001, 'n_layers': 3}, 'objective_value': 0.08},
                    {'parameters': {'learning_rate': 0.005, 'n_layers': 3}, 'objective_value': 0.05},
                ],
            },
            {
                'id': 2,
                'name': 'Historical Exp 2',
                'minimize': True,
                'parameters': [
                    {'name': 'learning_rate', 'param_type': 'real'},
                    {'name': 'n_layers', 'param_type': 'integer'},
                    {'name': 'dropout', 'param_type': 'real'},
                ],
                'trials': [
                    {'parameters': {'learning_rate': 0.002, 'n_layers': 3, 'dropout': 0.2}, 'objective_value': 0.07},
                    {'parameters': {'learning_rate': 0.003, 'n_layers': 4, 'dropout': 0.3}, 'objective_value': 0.06},
                    {'parameters': {'learning_rate': 0.004, 'n_layers': 3, 'dropout': 0.25}, 'objective_value': 0.04},
                ],
            },
        ]
        
        db = MockDatabase(experiments)
        
        # Build prior for new experiment
        target_params = [
            {'name': 'learning_rate', 'param_type': 'real', 'low': 0.0001, 'high': 0.1},
            {'name': 'n_layers', 'param_type': 'integer', 'low': 1, 'high': 5},
        ]
        
        builder = PriorKnowledgeBuilder(db, similarity_threshold=0.3)
        prior = builder.build_experiment_prior(target_params)
        
        # Should have found both experiments
        assert prior.n_experiments >= 1
        assert 'learning_rate' in prior.parameter_priors
        assert 'n_layers' in prior.parameter_priors
        
        # Prior for learning_rate should favor small values
        lr_prior = prior.parameter_priors['learning_rate']
        assert lr_prior.mean_optimal is not None
        assert lr_prior.mean_optimal < 0.01  # Historical best was around 0.003-0.005
        
        # Create optimizer with prior
        space = Space([
            Real(0.0001, 0.1, name="learning_rate", log_scale=True),
            Integer(1, 5, name="n_layers"),
        ])
        
        optimizer = PriorAwareBayesianOptimizer(
            space, prior, prior_weight=0.7, maximize=False
        )
        
        # Run a few iterations
        def objective(params):
            lr, n_layers = params
            return 0.1 - lr * 10 + n_layers * 0.01
        
        for _ in range(5):
            x = optimizer.suggest()
            y = objective(x)
            optimizer.tell(x, y)
        
        result = optimizer.get_result()
        assert result.n_iterations >= 5
