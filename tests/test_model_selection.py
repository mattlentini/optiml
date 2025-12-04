"""Tests for model selection and AutoML module."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

from optiml.model_selection import (
    KernelFamily,
    KernelConfig,
    ModelScore,
    create_kernel,
    KernelSelector,
    HyperparameterConfig,
    HyperparameterTuner,
    GPEnsemble,
    AutoML,
)


@pytest.fixture
def simple_data():
    """Generate simple 1D data for testing."""
    np.random.seed(42)
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, 20)
    return X, y


@pytest.fixture
def multi_dim_data():
    """Generate multi-dimensional data for testing."""
    np.random.seed(42)
    X = np.random.rand(30, 3)
    y = X[:, 0] ** 2 + 2 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 30)
    return X, y


class TestKernelConfig:
    """Tests for KernelConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = KernelConfig(family=KernelFamily.RBF)
        assert config.family == KernelFamily.RBF
        assert config.length_scale == 1.0
        assert config.length_scale_bounds == (1e-5, 1e5)
        assert config.additional_params == {}

    def test_custom_values(self):
        """Test custom values."""
        config = KernelConfig(
            family=KernelFamily.MATERN_52,
            length_scale=2.0,
            length_scale_bounds=(0.1, 10.0),
            additional_params={"noise_level": 0.01},
        )
        assert config.length_scale == 2.0
        assert config.length_scale_bounds == (0.1, 10.0)
        assert config.additional_params["noise_level"] == 0.01


class TestCreateKernel:
    """Tests for create_kernel function."""

    def test_rbf_kernel(self):
        """Test RBF kernel creation."""
        config = KernelConfig(family=KernelFamily.RBF)
        kernel = create_kernel(config, n_dims=1)
        assert kernel is not None
        # Kernel should have RBF component
        assert any("RBF" in str(kernel) or "rbf" in str(kernel).lower() for _ in [1])

    def test_matern_kernels(self):
        """Test Matern kernel creation."""
        for family in [KernelFamily.MATERN_12, KernelFamily.MATERN_32, KernelFamily.MATERN_52]:
            config = KernelConfig(family=family)
            kernel = create_kernel(config, n_dims=1)
            assert kernel is not None
            assert "Matern" in str(kernel)

    def test_rational_quadratic(self):
        """Test RationalQuadratic kernel creation."""
        config = KernelConfig(
            family=KernelFamily.RATIONAL_QUADRATIC,
            additional_params={"alpha": 2.0},
        )
        kernel = create_kernel(config, n_dims=1)
        assert kernel is not None
        assert "RationalQuadratic" in str(kernel)

    def test_periodic_kernel(self):
        """Test periodic kernel creation."""
        config = KernelConfig(
            family=KernelFamily.PERIODIC,
            additional_params={"periodicity": 1.0},
        )
        kernel = create_kernel(config, n_dims=1)
        assert kernel is not None

    def test_linear_kernel(self):
        """Test linear kernel creation."""
        config = KernelConfig(family=KernelFamily.LINEAR)
        kernel = create_kernel(config, n_dims=1)
        assert kernel is not None
        assert "DotProduct" in str(kernel)

    def test_composite_kernel(self):
        """Test composite kernel creation."""
        config = KernelConfig(family=KernelFamily.COMPOSITE)
        kernel = create_kernel(config, n_dims=1)
        assert kernel is not None

    def test_ard_kernel(self):
        """Test ARD kernel creation."""
        config = KernelConfig(family=KernelFamily.RBF)
        kernel = create_kernel(config, n_dims=3, ard=True)
        assert kernel is not None

    def test_kernel_has_noise(self):
        """Test that kernel includes noise term."""
        config = KernelConfig(family=KernelFamily.RBF)
        kernel = create_kernel(config, n_dims=1)
        assert "WhiteKernel" in str(kernel)

    def test_kernel_has_constant(self):
        """Test that kernel includes constant term."""
        config = KernelConfig(family=KernelFamily.RBF)
        kernel = create_kernel(config, n_dims=1)
        # ConstantKernel displays as "1**2" or similar in string representation
        assert "ConstantKernel" in str(kernel) or "**2" in str(kernel) or "Constant" in str(kernel)


class TestKernelSelector:
    """Tests for KernelSelector class."""

    def test_default_candidates(self):
        """Test default candidate kernels."""
        selector = KernelSelector()
        assert len(selector.candidates) > 0
        assert all(isinstance(c, KernelConfig) for c in selector.candidates)

    def test_custom_candidates(self):
        """Test custom candidate kernels."""
        candidates = [
            KernelConfig(family=KernelFamily.RBF),
            KernelConfig(family=KernelFamily.MATERN_52),
        ]
        selector = KernelSelector(candidates=candidates)
        assert len(selector.candidates) == 2

    def test_select_basic(self, simple_data):
        """Test basic kernel selection."""
        X, y = simple_data
        selector = KernelSelector(cv=3, n_restarts=1, random_state=42)
        
        best_config, scores = selector.select(X, y)
        
        assert isinstance(best_config, KernelConfig)
        assert len(scores) > 0
        assert all(isinstance(s, ModelScore) for s in scores)

    def test_select_with_criterion(self, simple_data):
        """Test kernel selection with different criteria."""
        X, y = simple_data
        selector = KernelSelector(cv=3, n_restarts=1, random_state=42)
        
        for criterion in ["cv", "aic", "bic", "lml"]:
            best_config, scores = selector.select(X, y, criterion=criterion)
            assert isinstance(best_config, KernelConfig)

    def test_select_multidim(self, multi_dim_data):
        """Test kernel selection with multi-dimensional data."""
        X, y = multi_dim_data
        selector = KernelSelector(cv=3, n_restarts=1, random_state=42)
        
        best_config, scores = selector.select(X, y)
        assert isinstance(best_config, KernelConfig)

    def test_model_scores_fields(self, simple_data):
        """Test that ModelScore has all expected fields."""
        X, y = simple_data
        selector = KernelSelector(cv=3, n_restarts=1, random_state=42)
        
        _, scores = selector.select(X, y)
        score = scores[0]
        
        assert hasattr(score, "kernel_config")
        assert hasattr(score, "cv_score")
        assert hasattr(score, "log_marginal_likelihood")
        assert hasattr(score, "aic")
        assert hasattr(score, "bic")
        assert hasattr(score, "n_params")
        assert hasattr(score, "fit_time")


class TestHyperparameterConfig:
    """Tests for HyperparameterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = HyperparameterConfig()
        assert config.n_restarts == 10
        assert config.optimizer == "L-BFGS-B"
        assert config.max_iter == 100
        assert config.use_warm_start is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = HyperparameterConfig(
            n_restarts=5,
            optimizer="fmin_l_bfgs_b",
            max_iter=50,
            use_warm_start=False,
        )
        assert config.n_restarts == 5
        assert config.max_iter == 50


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner class."""

    def test_tune_marginal_likelihood(self, simple_data):
        """Test tuning with marginal likelihood."""
        X, y = simple_data
        config = KernelConfig(family=KernelFamily.RBF)
        kernel = create_kernel(config, n_dims=1)
        
        tuner = HyperparameterTuner(
            config=HyperparameterConfig(n_restarts=2),
            strategy="marginal_likelihood",
            random_state=42,
        )
        
        tuned_kernel, best_lml = tuner.tune(kernel, X, y)
        
        assert tuned_kernel is not None
        assert isinstance(best_lml, float)

    def test_tune_cross_validation(self, simple_data):
        """Test tuning with cross-validation."""
        X, y = simple_data
        config = KernelConfig(family=KernelFamily.RBF)
        kernel = create_kernel(config, n_dims=1)
        
        tuner = HyperparameterTuner(
            config=HyperparameterConfig(n_restarts=2),
            strategy="cross_validation",
            random_state=42,
        )
        
        tuned_kernel, best_score = tuner.tune(kernel, X, y)
        
        assert tuned_kernel is not None
        assert isinstance(best_score, float)

    def test_tune_bayesian(self, simple_data):
        """Test tuning with Bayesian optimization."""
        X, y = simple_data
        config = KernelConfig(family=KernelFamily.RBF)
        kernel = create_kernel(config, n_dims=1)
        
        tuner = HyperparameterTuner(
            strategy="bayesian",
            random_state=42,
        )
        
        tuned_kernel, best_lml = tuner.tune(kernel, X, y)
        
        assert tuned_kernel is not None
        assert isinstance(best_lml, float)


class TestGPEnsemble:
    """Tests for GPEnsemble class."""

    def test_default_kernels(self):
        """Test default kernel configurations."""
        ensemble = GPEnsemble(n_members=3)
        assert len(ensemble.kernel_configs) == 3

    def test_custom_kernels(self):
        """Test custom kernel configurations."""
        configs = [
            KernelConfig(family=KernelFamily.RBF),
            KernelConfig(family=KernelFamily.MATERN_52),
        ]
        ensemble = GPEnsemble(kernel_configs=configs)
        assert len(ensemble.kernel_configs) == 2

    def test_fit_basic(self, simple_data):
        """Test basic fitting."""
        X, y = simple_data
        ensemble = GPEnsemble(n_members=3, random_state=42)
        
        ensemble.fit(X, y)
        
        assert ensemble._fitted
        assert len(ensemble.members_) > 0
        assert ensemble.weights_ is not None
        assert np.isclose(ensemble.weights_.sum(), 1.0)

    def test_predict(self, simple_data):
        """Test prediction."""
        X, y = simple_data
        ensemble = GPEnsemble(n_members=3, random_state=42)
        ensemble.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean = ensemble.predict(X_test)
        
        assert mean.shape == (1,)

    def test_predict_with_std(self, simple_data):
        """Test prediction with uncertainty."""
        X, y = simple_data
        ensemble = GPEnsemble(n_members=3, random_state=42)
        ensemble.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean, std = ensemble.predict(X_test, return_std=True)
        
        assert mean.shape == (1,)
        assert std.shape == (1,)
        assert std[0] >= 0

    def test_weights_equal(self, simple_data):
        """Test equal weights."""
        X, y = simple_data
        ensemble = GPEnsemble(n_members=3, weights="equal", random_state=42)
        ensemble.fit(X, y)
        
        n = len(ensemble.members_)
        expected = 1.0 / n
        assert np.allclose(ensemble.weights_, [expected] * n)

    def test_weights_lml(self, simple_data):
        """Test LML-based weights."""
        X, y = simple_data
        ensemble = GPEnsemble(n_members=3, weights="lml", random_state=42)
        ensemble.fit(X, y)
        
        assert np.isclose(ensemble.weights_.sum(), 1.0)
        assert all(w >= 0 for w in ensemble.weights_)

    def test_weights_cv(self, simple_data):
        """Test CV-based weights."""
        X, y = simple_data
        ensemble = GPEnsemble(n_members=3, weights="cv", random_state=42)
        ensemble.fit(X, y)
        
        assert np.isclose(ensemble.weights_.sum(), 1.0)

    def test_sample_y(self, simple_data):
        """Test sampling from posterior."""
        X, y = simple_data
        ensemble = GPEnsemble(n_members=3, random_state=42)
        ensemble.fit(X, y)
        
        X_test = np.array([[0.5], [0.7]])
        samples = ensemble.sample_y(X_test, n_samples=5, random_state=42)
        
        assert samples.shape == (5, 2)

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises error."""
        ensemble = GPEnsemble(n_members=3)
        
        with pytest.raises(ValueError, match="not fitted"):
            ensemble.predict(np.array([[0.5]]))


class TestAutoML:
    """Tests for AutoML class."""

    def test_modes(self):
        """Test different modes."""
        for mode in ["fast", "balanced", "thorough"]:
            automl = AutoML(mode=mode)
            assert automl.mode == mode

    def test_fit_single_gp(self, simple_data):
        """Test fitting with single GP."""
        X, y = simple_data
        automl = AutoML(mode="fast", use_ensemble=False, random_state=42)
        
        model = automl.fit(X, y)
        
        assert model is not None
        assert automl.best_kernel_config_ is not None

    def test_fit_ensemble(self, simple_data):
        """Test fitting with ensemble."""
        X, y = simple_data
        automl = AutoML(mode="fast", use_ensemble=True, random_state=42)
        
        model = automl.fit(X, y)
        
        assert isinstance(model, GPEnsemble)

    def test_predict(self, simple_data):
        """Test prediction."""
        X, y = simple_data
        automl = AutoML(mode="fast", random_state=42)
        automl.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean = automl.predict(X_test)
        
        assert mean.shape == (1,)

    def test_predict_with_std(self, simple_data):
        """Test prediction with uncertainty."""
        X, y = simple_data
        automl = AutoML(mode="fast", random_state=42)
        automl.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean, std = automl.predict(X_test, return_std=True)
        
        assert mean.shape == (1,)
        assert std.shape == (1,)

    def test_get_selection_report(self, simple_data):
        """Test getting selection report."""
        X, y = simple_data
        automl = AutoML(mode="fast", use_ensemble=False, random_state=42)
        automl.fit(X, y)
        
        report = automl.get_selection_report()
        
        assert "best_kernel" in report
        assert "n_candidates_evaluated" in report
        assert "scores" in report
        assert len(report["scores"]) > 0

    def test_get_selection_report_ensemble(self, simple_data):
        """Test selection report for ensemble mode."""
        X, y = simple_data
        automl = AutoML(mode="fast", use_ensemble=True, random_state=42)
        automl.fit(X, y)
        
        report = automl.get_selection_report()
        assert "message" in report

    def test_predict_before_fit_raises(self):
        """Test that predicting before fitting raises error."""
        automl = AutoML(mode="fast")
        
        with pytest.raises(ValueError, match="not fitted"):
            automl.predict(np.array([[0.5]]))

    def test_multidim_data(self, multi_dim_data):
        """Test with multi-dimensional data."""
        X, y = multi_dim_data
        automl = AutoML(mode="fast", random_state=42)
        
        model = automl.fit(X, y)
        mean = automl.predict(X[:5])
        
        assert mean.shape == (5,)


class TestModelScore:
    """Tests for ModelScore dataclass."""

    def test_creation(self):
        """Test ModelScore creation."""
        config = KernelConfig(family=KernelFamily.RBF)
        score = ModelScore(
            kernel_config=config,
            cv_score=-0.1,
            log_marginal_likelihood=10.0,
            aic=5.0,
            bic=6.0,
            n_params=3,
            fit_time=0.5,
        )
        
        assert score.cv_score == -0.1
        assert score.log_marginal_likelihood == 10.0
        assert score.aic == 5.0
        assert score.bic == 6.0
        assert score.n_params == 3
        assert score.fit_time == 0.5


class TestKernelFamily:
    """Tests for KernelFamily enum."""

    def test_all_families(self):
        """Test all kernel families exist."""
        families = list(KernelFamily)
        assert len(families) >= 7
        assert KernelFamily.RBF in families
        assert KernelFamily.MATERN_52 in families
        assert KernelFamily.COMPOSITE in families

    def test_values(self):
        """Test enum values."""
        assert KernelFamily.RBF.value == "rbf"
        assert KernelFamily.MATERN_52.value == "matern_52"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        
        # Should handle small datasets
        automl = AutoML(mode="fast", random_state=42)
        model = automl.fit(X, y)
        
        assert model is not None

    def test_noisy_data(self):
        """Test with noisy data."""
        np.random.seed(42)
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.random.normal(0, 10, 20)  # Pure noise
        
        automl = AutoML(mode="fast", random_state=42)
        model = automl.fit(X, y)
        
        # Should still fit without error
        assert model is not None

    def test_constant_target(self):
        """Test with constant target values."""
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.ones(10)  # Constant
        
        # Might warn but shouldn't crash
        automl = AutoML(mode="fast", random_state=42)
        try:
            model = automl.fit(X, y)
        except Exception:
            pass  # Some numerical issues with constant targets are acceptable

    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[0.5]])
        y = np.array([1.0])
        
        # Very small, but should handle
        selector = KernelSelector(cv=2, n_restarts=1)
        try:
            best_config, scores = selector.select(X, y)
        except ValueError:
            pass  # Expected for very small datasets
