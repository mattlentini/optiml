"""Tests for the robust optimization module."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from optiml import (
    Space,
    Real,
    RiskMeasure,
    UncertaintySet,
    RobustResult,
    compute_cvar,
    compute_var,
    compute_mean_variance,
    compute_entropic_risk,
    RobustAcquisition,
    RobustExpectedImprovement,
    WorstCaseAcquisition,
    CVaRAcquisition,
    RobustOptimizer,
    robust_evaluation,
    DistributionallyRobustOptimizer,
    create_robust_optimizer,
)


# Fixtures
@pytest.fixture
def simple_space():
    """Create a simple 2D search space."""
    return Space([
        Real(0.0, 1.0, name="x1"),
        Real(0.0, 1.0, name="x2"),
    ])


@pytest.fixture
def simple_gp():
    """Create a simple trained GP for testing."""
    np.random.seed(42)
    X = np.random.uniform(0, 1, (20, 2))
    y = np.sin(X[:, 0] * np.pi) + 0.5 * np.cos(X[:, 1] * np.pi) + np.random.normal(0, 0.1, 20)
    
    kernel = RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
    gp.fit(X, y)
    
    return gp, X, y


@pytest.fixture
def uncertainty_set():
    """Create a simple uncertainty set."""
    return UncertaintySet(
        type="box",
        center=np.zeros(2),
        radius=0.05
    )


# RiskMeasure Tests
class TestRiskMeasure:
    """Tests for RiskMeasure enum."""
    
    def test_all_values(self):
        """Test all risk measure values exist."""
        assert RiskMeasure.MEAN.value == "mean"
        assert RiskMeasure.WORST_CASE.value == "worst_case"
        assert RiskMeasure.CVAR.value == "cvar"
        assert RiskMeasure.VAR.value == "var"
        assert RiskMeasure.MEAN_VARIANCE.value == "mean_variance"
        assert RiskMeasure.ENTROPIC.value == "entropic"


# UncertaintySet Tests
class TestUncertaintySet:
    """Tests for UncertaintySet."""
    
    def test_creation_box(self):
        """Test basic box creation."""
        unc = UncertaintySet(
            type="box",
            center=np.array([0.0, 0.0]),
            radius=0.1
        )
        assert unc.type == "box"
        assert unc.radius == 0.1
    
    def test_creation_ellipsoidal(self):
        """Test ellipsoidal creation."""
        unc = UncertaintySet(
            type="ellipsoidal",
            center=np.array([0.0, 0.0]),
            radius=0.1,
            covariance=np.eye(2) * 0.01
        )
        assert unc.type == "ellipsoidal"
        assert unc.covariance is not None
    
    def test_creation_probabilistic(self):
        """Test probabilistic creation."""
        unc = UncertaintySet(
            type="probabilistic",
            center=np.array([0.0, 0.0]),
            radius=0.1
        )
        assert unc.type == "probabilistic"
    
    def test_sample_box(self):
        """Test sampling from box uncertainty."""
        unc = UncertaintySet(
            type="box",
            center=np.array([0.5, 0.5]),
            radius=0.1
        )
        
        samples = unc.sample(100, random_state=42)
        
        assert samples.shape == (100, 2)
        # Samples should be within box
        assert np.all(samples[:, 0] >= 0.4)
        assert np.all(samples[:, 0] <= 0.6)
        assert np.all(samples[:, 1] >= 0.4)
        assert np.all(samples[:, 1] <= 0.6)
    
    def test_sample_ellipsoidal(self):
        """Test sampling from ellipsoidal uncertainty."""
        unc = UncertaintySet(
            type="ellipsoidal",
            center=np.array([0.5, 0.5]),
            radius=0.1
        )
        
        samples = unc.sample(100, random_state=42)
        
        assert samples.shape == (100, 2)
        # Samples should be close to center
        distances = np.linalg.norm(samples - np.array([0.5, 0.5]), axis=1)
        assert np.all(distances <= 0.15)  # All within radius
    
    def test_sample_probabilistic(self):
        """Test sampling from probabilistic uncertainty."""
        unc = UncertaintySet(
            type="probabilistic",
            center=np.array([0.5, 0.5]),
            radius=0.1
        )
        
        samples = unc.sample(500, random_state=42)
        
        assert samples.shape == (500, 2)
        # Mean should be close to center
        assert abs(samples[:, 0].mean() - 0.5) < 0.03
        assert abs(samples[:, 1].mean() - 0.5) < 0.03
    
    def test_sample_with_explicit_samples(self):
        """Test sampling from provided samples."""
        explicit_samples = np.array([
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
        ])
        
        unc = UncertaintySet(
            type="samples",
            center=np.array([0.0, 0.0]),
            samples=explicit_samples
        )
        
        samples = unc.sample(100, random_state=42)
        
        assert samples.shape == (100, 2)
        # All samples should be from the explicit set
        for sample in samples:
            assert any(np.allclose(sample, exp) for exp in explicit_samples)


# Risk Measure Functions Tests
class TestRiskFunctions:
    """Tests for risk measure computation functions."""
    
    def test_compute_cvar_minimize(self):
        """Test CVaR computation for minimization."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 10.0])  # One bad value
        cvar = compute_cvar(values, alpha=0.2, minimize=True)
        
        # Should be around average of worst 20% (value 10)
        assert cvar >= 4.0
        assert np.isfinite(cvar)
    
    def test_compute_cvar_maximize(self):
        """Test CVaR computation for maximization."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cvar = compute_cvar(values, alpha=0.2, minimize=False)
        
        # Should be around average of worst 20% (value 1)
        assert cvar <= 2.0
    
    def test_compute_var_minimize(self):
        """Test VaR computation for minimization."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = compute_var(values, alpha=0.2, minimize=True)
        
        # 80th percentile
        assert var >= 4.0
    
    def test_compute_mean_variance(self):
        """Test mean-variance computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mv = compute_mean_variance(values, lambda_var=0.5)
        
        expected = np.mean(values) + 0.5 * np.var(values)
        assert abs(mv - expected) < 1e-10
    
    def test_compute_entropic_risk(self):
        """Test entropic risk computation."""
        values = np.array([1.0, 2.0, 3.0])
        er = compute_entropic_risk(values, theta=1.0, minimize=True)
        
        expected = np.log(np.mean(np.exp(values)))
        assert abs(er - expected) < 1e-10
        assert np.isfinite(er)


# RobustResult Tests
class TestRobustResult:
    """Tests for RobustResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        result = RobustResult(
            optimal_x=np.array([0.5, 0.5]),
            robust_value=0.1,
            mean_value=0.2,
            worst_case_value=0.5,
            confidence_interval=(0.05, 0.35),
            n_evaluations=100
        )
        
        assert result.robust_value == 0.1
        assert result.mean_value == 0.2
        assert result.n_evaluations == 100


# Robust Acquisition Functions Tests
class TestRobustExpectedImprovement:
    """Tests for RobustExpectedImprovement."""
    
    def test_creation(self):
        """Test creation."""
        acq = RobustExpectedImprovement(
            best_f=0.5,
            risk_measure=RiskMeasure.CVAR,
            alpha=0.1
        )
        assert acq.best_f == 0.5
        assert acq.risk_measure == RiskMeasure.CVAR
    
    def test_call(self, simple_gp, uncertainty_set):
        """Test acquisition function call."""
        gp, _, y = simple_gp
        acq = RobustExpectedImprovement(
            best_f=y.min(),
            risk_measure=RiskMeasure.CVAR
        )
        
        x = np.array([0.5, 0.5])
        value = acq(x, gp, uncertainty_set, n_samples=50)
        
        assert np.isfinite(value)
    
    def test_different_risk_measures(self, simple_gp, uncertainty_set):
        """Test with different risk measures."""
        gp, _, y = simple_gp
        x = np.array([0.5, 0.5])
        
        for risk_measure in [RiskMeasure.MEAN, RiskMeasure.CVAR, 
                             RiskMeasure.WORST_CASE, RiskMeasure.VAR]:
            acq = RobustExpectedImprovement(
                best_f=y.min(),
                risk_measure=risk_measure
            )
            value = acq(x, gp, uncertainty_set, n_samples=30)
            assert np.isfinite(value)


class TestWorstCaseAcquisition:
    """Tests for WorstCaseAcquisition."""
    
    def test_creation(self):
        """Test creation."""
        acq = WorstCaseAcquisition(minimize=True)
        assert acq.minimize is True
    
    def test_call_minimize(self, simple_gp, uncertainty_set):
        """Test worst-case for minimization."""
        gp, _, _ = simple_gp
        acq = WorstCaseAcquisition(minimize=True)
        
        x = np.array([0.5, 0.5])
        value = acq(x, gp, uncertainty_set, n_samples=50)
        
        # Should return max (worst for minimization)
        assert np.isfinite(value)
    
    def test_call_maximize(self, simple_gp, uncertainty_set):
        """Test worst-case for maximization."""
        gp, _, _ = simple_gp
        acq = WorstCaseAcquisition(minimize=False)
        
        x = np.array([0.5, 0.5])
        value = acq(x, gp, uncertainty_set, n_samples=50)
        
        # Should return min (worst for maximization)
        assert np.isfinite(value)


class TestCVaRAcquisition:
    """Tests for CVaRAcquisition."""
    
    def test_creation(self):
        """Test creation."""
        acq = CVaRAcquisition(alpha=0.1, minimize=True)
        assert acq.alpha == 0.1
    
    def test_call(self, simple_gp, uncertainty_set):
        """Test CVaR acquisition call."""
        gp, _, _ = simple_gp
        acq = CVaRAcquisition(alpha=0.1)
        
        x = np.array([0.5, 0.5])
        value = acq(x, gp, uncertainty_set, n_samples=50)
        
        assert np.isfinite(value)


# RobustOptimizer Tests
class TestRobustOptimizer:
    """Tests for RobustOptimizer."""
    
    def test_creation(self, simple_space):
        """Test optimizer creation."""
        uncertainty = UncertaintySet(
            type="box",
            center=np.zeros(2),
            radius=0.05
        )
        
        opt = RobustOptimizer(
            space=simple_space,
            uncertainty=uncertainty,
            risk_measure=RiskMeasure.CVAR
        )
        
        assert opt.risk_measure == RiskMeasure.CVAR
    
    def test_creation_with_dict(self, simple_space):
        """Test creation with dict uncertainty."""
        opt = RobustOptimizer(
            space=simple_space,
            uncertainty={"type": "box", "center": np.zeros(2), "radius": 0.05}
        )
        
        assert opt.uncertainty.type == "box"
    
    def test_suggest_initial(self, simple_space):
        """Test initial suggestions."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.05)
        opt = RobustOptimizer(simple_space, uncertainty)
        
        x = opt.suggest()
        
        assert len(x) == 2
        assert 0 <= x[0] <= 1
        assert 0 <= x[1] <= 1
    
    def test_tell(self, simple_space):
        """Test recording observations."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.05)
        opt = RobustOptimizer(simple_space, uncertainty)
        
        x = np.array([0.5, 0.5])
        opt.tell(x, 0.5)
        
        assert len(opt._X) == 1
        assert len(opt._y) == 1
    
    def test_tell_with_samples(self, simple_space):
        """Test recording with sample distribution."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.05)
        opt = RobustOptimizer(simple_space, uncertainty, risk_measure=RiskMeasure.CVAR)
        
        x = np.array([0.5, 0.5])
        y_samples = np.array([0.4, 0.5, 0.6, 0.8, 1.0])
        opt.tell(x, np.mean(y_samples), y_samples)
        
        assert len(opt._y_robust) == 1
        # CVaR should be different from mean
        assert opt._y_robust[0] != np.mean(y_samples)
    
    def test_suggest_after_observations(self, simple_space):
        """Test suggestions after multiple observations."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.05)
        opt = RobustOptimizer(simple_space, uncertainty)
        
        # Add initial observations
        for i in range(6):
            x = opt.suggest()
            y = x[0]**2 + x[1]**2
            opt.tell(x, y)
        
        # Now suggest should use model
        x = opt.suggest()
        assert len(x) == 2
    
    def test_get_best(self, simple_space):
        """Test getting best solution."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.05)
        opt = RobustOptimizer(simple_space, uncertainty)
        
        # Add observations
        opt.tell(np.array([0.5, 0.5]), 0.5)
        opt.tell(np.array([0.1, 0.1]), 0.02)
        opt.tell(np.array([0.8, 0.8]), 1.28)
        
        best_x, best_y, robust_y = opt.get_best()
        
        # Best should be (0.1, 0.1) with y=0.02
        np.testing.assert_array_almost_equal(best_x, [0.1, 0.1])
    
    def test_optimize(self, simple_space):
        """Test full optimization."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.02)
        opt = RobustOptimizer(simple_space, uncertainty, random_state=42)
        
        def objective(x):
            return x[0]**2 + x[1]**2
        
        result = opt.optimize(objective, n_iter=10)
        
        assert isinstance(result, RobustResult)
        assert result.n_evaluations == 10
        assert np.isfinite(result.robust_value)
        assert np.isfinite(result.mean_value)
    
    def test_optimize_with_samples(self, simple_space):
        """Test optimization with multiple samples per evaluation."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.02)
        opt = RobustOptimizer(simple_space, uncertainty, random_state=42)
        
        def noisy_objective(x):
            return x[0]**2 + x[1]**2 + np.random.normal(0, 0.1)
        
        result = opt.optimize(noisy_objective, n_iter=8, n_samples_per_eval=3)
        
        assert result.n_evaluations == 24  # 8 * 3
    
    def test_risk_measures(self, simple_space):
        """Test different risk measures."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.02)
        
        for measure in [RiskMeasure.MEAN, RiskMeasure.CVAR, 
                        RiskMeasure.WORST_CASE, RiskMeasure.VAR]:
            opt = RobustOptimizer(
                simple_space, 
                uncertainty, 
                risk_measure=measure,
                random_state=42
            )
            
            x = opt.suggest()
            assert len(x) == 2


# robust_evaluation Function Tests
class TestRobustEvaluation:
    """Tests for robust_evaluation function."""
    
    def test_basic_evaluation(self):
        """Test basic robust evaluation."""
        def objective(x):
            return x[0]**2 + x[1]**2
        
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.05)
        
        result = robust_evaluation(
            objective,
            np.array([0.5, 0.5]),
            uncertainty,
            n_samples=100
        )
        
        assert "mean" in result
        assert "std" in result
        assert "cvar" in result
        assert "var" in result
        assert "worst_case" in result
        assert "best_case" in result
        
        # Mean should be close to true value
        assert abs(result["mean"] - 0.5) < 0.1


# DistributionallyRobustOptimizer Tests
class TestDistributionallyRobustOptimizer:
    """Tests for DistributionallyRobustOptimizer."""
    
    def test_creation(self, simple_space):
        """Test creation."""
        def ref_dist():
            return np.random.normal(0, 0.1, 2)
        
        opt = DistributionallyRobustOptimizer(
            space=simple_space,
            reference_distribution=ref_dist,
            wasserstein_radius=0.1
        )
        
        assert opt.wasserstein_radius == 0.1
    
    def test_suggest(self, simple_space):
        """Test suggestion."""
        def ref_dist():
            return np.random.normal(0, 0.1, 2)
        
        opt = DistributionallyRobustOptimizer(
            simple_space,
            reference_distribution=ref_dist,
            random_state=42
        )
        
        x = opt.suggest()
        
        assert len(x) == 2
    
    def test_tell(self, simple_space):
        """Test recording observations."""
        def ref_dist():
            return np.random.normal(0, 0.1, 2)
        
        opt = DistributionallyRobustOptimizer(
            simple_space,
            reference_distribution=ref_dist
        )
        
        opt.tell(np.array([0.5, 0.5]), 0.5)
        
        assert len(opt._X) == 1


# create_robust_optimizer Factory Tests
class TestCreateRobustOptimizer:
    """Tests for create_robust_optimizer factory."""
    
    def test_create_cvar(self, simple_space):
        """Test creating CVaR optimizer."""
        opt = create_robust_optimizer(
            simple_space,
            uncertainty_type="box",
            uncertainty_radius=0.1,
            risk_measure="cvar"
        )
        
        assert isinstance(opt, RobustOptimizer)
        assert opt.risk_measure == RiskMeasure.CVAR
    
    def test_create_worst_case(self, simple_space):
        """Test creating worst-case optimizer."""
        opt = create_robust_optimizer(
            simple_space,
            risk_measure="worst_case"
        )
        
        assert opt.risk_measure == RiskMeasure.WORST_CASE
    
    def test_create_mean_variance(self, simple_space):
        """Test creating mean-variance optimizer."""
        opt = create_robust_optimizer(
            simple_space,
            risk_measure="mean_variance"
        )
        
        assert opt.risk_measure == RiskMeasure.MEAN_VARIANCE
    
    def test_create_ellipsoidal(self, simple_space):
        """Test creating with ellipsoidal uncertainty."""
        opt = create_robust_optimizer(
            simple_space,
            uncertainty_type="ellipsoidal",
            uncertainty_radius=0.05
        )
        
        assert opt.uncertainty.type == "ellipsoidal"
    
    def test_full_workflow(self, simple_space):
        """Test complete workflow with factory."""
        opt = create_robust_optimizer(
            simple_space,
            uncertainty_type="box",
            uncertainty_radius=0.02,
            risk_measure="cvar",
            alpha=0.1,
            random_state=42
        )
        
        def objective(x):
            return x[0]**2 + x[1]**2
        
        result = opt.optimize(objective, n_iter=8)
        
        assert result.robust_value >= 0
        assert result.n_evaluations == 8


# Edge Cases
class TestRobustEdgeCases:
    """Edge case tests for robust optimization."""
    
    def test_zero_radius_uncertainty(self, simple_space):
        """Test with zero uncertainty radius."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.0)
        opt = RobustOptimizer(simple_space, uncertainty)
        
        x = opt.suggest()
        opt.tell(x, x[0]**2 + x[1]**2)
        
        assert len(opt._X) == 1
    
    def test_single_dimension(self):
        """Test with single dimension."""
        space = Space([Real(0.0, 1.0, name="x")])
        uncertainty = UncertaintySet(type="box", center=np.zeros(1), radius=0.05)
        
        opt = RobustOptimizer(space, uncertainty, random_state=42)
        
        def objective(x):
            return (x[0] - 0.5)**2
        
        result = opt.optimize(objective, n_iter=8)
        
        assert np.isfinite(result.robust_value)
    
    def test_maximize(self, simple_space):
        """Test maximization mode."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.02)
        opt = RobustOptimizer(
            simple_space, 
            uncertainty, 
            minimize=False,
            random_state=42
        )
        
        def objective(x):
            return -((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
        
        result = opt.optimize(objective, n_iter=8)
        
        assert result.robust_value <= 0  # Maximum of negative function
    
    def test_large_uncertainty(self, simple_space):
        """Test with large uncertainty set."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.3)
        opt = RobustOptimizer(simple_space, uncertainty, random_state=42)
        
        def objective(x):
            return x[0]**2 + x[1]**2
        
        result = opt.optimize(objective, n_iter=8)
        
        # Should still work
        assert np.isfinite(result.robust_value)
    
    def test_get_best_empty(self, simple_space):
        """Test get_best with no observations."""
        uncertainty = UncertaintySet(type="box", center=np.zeros(2), radius=0.05)
        opt = RobustOptimizer(simple_space, uncertainty)
        
        with pytest.raises(ValueError):
            opt.get_best()


# Integration Tests
class TestRobustIntegration:
    """Integration tests for robust optimization."""
    
    def test_noisy_optimization(self, simple_space):
        """Test optimization with noisy objective."""
        np.random.seed(42)
        
        uncertainty = UncertaintySet(type="probabilistic", center=np.zeros(2), radius=0.05)
        opt = RobustOptimizer(
            simple_space,
            uncertainty,
            risk_measure=RiskMeasure.CVAR,
            alpha=0.1,
            random_state=42
        )
        
        def noisy_objective(x):
            true_value = (x[0] - 0.5)**2 + (x[1] - 0.5)**2
            return true_value + np.random.normal(0, 0.1)
        
        result = opt.optimize(noisy_objective, n_iter=15, n_samples_per_eval=3)
        
        assert result.n_evaluations == 45
        assert np.isfinite(result.robust_value)
    
    def test_comparison_risk_measures(self, simple_space):
        """Compare different risk measures on same problem."""
        np.random.seed(42)
        
        def objective(x):
            return x[0]**2 + x[1]**2 + np.random.normal(0, 0.1)
        
        results = {}
        for measure in ["mean", "cvar", "worst_case"]:
            opt = create_robust_optimizer(
                simple_space,
                risk_measure=measure,
                random_state=42
            )
            result = opt.optimize(objective, n_iter=10)
            results[measure] = result.robust_value
        
        # All should be finite
        for measure, value in results.items():
            assert np.isfinite(value), f"{measure} returned non-finite value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
