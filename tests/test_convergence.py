"""Tests for early stopping and convergence detection."""

import numpy as np
import pytest

from optiml.convergence import (
    StoppingCriteria,
    StoppingState,
    ConvergenceMonitor,
    PlateauDetector,
    BudgetAdvisor,
    create_convergence_monitor,
)
from optiml import BayesianOptimizer, Space, Real


class TestStoppingCriteria:
    """Tests for StoppingCriteria dataclass."""

    def test_default_values(self):
        """Test default values are None or 5."""
        criteria = StoppingCriteria()
        assert criteria.max_iterations is None
        assert criteria.improvement_threshold is None
        assert criteria.no_improvement_patience is None
        assert criteria.target_value is None
        assert criteria.confidence_level is None
        assert criteria.min_iterations == 5

    def test_custom_values(self):
        """Test custom values."""
        criteria = StoppingCriteria(
            max_iterations=100,
            improvement_threshold=0.01,
            no_improvement_patience=10,
            target_value=0.5,
            confidence_level=0.95,
            min_iterations=10,
        )
        assert criteria.max_iterations == 100
        assert criteria.improvement_threshold == 0.01
        assert criteria.no_improvement_patience == 10
        assert criteria.target_value == 0.5
        assert criteria.confidence_level == 0.95
        assert criteria.min_iterations == 10


class TestStoppingState:
    """Tests for StoppingState dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        state = StoppingState()
        assert state.iteration == 0
        assert state.best_value == float('-inf')
        assert state.best_iteration == 0
        assert state.no_improvement_count == 0
        assert state.history == []
        assert state.should_stop is False
        assert state.stop_reason == ""


class TestConvergenceMonitor:
    """Tests for ConvergenceMonitor class."""

    def test_max_iterations_stopping(self):
        """Test stopping after max iterations."""
        criteria = StoppingCriteria(max_iterations=5)
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        for i, val in enumerate(values):
            should_stop = monitor.update(val)
            if i >= 4:  # Should stop at 5th iteration
                assert should_stop
                break
        
        assert "Max iterations" in monitor.stop_reason

    def test_no_improvement_patience_stopping(self):
        """Test stopping after no improvement for patience iterations."""
        criteria = StoppingCriteria(
            no_improvement_patience=3,
            min_iterations=1,
        )
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        # First value is best, then worse values
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        for val in values:
            if monitor.update(val):
                break
        
        assert monitor.should_stop
        assert "No improvement" in monitor.stop_reason

    def test_target_value_stopping_maximize(self):
        """Test stopping when target reached (maximize)."""
        criteria = StoppingCriteria(
            target_value=10.0,
            min_iterations=1,
        )
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        values = [1.0, 5.0, 10.0, 15.0]
        for val in values:
            if monitor.update(val):
                break
        
        assert monitor.should_stop
        assert "Target value" in monitor.stop_reason

    def test_target_value_stopping_minimize(self):
        """Test stopping when target reached (minimize)."""
        criteria = StoppingCriteria(
            target_value=1.0,
            min_iterations=1,
        )
        monitor = ConvergenceMonitor(criteria, maximize=False)
        
        values = [10.0, 5.0, 0.5]
        for val in values:
            if monitor.update(val):
                break
        
        assert monitor.should_stop
        assert "Target value" in monitor.stop_reason

    def test_minimize_tracks_best(self):
        """Test that best value is tracked correctly for minimization."""
        criteria = StoppingCriteria(max_iterations=10)
        monitor = ConvergenceMonitor(criteria, maximize=False)
        
        values = [10.0, 5.0, 3.0, 1.0, 2.0]
        for val in values:
            monitor.update(val)
        
        assert monitor.state.best_value == 1.0
        assert monitor.state.best_iteration == 4

    def test_maximize_tracks_best(self):
        """Test that best value is tracked correctly for maximization."""
        criteria = StoppingCriteria(max_iterations=10)
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        values = [1.0, 5.0, 3.0, 10.0, 2.0]
        for val in values:
            monitor.update(val)
        
        assert monitor.state.best_value == 10.0
        assert monitor.state.best_iteration == 4

    def test_reset(self):
        """Test resetting the monitor."""
        criteria = StoppingCriteria(max_iterations=10)
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        # Add some values
        for val in [1.0, 2.0, 3.0]:
            monitor.update(val)
        
        # Reset
        monitor.reset()
        
        assert monitor.state.iteration == 0
        assert monitor.state.history == []
        assert monitor.state.best_value == float('-inf')

    def test_get_convergence_stats(self):
        """Test convergence statistics."""
        criteria = StoppingCriteria(max_iterations=20)
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5]
        for val in values:
            monitor.update(val)
        
        stats = monitor.get_convergence_stats()
        
        assert stats["iteration"] == 10
        assert stats["best_value"] == 5.5
        assert "recent_mean" in stats
        assert "recent_std" in stats
        assert "recent_cv" in stats

    def test_confidence_level_stopping(self):
        """Test stopping based on confidence level."""
        criteria = StoppingCriteria(
            confidence_level=0.99,  # Very high confidence required
            min_iterations=5,
        )
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        # Flat values should have low variance -> high confidence
        values = [10.0] * 15
        for val in values:
            if monitor.update(val):
                break
        
        assert monitor.should_stop
        assert "Confidence level" in monitor.stop_reason

    def test_min_iterations_respected(self):
        """Test that min_iterations is respected."""
        criteria = StoppingCriteria(
            target_value=5.0,
            min_iterations=10,
        )
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        # Should not stop before min_iterations even if target reached
        for i, val in enumerate([10.0] * 15):
            should_stop = monitor.update(val)
            if i < 9:  # First 10 iterations
                assert not should_stop


class TestPlateauDetector:
    """Tests for PlateauDetector class."""

    def test_detects_plateau(self):
        """Test that plateau is detected with constant values."""
        detector = PlateauDetector(window_size=5, threshold=0.01, min_iterations=5)
        
        # Flat values
        for val in [10.0] * 10:
            detector.update(val)
        
        assert detector.is_plateau

    def test_no_plateau_with_variance(self):
        """Test that varying values don't trigger plateau."""
        detector = PlateauDetector(window_size=5, threshold=0.01, min_iterations=5)
        
        # Values with high variance
        values = [1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0]
        for val in values:
            detector.update(val)
        
        assert not detector.is_plateau

    def test_needs_enough_history(self):
        """Test that detector needs enough history."""
        detector = PlateauDetector(window_size=10, min_iterations=10)
        
        for val in [10.0] * 5:
            detector.update(val)
        
        assert not detector.is_plateau  # Not enough history

    def test_reset(self):
        """Test resetting the detector."""
        detector = PlateauDetector(window_size=5, min_iterations=5)
        
        for val in [10.0] * 10:
            detector.update(val)
        
        detector.reset()
        
        assert detector.history == []
        assert not detector.is_plateau

    def test_get_plateau_info(self):
        """Test getting plateau info."""
        detector = PlateauDetector(window_size=5, min_iterations=5)
        
        for val in [10.0] * 10:
            detector.update(val)
        
        info = detector.get_plateau_info()
        
        assert "is_plateau" in info
        assert "coefficient_of_variation" in info
        assert "window_mean" in info
        assert "n_observations" in info
        assert info["n_observations"] == 10


class TestBudgetAdvisor:
    """Tests for BudgetAdvisor class."""

    def test_recommend_budget_basic(self):
        """Test basic budget recommendation."""
        advisor = BudgetAdvisor(n_dims=3)
        budget = advisor.recommend_budget()
        
        assert budget > 0
        assert isinstance(budget, int)

    def test_budget_scales_with_dims(self):
        """Test that budget increases with dimensionality."""
        advisor_low = BudgetAdvisor(n_dims=2)
        advisor_high = BudgetAdvisor(n_dims=10)
        
        budget_low = advisor_low.recommend_budget()
        budget_high = advisor_high.recommend_budget()
        
        assert budget_high > budget_low

    def test_budget_increases_with_constraints(self):
        """Test that constraints increase budget."""
        advisor_no = BudgetAdvisor(n_dims=5, has_constraints=False)
        advisor_yes = BudgetAdvisor(n_dims=5, has_constraints=True)
        
        budget_no = advisor_no.recommend_budget()
        budget_yes = advisor_yes.recommend_budget()
        
        assert budget_yes > budget_no

    def test_budget_increases_with_accuracy(self):
        """Test that higher accuracy requires more budget."""
        advisor = BudgetAdvisor(n_dims=5)
        
        budget_low = advisor.recommend_budget(target_accuracy=0.6)
        budget_high = advisor.recommend_budget(target_accuracy=0.95)
        
        assert budget_high > budget_low

    def test_budget_with_categorical(self):
        """Test budget with categorical variables."""
        advisor = BudgetAdvisor(n_dims=5, n_categorical=2)
        budget = advisor.recommend_budget()
        
        assert budget > 0

    def test_recommend_initial_samples(self):
        """Test initial samples recommendation."""
        advisor = BudgetAdvisor(n_dims=3)
        initial = advisor.recommend_initial_samples()
        
        assert initial >= 5
        assert isinstance(initial, int)

    def test_initial_samples_scales(self):
        """Test that initial samples scale with dims."""
        advisor_low = BudgetAdvisor(n_dims=2)
        advisor_high = BudgetAdvisor(n_dims=10)
        
        initial_low = advisor_low.recommend_initial_samples()
        initial_high = advisor_high.recommend_initial_samples()
        
        assert initial_high > initial_low

    def test_recommend_batch_size(self):
        """Test batch size recommendation."""
        advisor = BudgetAdvisor(n_dims=5)
        batch = advisor.recommend_batch_size()
        
        assert batch > 0
        assert isinstance(batch, int)

    def test_batch_size_with_parallel(self):
        """Test batch size with parallel workers."""
        advisor = BudgetAdvisor(n_dims=5)
        
        batch = advisor.recommend_batch_size(n_parallel=4)
        assert batch <= 4

    def test_estimate_time_to_convergence(self):
        """Test time estimation."""
        advisor = BudgetAdvisor(n_dims=5)
        times = advisor.estimate_time_to_convergence(evaluation_time=10.0)
        
        assert "budget" in times
        assert "sequential_seconds" in times
        assert "sequential_hours" in times
        assert times["budget"] > 0
        assert times["sequential_seconds"] > 0

    def test_complexity_factor(self):
        """Test that complexity affects budget."""
        advisor = BudgetAdvisor(n_dims=5)
        
        budget_low = advisor.recommend_budget(problem_complexity="low")
        budget_high = advisor.recommend_budget(problem_complexity="high")
        
        assert budget_high > budget_low


class TestCreateConvergenceMonitor:
    """Tests for create_convergence_monitor factory function."""

    def test_basic_creation(self):
        """Test basic monitor creation."""
        monitor = create_convergence_monitor(max_iterations=50)
        
        assert isinstance(monitor, ConvergenceMonitor)
        assert monitor.criteria.max_iterations == 50

    def test_all_parameters(self):
        """Test creation with all parameters."""
        monitor = create_convergence_monitor(
            max_iterations=100,
            improvement_threshold=0.01,
            patience=10,
            target=0.95,
            confidence=0.99,
            maximize=False,
        )
        
        assert monitor.criteria.max_iterations == 100
        assert monitor.criteria.improvement_threshold == 0.01
        assert monitor.criteria.no_improvement_patience == 10
        assert monitor.criteria.target_value == 0.95
        assert monitor.criteria.confidence_level == 0.99
        assert not monitor.maximize


class TestIntegrationWithOptimizer:
    """Integration tests with BayesianOptimizer."""

    def test_optimize_with_early_stopping(self):
        """Test optimization with early stopping."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space=space, random_state=42)
        
        criteria = StoppingCriteria(
            max_iterations=20,
            target_value=0.01,
        )
        monitor = ConvergenceMonitor(criteria, maximize=False)
        
        for i in range(100):  # Max 100 iterations
            x = optimizer.suggest()
            # Objective: minimize x^2, optimal at x=0
            y = float(x[0] ** 2)
            optimizer.tell([x], [y])
            
            if monitor.update(y):
                break
        
        # Should stop before 100 iterations
        assert monitor.state.iteration <= 20
        
        # Should have found a good solution
        assert monitor.state.best_value < 0.5

    def test_no_improvement_stopping(self):
        """Test stopping on no improvement."""
        space = Space([Real(-5, 5, name="x")])
        optimizer = BayesianOptimizer(space=space, random_state=42)
        
        criteria = StoppingCriteria(
            no_improvement_patience=5,
            min_iterations=3,
        )
        monitor = ConvergenceMonitor(criteria, maximize=False)
        
        for i in range(50):
            x = optimizer.suggest()
            # Sinc function - has many local minima
            y = np.sinc(x[0])
            optimizer.tell([x], [y])
            
            if monitor.update(y):
                break
        
        # Should stop due to no improvement
        assert monitor.should_stop

    def test_with_plateau_detector(self):
        """Test combining with plateau detector."""
        space = Space([Real(0, 1, name="x")])
        optimizer = BayesianOptimizer(space=space, random_state=42)
        
        detector = PlateauDetector(window_size=5, threshold=0.1, min_iterations=5)
        
        for i in range(30):
            x = optimizer.suggest()
            y = x[0] ** 2
            optimizer.tell([x], [y])
            
            if detector.update(y):
                # On a plateau, might want to adjust exploration
                break
        
        # Just verify it runs without error
        info = detector.get_plateau_info()
        assert "is_plateau" in info


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_history_stats(self):
        """Test statistics with no history."""
        criteria = StoppingCriteria(max_iterations=10)
        monitor = ConvergenceMonitor(criteria)
        
        stats = monitor.get_convergence_stats()
        assert stats["iteration"] == 0

    def test_single_value_stats(self):
        """Test statistics with single value."""
        criteria = StoppingCriteria(max_iterations=10)
        monitor = ConvergenceMonitor(criteria)
        monitor.update(5.0)
        
        stats = monitor.get_convergence_stats()
        assert stats["iteration"] == 1
        assert stats["best_value"] == 5.0

    def test_nan_value(self):
        """Test handling of NaN values."""
        criteria = StoppingCriteria(max_iterations=10)
        monitor = ConvergenceMonitor(criteria)
        
        # NaN should not break the monitor
        monitor.update(1.0)
        monitor.update(float('nan'))
        monitor.update(3.0)
        
        # Just verify it runs
        assert monitor.state.iteration == 3

    def test_inf_value(self):
        """Test handling of infinite values."""
        criteria = StoppingCriteria(max_iterations=10)
        monitor = ConvergenceMonitor(criteria, maximize=True)
        
        monitor.update(1.0)
        monitor.update(float('inf'))
        monitor.update(3.0)
        
        # Inf should become the best for maximization
        assert monitor.state.best_value == float('inf')

    def test_negative_values(self):
        """Test with negative values."""
        criteria = StoppingCriteria(
            target_value=-10.0,
            min_iterations=1,
        )
        monitor = ConvergenceMonitor(criteria, maximize=False)
        
        for val in [-1.0, -5.0, -15.0]:
            if monitor.update(val):
                break
        
        assert monitor.should_stop
        assert monitor.state.best_value == -15.0

    def test_zero_dimensions(self):
        """Test budget advisor with edge case dimensions."""
        # 1 dimension
        advisor = BudgetAdvisor(n_dims=1)
        budget = advisor.recommend_budget()
        assert budget > 0

    def test_high_dimensions(self):
        """Test budget advisor with high dimensions."""
        advisor = BudgetAdvisor(n_dims=50)
        budget = advisor.recommend_budget()
        assert budget > 100  # Should recommend substantial budget

    def test_all_categorical(self):
        """Test with all categorical dimensions."""
        advisor = BudgetAdvisor(n_dims=5, n_categorical=5)
        budget = advisor.recommend_budget()
        assert budget > 0
