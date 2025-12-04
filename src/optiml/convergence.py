"""Early stopping and convergence detection for Bayesian optimization.

This module provides utilities for detecting when optimization has converged
and should be stopped, saving computational resources and preventing over-fitting.

Features:
- Multiple convergence criteria
- Budget advisor for estimating required iterations
- Plateau detection
- Target-based stopping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class StoppingCriteria:
    """Configuration for early stopping criteria.
    
    Attributes
    ----------
    max_iterations : int, optional
        Maximum number of iterations. Always stops if reached.
    improvement_threshold : float, optional
        Minimum relative improvement required (e.g., 0.01 = 1%).
    no_improvement_patience : int, optional
        Stop after this many iterations without improvement.
    target_value : float, optional
        Stop when this target value is achieved.
    confidence_level : float, optional
        Stop when this confident at optimum (0-1).
    min_iterations : int, default=5
        Minimum iterations before early stopping is considered.
    """
    max_iterations: Optional[int] = None
    improvement_threshold: Optional[float] = None
    no_improvement_patience: Optional[int] = None
    target_value: Optional[float] = None
    confidence_level: Optional[float] = None
    min_iterations: int = 5


@dataclass
class StoppingState:
    """Internal state for tracking convergence.
    
    Attributes
    ----------
    iteration : int
        Current iteration number.
    best_value : float
        Best objective value seen so far.
    best_iteration : int
        Iteration when best was found.
    no_improvement_count : int
        Consecutive iterations without improvement.
    history : list
        History of objective values.
    should_stop : bool
        Whether optimization should stop.
    stop_reason : str
        Reason for stopping (if applicable).
    """
    iteration: int = 0
    best_value: float = float('-inf')
    best_iteration: int = 0
    no_improvement_count: int = 0
    history: List[float] = field(default_factory=list)
    should_stop: bool = False
    stop_reason: str = ""


class ConvergenceMonitor:
    """Monitors optimization progress and detects convergence.
    
    Parameters
    ----------
    criteria : StoppingCriteria
        Stopping criteria configuration.
    maximize : bool, default=True
        Whether the objective is being maximized.
        
    Examples
    --------
    >>> criteria = StoppingCriteria(
    ...     max_iterations=100,
    ...     improvement_threshold=0.01,
    ...     no_improvement_patience=10,
    ... )
    >>> monitor = ConvergenceMonitor(criteria)
    >>> 
    >>> for i in range(100):
    ...     x = optimizer.suggest()
    ...     y = objective(x)
    ...     optimizer.tell(x, y)
    ...     
    ...     if monitor.update(y):
    ...         print(f"Stopped: {monitor.stop_reason}")
    ...         break
    """
    
    def __init__(
        self,
        criteria: StoppingCriteria,
        maximize: bool = True,
    ) -> None:
        self.criteria = criteria
        self.maximize = maximize
        self.state = StoppingState()
        
        # Initialize best value based on optimization direction
        if maximize:
            self.state.best_value = float('-inf')
        else:
            self.state.best_value = float('inf')
    
    def update(self, value: float) -> bool:
        """Update the monitor with a new observation.
        
        Parameters
        ----------
        value : float
            The observed objective value.
            
        Returns
        -------
        bool
            True if optimization should stop, False otherwise.
        """
        self.state.iteration += 1
        self.state.history.append(value)
        
        # Check if this is an improvement
        is_improvement = self._is_improvement(value)
        
        if is_improvement:
            old_best = self.state.best_value
            self.state.best_value = value
            self.state.best_iteration = self.state.iteration
            self.state.no_improvement_count = 0
            
            # Check relative improvement threshold
            if (self.criteria.improvement_threshold is not None and 
                self.state.iteration > self.criteria.min_iterations and
                old_best != float('-inf') and old_best != float('inf')):
                
                relative_improvement = abs(value - old_best) / (abs(old_best) + 1e-10)
                if relative_improvement < self.criteria.improvement_threshold:
                    self.state.no_improvement_count += 1
        else:
            self.state.no_improvement_count += 1
        
        # Check stopping criteria
        self._check_stopping_criteria(value)
        
        return self.state.should_stop
    
    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement over current best."""
        if self.maximize:
            return value > self.state.best_value
        else:
            return value < self.state.best_value
    
    def _check_stopping_criteria(self, value: float) -> None:
        """Check all stopping criteria."""
        # Skip if not enough iterations
        if self.state.iteration < self.criteria.min_iterations:
            return
        
        # Max iterations
        if (self.criteria.max_iterations is not None and 
            self.state.iteration >= self.criteria.max_iterations):
            self.state.should_stop = True
            self.state.stop_reason = f"Max iterations ({self.criteria.max_iterations}) reached"
            return
        
        # No improvement patience
        if (self.criteria.no_improvement_patience is not None and
            self.state.no_improvement_count >= self.criteria.no_improvement_patience):
            self.state.should_stop = True
            self.state.stop_reason = (
                f"No improvement for {self.criteria.no_improvement_patience} iterations"
            )
            return
        
        # Target value
        if self.criteria.target_value is not None:
            if self.maximize:
                target_reached = value >= self.criteria.target_value
            else:
                target_reached = value <= self.criteria.target_value
            
            if target_reached:
                self.state.should_stop = True
                self.state.stop_reason = f"Target value ({self.criteria.target_value}) achieved"
                return
        
        # Confidence level (simplified check based on recent variance)
        if (self.criteria.confidence_level is not None and
            len(self.state.history) >= 10):
            
            recent = self.state.history[-10:]
            cv = np.std(recent) / (abs(np.mean(recent)) + 1e-10)
            
            # If coefficient of variation is low enough, we're confident
            confidence = 1.0 - cv
            if confidence >= self.criteria.confidence_level:
                self.state.should_stop = True
                self.state.stop_reason = (
                    f"Confidence level ({self.criteria.confidence_level:.0%}) reached"
                )
                return
    
    @property
    def should_stop(self) -> bool:
        """Whether optimization should stop."""
        return self.state.should_stop
    
    @property
    def stop_reason(self) -> str:
        """Reason for stopping."""
        return self.state.stop_reason
    
    def reset(self) -> None:
        """Reset the monitor state."""
        self.state = StoppingState()
        if self.maximize:
            self.state.best_value = float('-inf')
        else:
            self.state.best_value = float('inf')
    
    def get_convergence_stats(self) -> Dict[str, Any]:
        """Get statistics about convergence progress.
        
        Returns
        -------
        dict
            Dictionary with convergence statistics.
        """
        history = self.state.history
        
        if not history:
            return {"iteration": 0}
        
        stats_dict = {
            "iteration": self.state.iteration,
            "best_value": self.state.best_value,
            "best_iteration": self.state.best_iteration,
            "current_value": history[-1],
            "no_improvement_count": self.state.no_improvement_count,
        }
        
        if len(history) >= 2:
            stats_dict["improvement"] = history[-1] - history[-2]
            stats_dict["total_improvement"] = history[-1] - history[0]
        
        if len(history) >= 5:
            recent = history[-5:]
            stats_dict["recent_mean"] = np.mean(recent)
            stats_dict["recent_std"] = np.std(recent)
            stats_dict["recent_cv"] = np.std(recent) / (abs(np.mean(recent)) + 1e-10)
        
        return stats_dict


class PlateauDetector:
    """Detects plateaus in optimization progress.
    
    A plateau is detected when the objective value has not improved
    significantly for a number of iterations.
    
    Parameters
    ----------
    window_size : int, default=10
        Number of recent iterations to consider.
    threshold : float, default=0.01
        Minimum coefficient of variation to not be considered a plateau.
    min_iterations : int, default=10
        Minimum iterations before plateau detection starts.
        
    Examples
    --------
    >>> detector = PlateauDetector(window_size=10, threshold=0.01)
    >>> for y in objective_values:
    ...     if detector.update(y):
    ...         print("Plateau detected - consider stopping or increasing exploration")
    """
    
    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.01,
        min_iterations: int = 10,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.history: List[float] = []
        self.is_plateau = False
    
    def update(self, value: float) -> bool:
        """Update with new observation and check for plateau.
        
        Parameters
        ----------
        value : float
            New objective value.
            
        Returns
        -------
        bool
            True if currently on a plateau.
        """
        self.history.append(value)
        
        if len(self.history) < max(self.window_size, self.min_iterations):
            self.is_plateau = False
            return False
        
        window = self.history[-self.window_size:]
        cv = np.std(window) / (abs(np.mean(window)) + 1e-10)
        
        self.is_plateau = cv < self.threshold
        return self.is_plateau
    
    def get_plateau_info(self) -> Dict[str, Any]:
        """Get information about current plateau status."""
        if len(self.history) < self.window_size:
            return {"is_plateau": False, "n_observations": len(self.history)}
        
        window = self.history[-self.window_size:]
        cv = np.std(window) / (abs(np.mean(window)) + 1e-10)
        
        return {
            "is_plateau": self.is_plateau,
            "coefficient_of_variation": cv,
            "window_mean": np.mean(window),
            "window_std": np.std(window),
            "n_observations": len(self.history),
        }
    
    def reset(self) -> None:
        """Reset the detector."""
        self.history = []
        self.is_plateau = False


class BudgetAdvisor:
    """Advises on the budget needed for optimization.
    
    Provides recommendations on how many evaluations are likely needed
    to find a good optimum based on problem characteristics.
    
    Parameters
    ----------
    n_dims : int
        Number of dimensions in the search space.
    n_categorical : int, default=0
        Number of categorical dimensions.
    has_constraints : bool, default=False
        Whether the problem has constraints.
        
    Examples
    --------
    >>> advisor = BudgetAdvisor(n_dims=5, n_categorical=1)
    >>> budget = advisor.recommend_budget(target_accuracy=0.9)
    >>> print(f"Recommended budget: {budget} evaluations")
    """
    
    def __init__(
        self,
        n_dims: int,
        n_categorical: int = 0,
        has_constraints: bool = False,
    ) -> None:
        self.n_dims = n_dims
        self.n_categorical = n_categorical
        self.has_constraints = has_constraints
    
    def recommend_budget(
        self,
        target_accuracy: float = 0.9,
        problem_complexity: str = "medium",
    ) -> int:
        """Recommend a budget for optimization.
        
        Parameters
        ----------
        target_accuracy : float, default=0.9
            Probability of finding a near-optimal solution (0-1).
        problem_complexity : str, default="medium"
            Subjective problem complexity: "low", "medium", "high".
            
        Returns
        -------
        int
            Recommended number of function evaluations.
        """
        # Base budget scales with dimensionality
        # Rule of thumb: ~10 * n_dims for low dimensions, more for high
        n_continuous = self.n_dims - self.n_categorical
        
        if n_continuous <= 3:
            base = 10 * n_continuous
        elif n_continuous <= 10:
            base = 15 * n_continuous
        else:
            base = 20 * n_continuous
        
        # Adjust for categorical variables
        # Each categorical variable can multiply the search space
        categorical_factor = 1.0 + 0.2 * self.n_categorical
        
        # Adjust for constraints
        constraint_factor = 1.5 if self.has_constraints else 1.0
        
        # Adjust for complexity
        complexity_factors = {"low": 0.7, "medium": 1.0, "high": 1.5}
        complexity_factor = complexity_factors.get(problem_complexity.lower(), 1.0)
        
        # Adjust for target accuracy
        # Higher accuracy requires more samples
        accuracy_factor = 1.0 + (target_accuracy - 0.5) * 2
        
        # Compute recommended budget
        budget = int(
            base * 
            categorical_factor * 
            constraint_factor * 
            complexity_factor * 
            accuracy_factor
        )
        
        # Minimum budget
        budget = max(budget, 10)
        
        return budget
    
    def recommend_initial_samples(self) -> int:
        """Recommend number of initial random samples.
        
        Returns
        -------
        int
            Recommended number of initial samples before Bayesian optimization.
        """
        # Rule of thumb: 2*n_dims to 5*n_dims
        n_continuous = self.n_dims - self.n_categorical
        
        if n_continuous <= 5:
            return max(5, 2 * self.n_dims + 1)
        else:
            return max(10, 3 * self.n_dims)
    
    def recommend_batch_size(
        self,
        n_parallel: Optional[int] = None,
    ) -> int:
        """Recommend batch size for parallel optimization.
        
        Parameters
        ----------
        n_parallel : int, optional
            Number of parallel workers available.
            
        Returns
        -------
        int
            Recommended batch size.
        """
        if n_parallel is not None:
            return min(n_parallel, 10)
        
        # Default recommendation based on dimensionality
        if self.n_dims <= 3:
            return 3
        elif self.n_dims <= 10:
            return 5
        else:
            return min(10, self.n_dims)
    
    def estimate_time_to_convergence(
        self,
        evaluation_time: float,
        target_accuracy: float = 0.9,
    ) -> Dict[str, float]:
        """Estimate time to convergence.
        
        Parameters
        ----------
        evaluation_time : float
            Time (in seconds) for one function evaluation.
        target_accuracy : float
            Target accuracy (0-1).
            
        Returns
        -------
        dict
            Dictionary with time estimates.
        """
        budget = self.recommend_budget(target_accuracy)
        
        # Sequential time
        sequential_time = budget * evaluation_time
        
        # With different batch sizes
        batch_sizes = [2, 4, 8]
        times = {}
        
        for batch_size in batch_sizes:
            n_batches = int(np.ceil(budget / batch_size))
            times[f"batch_{batch_size}"] = n_batches * evaluation_time
        
        return {
            "budget": budget,
            "sequential_seconds": sequential_time,
            "sequential_hours": sequential_time / 3600,
            **{k: v for k, v in times.items()},
        }


def create_convergence_monitor(
    max_iterations: Optional[int] = None,
    improvement_threshold: Optional[float] = None,
    patience: Optional[int] = None,
    target: Optional[float] = None,
    confidence: Optional[float] = None,
    maximize: bool = True,
) -> ConvergenceMonitor:
    """Create a convergence monitor with common settings.
    
    Parameters
    ----------
    max_iterations : int, optional
        Maximum iterations.
    improvement_threshold : float, optional
        Minimum relative improvement.
    patience : int, optional
        Iterations without improvement before stopping.
    target : float, optional
        Target value to achieve.
    confidence : float, optional
        Confidence level required.
    maximize : bool
        Whether maximizing the objective.
        
    Returns
    -------
    ConvergenceMonitor
        Configured convergence monitor.
        
    Examples
    --------
    >>> monitor = create_convergence_monitor(
    ...     max_iterations=50,
    ...     patience=10,
    ...     target=0.95,
    ... )
    """
    criteria = StoppingCriteria(
        max_iterations=max_iterations,
        improvement_threshold=improvement_threshold,
        no_improvement_patience=patience,
        target_value=target,
        confidence_level=confidence,
    )
    
    return ConvergenceMonitor(criteria, maximize=maximize)
