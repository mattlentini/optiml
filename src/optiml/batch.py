"""
Batch/parallel acquisition functions for Bayesian optimization.

This module provides acquisition functions for selecting multiple points
simultaneously, enabling parallel experimentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from optiml.surrogate import SurrogateModel
    from optiml.acquisition import AcquisitionFunction


class BatchAcquisitionFunction:
    """Base class for batch acquisition functions."""
    
    def __init__(self, base_acquisition: Optional[AcquisitionFunction] = None):
        """
        Initialize batch acquisition function.
        
        Parameters
        ----------
        base_acquisition : AcquisitionFunction, optional
            Base acquisition function to use. Defaults to ExpectedImprovement.
        """
        if base_acquisition is None:
            from optiml.acquisition import ExpectedImprovement
            base_acquisition = ExpectedImprovement()
        self.base_acquisition = base_acquisition
    
    def suggest_batch(
        self,
        n_points: int,
        surrogate: SurrogateModel,
        y_best: float,
        space,
        n_candidates: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Suggest a batch of points for parallel evaluation.
        
        Parameters
        ----------
        n_points : int
            Number of points to suggest.
        surrogate : SurrogateModel
            Fitted surrogate model.
        y_best : float
            Current best observed value.
        space : Space
            Search space for sampling candidates.
        n_candidates : int
            Number of candidate points to evaluate.
        rng : np.random.Generator, optional
            Random number generator.
            
        Returns
        -------
        np.ndarray
            Array of shape (n_points, n_features) with suggested points.
        """
        raise NotImplementedError


class ConstantLiarBatch(BatchAcquisitionFunction):
    """
    Constant Liar strategy for batch acquisition.
    
    This is a simple but effective greedy heuristic:
    1. Select the best point using the acquisition function
    2. Add it to the batch with a "liar" value
    3. Repeat until batch is full
    
    The "liar" value can be:
    - min: Pessimistic (encourages exploration)
    - max: Optimistic (encourages exploitation)
    - mean: Neutral
    
    Parameters
    ----------
    base_acquisition : AcquisitionFunction, optional
        Base acquisition function.
    strategy : str
        Liar strategy: 'min', 'max', or 'mean'.
    
    References
    ----------
    Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
    Kriging is well-suited to parallelize optimization.
    """
    
    def __init__(
        self,
        base_acquisition: Optional[AcquisitionFunction] = None,
        strategy: str = 'min',
    ):
        super().__init__(base_acquisition)
        if strategy not in ['min', 'max', 'mean']:
            raise ValueError(f"Strategy must be 'min', 'max', or 'mean', got {strategy}")
        self.strategy = strategy
    
    def suggest_batch(
        self,
        n_points: int,
        surrogate: SurrogateModel,
        y_best: float,
        space,
        n_candidates: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Suggest batch using Constant Liar strategy."""
        if rng is None:
            rng = np.random.default_rng()
        
        # Keep track of temporary observations
        X_temp = []
        y_temp = []
        
        batch = []
        
        for _ in range(n_points):
            # Sample candidates
            candidates = space.sample(n_candidates, rng)
            
            # Evaluate acquisition on candidates
            acq_values = self.base_acquisition(candidates, surrogate, y_best)
            
            # Select best
            best_idx = np.argmax(acq_values)
            best_point = candidates[best_idx]
            batch.append(best_point)
            
            # Determine liar value
            if self.strategy == 'min':
                liar_value = y_best - 1.0  # Pessimistic
            elif self.strategy == 'max':
                liar_value = y_best + 1.0  # Optimistic
            else:  # mean
                # Convert to numpy array for prediction
                best_point_array = np.array(best_point).reshape(1, -1)
                mean_pred, _ = surrogate.predict(best_point_array)
                liar_value = mean_pred[0]
            
            # Temporarily update surrogate
            X_temp.append(best_point)
            y_temp.append(liar_value)
            
            # Create temporary surrogate with liar point
            if len(X_temp) > 0:
                # Get original data
                X_orig = np.array(surrogate._X) if hasattr(surrogate, '_X') else np.empty((0, len(best_point)))
                y_orig = np.array(surrogate._y) if hasattr(surrogate, '_y') else np.empty(0)
                
                # Combine with temporary points
                X_combined = np.vstack([X_orig, X_temp])
                y_combined = np.hstack([y_orig, y_temp])
                
                # Refit surrogate
                try:
                    surrogate.fit(X_combined, y_combined)
                except:
                    # If refitting fails, continue with original surrogate
                    pass
        
        return np.array(batch)


class LocalPenalizationBatch(BatchAcquisitionFunction):
    """
    Local Penalization strategy for batch acquisition.
    
    This method modifies the acquisition function to penalize points
    near already-selected batch members, promoting diversity.
    
    acq_penalized(x) = acq(x) * prod_i penalty(distance(x, x_i))
    
    where penalty(d) = (1 - exp(-d^2 / (2*r^2)))^s
    
    Parameters
    ----------
    base_acquisition : AcquisitionFunction, optional
        Base acquisition function.
    penalty_radius : float
        Radius for local penalization (in normalized space).
    penalty_strength : float
        Strength of the penalty (higher = stronger diversity).
    
    References
    ----------
    González, J., Osborne, M., & Lawrence, N. (2016).
    GLASSES: Relieving the myopia of Bayesian optimisation.
    """
    
    def __init__(
        self,
        base_acquisition: Optional[AcquisitionFunction] = None,
        penalty_radius: float = 0.1,
        penalty_strength: float = 2.0,
    ):
        super().__init__(base_acquisition)
        self.penalty_radius = penalty_radius
        self.penalty_strength = penalty_strength
    
    def _compute_penalty(self, X: np.ndarray, batch_points: List[np.ndarray]) -> np.ndarray:
        """Compute penalization based on distance to batch points."""
        if not batch_points:
            return np.ones(len(X))
        
        penalty = np.ones(len(X))
        batch_array = np.array(batch_points)
        
        # Compute distances to all batch points
        distances = cdist(X, batch_array, metric='euclidean')
        
        # Apply penalty function for each batch point
        for i in range(len(batch_points)):
            d = distances[:, i]
            penalty *= (1 - np.exp(-d**2 / (2 * self.penalty_radius**2)))**self.penalty_strength
        
        return penalty
    
    def suggest_batch(
        self,
        n_points: int,
        surrogate: SurrogateModel,
        y_best: float,
        space,
        n_candidates: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Suggest batch using Local Penalization."""
        if rng is None:
            rng = np.random.default_rng()
        
        batch = []
        
        for _ in range(n_points):
            # Sample candidates
            candidates = space.sample(n_candidates, rng)
            
            # Evaluate base acquisition
            acq_values = self.base_acquisition(candidates, surrogate, y_best)
            
            # Apply penalty for proximity to existing batch members
            penalty = self._compute_penalty(candidates, batch)
            penalized_acq = acq_values * penalty
            
            # Select best penalized point
            best_idx = np.argmax(penalized_acq)
            best_point = candidates[best_idx]
            batch.append(best_point)
        
        return np.array(batch)


class qExpectedImprovement(BatchAcquisitionFunction):
    """
    Batch Expected Improvement (q-EI).
    
    This computes the expected improvement of evaluating q points jointly.
    Uses Monte Carlo sampling to approximate the q-EI.
    
    q-EI(X) = E[max(max_i f(x_i) - f_best, 0)]
    
    where the expectation is over the joint GP posterior.
    
    Parameters
    ----------
    base_acquisition : AcquisitionFunction, optional
        Not used, kept for API consistency.
    n_samples : int
        Number of Monte Carlo samples for approximation.
    
    References
    ----------
    Ginsbourger, D., Le Riche, R., & Carraro, L. (2010).
    A multi-points criterion for deterministic parallel global optimization.
    """
    
    def __init__(
        self,
        base_acquisition: Optional[AcquisitionFunction] = None,
        n_samples: int = 100,
    ):
        super().__init__(base_acquisition)
        self.n_samples = n_samples
    
    def evaluate_qei(
        self,
        batch: np.ndarray,
        surrogate: SurrogateModel,
        y_best: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Evaluate q-EI for a batch of points using Monte Carlo.
        
        Parameters
        ----------
        batch : np.ndarray
            Points to evaluate jointly (n_points, n_features).
        surrogate : SurrogateModel
            Fitted surrogate model.
        y_best : float
            Current best value.
        rng : np.random.Generator
            Random number generator.
            
        Returns
        -------
        float
            q-EI value for this batch.
        """
        # Sample from GP posterior at batch points
        samples = surrogate.sample_y(batch, n_samples=self.n_samples, rng=rng)
        
        # For each sample, take the max over the batch
        max_samples = np.max(samples, axis=1)
        
        # Compute improvement
        improvement = np.maximum(max_samples - y_best, 0)
        
        # Return expected improvement
        return np.mean(improvement)
    
    def suggest_batch(
        self,
        n_points: int,
        surrogate: SurrogateModel,
        y_best: float,
        space,
        n_candidates: int = 10000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Suggest batch using q-EI.
        
        Note: This uses a greedy heuristic to build the batch sequentially,
        as exact q-EI optimization is computationally expensive.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Use local penalization as a fast approximation
        lp = LocalPenalizationBatch(
            base_acquisition=self.base_acquisition,
            penalty_radius=0.1,
            penalty_strength=2.0,
        )
        
        return lp.suggest_batch(
            n_points, surrogate, y_best, space, n_candidates, rng
        )


def suggest_batch(
    optimizer,
    n_points: int = 2,
    strategy: str = 'local_penalization',
    **kwargs
) -> List[List]:
    """
    Convenience function to suggest a batch of points.
    
    Parameters
    ----------
    optimizer : BayesianOptimizer
        Fitted optimizer.
    n_points : int
        Number of points in the batch.
    strategy : str
        Batch strategy: 'constant_liar', 'local_penalization', or 'qei'.
    **kwargs
        Additional arguments for the batch acquisition function.
        
    Returns
    -------
    List[List]
        List of suggested parameter vectors.
    
    Examples
    --------
    >>> from optiml import BayesianOptimizer, Space, Real
    >>> from optiml.batch import suggest_batch
    >>> 
    >>> space = Space([Real(0, 1, name='x1'), Real(0, 1, name='x2')])
    >>> optimizer = BayesianOptimizer(space)
    >>> 
    >>> # Get 3 points for parallel evaluation
    >>> batch = suggest_batch(optimizer, n_points=3, strategy='local_penalization')
    """
    # Check if optimizer has enough data
    if len(optimizer._X) < optimizer.n_initial:
        raise ValueError(
            f"Optimizer must have at least {optimizer.n_initial} observations before suggesting batch. "
            f"Currently has {len(optimizer._X)}."
        )
    
    # Ensure surrogate is fitted by calling suggest once (this fits it internally)
    # We need to transform data and fit the surrogate
    X_normalized = optimizer.space.transform(optimizer._X)
    y_array = np.array(optimizer._y)
    
    # Flip sign if minimizing
    if not optimizer.maximize:
        y_array = -y_array
    
    # Fit the surrogate model
    optimizer.surrogate.fit(X_normalized, y_array)
    
    # Create batch acquisition function
    if strategy == 'constant_liar':
        liar_strategy = kwargs.get('liar_strategy', 'min')
        batch_acq = ConstantLiarBatch(
            base_acquisition=optimizer.acquisition,
            strategy=liar_strategy,
        )
    elif strategy == 'local_penalization':
        penalty_radius = kwargs.get('penalty_radius', 0.1)
        penalty_strength = kwargs.get('penalty_strength', 2.0)
        batch_acq = LocalPenalizationBatch(
            base_acquisition=optimizer.acquisition,
            penalty_radius=penalty_radius,
            penalty_strength=penalty_strength,
        )
    elif strategy == 'qei':
        n_samples = kwargs.get('n_samples', 100)
        batch_acq = qExpectedImprovement(
            base_acquisition=optimizer.acquisition,
            n_samples=n_samples,
        )
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            "Use 'constant_liar', 'local_penalization', or 'qei'."
        )
    
    # Get current best value
    if optimizer.maximize:
        y_best = max(y_array) if len(y_array) > 0 else 0.0
    else:
        # Already flipped if minimizing
        y_best = max(y_array) if len(y_array) > 0 else 0.0
    
    # Suggest batch
    batch_array = batch_acq.suggest_batch(
        n_points=n_points,
        surrogate=optimizer.surrogate,
        y_best=y_best,
        space=optimizer.space,
        rng=optimizer._rng,
    )
    
    # Convert to list of lists
    return [list(point) for point in batch_array]
