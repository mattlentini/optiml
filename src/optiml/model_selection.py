"""Model selection and AutoML for Gaussian Process surrogates.

This module provides automatic model selection, kernel selection, and
hyperparameter tuning for Gaussian Process surrogate models in Bayesian
optimization.

Features:
- Automatic kernel selection via cross-validation
- Hyperparameter optimization with multiple strategies
- Model comparison using information criteria (AIC, BIC)
- Ensemble of GPs for improved predictions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel,
    RBF,
    Matern,
    RationalQuadratic,
    WhiteKernel,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
)


class KernelFamily(Enum):
    """Families of kernels for automatic selection."""
    RBF = "rbf"
    MATERN_12 = "matern_12"
    MATERN_32 = "matern_32"
    MATERN_52 = "matern_52"
    RATIONAL_QUADRATIC = "rational_quadratic"
    PERIODIC = "periodic"
    LINEAR = "linear"
    COMPOSITE = "composite"


@dataclass
class KernelConfig:
    """Configuration for a kernel.
    
    Attributes
    ----------
    family : KernelFamily
        The kernel family.
    length_scale : float or array-like
        Initial length scale(s).
    length_scale_bounds : tuple
        Bounds for length scale optimization.
    additional_params : dict
        Additional kernel-specific parameters.
    """
    family: KernelFamily
    length_scale: Union[float, np.ndarray] = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-5, 1e5)
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelScore:
    """Score for a model configuration.
    
    Attributes
    ----------
    kernel_config : KernelConfig
        The kernel configuration.
    cv_score : float
        Cross-validation score (negative MSE).
    log_marginal_likelihood : float
        Log marginal likelihood of the fitted model.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    n_params : int
        Number of hyperparameters.
    fit_time : float
        Time to fit the model in seconds.
    """
    kernel_config: KernelConfig
    cv_score: float
    log_marginal_likelihood: float
    aic: float
    bic: float
    n_params: int
    fit_time: float = 0.0


def create_kernel(
    config: KernelConfig,
    n_dims: int = 1,
    ard: bool = False,
) -> Kernel:
    """Create a sklearn kernel from configuration.
    
    Parameters
    ----------
    config : KernelConfig
        Kernel configuration.
    n_dims : int
        Number of input dimensions.
    ard : bool
        Whether to use Automatic Relevance Determination (per-dimension length scales).
        
    Returns
    -------
    Kernel
        The sklearn kernel object.
    """
    # Determine length scale
    if ard and n_dims > 1:
        length_scale = np.ones(n_dims) * config.length_scale if np.isscalar(config.length_scale) else config.length_scale
    else:
        length_scale = config.length_scale
    
    bounds = config.length_scale_bounds
    
    # Create base kernel
    if config.family == KernelFamily.RBF:
        base_kernel = RBF(length_scale=length_scale, length_scale_bounds=bounds)
    
    elif config.family == KernelFamily.MATERN_12:
        base_kernel = Matern(length_scale=length_scale, length_scale_bounds=bounds, nu=0.5)
    
    elif config.family == KernelFamily.MATERN_32:
        base_kernel = Matern(length_scale=length_scale, length_scale_bounds=bounds, nu=1.5)
    
    elif config.family == KernelFamily.MATERN_52:
        base_kernel = Matern(length_scale=length_scale, length_scale_bounds=bounds, nu=2.5)
    
    elif config.family == KernelFamily.RATIONAL_QUADRATIC:
        alpha = config.additional_params.get("alpha", 1.0)
        alpha_bounds = config.additional_params.get("alpha_bounds", (1e-5, 1e5))
        base_kernel = RationalQuadratic(
            length_scale=length_scale,
            length_scale_bounds=bounds,
            alpha=alpha,
            alpha_bounds=alpha_bounds,
        )
    
    elif config.family == KernelFamily.PERIODIC:
        periodicity = config.additional_params.get("periodicity", 1.0)
        periodicity_bounds = config.additional_params.get("periodicity_bounds", (1e-5, 1e5))
        base_kernel = ExpSineSquared(
            length_scale=length_scale,
            length_scale_bounds=bounds,
            periodicity=periodicity,
            periodicity_bounds=periodicity_bounds,
        )
    
    elif config.family == KernelFamily.LINEAR:
        sigma_0 = config.additional_params.get("sigma_0", 1.0)
        sigma_0_bounds = config.additional_params.get("sigma_0_bounds", (1e-5, 1e5))
        base_kernel = DotProduct(sigma_0=sigma_0, sigma_0_bounds=sigma_0_bounds)
    
    elif config.family == KernelFamily.COMPOSITE:
        # Composite kernel: RBF + Linear (for non-stationary functions)
        base_kernel = RBF(length_scale=length_scale, length_scale_bounds=bounds) + \
                      DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5))
    
    else:
        raise ValueError(f"Unknown kernel family: {config.family}")
    
    # Add constant kernel for output scale
    constant = config.additional_params.get("constant", 1.0)
    constant_bounds = config.additional_params.get("constant_bounds", (1e-5, 1e5))
    
    # Add white noise kernel
    noise_level = config.additional_params.get("noise_level", 1e-5)
    noise_bounds = config.additional_params.get("noise_bounds", (1e-10, 1e1))
    
    kernel = ConstantKernel(constant, constant_bounds) * base_kernel + \
             WhiteKernel(noise_level, noise_bounds)
    
    return kernel


class KernelSelector:
    """Automatic kernel selection for Gaussian Processes.
    
    Uses cross-validation and/or information criteria to select the best
    kernel from a set of candidates.
    
    Parameters
    ----------
    candidates : list of KernelConfig, optional
        List of kernel configurations to try. If None, uses default set.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default="neg_mean_squared_error"
        Scoring metric for cross-validation.
    use_ard : bool, default=True
        Whether to try ARD (Automatic Relevance Determination) kernels.
    n_restarts : int, default=5
        Number of optimizer restarts for GP fitting.
    random_state : int, optional
        Random state for reproducibility.
        
    Examples
    --------
    >>> selector = KernelSelector(cv=5)
    >>> best_config, scores = selector.select(X, y)
    >>> print(f"Best kernel: {best_config.family}")
    """
    
    def __init__(
        self,
        candidates: Optional[List[KernelConfig]] = None,
        cv: int = 5,
        scoring: str = "neg_mean_squared_error",
        use_ard: bool = True,
        n_restarts: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.candidates = candidates or self._default_candidates()
        self.cv = cv
        self.scoring = scoring
        self.use_ard = use_ard
        self.n_restarts = n_restarts
        self.random_state = random_state
    
    def _default_candidates(self) -> List[KernelConfig]:
        """Return default set of kernel candidates."""
        return [
            KernelConfig(family=KernelFamily.RBF),
            KernelConfig(family=KernelFamily.MATERN_32),
            KernelConfig(family=KernelFamily.MATERN_52),
            KernelConfig(family=KernelFamily.RATIONAL_QUADRATIC),
            KernelConfig(family=KernelFamily.COMPOSITE),
        ]
    
    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        criterion: str = "cv",
    ) -> Tuple[KernelConfig, List[ModelScore]]:
        """Select the best kernel configuration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        criterion : str, default="cv"
            Selection criterion: "cv", "aic", "bic", or "lml".
            
        Returns
        -------
        best_config : KernelConfig
            The best kernel configuration.
        scores : list of ModelScore
            Scores for all configurations.
        """
        import time
        
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_dims = X.shape[1] if X.ndim > 1 else 1
        n_samples = len(y)
        
        scores = []
        
        for config in self.candidates:
            start_time = time.time()
            
            # Create kernel (with and without ARD if enabled)
            kernels_to_try = [create_kernel(config, n_dims, ard=False)]
            if self.use_ard and n_dims > 1:
                kernels_to_try.append(create_kernel(config, n_dims, ard=True))
            
            for kernel in kernels_to_try:
                try:
                    # Create and fit GP
                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=self.n_restarts,
                        random_state=self.random_state,
                        normalize_y=True,
                    )
                    
                    # Cross-validation score
                    if n_samples >= self.cv:
                        cv_scores = cross_val_score(
                            gp, X, y,
                            cv=min(self.cv, n_samples),
                            scoring=self.scoring,
                        )
                        cv_score = np.mean(cv_scores)
                    else:
                        cv_score = float('-inf')
                    
                    # Fit to get log marginal likelihood
                    gp.fit(X, y)
                    lml = gp.log_marginal_likelihood_value_
                    
                    # Count parameters
                    n_params = len(gp.kernel_.theta)
                    
                    # Information criteria
                    # AIC = 2k - 2ln(L)
                    # BIC = k*ln(n) - 2ln(L)
                    aic = 2 * n_params - 2 * lml
                    bic = n_params * np.log(n_samples) - 2 * lml
                    
                    fit_time = time.time() - start_time
                    
                    score = ModelScore(
                        kernel_config=config,
                        cv_score=cv_score,
                        log_marginal_likelihood=lml,
                        aic=aic,
                        bic=bic,
                        n_params=n_params,
                        fit_time=fit_time,
                    )
                    scores.append(score)
                    
                except Exception as e:
                    warnings.warn(f"Failed to fit kernel {config.family}: {e}")
                    continue
        
        if not scores:
            raise ValueError("No kernel could be fitted successfully")
        
        # Select best based on criterion
        if criterion == "cv":
            best_idx = np.argmax([s.cv_score for s in scores])
        elif criterion == "aic":
            best_idx = np.argmin([s.aic for s in scores])
        elif criterion == "bic":
            best_idx = np.argmin([s.bic for s in scores])
        elif criterion == "lml":
            best_idx = np.argmax([s.log_marginal_likelihood for s in scores])
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return scores[best_idx].kernel_config, scores


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization.
    
    Attributes
    ----------
    n_restarts : int
        Number of optimizer restarts.
    optimizer : str
        Optimizer to use: "L-BFGS-B", "fmin_l_bfgs_b", or "gradient_descent".
    max_iter : int
        Maximum iterations per restart.
    learning_rate : float
        Learning rate for gradient descent.
    use_warm_start : bool
        Whether to use warm starts from previous optimization.
    """
    n_restarts: int = 10
    optimizer: str = "L-BFGS-B"
    max_iter: int = 100
    learning_rate: float = 0.1
    use_warm_start: bool = True


class HyperparameterTuner:
    """Advanced hyperparameter tuning for Gaussian Processes.
    
    Provides multiple strategies for optimizing GP hyperparameters including
    gradient-based optimization, random search, and Bayesian optimization.
    
    Parameters
    ----------
    config : HyperparameterConfig, optional
        Configuration for tuning.
    strategy : str, default="marginal_likelihood"
        Tuning strategy: "marginal_likelihood", "cross_validation", or "bayesian".
    random_state : int, optional
        Random state for reproducibility.
        
    Examples
    --------
    >>> tuner = HyperparameterTuner(strategy="marginal_likelihood")
    >>> best_kernel = tuner.tune(kernel, X, y)
    """
    
    def __init__(
        self,
        config: Optional[HyperparameterConfig] = None,
        strategy: str = "marginal_likelihood",
        random_state: Optional[int] = None,
    ) -> None:
        self.config = config or HyperparameterConfig()
        self.strategy = strategy
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
    
    def tune(
        self,
        kernel: Kernel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Kernel, float]:
        """Tune hyperparameters for a kernel.
        
        Parameters
        ----------
        kernel : Kernel
            The kernel to tune.
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        tuned_kernel : Kernel
            The kernel with optimized hyperparameters.
        best_score : float
            The best score achieved.
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if self.strategy == "marginal_likelihood":
            return self._tune_marginal_likelihood(kernel, X, y)
        elif self.strategy == "cross_validation":
            return self._tune_cross_validation(kernel, X, y)
        elif self.strategy == "bayesian":
            return self._tune_bayesian(kernel, X, y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _tune_marginal_likelihood(
        self,
        kernel: Kernel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Kernel, float]:
        """Tune using log marginal likelihood maximization."""
        best_lml = float('-inf')
        best_kernel = kernel.clone_with_theta(kernel.theta)
        
        bounds = kernel.bounds
        n_params = len(kernel.theta)
        
        for i in range(self.config.n_restarts):
            # Random initialization
            if i == 0 and self.config.use_warm_start:
                theta_init = kernel.theta
            else:
                theta_init = self._rng.uniform(bounds[:, 0], bounds[:, 1])
            
            try:
                # Create GP and optimize
                gp = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(theta_init),
                    optimizer=self.config.optimizer,
                    n_restarts_optimizer=0,  # We handle restarts ourselves
                    random_state=self.random_state,
                    normalize_y=True,
                )
                gp.fit(X, y)
                
                if gp.log_marginal_likelihood_value_ > best_lml:
                    best_lml = gp.log_marginal_likelihood_value_
                    best_kernel = gp.kernel_
                    
            except Exception:
                continue
        
        return best_kernel, best_lml
    
    def _tune_cross_validation(
        self,
        kernel: Kernel,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> Tuple[Kernel, float]:
        """Tune using cross-validation score."""
        best_score = float('-inf')
        best_kernel = kernel.clone_with_theta(kernel.theta)
        
        bounds = kernel.bounds
        
        for i in range(self.config.n_restarts):
            if i == 0 and self.config.use_warm_start:
                theta_init = kernel.theta
            else:
                theta_init = self._rng.uniform(bounds[:, 0], bounds[:, 1])
            
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(theta_init),
                    n_restarts_optimizer=0,
                    random_state=self.random_state,
                    normalize_y=True,
                )
                
                scores = cross_val_score(gp, X, y, cv=min(cv, len(y)), scoring="neg_mean_squared_error")
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    gp.fit(X, y)
                    best_kernel = gp.kernel_
                    
            except Exception:
                continue
        
        return best_kernel, best_score
    
    def _tune_bayesian(
        self,
        kernel: Kernel,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 20,
    ) -> Tuple[Kernel, float]:
        """Tune using Bayesian optimization over hyperparameters."""
        # Use a simple GP to optimize hyperparameters
        bounds = kernel.bounds
        n_params = len(kernel.theta)
        
        # Initialize with random samples
        n_init = min(5, n_iter // 2)
        theta_samples = self._rng.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_init, n_params)
        )
        
        lml_values = []
        for theta in theta_samples:
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(theta),
                    n_restarts_optimizer=0,
                    random_state=self.random_state,
                    normalize_y=True,
                )
                gp.fit(X, y)
                lml_values.append(gp.log_marginal_likelihood_value_)
            except Exception:
                lml_values.append(float('-inf'))
        
        theta_samples = list(theta_samples)
        
        # Bayesian optimization loop
        for _ in range(n_iter - n_init):
            # Fit GP to hyperparameter evaluations
            valid_mask = np.isfinite(lml_values)
            if np.sum(valid_mask) < 2:
                # Not enough valid samples, fall back to random
                next_theta = self._rng.uniform(bounds[:, 0], bounds[:, 1])
            else:
                X_hp = np.array(theta_samples)[valid_mask]
                y_hp = np.array(lml_values)[valid_mask]
                
                hp_gp = GaussianProcessRegressor(
                    kernel=RBF(length_scale=np.ones(n_params)),
                    n_restarts_optimizer=2,
                    random_state=self.random_state,
                    normalize_y=True,
                )
                hp_gp.fit(X_hp, y_hp)
                
                # Maximize expected improvement
                next_theta = self._maximize_ei(hp_gp, bounds, np.max(y_hp))
            
            # Evaluate
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(next_theta),
                    n_restarts_optimizer=0,
                    random_state=self.random_state,
                    normalize_y=True,
                )
                gp.fit(X, y)
                lml = gp.log_marginal_likelihood_value_
            except Exception:
                lml = float('-inf')
            
            theta_samples.append(next_theta)
            lml_values.append(lml)
        
        # Return best
        best_idx = np.argmax(lml_values)
        best_theta = theta_samples[best_idx]
        best_lml = lml_values[best_idx]
        
        return kernel.clone_with_theta(best_theta), best_lml
    
    def _maximize_ei(
        self,
        gp: GaussianProcessRegressor,
        bounds: np.ndarray,
        best_y: float,
        n_samples: int = 100,
    ) -> np.ndarray:
        """Maximize expected improvement to find next hyperparameters."""
        n_params = len(bounds)
        
        # Random candidates
        candidates = self._rng.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_samples, n_params)
        )
        
        # Compute EI for each
        mu, sigma = gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-10)
        
        z = (mu - best_y) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        
        return candidates[np.argmax(ei)]


class GPEnsemble:
    """Ensemble of Gaussian Processes for improved predictions.
    
    Combines multiple GPs with different kernels to improve robustness
    and prediction accuracy.
    
    Parameters
    ----------
    kernel_configs : list of KernelConfig, optional
        Kernel configurations for ensemble members.
    weights : str or array-like, default="equal"
        Weights for ensemble: "equal", "cv", "lml", or explicit weights.
    n_members : int, default=5
        Number of ensemble members if kernel_configs not provided.
    random_state : int, optional
        Random state for reproducibility.
        
    Examples
    --------
    >>> ensemble = GPEnsemble(weights="lml")
    >>> ensemble.fit(X, y)
    >>> mean, std = ensemble.predict(X_new, return_std=True)
    """
    
    def __init__(
        self,
        kernel_configs: Optional[List[KernelConfig]] = None,
        weights: Union[str, np.ndarray] = "equal",
        n_members: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.kernel_configs = kernel_configs or self._default_kernels(n_members)
        self.weights_strategy = weights
        self.n_members = n_members
        self.random_state = random_state
        
        self.members_: List[GaussianProcessRegressor] = []
        self.weights_: Optional[np.ndarray] = None
        self._fitted = False
    
    def _default_kernels(self, n: int) -> List[KernelConfig]:
        """Generate default kernel configurations."""
        defaults = [
            KernelConfig(family=KernelFamily.RBF),
            KernelConfig(family=KernelFamily.MATERN_32),
            KernelConfig(family=KernelFamily.MATERN_52),
            KernelConfig(family=KernelFamily.RATIONAL_QUADRATIC),
            KernelConfig(family=KernelFamily.COMPOSITE),
        ]
        # Cycle through defaults if n > len(defaults)
        return [defaults[i % len(defaults)] for i in range(n)]
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "GPEnsemble":
        """Fit the ensemble to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_dims = X.shape[1] if X.ndim > 1 else 1
        
        self.members_ = []
        lmls = []
        cv_scores = []
        
        for config in self.kernel_configs:
            kernel = create_kernel(config, n_dims)
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                random_state=self.random_state,
                normalize_y=True,
            )
            
            try:
                gp.fit(X, y)
                self.members_.append(gp)
                lmls.append(gp.log_marginal_likelihood_value_)
                
                # CV score if needed
                if self.weights_strategy == "cv":
                    scores = cross_val_score(
                        gp, X, y,
                        cv=min(5, len(y)),
                        scoring="neg_mean_squared_error",
                    )
                    cv_scores.append(np.mean(scores))
                    
            except Exception as e:
                warnings.warn(f"Failed to fit member with kernel {config.family}: {e}")
                continue
        
        if not self.members_:
            raise ValueError("No ensemble members could be fitted")
        
        # Compute weights
        n_members = len(self.members_)
        
        if isinstance(self.weights_strategy, np.ndarray):
            self.weights_ = self.weights_strategy[:n_members]
            self.weights_ = self.weights_ / self.weights_.sum()
            
        elif self.weights_strategy == "equal":
            self.weights_ = np.ones(n_members) / n_members
            
        elif self.weights_strategy == "lml":
            # Softmax of log marginal likelihoods
            lmls = np.array(lmls)
            lmls = lmls - lmls.max()  # Numerical stability
            weights = np.exp(lmls)
            self.weights_ = weights / weights.sum()
            
        elif self.weights_strategy == "cv":
            # Softmax of CV scores
            cv_scores = np.array(cv_scores)
            cv_scores = cv_scores - cv_scores.max()
            weights = np.exp(cv_scores)
            self.weights_ = weights / weights.sum()
            
        else:
            raise ValueError(f"Unknown weights strategy: {self.weights_strategy}")
        
        self._fitted = True
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict using the ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to predict at.
        return_std : bool, default=False
            Whether to return standard deviation.
            
        Returns
        -------
        mean : array-like of shape (n_samples,)
            Predicted mean.
        std : array-like of shape (n_samples,), optional
            Predicted standard deviation (if return_std=True).
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        X = np.asarray(X)
        
        # Collect predictions from all members
        means = []
        variances = []
        
        for gp in self.members_:
            mu, sigma = gp.predict(X, return_std=True)
            means.append(mu)
            variances.append(sigma ** 2)
        
        means = np.array(means)
        variances = np.array(variances)
        
        # Weighted average of means
        ensemble_mean = np.average(means, axis=0, weights=self.weights_)
        
        if return_std:
            # Total variance = weighted average of (variance + squared mean)
            # minus squared weighted mean
            weighted_second_moment = np.average(
                variances + means ** 2, axis=0, weights=self.weights_
            )
            ensemble_var = weighted_second_moment - ensemble_mean ** 2
            ensemble_std = np.sqrt(np.maximum(ensemble_var, 0))
            return ensemble_mean, ensemble_std
        
        return ensemble_mean
    
    def sample_y(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample from the posterior distribution.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to sample at.
        n_samples : int, default=1
            Number of samples.
        random_state : int, optional
            Random state.
            
        Returns
        -------
        samples : array-like of shape (n_samples, n_points)
            Samples from the posterior.
        """
        if not self._fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        rng = np.random.RandomState(random_state)
        X = np.asarray(X)
        
        # Sample which member to use for each sample
        member_indices = rng.choice(
            len(self.members_),
            size=n_samples,
            p=self.weights_,
        )
        
        samples = np.zeros((n_samples, len(X)))
        for i, idx in enumerate(member_indices):
            sample = self.members_[idx].sample_y(X, n_samples=1, random_state=rng.randint(2**31))
            samples[i] = sample.ravel()
        
        return samples


class AutoML:
    """Automatic machine learning for Gaussian Process surrogate models.
    
    Combines kernel selection, hyperparameter tuning, and ensemble creation
    into a single automated pipeline.
    
    Parameters
    ----------
    mode : str, default="balanced"
        Optimization mode: "fast", "balanced", or "thorough".
    use_ensemble : bool, default=False
        Whether to use an ensemble instead of a single GP.
    random_state : int, optional
        Random state for reproducibility.
        
    Examples
    --------
    >>> automl = AutoML(mode="balanced")
    >>> model = automl.fit(X, y)
    >>> predictions = model.predict(X_new)
    """
    
    def __init__(
        self,
        mode: str = "balanced",
        use_ensemble: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.mode = mode
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        
        self.model_: Optional[Union[GaussianProcessRegressor, GPEnsemble]] = None
        self.best_kernel_config_: Optional[KernelConfig] = None
        self.selection_scores_: Optional[List[ModelScore]] = None
    
    def _get_config(self) -> Tuple[int, int, int]:
        """Get configuration based on mode."""
        if self.mode == "fast":
            return 3, 3, 3  # cv_folds, n_restarts, n_candidates
        elif self.mode == "balanced":
            return 5, 5, 5
        elif self.mode == "thorough":
            return 10, 10, 7
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Union[GaussianProcessRegressor, GPEnsemble]:
        """Fit the best model to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        model
            The fitted model (GP or ensemble).
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        cv_folds, n_restarts, n_candidates = self._get_config()
        
        if self.use_ensemble:
            # Use ensemble
            ensemble = GPEnsemble(
                weights="lml",
                n_members=n_candidates,
                random_state=self.random_state,
            )
            ensemble.fit(X, y)
            self.model_ = ensemble
            return ensemble
        
        else:
            # Select best kernel
            selector = KernelSelector(
                cv=cv_folds,
                n_restarts=n_restarts,
                random_state=self.random_state,
            )
            
            self.best_kernel_config_, self.selection_scores_ = selector.select(X, y)
            
            # Tune hyperparameters
            n_dims = X.shape[1] if X.ndim > 1 else 1
            kernel = create_kernel(self.best_kernel_config_, n_dims)
            
            tuner = HyperparameterTuner(
                config=HyperparameterConfig(n_restarts=n_restarts),
                random_state=self.random_state,
            )
            tuned_kernel, _ = tuner.tune(kernel, X, y)
            
            # Fit final model
            self.model_ = GaussianProcessRegressor(
                kernel=tuned_kernel,
                n_restarts_optimizer=0,  # Already tuned
                random_state=self.random_state,
                normalize_y=True,
            )
            self.model_.fit(X, y)
            
            return self.model_
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict using the fitted model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to predict at.
        return_std : bool, default=False
            Whether to return standard deviation.
            
        Returns
        -------
        mean : array-like of shape (n_samples,)
            Predicted mean.
        std : array-like of shape (n_samples,), optional
            Predicted standard deviation.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model_.predict(X, return_std=return_std)
    
    def get_selection_report(self) -> Dict[str, Any]:
        """Get a report of the model selection process.
        
        Returns
        -------
        dict
            Report with selection details.
        """
        if self.selection_scores_ is None:
            return {"message": "No selection performed (ensemble mode or not fitted)"}
        
        report = {
            "best_kernel": self.best_kernel_config_.family.value if self.best_kernel_config_ else None,
            "n_candidates_evaluated": len(self.selection_scores_),
            "scores": [],
        }
        
        for score in self.selection_scores_:
            report["scores"].append({
                "kernel": score.kernel_config.family.value,
                "cv_score": score.cv_score,
                "aic": score.aic,
                "bic": score.bic,
                "lml": score.log_marginal_likelihood,
                "n_params": score.n_params,
                "fit_time": score.fit_time,
            })
        
        return report
