"""
Prior Knowledge Module for Bayesian Optimization
=================================================

This module provides functionality to build prior knowledge from historical
experiments stored in the database and use it to warm-start future optimizations.

Key Features:
- Extract prior distributions from historical experiment data
- Find similar experiments based on parameter structure
- Transfer learning between related experiments
- Informative GP priors based on observed data patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
import json


@dataclass
class ParameterPrior:
    """Prior knowledge about a single parameter.
    
    Attributes
    ----------
    name : str
        Parameter name.
    param_type : str
        Type of parameter ('real', 'integer', 'categorical').
    mean_optimal : float
        Mean of optimal values across experiments.
    std_optimal : float
        Std dev of optimal values.
    best_values : list
        List of best values from historical experiments.
    value_counts : dict
        For categorical: counts of each category in best trials.
    confidence : float
        Confidence in this prior (0-1), based on amount of data.
    """
    name: str
    param_type: str
    mean_optimal: Optional[float] = None
    std_optimal: Optional[float] = None
    best_values: List[Any] = field(default_factory=list)
    value_counts: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    low: Optional[float] = None
    high: Optional[float] = None
    
    def get_prior_distribution(self) -> Optional[stats.rv_continuous]:
        """Get a scipy distribution representing the prior.
        
        Returns truncated normal for real/integer, or None for categorical.
        """
        if self.param_type == 'categorical':
            return None
        
        if self.mean_optimal is None or self.std_optimal is None:
            return None
        
        if self.low is not None and self.high is not None:
            # Truncated normal within bounds
            a = (self.low - self.mean_optimal) / max(self.std_optimal, 1e-6)
            b = (self.high - self.mean_optimal) / max(self.std_optimal, 1e-6)
            return stats.truncnorm(a, b, loc=self.mean_optimal, scale=self.std_optimal)
        
        return stats.norm(loc=self.mean_optimal, scale=self.std_optimal)
    
    def sample(self, n: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        """Sample from the prior distribution.
        
        Parameters
        ----------
        n : int
            Number of samples.
        rng : np.random.Generator
            Random number generator.
            
        Returns
        -------
        np.ndarray
            Samples from the prior.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if self.param_type == 'categorical':
            if not self.value_counts:
                return None
            categories = list(self.value_counts.keys())
            weights = np.array(list(self.value_counts.values()), dtype=float)
            weights /= weights.sum()
            return rng.choice(categories, size=n, p=weights)
        
        dist = self.get_prior_distribution()
        if dist is not None:
            return dist.rvs(size=n, random_state=rng)
        
        # Fallback to uniform
        if self.low is not None and self.high is not None:
            return rng.uniform(self.low, self.high, size=n)
        return None
    
    def get_category_probabilities(self) -> Dict[str, float]:
        """Get probability distribution over categories."""
        if not self.value_counts:
            return {}
        total = sum(self.value_counts.values())
        return {k: v / total for k, v in self.value_counts.items()}


@dataclass
class ExperimentPrior:
    """Prior knowledge from a collection of similar experiments.
    
    Attributes
    ----------
    parameter_priors : dict
        Mapping of parameter name to ParameterPrior.
    n_experiments : int
        Number of experiments used to build this prior.
    n_trials : int
        Total number of trials used.
    expected_best : float
        Expected best objective value based on history.
    objective_variance : float
        Variance in best objective values.
    warm_start_points : list
        Suggested warm-start points from best historical trials.
    metadata : dict
        Additional metadata about the prior.
    """
    parameter_priors: Dict[str, ParameterPrior] = field(default_factory=dict)
    n_experiments: int = 0
    n_trials: int = 0
    expected_best: Optional[float] = None
    objective_variance: Optional[float] = None
    warm_start_points: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_prior_mean_function(self) -> callable:
        """Get a prior mean function for the GP.
        
        Returns a function that maps normalized X to prior mean predictions.
        This can be used as a custom mean function in the GP surrogate.
        """
        if not self.warm_start_points or self.expected_best is None:
            return lambda X: np.zeros(len(X))
        
        # Use expected best as constant mean
        expected = self.expected_best
        return lambda X: np.full(len(X), expected)
    
    def sample_warm_start(self, n: int = 5, rng: np.random.Generator = None) -> List[Dict[str, Any]]:
        """Sample warm-start points biased toward historically good regions.
        
        Parameters
        ----------
        n : int
            Number of points to sample.
        rng : np.random.Generator
            Random number generator.
            
        Returns
        -------
        list
            List of parameter dictionaries.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        points = []
        for _ in range(n):
            point = {}
            for name, prior in self.parameter_priors.items():
                sample = prior.sample(1, rng)
                if sample is not None:
                    point[name] = sample[0] if len(sample) == 1 else sample
            if point:
                points.append(point)
        
        return points


class PriorKnowledgeBuilder:
    """Builds prior knowledge from historical experiment data.
    
    This class analyzes the database to extract useful priors for new
    experiments based on similar past experiments.
    
    Parameters
    ----------
    db : Database
        The OptiML database instance.
    similarity_threshold : float
        Minimum similarity score (0-1) to consider experiments related.
    min_trials : int
        Minimum number of completed trials to use an experiment.
    """
    
    def __init__(
        self,
        db,
        similarity_threshold: float = 0.5,
        min_trials: int = 3,
    ):
        self.db = db
        self.similarity_threshold = similarity_threshold
        self.min_trials = min_trials
    
    def compute_parameter_similarity(
        self,
        params1: List[Dict[str, Any]],
        params2: List[Dict[str, Any]],
    ) -> float:
        """Compute similarity between two parameter sets.
        
        Uses Jaccard similarity on parameter names and type matching.
        
        Returns
        -------
        float
            Similarity score between 0 and 1.
        """
        if not params1 or not params2:
            return 0.0
        
        names1 = {p['name'].lower() for p in params1}
        names2 = {p['name'].lower() for p in params2}
        
        # Jaccard similarity on names
        intersection = len(names1 & names2)
        union = len(names1 | names2)
        name_similarity = intersection / union if union > 0 else 0
        
        # Type matching for common parameters
        type_matches = 0
        common = names1 & names2
        if common:
            types1 = {p['name'].lower(): p['param_type'] for p in params1}
            types2 = {p['name'].lower(): p['param_type'] for p in params2}
            for name in common:
                if types1.get(name) == types2.get(name):
                    type_matches += 1
            type_similarity = type_matches / len(common)
        else:
            type_similarity = 0
        
        # Weighted combination
        return 0.6 * name_similarity + 0.4 * type_similarity
    
    def find_similar_experiments(
        self,
        target_params: List[Dict[str, Any]],
        template_id: Optional[str] = None,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find experiments similar to the target parameter structure.
        
        Parameters
        ----------
        target_params : list
            List of parameter dictionaries for the new experiment.
        template_id : str, optional
            If provided, prioritize experiments with the same template.
        exclude_ids : list, optional
            Experiment IDs to exclude from search.
            
        Returns
        -------
        list
            List of (experiment_dict, similarity_score) tuples, sorted by score.
        """
        exclude_ids = exclude_ids or []
        all_experiments = self.db.list_experiments(include_archived=False)
        
        similar = []
        for exp in all_experiments:
            if exp['id'] in exclude_ids:
                continue
            
            # Check minimum trials
            completed_trials = [t for t in exp.get('trials', []) 
                              if t.get('objective_value') is not None]
            if len(completed_trials) < self.min_trials:
                continue
            
            # Compute similarity
            similarity = self.compute_parameter_similarity(
                target_params, exp.get('parameters', [])
            )
            
            # Boost score if same template
            if template_id and exp.get('template_id') == template_id:
                similarity = min(1.0, similarity + 0.2)
            
            if similarity >= self.similarity_threshold:
                similar.append((exp, similarity))
        
        # Sort by similarity (descending)
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def extract_best_trials(
        self,
        experiments: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float, bool]]:
        """Extract top trials from a set of experiments.
        
        Parameters
        ----------
        experiments : list
            List of experiment dictionaries.
        top_k : int
            Number of top trials to extract per experiment.
            
        Returns
        -------
        list
            List of (parameters, objective_value, minimize) tuples.
        """
        all_best = []
        
        for exp in experiments:
            minimize = exp.get('minimize', True)
            trials = exp.get('trials', [])
            completed = [t for t in trials if t.get('objective_value') is not None]
            
            if not completed:
                continue
            
            # Sort by objective value
            if minimize:
                completed.sort(key=lambda t: t['objective_value'])
            else:
                completed.sort(key=lambda t: t['objective_value'], reverse=True)
            
            # Take top k
            for trial in completed[:top_k]:
                all_best.append((
                    trial['parameters'],
                    trial['objective_value'],
                    minimize
                ))
        
        return all_best
    
    def build_parameter_prior(
        self,
        param_name: str,
        param_type: str,
        best_trials: List[Tuple[Dict[str, Any], float, bool]],
        low: Optional[float] = None,
        high: Optional[float] = None,
    ) -> ParameterPrior:
        """Build a prior for a single parameter from best trials.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter.
        param_type : str
            Type of parameter.
        best_trials : list
            List of best trial data.
        low, high : float, optional
            Parameter bounds.
            
        Returns
        -------
        ParameterPrior
            The constructed prior.
        """
        # Extract values for this parameter from best trials
        values = []
        for params, obj_value, minimize in best_trials:
            # Handle case-insensitive matching
            for key in params:
                if key.lower() == param_name.lower():
                    values.append(params[key])
                    break
        
        if not values:
            return ParameterPrior(
                name=param_name,
                param_type=param_type,
                confidence=0.0,
                low=low,
                high=high,
            )
        
        prior = ParameterPrior(
            name=param_name,
            param_type=param_type,
            best_values=values,
            low=low,
            high=high,
        )
        
        if param_type == 'categorical':
            # Count category occurrences
            counts = {}
            for v in values:
                counts[str(v)] = counts.get(str(v), 0) + 1
            prior.value_counts = counts
            prior.confidence = min(1.0, len(values) / 10)  # Cap at 10 samples
        else:
            # Compute statistics for numerical parameters
            numeric_values = [float(v) for v in values if v is not None]
            if numeric_values:
                prior.mean_optimal = np.mean(numeric_values)
                prior.std_optimal = np.std(numeric_values) if len(numeric_values) > 1 else (
                    (high - low) / 4 if low is not None and high is not None else 1.0
                )
                prior.confidence = min(1.0, len(numeric_values) / 10)
        
        return prior
    
    def build_experiment_prior(
        self,
        target_params: List[Dict[str, Any]],
        template_id: Optional[str] = None,
        exclude_ids: Optional[List[int]] = None,
        max_experiments: int = 10,
        top_k_trials: int = 5,
    ) -> ExperimentPrior:
        """Build a complete prior for a new experiment.
        
        Parameters
        ----------
        target_params : list
            Parameter definitions for the new experiment.
        template_id : str, optional
            Template ID for the new experiment.
        exclude_ids : list, optional
            Experiment IDs to exclude.
        max_experiments : int
            Maximum number of similar experiments to use.
        top_k_trials : int
            Number of top trials to use per experiment.
            
        Returns
        -------
        ExperimentPrior
            The constructed prior with all available knowledge.
        """
        # Find similar experiments
        similar = self.find_similar_experiments(
            target_params, template_id, exclude_ids
        )[:max_experiments]
        
        if not similar:
            return ExperimentPrior(
                metadata={'message': 'No similar experiments found'}
            )
        
        experiments = [exp for exp, score in similar]
        
        # Extract best trials
        best_trials = self.extract_best_trials(experiments, top_k_trials)
        
        # Build parameter priors
        param_priors = {}
        for param in target_params:
            name = param['name']
            ptype = param['param_type']
            low = param.get('low')
            high = param.get('high')
            
            prior = self.build_parameter_prior(name, ptype, best_trials, low, high)
            param_priors[name] = prior
        
        # Compute objective statistics
        # Normalize objectives to comparable scale
        objective_values = []
        for params, obj_value, minimize in best_trials:
            # Flip sign for maximization problems to make comparable
            if not minimize:
                objective_values.append(-obj_value)
            else:
                objective_values.append(obj_value)
        
        expected_best = np.mean(objective_values) if objective_values else None
        obj_variance = np.var(objective_values) if len(objective_values) > 1 else None
        
        # Build warm-start points
        warm_start = []
        for params, obj_value, minimize in best_trials[:10]:
            warm_start.append((params, obj_value))
        
        return ExperimentPrior(
            parameter_priors=param_priors,
            n_experiments=len(experiments),
            n_trials=len(best_trials),
            expected_best=expected_best,
            objective_variance=obj_variance,
            warm_start_points=warm_start,
            metadata={
                'similar_experiments': [
                    {'id': exp['id'], 'name': exp['name'], 'similarity': score}
                    for exp, score in similar
                ],
                'template_id': template_id,
            }
        )


class PriorAwareBayesianOptimizer:
    """Bayesian Optimizer that uses prior knowledge from historical data.
    
    This optimizer wraps the standard BayesianOptimizer but uses historical
    experiment data to:
    1. Generate informed initial samples instead of random
    2. Warm-start with data from similar experiments
    3. Use informative prior mean for the GP
    
    Parameters
    ----------
    space : Space
        The search space.
    prior : ExperimentPrior
        Prior knowledge from historical experiments.
    prior_weight : float
        How much to weight prior vs exploration (0-1).
    **kwargs
        Additional arguments passed to BayesianOptimizer.
    """
    
    def __init__(
        self,
        space,
        prior: ExperimentPrior,
        prior_weight: float = 0.5,
        **kwargs
    ):
        from optiml import BayesianOptimizer
        
        self.space = space
        self.prior = prior
        self.prior_weight = np.clip(prior_weight, 0, 1)
        self.kwargs = kwargs
        
        # Create the base optimizer
        self._optimizer = BayesianOptimizer(space, **kwargs)
        
        # Inject warm-start data if available
        self._apply_warm_start()
    
    def _apply_warm_start(self) -> None:
        """Apply warm-start points from prior to the optimizer."""
        if not self.prior.warm_start_points:
            return
        
        # Only use points that match our parameter structure
        param_names = [d.name for d in self.space.dimensions]
        
        for params, obj_value in self.prior.warm_start_points[:3]:  # Limit to 3
            try:
                # Build parameter list in correct order
                x = []
                for name in param_names:
                    if name in params:
                        x.append(params[name])
                    elif name.lower() in {k.lower() for k in params}:
                        # Case-insensitive fallback
                        for k, v in params.items():
                            if k.lower() == name.lower():
                                x.append(v)
                                break
                    else:
                        # Missing parameter - skip this point
                        raise KeyError(f"Missing parameter: {name}")
                
                # Scale objective if needed (prior weight)
                # Don't actually add these as real observations, but use for initial model
                self._optimizer._X.append(x)
                self._optimizer._y.append(obj_value)
            except (KeyError, ValueError):
                continue
    
    def _suggest_with_prior(self) -> list:
        """Suggest a point biased by the prior."""
        rng = self._optimizer._rng
        
        # With probability (1 - prior_weight), use standard suggestion
        if rng.random() > self.prior_weight or not self.prior.parameter_priors:
            return self._optimizer.suggest()
        
        # Sample from prior distributions
        param_names = [d.name for d in self.space.dimensions]
        x = []
        
        for name in param_names:
            prior = self.prior.parameter_priors.get(name)
            if prior is not None and prior.confidence > 0.3:
                sample = prior.sample(1, rng)
                if sample is not None and len(sample) > 0:
                    x.append(sample[0])
                    continue
            
            # Fallback to random from space
            sample = self.space.sample(1, rng)[0]
            idx = param_names.index(name)
            x.append(sample[idx])
        
        return x
    
    def suggest(self) -> list:
        """Suggest the next point, using prior knowledge if available."""
        # Use prior-informed suggestions for initial phase
        if len(self._optimizer._X) < self._optimizer.n_initial:
            return self._suggest_with_prior()
        
        # After initial phase, use standard Bayesian optimization
        return self._optimizer.suggest()
    
    def tell(self, x: list, y: float) -> None:
        """Record an observation."""
        self._optimizer.tell(x, y)
    
    def optimize(self, objective, n_iterations: int = 20, callback=None):
        """Run the full optimization loop."""
        return self._optimizer.optimize(objective, n_iterations, callback)
    
    def get_result(self):
        """Get the current optimization result."""
        return self._optimizer.get_result()
    
    def reset(self) -> None:
        """Reset the optimizer state."""
        self._optimizer.reset()
        self._apply_warm_start()


def get_prior_for_experiment(
    db,
    target_params: List[Dict[str, Any]],
    template_id: Optional[str] = None,
    current_experiment_id: Optional[int] = None,
) -> ExperimentPrior:
    """Convenience function to get prior knowledge for a new experiment.
    
    Parameters
    ----------
    db : Database
        The OptiML database.
    target_params : list
        Parameter definitions for the new experiment.
    template_id : str, optional
        Template ID if using a template.
    current_experiment_id : int, optional
        ID of current experiment to exclude from prior.
        
    Returns
    -------
    ExperimentPrior
        Prior knowledge extracted from similar experiments.
    """
    builder = PriorKnowledgeBuilder(db)
    exclude = [current_experiment_id] if current_experiment_id else None
    return builder.build_experiment_prior(target_params, template_id, exclude)


def create_prior_aware_optimizer(
    space,
    prior: ExperimentPrior,
    prior_weight: float = 0.5,
    **kwargs
) -> PriorAwareBayesianOptimizer:
    """Create a Bayesian optimizer that uses prior knowledge.
    
    Parameters
    ----------
    space : Space
        The search space.
    prior : ExperimentPrior
        Prior knowledge from historical experiments.
    prior_weight : float
        Balance between prior and exploration (0-1).
    **kwargs
        Additional arguments for BayesianOptimizer.
        
    Returns
    -------
    PriorAwareBayesianOptimizer
        Optimizer configured with prior knowledge.
    """
    return PriorAwareBayesianOptimizer(space, prior, prior_weight, **kwargs)
