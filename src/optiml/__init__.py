"""OptiML: Advanced Statistical Modeling with Easy-to-Use Bayesian Optimization."""

from optiml.optimizer import BayesianOptimizer
from optiml.space import Space, Real, Integer, Categorical
from optiml.surrogate import GaussianProcessSurrogate
from optiml.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
)

__version__ = "0.1.0"
__all__ = [
    "BayesianOptimizer",
    "Space",
    "Real",
    "Integer",
    "Categorical",
    "GaussianProcessSurrogate",
    "AcquisitionFunction",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ProbabilityOfImprovement",
]
