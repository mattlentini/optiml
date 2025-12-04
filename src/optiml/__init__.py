"""OptiML: Advanced Statistical Modeling with Easy-to-Use Bayesian Optimization."""

# Core optimizer and space
from optiml.optimizer import BayesianOptimizer
from optiml.space import Space, Real, Integer, Categorical
from optiml.surrogate import GaussianProcessSurrogate

# Acquisition functions
from optiml.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    LowerConfidenceBound,
)

# Experimental designs (DOE)
from optiml.designs import (
    Design,
    RandomDesign,
    LatinHypercubeDesign,
    SobolDesign,
    HaltonDesign,
    FullFactorialDesign,
    FractionalFactorialDesign,
    CentralCompositeDesign,
    BoxBehnkenDesign,
    PlackettBurmanDesign,
    DesignMetrics,
    evaluate_design,
    compare_designs,
)

# Kernel functions
from optiml.kernels import (
    Kernel,
    RBF,
    Matern,
    Matern12,
    Matern32,
    Matern52,
    RationalQuadratic,
    Periodic,
    WhiteNoise,
    ConstantKernel,
    SumKernel,
    ProductKernel,
    create_kernel,
    default_kernel,
)

# Statistical analysis
from optiml.statistics import (
    SummaryStatistics,
    calculate_summary_statistics,
    ParameterEffect,
    InteractionEffect,
    EffectsAnalysis,
    analyze_effects,
    ANOVAResult,
    ANOVATable,
    perform_anova,
    Residuals,
    calculate_residuals,
    ConfidenceInterval,
    confidence_interval_mean,
    prediction_interval,
    NormalityTest,
    check_normality,
    calculate_partial_dependence,
    calculate_all_partial_dependence,
)

# Visualization
from optiml.visualization import (
    plot_convergence,
    plot_parameter_importance,
    plot_partial_dependence,
    plot_partial_dependence_grid,
    plot_contour,
    plot_slice,
    plot_pareto_front,
    compute_pareto_mask,
    plot_acquisition,
    plot_optimization_summary,
    plot_residuals_diagnostic,
)

# Constraints
from optiml.constraints import (
    Constraint,
    LinearConstraint,
    NonlinearConstraint,
    BoundConstraint,
    SumConstraint,
    BlackBoxConstraint,
    ConstraintHandler,
    PenaltyMethod,
    ConstrainedExpectedImprovement,
    sample_feasible_points,
)

# Multi-objective optimization
from optiml.multi_objective import (
    ParetoFront,
    is_pareto_optimal,
    compute_pareto_front,
    compute_hypervolume,
    compute_crowding_distance,
    Scalarization,
    WeightedSum,
    Chebyshev,
    AugmentedChebyshev,
    ParEGO,
    ExpectedHypervolumeImprovement,
    MultiObjectiveOptimizer,
    generate_weight_vectors,
)


__version__ = "0.1.0"

__all__ = [
    # Core
    "BayesianOptimizer",
    "Space",
    "Real",
    "Integer",
    "Categorical",
    "GaussianProcessSurrogate",
    # Acquisition functions
    "AcquisitionFunction",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ProbabilityOfImprovement",
    "LowerConfidenceBound",
    # Designs
    "Design",
    "RandomDesign",
    "LatinHypercubeDesign",
    "SobolDesign",
    "HaltonDesign",
    "FullFactorialDesign",
    "FractionalFactorialDesign",
    "CentralCompositeDesign",
    "BoxBehnkenDesign",
    "PlackettBurmanDesign",
    "DesignMetrics",
    "evaluate_design",
    "compare_designs",
    # Kernels
    "Kernel",
    "RBF",
    "Matern",
    "Matern12",
    "Matern32",
    "Matern52",
    "RationalQuadratic",
    "Periodic",
    "WhiteNoise",
    "ConstantKernel",
    "SumKernel",
    "ProductKernel",
    "create_kernel",
    "default_kernel",
    # Statistics
    "SummaryStatistics",
    "calculate_summary_statistics",
    "ParameterEffect",
    "InteractionEffect",
    "EffectsAnalysis",
    "analyze_effects",
    "ANOVAResult",
    "ANOVATable",
    "perform_anova",
    "Residuals",
    "calculate_residuals",
    "ConfidenceInterval",
    "confidence_interval_mean",
    "prediction_interval",
    "NormalityTest",
    "check_normality",
    "calculate_partial_dependence",
    "calculate_all_partial_dependence",
    # Visualization
    "plot_convergence",
    "plot_parameter_importance",
    "plot_partial_dependence",
    "plot_partial_dependence_grid",
    "plot_contour",
    "plot_slice",
    "plot_pareto_front",
    "compute_pareto_mask",
    "plot_acquisition",
    "plot_optimization_summary",
    "plot_residuals_diagnostic",
    # Constraints
    "Constraint",
    "LinearConstraint",
    "NonlinearConstraint",
    "BoundConstraint",
    "SumConstraint",
    "BlackBoxConstraint",
    "ConstraintHandler",
    "PenaltyMethod",
    "ConstrainedExpectedImprovement",
    "sample_feasible_points",
    # Multi-objective
    "ParetoFront",
    "is_pareto_optimal",
    "compute_pareto_front",
    "compute_hypervolume",
    "compute_crowding_distance",
    "Scalarization",
    "WeightedSum",
    "Chebyshev",
    "AugmentedChebyshev",
    "ParEGO",
    "ExpectedHypervolumeImprovement",
    "MultiObjectiveOptimizer",
    "generate_weight_vectors",
]
