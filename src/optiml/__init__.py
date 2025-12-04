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

# Prior knowledge from historical data
from optiml.priors import (
    ParameterPrior,
    ExperimentPrior,
    PriorKnowledgeBuilder,
    PriorAwareBayesianOptimizer,
    get_prior_for_experiment,
    create_prior_aware_optimizer,
)

# Batch/parallel acquisition
from optiml.batch import (
    BatchAcquisitionFunction,
    ConstantLiarBatch,
    LocalPenalizationBatch,
    qExpectedImprovement,
    suggest_batch,
)

# Early stopping and convergence detection
from optiml.convergence import (
    StoppingCriteria,
    StoppingState,
    ConvergenceMonitor,
    PlateauDetector,
    BudgetAdvisor,
    create_convergence_monitor,
)

# Model selection and AutoML
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

# Sensitivity analysis
from optiml.sensitivity import (
    SobolIndices,
    MorrisResult,
    LocalSensitivity,
    compute_sobol_indices,
    compute_sobol_from_surrogate,
    compute_morris,
    compute_morris_from_surrogate,
    compute_local_sensitivity,
    compute_local_sensitivity_from_surrogate,
    correlation_sensitivity,
    main_effect_indices,
    SensitivityAnalyzer,
)

# Robust optimization
from optiml.robust import (
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

# Interactive Plotly visualizations (optional)
try:
    from optiml.plotly_viz import (
        PlotlyTheme,
        LIGHT_THEME,
        DARK_THEME,
        surface_3d,
        contour_plot,
        parallel_coordinates,
        pareto_front as plotly_pareto_front,
        convergence_animation,
        acquisition_landscape,
        uncertainty_plot,
        effects_plot,
        slice_plot,
        save_figure,
    )
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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
    # Prior knowledge
    "ParameterPrior",
    "ExperimentPrior",
    "PriorKnowledgeBuilder",
    "PriorAwareBayesianOptimizer",
    "get_prior_for_experiment",
    "create_prior_aware_optimizer",
    # Batch acquisition
    "BatchAcquisitionFunction",
    "ConstantLiarBatch",
    "LocalPenalizationBatch",
    "qExpectedImprovement",
    "suggest_batch",
    # Convergence detection
    "StoppingCriteria",
    "StoppingState",
    "ConvergenceMonitor",
    "PlateauDetector",
    "BudgetAdvisor",
    "create_convergence_monitor",
    # Model selection
    "KernelFamily",
    "KernelConfig",
    "ModelScore",
    "create_kernel",
    "KernelSelector",
    "HyperparameterConfig",
    "HyperparameterTuner",
    "GPEnsemble",
    "AutoML",
    # Sensitivity analysis
    "SobolIndices",
    "MorrisResult",
    "LocalSensitivity",
    "compute_sobol_indices",
    "compute_sobol_from_surrogate",
    "compute_morris",
    "compute_morris_from_surrogate",
    "compute_local_sensitivity",
    "compute_local_sensitivity_from_surrogate",
    "correlation_sensitivity",
    "main_effect_indices",
    "SensitivityAnalyzer",
    # Robust optimization
    "RiskMeasure",
    "UncertaintySet",
    "RobustResult",
    "compute_cvar",
    "compute_var",
    "compute_mean_variance",
    "compute_entropic_risk",
    "RobustAcquisition",
    "RobustExpectedImprovement",
    "WorstCaseAcquisition",
    "CVaRAcquisition",
    "RobustOptimizer",
    "robust_evaluation",
    "DistributionallyRobustOptimizer",
    "create_robust_optimizer",
    # Interactive Plotly visualizations (when available)
    "PlotlyTheme",
    "LIGHT_THEME",
    "DARK_THEME",
    "surface_3d",
    "contour_plot",
    "parallel_coordinates",
    "plotly_pareto_front",
    "convergence_animation",
    "acquisition_landscape",
    "uncertainty_plot",
    "effects_plot",
    "slice_plot",
    "save_figure",
    "PLOTLY_AVAILABLE",
]
