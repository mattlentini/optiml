# Advanced Features Guide

This guide covers the advanced statistical and optimization features available in OptiML.

## Table of Contents

1. [Batch/Parallel Acquisition](#batchparallel-acquisition)
2. [Prior Knowledge & Transfer Learning](#prior-knowledge--transfer-learning)
3. [Multi-Objective Optimization](#multi-objective-optimization)
4. [Constraint Handling](#constraint-handling)
5. [Advanced Statistical Analysis](#advanced-statistical-analysis)
6. [Design of Experiments](#design-of-experiments)

---

## Batch/Parallel Acquisition

When you can run multiple experiments simultaneously, batch acquisition functions suggest multiple parameter sets that balance exploration and diversity.

### Why Use Batch Acquisition?

- **Save Time**: Run experiments in parallel instead of sequentially
- **Efficient Resource Use**: Fully utilize available instruments/equipment
- **Cost-Effective**: Reduce total turnaround time for optimization studies

### Available Strategies

#### 1. Local Penalization (Recommended)

Best for most use cases. Promotes diversity by penalizing points near already-selected batch members.

```python
from optiml import BayesianOptimizer, Space, Real
from optiml.batch import suggest_batch

space = Space([
    Real(0, 100, name='Temperature'),
    Real(1, 10, name='pH'),
])

optimizer = BayesianOptimizer(space, maximize=True)

# Add initial data
for _ in range(5):
    x = optimizer.suggest()
    y = objective(x)
    optimizer.tell(x, y)

# Get 3 diverse points for parallel evaluation
batch = suggest_batch(
    optimizer,
    n_points=3,
    strategy='local_penalization',
    penalty_radius=0.1,  # Control diversity (0.01-0.5)
)

# Run all experiments in parallel
for x in batch:
    y = objective(x)
    optimizer.tell(x, y)
```

**Parameters:**
- `penalty_radius`: Smaller = less diversity (default: 0.1)
- `penalty_strength`: Higher = stronger diversity (default: 2.0)

#### 2. Constant Liar

Greedy heuristic that assumes each suggested point has a certain value ("liar" value).

```python
batch = suggest_batch(
    optimizer,
    n_points=3,
    strategy='constant_liar',
    liar_strategy='min',  # 'min', 'max', or 'mean'
)
```

**Strategies:**
- `'min'`: Pessimistic - encourages exploration
- `'max'`: Optimistic - encourages exploitation
- `'mean'`: Neutral - balanced approach

#### 3. q-Expected Improvement (qEI)

Theoretically optimal but computationally expensive. Uses Monte Carlo approximation.

```python
batch = suggest_batch(
    optimizer,
    n_points=3,
    strategy='qei',
    n_samples=100,  # MC samples (more = better but slower)
)
```

### Desktop App Usage

In the OptiML desktop app:

1. Go to Optimization view
2. Click "Batch Suggestions" button
3. Select number of experiments (2-10)
4. Choose strategy (local penalization recommended)
5. Click "Generate Batch"
6. Run all experiments
7. Record results for each one

---

## Prior Knowledge & Transfer Learning

Leverage historical experiment data to warm-start new optimizations, saving time and resources.

### Why Use Prior Knowledge?

- **Faster Convergence**: Start from informed initial guesses
- **Fewer Wasted Experiments**: Avoid known poor regions
- **Organizational Learning**: Leverage accumulated knowledge
- **Cost Savings**: Reduce total experiments needed

### How It Works

1. **Database Analysis**: OptiML scans your experiment database for similar studies
2. **Prior Building**: Extracts optimal parameter distributions from best trials
3. **Warm Starting**: Biases initial suggestions toward historically successful regions
4. **Adaptive**: Balances prior knowledge with exploration

### Using Prior Knowledge

#### In Python Code

```python
from optiml import BayesianOptimizer, Space, Real
from optiml.priors import PriorAwareBayesianOptimizer, ExperimentPrior, ParameterPrior
import numpy as np

# Define your space
space = Space([
    Real(0, 100, name='Temperature'),
    Real(1, 10, name='pH'),
])

# Build prior from historical data
param_priors = {
    'Temperature': ParameterPrior(
        name='Temperature',
        param_type='real',
        mean_optimal=75.0,  # Historical optimal mean
        std_optimal=5.0,    # Spread of optima
        confidence=0.7,     # Confidence in prior (0-1)
        low=0, high=100,
    ),
    'pH': ParameterPrior(
        name='pH',
        param_type='real',
        mean_optimal=7.5,
        std_optimal=0.5,
        confidence=0.8,
        low=1, high=10,
    ),
}

prior = ExperimentPrior(
    parameter_priors=param_priors,
    n_experiments=5,      # Number of similar experiments
    n_trials=75,          # Total historical trials
)

# Create prior-aware optimizer
optimizer = PriorAwareBayesianOptimizer(
    space,
    prior=prior,
    prior_weight=0.5,  # 0=ignore prior, 1=max influence
    maximize=True,
)

# Optimize as usual
for _ in range(20):
    x = optimizer.suggest()  # Biased by prior initially
    y = objective(x)
    optimizer.tell(x, y)
```

#### In Desktop App with Database

The app automatically finds similar experiments:

```python
# In your experiment setup
from app.core.session import Session

session = Session()
experiment = session.new_experiment(
    name="New Purification Study",
    description="Optimizing buffer conditions"
)

# Check available prior knowledge
prior_info = session.get_prior_knowledge_info()
print(f"Similar experiments: {prior_info['n_similar_experiments']}")
print(f"Historical trials: {prior_info['n_historical_trials']}")

# Get suggestion with prior
params = session.suggest_next(
    use_prior=True,
    prior_weight=0.6  # 60% prior, 40% exploration
)
```

### Prior Weight Tuning

| prior_weight | Behavior | When to Use |
|-------------|----------|-------------|
| 0.0 | Ignore prior completely | First time optimization, very different system |
| 0.3 | Slight bias toward prior | Low confidence in similarity |
| 0.5 | Balanced (recommended) | Moderate confidence, good starting point |
| 0.7 | Strong trust in prior | High similarity, validated methods |
| 1.0 | Maximum prior influence | Nearly identical system, known optimal region |

### Database Integration

OptiML automatically:
1. Finds experiments with similar parameter structures
2. Computes similarity scores (Jaccard index + type matching)
3. Extracts best trials from similar experiments
4. Builds parameter-specific priors
5. Provides confidence metrics

Experiments are considered similar if they share:
- Common parameter names
- Matching parameter types (real/integer/categorical)
- Same optimization objective direction

---

## Multi-Objective Optimization

Optimize multiple competing objectives simultaneously (e.g., maximize yield while minimizing cost).

### Features

- **Pareto Front Discovery**: Find all trade-off solutions
- **Multiple Scalarization Methods**: Different ways to combine objectives
- **EHVI Acquisition**: Expected Hypervolume Improvement
- **Visual Analysis**: 2D/3D Pareto front plots

### Example: Resolution vs Run Time

```python
from optiml.multi_objective import MultiObjectiveOptimizer
from optiml import Space, Real

space = Space([
    Real(5, 95, name='Organic %'),
    Real(5, 60, name='Gradient Time'),
])

# Define objective function returning multiple values
def chromatography(params):
    organic, time = params
    
    # Compute both objectives
    resolution = compute_resolution(organic, time)
    run_time = compute_run_time(organic, time)
    
    return {
        'resolution': resolution,  # Maximize
        'run_time': run_time,      # Minimize
    }

# Create multi-objective optimizer
optimizer = MultiObjectiveOptimizer(
    space,
    objectives=['resolution', 'run_time'],
    directions=['maximize', 'minimize'],
)

# Optimize
result = optimizer.optimize(chromatography, n_iterations=30)

# Get Pareto front
pareto_front = result.pareto_front
print(f"Found {len(pareto_front)} Pareto-optimal solutions")

# Visualize trade-offs
from optiml.visualization import plot_pareto_front
plot_pareto_front(result)
```

### Scalarization Strategies

Convert multi-objective to single-objective:

```python
from optiml.multi_objective import WeightedSum, Chebyshev, ParEGO

# Weighted sum: simple linear combination
scalarizer = WeightedSum(weights=[0.7, 0.3])

# Chebyshev: minimize worst-case deviation
scalarizer = Chebyshev(weights=[0.5, 0.5], reference=[10, 5])

# ParEGO: randomized weights per iteration
scalarizer = ParEGO()
```

---

## Constraint Handling

Handle experimental constraints beyond simple bounds.

### Types of Constraints

#### 1. Black-Box Constraints

Unknown in advance, learned from experiments (e.g., pressure, stability).

```python
from optiml.constraints import BlackBoxConstraint

constraint = BlackBoxConstraint(
    name="Backpressure",
    max_value=400,  # Must stay below 400 bar
)

optimizer = BayesianOptimizer(
    space,
    constraints=[constraint]
)
```

#### 2. Linear Constraints

Known analytical relationships.

```python
from optiml.constraints import LinearConstraint

# Sum of components must equal 100%
constraint = LinearConstraint(
    coefficients=[1, 1, 1],
    bound=100,
    bound_type='eq',  # 'eq', 'ineq', 'leq', 'geq'
)
```

#### 3. Nonlinear Constraints

Complex analytical constraints.

```python
from optiml.constraints import NonlinearConstraint

# Custom constraint function
def stability_constraint(x):
    ph, temp = x
    return ph * temp - 200  # Must be >= 0

constraint = NonlinearConstraint(
    function=stability_constraint,
    lower_bound=0,
)
```

---

## Advanced Statistical Analysis

### ANOVA (Analysis of Variance)

Identify statistically significant factors.

```python
from optiml.statistics import perform_anova
import numpy as np

X = np.array([...])  # Parameter values
y = np.array([...])  # Response values
param_names = ['Temperature', 'pH', 'Concentration']

result = perform_anova(X, y, param_names)

print(f"Model R²: {result.r_squared:.3f}")
for table in result.tables:
    if table.p_value < 0.05:
        print(f"{table.factor} is significant (p={table.p_value:.4f})")
```

### Effects Analysis

Quantify parameter impacts and interactions.

```python
from optiml.statistics import analyze_effects

effects = analyze_effects(X, y, param_names)

# Main effects
for effect in effects.main_effects:
    print(f"{effect.name}: {effect.value:.3f} ± {effect.stderr:.3f}")

# Interactions
for interaction in effects.interactions:
    print(f"{interaction.factors}: {interaction.value:.3f}")
```

### Partial Dependence Plots

Visualize parameter-response relationships.

```python
from optiml import calculate_all_partial_dependence
from optiml.visualization import plot_partial_dependence_grid

optimizer = BayesianOptimizer(space, maximize=True)
# ... run optimization ...

pdp_data = calculate_all_partial_dependence(
    optimizer.surrogate,
    space,
    n_grid=50
)

plot_partial_dependence_grid(pdp_data, space)
```

---

## Design of Experiments

Generate initial experiment plans.

### Latin Hypercube Sampling (Recommended)

Space-filling design ensuring good coverage.

```python
from optiml.designs import LatinHypercubeDesign

design = LatinHypercubeDesign(space, n_samples=20)
points = design.generate()

for i, x in enumerate(points):
    print(f"Experiment {i+1}: {x}")
    # Run experiment...
```

### Sobol Sequence

Low-discrepancy quasi-random sequence.

```python
from optiml.designs import SobolDesign

design = SobolDesign(space, n_samples=16)  # Power of 2 recommended
points = design.generate()
```

### Factorial Designs

Classic DOE for screening studies.

```python
from optiml.designs import (
    FullFactorialDesign,
    FractionalFactorialDesign,
    CentralCompositeDesign,
)

# Full factorial (all combinations)
design = FullFactorialDesign(space, levels=3)

# Fractional factorial (subset)
design = FractionalFactorialDesign(space, resolution=4)

# Central composite (response surface)
design = CentralCompositeDesign(space)
```

### Comparing Designs

```python
from optiml.designs import compare_designs

designs = {
    'LHS': LatinHypercubeDesign(space, 20),
    'Sobol': SobolDesign(space, 16),
    'Random': RandomDesign(space, 20),
}

comparison = compare_designs(designs, space)
print(comparison)
```

---

## Best Practices

### General Guidelines

1. **Start Simple**: Begin with standard Bayesian optimization before adding complexity
2. **Use Defaults**: The default settings work well for most problems
3. **Validate Results**: Always verify optimal conditions with replicate experiments
4. **Document Everything**: Use the notebook feature to record observations

### When to Use Each Feature

| Feature | Use When | Don't Use When |
|---------|----------|----------------|
| **Batch Acquisition** | Can run experiments in parallel | Must run sequentially |
| **Prior Knowledge** | Similar experiments exist in database | Completely new system |
| **Multi-Objective** | Multiple competing goals | Single clear objective |
| **Constraints** | Hard limits on feasible region | Only simple bounds |
| **ANOVA** | Want to understand factor significance | Just want optimal conditions |
| **LHS/Sobol** | Starting new study | Already have data |

### Performance Tips

- Use `n_initial=5` for 2-3 parameters, scale up for higher dimensions
- For expensive experiments, start with LHS design
- Enable prior knowledge when similarity > 0.5
- Use local penalization for batch sizes 2-5
- Set `prior_weight=0.5` as starting point, adjust based on results

---

## Support & Resources

- **Examples Directory**: See `/examples` for working code
- **Tests**: Check `/tests` for detailed usage patterns
- **Documentation**: See README.md and docstrings
- **Issues**: Report bugs on GitHub

---

*Last Updated: December 2024*
