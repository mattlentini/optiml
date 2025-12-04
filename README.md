# OptiML

**Advanced Statistical Modeling with Easy-to-Use Bayesian Optimization**

OptiML is a Python library for black-box optimization using Bayesian optimization techniques. It provides an intuitive API for optimizing expensive-to-evaluate functions, making it ideal for hyperparameter tuning, experimental design, and other optimization tasks.

## Features

- 🚀 **Easy-to-use API** - Simple `suggest`/`tell` interface or all-in-one `optimize` method
- 📊 **Gaussian Process surrogate** - Automatic hyperparameter tuning with marginal likelihood
- 🎯 **Multiple acquisition functions** - Expected Improvement, UCB, Probability of Improvement
- 🔢 **Flexible search spaces** - Real, Integer, and Categorical dimensions with log-scale support
- 🎲 **Reproducible results** - Full random state control for experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/OptiML.git
cd OptiML

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from optiml import BayesianOptimizer, Space, Real

# Define your objective function
def objective(params):
    x, y = params
    return -(x - 2)**2 - (y - 3)**2  # Maximum at (2, 3)

# Define the search space
space = Space([
    Real(0, 5, name="x"),
    Real(0, 5, name="y"),
])

# Create optimizer and run
optimizer = BayesianOptimizer(space, maximize=True)
result = optimizer.optimize(objective, n_iterations=25)

print(f"Best parameters: {result.x_best}")
print(f"Best value: {result.y_best}")
```

### Hyperparameter Tuning

```python
from optiml import BayesianOptimizer, Space, Real, Integer, Categorical

# Define hyperparameter search space
space = Space([
    Real(1e-5, 1e-1, name="learning_rate", log_scale=True),
    Integer(1, 5, name="n_layers"),
    Integer(32, 256, name="hidden_size"),
    Categorical(["relu", "tanh", "gelu"], name="activation"),
])

def train_and_evaluate(params):
    lr, n_layers, hidden_size, activation = params
    # Your model training code here
    # Return validation metric
    return validation_accuracy

optimizer = BayesianOptimizer(space, maximize=True, n_initial=10)
result = optimizer.optimize(train_and_evaluate, n_iterations=50)
```

### Manual Suggest/Tell Interface

For more control, use the suggest/tell interface:

```python
optimizer = BayesianOptimizer(space)

for i in range(50):
    # Get next point to evaluate
    params = optimizer.suggest()
    
    # Evaluate (could be an expensive experiment)
    value = expensive_experiment(params)
    
    # Record the observation
    optimizer.tell(params, value)

result = optimizer.get_result()
```

## Search Space Dimensions

### Real (Continuous)

```python
# Linear scale
x = Real(0.0, 10.0, name="x")

# Log scale (for learning rates, regularization, etc.)
lr = Real(1e-5, 1e-1, name="learning_rate", log_scale=True)
```

### Integer (Discrete)

```python
n_layers = Integer(1, 10, name="n_layers")
batch_size = Integer(16, 256, name="batch_size")
```

### Categorical

```python
activation = Categorical(["relu", "tanh", "sigmoid"], name="activation")
optimizer_type = Categorical(["adam", "sgd", "rmsprop"], name="optimizer")
```

## Acquisition Functions

OptiML supports multiple acquisition functions:

```python
from optiml import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement

# Expected Improvement (default)
optimizer = BayesianOptimizer(space, acquisition=ExpectedImprovement(xi=0.01))

# Upper Confidence Bound
optimizer = BayesianOptimizer(space, acquisition=UpperConfidenceBound(kappa=2.0))

# Probability of Improvement
optimizer = BayesianOptimizer(space, acquisition=ProbabilityOfImprovement(xi=0.01))
```

## Examples

Check out the `examples/` directory for more detailed examples:

- `basic_usage.py` - Simple function optimization
- `hyperparameter_tuning.py` - ML hyperparameter optimization with scikit-learn

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=optiml

# Run specific test file
pytest tests/test_optimizer.py
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/optiml tests

# Lint
ruff check src/optiml tests

# Type check
mypy src/optiml
```

## License

MIT License - see LICENSE file for details.
