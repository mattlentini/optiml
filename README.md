# OptiML 🎯

<p align="center">
  <img src="optiml_logo.png" alt="OptiML Logo" width="200"/>
</p>

<p align="center">
  <strong>Free, Open-Source Bayesian Optimization for Analytical Development</strong>
</p>

<p align="center">
  <em>The scientist's tool for method development and experiment optimization</em>
</p>

---

OptiML is a powerful yet accessible Bayesian optimization tool designed specifically for **analytical development** in biotechnology, pharmaceuticals, and research. It provides both a no-code desktop application and a Python library for programmatic access.

> **Perfect for:** Analytical scientists, method developers, process engineers, and researchers who need to optimize HPLC methods, formulations, bioassays, and other complex experiments with minimal trial runs.

## ✨ Key Features

### 🖥️ Desktop Application (No Code Required)
- **Modern Dark UI** - Clean interface designed for long lab sessions
- **Method Templates** - Pre-built templates for HPLC, LC-MS, bioassays, formulation, and more
- **Guided Workflow** - Step-by-step wizard for setting up experiments
- **AI-Powered Suggestions** - Intelligent parameter recommendations using Bayesian optimization
- **Lab Notebook** - Document observations, issues, decisions, and milestones
- **QbD Reports** - Generate Quality by Design reports with design space visualization
- **SQLite Database** - Persistent storage for all your experiments
- **Export Results** - Export trials to CSV or JSON for further analysis

### 🐍 Python Library (For Developers)
- **Easy-to-use API** - Simple `suggest`/`tell` interface or all-in-one `optimize` method
- **Gaussian Process surrogate** - Automatic hyperparameter tuning with marginal likelihood
- **Multiple acquisition functions** - Expected Improvement, UCB, Probability of Improvement, LCB
- **Flexible search spaces** - Real, Integer, and Categorical dimensions with log-scale support
- **Reproducible results** - Full random state control for experiments

## 🚀 Quick Start

### Desktop Application

```bash
# Clone the repository
git clone https://github.com/mattlentini/optiml.git
cd optiml

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[app]"

# Launch the app
python app/main.py
```

### Python Library

```bash
pip install -e "."
```

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

## 🧪 Analytical Development Examples

### HPLC Method Optimization

```python
from optiml import BayesianOptimizer, Space, Real, Integer, Categorical

# Define HPLC method parameters
space = Space([
    Real(20, 50, name="column_temp"),           # °C
    Real(0.5, 2.0, name="flow_rate"),           # mL/min
    Real(5, 40, name="organic_initial"),        # % B
    Real(60, 95, name="organic_final"),         # % B
    Real(5, 30, name="gradient_time"),          # min
    Categorical(["ACN", "MeOH"], name="organic_modifier"),
])

def evaluate_separation(params):
    # Run your HPLC method and return resolution
    # (In practice, this would interface with your instrument)
    return measured_resolution

optimizer = BayesianOptimizer(space, maximize=True)
result = optimizer.optimize(evaluate_separation, n_iterations=20)
```

### Bioassay Optimization

```python
space = Space([
    Real(0.1, 10, name="cell_density", log_scale=True),  # cells/mL
    Real(1, 48, name="incubation_time"),                  # hours
    Real(100, 1000, name="substrate_conc"),               # µg/mL
    Integer(4, 10, name="ph"),
])

optimizer = BayesianOptimizer(space, minimize=True)  # Minimize CV%
```

## 📁 Project Structure

```
OptiML/
├── app/                          # Desktop Application (Flet)
│   ├── main.py                   # App entry point, navigation
│   ├── assets/                   # Logo and images
│   ├── core/
│   │   ├── colors.py            # Theme color palette
│   │   ├── database.py          # SQLite persistence layer
│   │   ├── reports.py           # QbD report generation
│   │   ├── session.py           # Data models (Experiment, Trial, etc.)
│   │   └── templates.py         # Method development templates
│   └── views/
│       ├── home.py              # Landing page
│       ├── new_experiment.py    # Experiment creation wizard
│       ├── optimization.py      # Main optimization workflow
│       ├── notebook.py          # Lab notebook/journal
│       └── results.py           # Visualizations and export
├── src/optiml/                   # Python Library
│   ├── __init__.py              # Package exports
│   ├── optimizer.py             # BayesianOptimizer class
│   ├── space.py                 # Search space (Real, Integer, Categorical)
│   ├── surrogate.py             # Gaussian Process model
│   └── acquisition.py           # Acquisition functions
├── tests/                        # Test suite (pytest)
├── examples/                     # Usage examples
└── pyproject.toml               # Package configuration
```

## 🔧 Search Space Dimensions

### Real (Continuous)
```python
# Linear scale
column_temp = Real(20.0, 50.0, name="column_temp")

# Log scale (for concentrations spanning orders of magnitude)
concentration = Real(0.01, 100, name="concentration", log_scale=True)
```

### Integer (Discrete)
```python
ph = Integer(4, 10, name="ph")
replicate_count = Integer(3, 6, name="replicates")
```

### Categorical
```python
buffer = Categorical(["phosphate", "tris", "acetate"], name="buffer")
column = Categorical(["C18", "C8", "phenyl"], name="column_type")
```

## 📊 Acquisition Functions

```python
from optiml import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement

# Expected Improvement (default) - balanced exploration/exploitation
optimizer = BayesianOptimizer(space, acquisition=ExpectedImprovement(xi=0.01))

# Upper Confidence Bound - more exploration
optimizer = BayesianOptimizer(space, acquisition=UpperConfidenceBound(kappa=2.0))

# Probability of Improvement - more exploitation
optimizer = BayesianOptimizer(space, acquisition=ProbabilityOfImprovement(xi=0.01))
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=optiml

# Run specific test file
pytest tests/test_optimizer.py -v
```

## 🛠️ Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/optiml tests app

# Lint
ruff check src/optiml tests

# Type check
mypy src/optiml
```

## 🎯 Why OptiML?

| Feature | OptiML | JMP | Minitab | Optuna |
|---------|--------|-----|---------|--------|
| Free & Open Source | ✅ | ❌ ($2,500+) | ❌ ($1,500+) | ✅ |
| No Coding Required | ✅ | ✅ | ✅ | ❌ |
| Bayesian Optimization | ✅ | ✅ | Limited | ✅ |
| Desktop App | ✅ | ✅ | ✅ | ❌ |
| Python API | ✅ | Limited | Limited | ✅ |
| Method Templates | ✅ | Limited | Limited | ❌ |
| Lab Notebook | ✅ | ❌ | ❌ | ❌ |
| QbD Reports | ✅ | ✅ | Limited | ❌ |
| SQLite Database | ✅ | Proprietary | Proprietary | ❌ |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details.

---

<p align="center">
  <strong>Made with ❤️ for scientists who deserve free, powerful optimization tools.</strong>
</p>
