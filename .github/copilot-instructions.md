# OptiML - Bayesian Optimization Library

## Project Overview
Advanced statistical modeling tool with easy-to-use Bayesian optimizer in Python.

## Project Structure
```
src/optiml/
├── __init__.py          # Package exports
├── optimizer.py         # Main BayesianOptimizer class
├── space.py             # Search space definitions (Real, Integer, Categorical)
├── surrogate.py         # Gaussian Process surrogate model
└── acquisition.py       # Acquisition functions (EI, UCB, PI)

tests/                   # Test suite with pytest
examples/                # Usage examples
```

## Development Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=optiml

# Format code
black src/optiml tests

# Lint
ruff check src/optiml tests

# Type check
mypy src/optiml
```

## Key Components
- **BayesianOptimizer**: Main optimizer with suggest/tell interface
- **Space**: Container for search space dimensions
- **Real/Integer/Categorical**: Dimension types for different parameter kinds
- **GaussianProcessSurrogate**: GP model with automatic hyperparameter tuning
- **ExpectedImprovement/UCB/PI**: Acquisition functions for exploration-exploitation

## Dependencies
- numpy, scipy, scikit-learn (core)
- pytest, black, ruff, mypy (development)
