# OptiML - Bayesian Optimization for Analytical Development

## Project Overview
OptiML is a free, open-source Bayesian optimization tool designed for analytical scientists and engineers. It provides both a no-code desktop application and a Python library for programmatic access. Think of it as a free alternative to JMP or Minitab, specifically focused on Bayesian optimization for method development.

**Target Users:** Analytical scientists, method developers, process engineers, and researchers in biotechnology, pharmaceuticals, and academia.

## Project Structure
```
OptiML/
├── app/                          # Desktop Application (Flet)
│   ├── main.py                   # App entry point, navigation, theming
│   ├── assets/                   # Logo (PNG/SVG) and images
│   ├── core/
│   │   ├── __init__.py          # Core module exports
│   │   ├── colors.py            # Theme color palette (matches logo)
│   │   ├── database.py          # SQLite persistence layer
│   │   ├── reports.py           # QbD report generation (HTML/charts)
│   │   ├── session.py           # Data models (Experiment, Trial, Parameter, NotebookEntry)
│   │   └── templates.py         # Method development templates (14 categories)
│   └── views/
│       ├── __init__.py          # View exports
│       ├── home.py              # Landing page with features overview
│       ├── new_experiment.py    # 4-step wizard (Details → Template → Parameters → Review)
│       ├── optimization.py      # Main optimization workflow (suggest/record)
│       ├── notebook.py          # Lab notebook/journal for documentation
│       └── results.py           # Visualizations, statistics, export
├── src/optiml/                   # Python Library
│   ├── __init__.py              # Package exports
│   ├── optimizer.py             # Main BayesianOptimizer class
│   ├── space.py                 # Search space definitions (Real, Integer, Categorical)
│   ├── surrogate.py             # Gaussian Process surrogate model
│   └── acquisition.py           # Acquisition functions (EI, UCB, PI, LCB)
├── tests/                        # Test suite with pytest (33 tests)
├── examples/                     # Usage examples
└── pyproject.toml               # Package configuration
```

## Development Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Install in development mode (library only)
pip install -e ".[dev]"

# Install with app dependencies
pip install -e ".[app]"

# Run the desktop app
python app/main.py

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=optiml

# Format code
black src/optiml tests app

# Lint
ruff check src/optiml tests

# Type check
mypy src/optiml
```

## Desktop App Architecture

### Views (5 total)
- **HomeView** (`/`): Landing page with logo, feature overview, recent experiments from database
- **NewExperimentView** (`/new`): 4-step wizard (Details → Template → Parameters → Review)
- **OptimizationView** (`/optimize`): Main workflow - get AI suggestions, record results
- **NotebookView** (`/notebook`): Lab journal with entries (notes, observations, issues, decisions, milestones)
- **ResultsView** (`/results`): Charts, statistics, QbD reports, CSV export

### Core Components
- **Session**: Manages current experiment, optimizer state, database persistence
- **Experiment**: Contains parameters, trials, notebook entries, objective settings
- **Parameter**: Real, Integer, or Categorical with units and constraints
- **Trial**: Single experimental run with parameters, result, and metadata
- **NotebookEntry**: Lab journal entry with type, tags, and optional trial link
- **Database**: SQLite persistence at `~/.optiml/optiml.db`

### Theme (from logo)
- Dark navy background: `#060F1F`
- Primary blue: `#2951AA`
- Light blue accent: `#8AAAE9`
- Text: `#E9F0FB`
- Color palette centralized in `app/core/colors.py`

### Method Templates (14 categories)
Categories: Chromatography, Spectroscopy, Mass Spectrometry, Bioassays, Sample Preparation, Formulation, Process Development, Stability

## Python Library Components
- **BayesianOptimizer**: Main optimizer with `suggest()`/`tell()` interface or `optimize()` method
- **Space**: Container for search space dimensions
- **Real/Integer/Categorical**: Dimension types with validation
- **GaussianProcessSurrogate**: GP model with automatic hyperparameter tuning
- **ExpectedImprovement/UCB/PI/LCB**: Acquisition functions for exploration-exploitation

## Key Design Principles
1. **No-code first**: Desktop app should be usable without programming knowledge
2. **Biotech-focused**: Terminology and templates tailored for analytical development
3. **Guided workflows**: Step-by-step wizards help users set up experiments correctly
4. **Visual feedback**: Show optimization progress with charts and statistics
5. **Data persistence**: SQLite database for experiments, JSON export for portability
6. **Cross-platform**: Works on macOS, Windows, and Linux via Flet

## Dependencies
### Core Library
- numpy, scipy, scikit-learn

### Desktop App
- flet (cross-platform UI framework)
- matplotlib (for charts)
- pillow (for image handling)

### Development
- pytest, pytest-cov, black, ruff, mypy

## Database Schema
Location: `~/.optiml/optiml.db`

Tables:
- `experiments`: id, name, description, objective_name, minimize, template_id, created_at, updated_at, archived
- `parameters`: id, experiment_id, name, param_type, low, high, log_scale, categories, unit, description
- `trials`: id, experiment_id, trial_number, parameters (JSON), objective_value, response_values (JSON), timestamp, notes, metadata (JSON)
- `settings`: key, value (for app preferences)
