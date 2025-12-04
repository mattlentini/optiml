"""
Session management for OptiML application.
Tailored for Analytical Development in Biotechnology.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import sys
import os

# Add src to path for optiml imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from optiml import BayesianOptimizer, Space, Real, Integer, Categorical


@dataclass
class Parameter:
    """Represents a single method parameter (factor)."""
    name: str
    param_type: str  # 'real', 'integer', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    log_scale: bool = False
    categories: Optional[List[str]] = None
    # New fields for biotech AD
    unit: Optional[str] = None  # e.g., "µg/mL", "°C", "min"
    description: str = ""
    constraint_min: Optional[float] = None  # Hard constraint minimum
    constraint_max: Optional[float] = None  # Hard constraint maximum
    
    def to_dimension(self):
        """Convert to optiml dimension."""
        if self.param_type == 'real':
            return Real(self.low, self.high, name=self.name, log_scale=self.log_scale)
        elif self.param_type == 'integer':
            return Integer(int(self.low), int(self.high), name=self.name)
        elif self.param_type == 'categorical':
            return Categorical(self.categories, name=self.name)
        raise ValueError(f"Unknown parameter type: {self.param_type}")
    
    def format_value(self, value: Any) -> str:
        """Format a value with its unit."""
        if isinstance(value, float):
            formatted = f"{value:.6g}"
        else:
            formatted = str(value)
        if self.unit:
            return f"{formatted} {self.unit}"
        return formatted
    
    def display_range(self) -> str:
        """Get display string for parameter range."""
        if self.param_type == 'categorical':
            return f"Options: {', '.join(self.categories)}"
        unit_str = f" {self.unit}" if self.unit else ""
        scale_str = " (log)" if self.log_scale else ""
        return f"{self.low} - {self.high}{unit_str}{scale_str}"


@dataclass
class Response:
    """Represents a response/objective to optimize."""
    name: str
    minimize: bool = True
    unit: Optional[str] = None
    weight: float = 1.0  # For multi-objective weighting
    target_value: Optional[float] = None  # For target-based optimization
    description: str = ""


@dataclass
class Trial:
    """Represents a single experimental run."""
    trial_number: int
    parameters: Dict[str, Any]
    objective_value: Optional[float] = None
    # Support for multiple responses
    response_values: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    # Additional metadata for AD
    run_order: Optional[int] = None  # Randomized run order
    operator: str = ""
    instrument_id: str = ""


@dataclass
class NotebookEntry:
    """Represents a notebook/lab journal entry."""
    entry_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    title: str = ""
    content: str = ""
    entry_type: str = "note"  # 'note', 'observation', 'issue', 'decision', 'milestone'
    related_trial: Optional[int] = None  # Link to a specific trial
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "title": self.title,
            "content": self.content,
            "entry_type": self.entry_type,
            "related_trial": self.related_trial,
            "tags": self.tags,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NotebookEntry":
        return NotebookEntry(
            entry_id=data.get("entry_id", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            title=data.get("title", ""),
            content=data.get("content", ""),
            entry_type=data.get("entry_type", "note"),
            related_trial=data.get("related_trial"),
            tags=data.get("tags", []),
        )


@dataclass 
class Experiment:
    """Represents a method development study."""
    name: str
    description: str = ""
    objective_name: str = "Response"  # Changed from "Objective"
    minimize: bool = True
    parameters: List[Parameter] = field(default_factory=list)
    trials: List[Trial] = field(default_factory=list)
    notebook_entries: List[NotebookEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_notebook_entry(self, title: str, content: str, entry_type: str = "note",
                           related_trial: Optional[int] = None, tags: List[str] = None) -> NotebookEntry:
        """Add a new notebook entry."""
        entry_id = len(self.notebook_entries) + 1
        entry = NotebookEntry(
            entry_id=entry_id,
            title=title,
            content=content,
            entry_type=entry_type,
            related_trial=related_trial,
            tags=tags or [],
        )
        self.notebook_entries.append(entry)
        return entry
    
    def get_space(self) -> Space:
        """Create optiml Space from parameters."""
        dimensions = [p.to_dimension() for p in self.parameters]
        return Space(dimensions)
    
    def get_optimizer(self, use_prior: bool = False, prior_weight: float = 0.5, db=None) -> BayesianOptimizer:
        """Create optimizer with current trials.
        
        Args:
            use_prior: If True, use prior knowledge from historical experiments.
            prior_weight: Weight for prior vs exploration (0-1). Higher = more prior influence.
            db: Database instance for prior knowledge. Uses global db if not provided.
        
        Returns:
            BayesianOptimizer configured with optional prior knowledge.
        """
        space = self.get_space()
        # BayesianOptimizer uses maximize=True/False, so we invert minimize
        
        if use_prior:
            # Import prior knowledge components
            from optiml.priors import get_prior_for_experiment, PriorAwareBayesianOptimizer
            
            if db is None:
                from .database import get_database
                db = get_database()
            
            # Convert parameters to dict format for prior lookup
            param_dicts = [
                {
                    'name': p.name,
                    'param_type': p.param_type,
                    'low': p.low,
                    'high': p.high,
                }
                for p in self.parameters
            ]
            
            # Get prior knowledge from similar experiments
            prior = get_prior_for_experiment(db, param_dicts)
            
            # Create prior-aware optimizer
            optimizer = PriorAwareBayesianOptimizer(
                space, 
                prior, 
                prior_weight=prior_weight,
                maximize=not self.minimize
            )
        else:
            optimizer = BayesianOptimizer(space, maximize=not self.minimize)
        
        # Add existing trials
        for trial in self.trials:
            if trial.objective_value is not None:
                x = [trial.parameters[p.name] for p in self.parameters]
                optimizer.tell(x, trial.objective_value)
        
        return optimizer
    
    def get_prior_info(self, db=None) -> Dict[str, Any]:
        """Get information about available prior knowledge.
        
        Returns a summary of similar experiments and what prior knowledge
        would be used for optimization.
        
        Args:
            db: Database instance. Uses global db if not provided.
            
        Returns:
            Dict with prior knowledge summary.
        """
        from optiml.priors import PriorKnowledgeBuilder
        
        if db is None:
            from .database import get_database
            db = get_database()
        
        # Convert parameters to dict format
        param_dicts = [
            {
                'name': p.name,
                'param_type': p.param_type,
                'low': p.low,
                'high': p.high,
            }
            for p in self.parameters
        ]
        
        builder = PriorKnowledgeBuilder(db)
        prior = builder.build_experiment_prior(param_dicts)
        
        # Build summary
        summary = {
            'has_prior': prior.n_experiments > 0,
            'n_similar_experiments': prior.n_experiments,
            'n_historical_trials': prior.n_trials,
            'similar_experiments': prior.metadata.get('similar_experiments', []),
            'parameter_priors': {},
        }
        
        for name, param_prior in prior.parameter_priors.items():
            summary['parameter_priors'][name] = {
                'confidence': param_prior.confidence,
                'mean_optimal': param_prior.mean_optimal,
                'std_optimal': param_prior.std_optimal,
                'n_samples': len(param_prior.best_values),
            }
            if param_prior.value_counts:
                summary['parameter_priors'][name]['category_probs'] = param_prior.get_category_probabilities()
        
        return summary
    
    def get_best_trial(self) -> Optional[Trial]:
        """Get the best trial so far."""
        completed = [t for t in self.trials if t.objective_value is not None]
        if not completed:
            return None
        
        if self.minimize:
            return min(completed, key=lambda t: t.objective_value)
        return max(completed, key=lambda t: t.objective_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "objective_name": self.objective_name,
            "minimize": self.minimize,
            "created_at": self.created_at,
            "parameters": [
                {
                    "name": p.name,
                    "param_type": p.param_type,
                    "low": p.low,
                    "high": p.high,
                    "log_scale": p.log_scale,
                    "categories": p.categories,
                    "unit": p.unit,
                    "description": p.description,
                    "constraint_min": p.constraint_min,
                    "constraint_max": p.constraint_max,
                }
                for p in self.parameters
            ],
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "parameters": t.parameters,
                    "objective_value": t.objective_value,
                    "timestamp": t.timestamp,
                    "notes": t.notes,
                    "response_values": t.response_values,
                    "run_order": t.run_order,
                    "operator": t.operator,
                    "instrument_id": t.instrument_id,
                }
                for t in self.trials
            ],
            "notebook_entries": [e.to_dict() for e in self.notebook_entries],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        # Parse parameters with backward compatibility for new fields
        params = []
        for p in data.get("parameters", []):
            params.append(Parameter(
                name=p["name"],
                param_type=p["param_type"],
                low=p.get("low"),
                high=p.get("high"),
                log_scale=p.get("log_scale", False),
                categories=p.get("categories"),
                unit=p.get("unit", ""),
                description=p.get("description", ""),
                constraint_min=p.get("constraint_min"),
                constraint_max=p.get("constraint_max"),
            ))
        
        # Parse trials with backward compatibility for new fields
        trials = []
        for t in data.get("trials", []):
            trials.append(Trial(
                trial_number=t["trial_number"],
                parameters=t["parameters"],
                objective_value=t.get("objective_value"),
                timestamp=t.get("timestamp", datetime.now().isoformat()),
                notes=t.get("notes", ""),
                response_values=t.get("response_values", {}),
                run_order=t.get("run_order"),
                operator=t.get("operator", ""),
                instrument_id=t.get("instrument_id", ""),
            ))
        
        # Parse notebook entries
        notebook_entries = []
        for e in data.get("notebook_entries", []):
            notebook_entries.append(NotebookEntry.from_dict(e))
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            objective_name=data.get("objective_name", "Response"),
            minimize=data.get("minimize", True),
            parameters=params,
            trials=trials,
            notebook_entries=notebook_entries,
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


class Session:
    """Application session state with SQLite database persistence."""
    
    def __init__(self, use_database: bool = True):
        """Initialize session.
        
        Args:
            use_database: If True, persist to SQLite. If False, memory only.
        """
        self.current_experiment: Optional[Experiment] = None
        self.current_experiment_id: Optional[int] = None  # Database ID
        self.experiments: List[Experiment] = []
        self.suggested_params: Optional[Dict[str, Any]] = None
        self.use_database = use_database
        self._db = None
        
        if use_database:
            from .database import get_database
            self._db = get_database()
    
    @property
    def db(self):
        """Get database instance."""
        if self._db is None and self.use_database:
            from .database import get_database
            self._db = get_database()
        return self._db
    
    def new_experiment(self, name: str, description: str = "", template_id: Optional[str] = None) -> Experiment:
        """Create a new experiment."""
        exp = Experiment(name=name, description=description)
        self.current_experiment = exp
        self.experiments.append(exp)
        
        # Save to database
        if self.db:
            self.current_experiment_id = self.db.create_experiment(
                name=name,
                description=description,
                objective_name=exp.objective_name,
                minimize=exp.minimize,
                template_id=template_id,
            )
        
        return exp
    
    def save_current_to_db(self):
        """Save current experiment state to database."""
        if not self.db or not self.current_experiment:
            return
        
        exp = self.current_experiment
        
        # Create new if no ID
        if not self.current_experiment_id:
            self.current_experiment_id = self.db.create_experiment(
                name=exp.name,
                description=exp.description,
                objective_name=exp.objective_name,
                minimize=exp.minimize,
            )
        
        # Update experiment
        self.db.update_experiment(
            self.current_experiment_id,
            name=exp.name,
            description=exp.description,
            objective_name=exp.objective_name,
            minimize=exp.minimize,
        )
        
        # Note: Parameters and trials are saved incrementally
    
    def load_experiment_from_db(self, experiment_id: int) -> Optional[Experiment]:
        """Load an experiment from the database."""
        if not self.db:
            return None
        
        data = self.db.get_experiment(experiment_id)
        if not data:
            return None
        
        # Convert database format to Experiment
        params = []
        for p in data['parameters']:
            params.append(Parameter(
                name=p['name'],
                param_type=p['param_type'],
                low=p.get('low'),
                high=p.get('high'),
                log_scale=p.get('log_scale', False),
                categories=p.get('categories'),
                unit=p.get('unit', ''),
                description=p.get('description', ''),
                constraint_min=p.get('constraint_min'),
                constraint_max=p.get('constraint_max'),
            ))
        
        trials = []
        for t in data['trials']:
            trials.append(Trial(
                trial_number=t['trial_number'],
                parameters=t['parameters'],
                objective_value=t.get('objective_value'),
                response_values=t.get('response_values', {}),
                timestamp=t.get('timestamp', datetime.now().isoformat()),
                notes=t.get('notes', ''),
                run_order=t.get('run_order'),
                operator=t.get('operator', ''),
                instrument_id=t.get('instrument_id', ''),
            ))
        
        exp = Experiment(
            name=data['name'],
            description=data.get('description', ''),
            objective_name=data.get('objective_name', 'Response'),
            minimize=data.get('minimize', True),
            parameters=params,
            trials=trials,
            created_at=data.get('created_at', datetime.now().isoformat()),
        )
        
        self.current_experiment = exp
        self.current_experiment_id = experiment_id
        if exp not in self.experiments:
            self.experiments.append(exp)
        
        return exp
    
    def list_experiments_from_db(self) -> List[Dict[str, Any]]:
        """List all experiments from database (summary info only)."""
        if not self.db:
            return []
        
        experiments = self.db.list_experiments()
        # Return summary info for display
        return [{
            'id': e['id'],
            'name': e['name'],
            'description': e['description'],
            'created_at': e['created_at'],
            'updated_at': e['updated_at'],
            'trial_count': len(e.get('trials', [])),
        } for e in experiments]
    
    def delete_experiment_from_db(self, experiment_id: int) -> bool:
        """Delete an experiment from the database."""
        if not self.db:
            return False
        return self.db.delete_experiment(experiment_id)
    
    def suggest_next(self) -> Dict[str, Any]:
        """Get next suggested parameters."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        optimizer = self.current_experiment.get_optimizer()
        x = optimizer.suggest()
        
        # Convert to named dict
        params = {}
        for i, p in enumerate(self.current_experiment.parameters):
            params[p.name] = x[i]
        
        self.suggested_params = params
        return params
    
    def record_result(self, objective_value: float, notes: str = "") -> Trial:
        """Record the result of the suggested parameters."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        if not self.suggested_params:
            raise ValueError("No suggested parameters")
        
        trial = Trial(
            trial_number=len(self.current_experiment.trials) + 1,
            parameters=self.suggested_params.copy(),
            objective_value=objective_value,
            notes=notes,
        )
        self.current_experiment.trials.append(trial)
        
        # Save to database
        if self.db and self.current_experiment_id:
            self.db.add_trial(
                experiment_id=self.current_experiment_id,
                trial_number=trial.trial_number,
                parameters=trial.parameters,
                objective_value=trial.objective_value,
                notes=trial.notes,
            )
        
        self.suggested_params = None
        return trial
    
    def add_parameters_to_db(self):
        """Save current experiment's parameters to database."""
        if not self.db or not self.current_experiment or not self.current_experiment_id:
            return
        
        for i, p in enumerate(self.current_experiment.parameters):
            self.db.add_parameter(
                experiment_id=self.current_experiment_id,
                name=p.name,
                param_type=p.param_type,
                low=p.low,
                high=p.high,
                log_scale=p.log_scale,
                categories=p.categories,
                unit=p.unit or '',
                description=p.description,
                constraint_min=p.constraint_min,
                constraint_max=p.constraint_max,
                sort_order=i,
            )
    
    def save_experiment(self, filepath: str):
        """Save current experiment to file."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        with open(filepath, 'w') as f:
            json.dump(self.current_experiment.to_dict(), f, indent=2)
    
    def load_experiment(self, filepath: str) -> Experiment:
        """Load experiment from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        exp = Experiment.from_dict(data)
        self.current_experiment = exp
        self.experiments.append(exp)
        return exp
    
    def export_trials_csv(self, filepath: str):
        """Export trials to CSV."""
        if not self.current_experiment:
            raise ValueError("No active experiment")
        
        import csv
        
        exp = self.current_experiment
        if not exp.trials:
            raise ValueError("No trials to export")
        
        param_names = [p.name for p in exp.parameters]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Trial', *param_names, exp.objective_name, 'Timestamp', 'Notes']
            writer.writerow(header)
            
            # Data
            for trial in exp.trials:
                row = [
                    trial.trial_number,
                    *[trial.parameters.get(name, '') for name in param_names],
                    trial.objective_value if trial.objective_value is not None else '',
                    trial.timestamp,
                    trial.notes,
                ]
                writer.writerow(row)
