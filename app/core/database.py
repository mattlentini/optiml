"""
OptiML Database Layer
======================
SQLite-based persistence for experiments, trials, and settings.

Why SQLite?
- Zero configuration, single-file database
- Built into Python (no extra dependencies)
- ACID compliant for data integrity
- Fast queries for finding best trials, filtering data
- Easy backup (just copy the .db file)
- Perfect for desktop applications
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

# Default database location
DEFAULT_DB_PATH = Path.home() / ".optiml" / "optiml.db"


class Database:
    """SQLite database manager for OptiML."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to database file. Defaults to ~/.optiml/optiml.db
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_schema(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    objective_name TEXT DEFAULT 'Response',
                    minimize INTEGER DEFAULT 1,
                    template_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived INTEGER DEFAULT 0
                )
            """)
            
            # Parameters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    param_type TEXT NOT NULL,
                    low REAL,
                    high REAL,
                    log_scale INTEGER DEFAULT 0,
                    categories TEXT,
                    unit TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    constraint_min REAL,
                    constraint_max REAL,
                    sort_order INTEGER DEFAULT 0,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                )
            """)
            
            # Trials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    trial_number INTEGER NOT NULL,
                    parameters TEXT NOT NULL,
                    objective_value REAL,
                    response_values TEXT,
                    timestamp TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    run_order INTEGER,
                    operator TEXT DEFAULT '',
                    instrument_id TEXT DEFAULT '',
                    status TEXT DEFAULT 'completed',
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                )
            """)
            
            # Settings table (key-value store)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trials_experiment 
                ON trials(experiment_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trials_objective 
                ON trials(experiment_id, objective_value)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_archived 
                ON experiments(archived)
            """)
    
    # ==================== Experiment Operations ====================
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        objective_name: str = "Response",
        minimize: bool = True,
        template_id: Optional[str] = None,
    ) -> int:
        """Create a new experiment and return its ID."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments 
                (name, description, objective_name, minimize, template_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, description, objective_name, int(minimize), template_id, now, now))
            return cursor.lastrowid
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM experiments WHERE id = ?
            """, (experiment_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_experiment(row, conn)
            return None
    
    def list_experiments(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """List all experiments."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if include_archived:
                cursor.execute("SELECT * FROM experiments ORDER BY updated_at DESC")
            else:
                cursor.execute("SELECT * FROM experiments WHERE archived = 0 ORDER BY updated_at DESC")
            
            return [self._row_to_experiment(row, conn) for row in cursor.fetchall()]
    
    def update_experiment(self, experiment_id: int, **kwargs) -> bool:
        """Update experiment fields."""
        allowed_fields = {'name', 'description', 'objective_name', 'minimize', 'archived'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        # Convert bool to int for SQLite
        if 'minimize' in updates:
            updates['minimize'] = int(updates['minimize'])
        if 'archived' in updates:
            updates['archived'] = int(updates['archived'])
        
        updates['updated_at'] = datetime.now().isoformat()
        
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [experiment_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE experiments SET {set_clause} WHERE id = ?
            """, values)
            return cursor.rowcount > 0
    
    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment and all its data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            return cursor.rowcount > 0
    
    def archive_experiment(self, experiment_id: int) -> bool:
        """Archive an experiment (soft delete)."""
        return self.update_experiment(experiment_id, archived=True)
    
    def _row_to_experiment(self, row: sqlite3.Row, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Convert database row to experiment dict with parameters and trials."""
        exp = dict(row)
        exp['minimize'] = bool(exp['minimize'])
        exp['archived'] = bool(exp['archived'])
        
        # Get parameters
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM parameters WHERE experiment_id = ? ORDER BY sort_order
        """, (exp['id'],))
        exp['parameters'] = [self._row_to_parameter(r) for r in cursor.fetchall()]
        
        # Get trials
        cursor.execute("""
            SELECT * FROM trials WHERE experiment_id = ? ORDER BY trial_number
        """, (exp['id'],))
        exp['trials'] = [self._row_to_trial(r) for r in cursor.fetchall()]
        
        return exp
    
    # ==================== Parameter Operations ====================
    
    def add_parameter(
        self,
        experiment_id: int,
        name: str,
        param_type: str,
        low: Optional[float] = None,
        high: Optional[float] = None,
        log_scale: bool = False,
        categories: Optional[List[str]] = None,
        unit: str = "",
        description: str = "",
        constraint_min: Optional[float] = None,
        constraint_max: Optional[float] = None,
        sort_order: int = 0,
    ) -> int:
        """Add a parameter to an experiment."""
        categories_json = json.dumps(categories) if categories else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO parameters
                (experiment_id, name, param_type, low, high, log_scale, categories, 
                 unit, description, constraint_min, constraint_max, sort_order)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, name, param_type, low, high, int(log_scale),
                  categories_json, unit, description, constraint_min, constraint_max, sort_order))
            
            # Update experiment timestamp
            cursor.execute("""
                UPDATE experiments SET updated_at = ? WHERE id = ?
            """, (datetime.now().isoformat(), experiment_id))
            
            return cursor.lastrowid
    
    def add_parameters_bulk(self, experiment_id: int, parameters: List[Dict[str, Any]]) -> List[int]:
        """Add multiple parameters at once."""
        ids = []
        for i, param in enumerate(parameters):
            param['sort_order'] = i
            param_id = self.add_parameter(experiment_id, **param)
            ids.append(param_id)
        return ids
    
    def _row_to_parameter(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to parameter dict."""
        param = dict(row)
        param['log_scale'] = bool(param['log_scale'])
        if param['categories']:
            param['categories'] = json.loads(param['categories'])
        return param
    
    # ==================== Trial Operations ====================
    
    def add_trial(
        self,
        experiment_id: int,
        trial_number: int,
        parameters: Dict[str, Any],
        objective_value: Optional[float] = None,
        response_values: Optional[Dict[str, float]] = None,
        notes: str = "",
        run_order: Optional[int] = None,
        operator: str = "",
        instrument_id: str = "",
        status: str = "completed",
    ) -> int:
        """Add a trial to an experiment."""
        now = datetime.now().isoformat()
        params_json = json.dumps(parameters)
        responses_json = json.dumps(response_values) if response_values else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trials
                (experiment_id, trial_number, parameters, objective_value, response_values,
                 timestamp, notes, run_order, operator, instrument_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, trial_number, params_json, objective_value, responses_json,
                  now, notes, run_order, operator, instrument_id, status))
            
            # Update experiment timestamp
            cursor.execute("""
                UPDATE experiments SET updated_at = ? WHERE id = ?
            """, (now, experiment_id))
            
            return cursor.lastrowid
    
    def update_trial(self, trial_id: int, **kwargs) -> bool:
        """Update trial fields."""
        allowed_fields = {'objective_value', 'response_values', 'notes', 
                          'run_order', 'operator', 'instrument_id', 'status'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        # JSON encode dict fields
        if 'response_values' in updates and updates['response_values'] is not None:
            updates['response_values'] = json.dumps(updates['response_values'])
        
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [trial_id]
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE trials SET {set_clause} WHERE id = ?", values)
            return cursor.rowcount > 0
    
    def get_best_trial(self, experiment_id: int, minimize: bool = True) -> Optional[Dict[str, Any]]:
        """Get the best trial for an experiment."""
        order = "ASC" if minimize else "DESC"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM trials 
                WHERE experiment_id = ? AND objective_value IS NOT NULL
                ORDER BY objective_value {order}
                LIMIT 1
            """, (experiment_id,))
            row = cursor.fetchone()
            return self._row_to_trial(row) if row else None
    
    def get_trial_count(self, experiment_id: int) -> int:
        """Get number of trials for an experiment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM trials WHERE experiment_id = ?
            """, (experiment_id,))
            return cursor.fetchone()[0]
    
    def _row_to_trial(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to trial dict."""
        trial = dict(row)
        trial['parameters'] = json.loads(trial['parameters'])
        if trial['response_values']:
            trial['response_values'] = json.loads(trial['response_values'])
        else:
            trial['response_values'] = {}
        return trial
    
    # ==================== Settings Operations ====================
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row['value'])
            return default
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value."""
        now = datetime.now().isoformat()
        value_json = json.dumps(value)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value_json, now))
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings")
            return {row['key']: json.loads(row['value']) for row in cursor.fetchall()}
    
    # ==================== Statistics & Queries ====================
    
    def get_experiment_stats(self, experiment_id: int) -> Dict[str, Any]:
        """Get statistics for an experiment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get experiment info
            cursor.execute("SELECT minimize FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()
            if not row:
                return {}
            
            minimize = bool(row['minimize'])
            
            # Get trial statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(objective_value) as completed,
                    MIN(objective_value) as min_value,
                    MAX(objective_value) as max_value,
                    AVG(objective_value) as mean_value
                FROM trials WHERE experiment_id = ?
            """, (experiment_id,))
            
            stats = dict(cursor.fetchone())
            
            # Calculate std dev
            if stats['completed'] > 1:
                cursor.execute("""
                    SELECT objective_value FROM trials 
                    WHERE experiment_id = ? AND objective_value IS NOT NULL
                """, (experiment_id,))
                values = [r['objective_value'] for r in cursor.fetchall()]
                mean = stats['mean_value']
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                stats['std_value'] = variance ** 0.5
            else:
                stats['std_value'] = 0
            
            stats['best_value'] = stats['min_value'] if minimize else stats['max_value']
            
            return stats
    
    # ==================== Import/Export ====================
    
    def export_experiment_json(self, experiment_id: int) -> Optional[str]:
        """Export experiment as JSON string."""
        exp = self.get_experiment(experiment_id)
        if exp:
            return json.dumps(exp, indent=2, default=str)
        return None
    
    def import_experiment_json(self, json_str: str) -> int:
        """Import experiment from JSON string. Returns new experiment ID."""
        data = json.loads(json_str)
        
        # Create experiment
        exp_id = self.create_experiment(
            name=data.get('name', 'Imported Experiment'),
            description=data.get('description', ''),
            objective_name=data.get('objective_name', 'Response'),
            minimize=data.get('minimize', True),
            template_id=data.get('template_id'),
        )
        
        # Add parameters
        for param in data.get('parameters', []):
            self.add_parameter(
                experiment_id=exp_id,
                name=param['name'],
                param_type=param['param_type'],
                low=param.get('low'),
                high=param.get('high'),
                log_scale=param.get('log_scale', False),
                categories=param.get('categories'),
                unit=param.get('unit', ''),
                description=param.get('description', ''),
                constraint_min=param.get('constraint_min'),
                constraint_max=param.get('constraint_max'),
            )
        
        # Add trials
        for trial in data.get('trials', []):
            self.add_trial(
                experiment_id=exp_id,
                trial_number=trial['trial_number'],
                parameters=trial['parameters'],
                objective_value=trial.get('objective_value'),
                response_values=trial.get('response_values'),
                notes=trial.get('notes', ''),
                run_order=trial.get('run_order'),
                operator=trial.get('operator', ''),
                instrument_id=trial.get('instrument_id', ''),
            )
        
        return exp_id
    
    def export_trials_csv(self, experiment_id: int) -> Optional[str]:
        """Export trials as CSV string."""
        exp = self.get_experiment(experiment_id)
        if not exp or not exp['trials']:
            return None
        
        param_names = [p['name'] for p in exp['parameters']]
        
        # Header
        lines = [",".join(["trial_number"] + param_names + [exp['objective_name'], "notes"])]
        
        # Data rows
        for trial in exp['trials']:
            row = [str(trial['trial_number'])]
            for name in param_names:
                val = trial['parameters'].get(name, '')
                row.append(str(val) if val is not None else '')
            row.append(str(trial['objective_value']) if trial['objective_value'] is not None else '')
            row.append(f'"{trial.get("notes", "")}"')
            lines.append(",".join(row))
        
        return "\n".join(lines)


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def init_database(db_path: Optional[Path] = None) -> Database:
    """Initialize the global database with optional custom path."""
    global _db
    _db = Database(db_path)
    return _db
