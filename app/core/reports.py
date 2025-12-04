"""
QbD Report Generation - Generate Quality by Design reports for method development studies.

Supports:
- Summary statistics
- Design space visualization  
- Optimal conditions identification
- Export to PDF/HTML
"""

import io
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.session import Experiment

# Import centralized color scheme
from core.colors import (
    PRIMARY, PRIMARY_LIGHT, PRIMARY_DARK,
    BG_DARK, BG_SURFACE, BG_ELEVATED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED
)

# Try to import matplotlib for charts
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_design_space_plot(experiment: "Experiment") -> Optional[str]:
    """
    Generate a design space visualization showing the explored region.
    Returns base64-encoded PNG image or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if len(experiment.trials) < 3:
        return None
    
    # Get parameter names for numeric parameters only
    numeric_params = [p for p in experiment.parameters if p.param_type != "categorical"]
    
    if len(numeric_params) < 2:
        return None
    
    # Use first two numeric parameters for 2D plot
    p1, p2 = numeric_params[0], numeric_params[1]
    
    # Extract data
    x_vals = []
    y_vals = []
    z_vals = []  # Objective values for coloring
    
    for trial in experiment.trials:
        if trial.objective_value is not None:
            x_vals.append(trial.parameters.get(p1.name, 0))
            y_vals.append(trial.parameters.get(p2.name, 0))
            z_vals.append(trial.objective_value)
    
    if not x_vals:
        return None
    
    # Create figure with theme colors
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_DARK)
    ax.set_facecolor(BG_SURFACE)
    
    # Scatter plot with color based on objective
    scatter = ax.scatter(
        x_vals, y_vals, c=z_vals, 
        cmap='RdYlGn_r' if experiment.minimize else 'RdYlGn',
        s=100, alpha=0.8, edgecolors='white', linewidth=1
    )
    
    # Mark best point
    best = experiment.get_best_trial()
    if best:
        best_x = best.parameters.get(p1.name, 0)
        best_y = best.parameters.get(p2.name, 0)
        ax.scatter([best_x], [best_y], c=PRIMARY, s=200, marker='*', 
                   edgecolors='white', linewidth=2, zorder=10, label='Optimum')
    
    # Styling
    unit1 = f" ({p1.unit})" if p1.unit else ""
    unit2 = f" ({p2.unit})" if p2.unit else ""
    ax.set_xlabel(f"{p1.name}{unit1}", color='white', fontsize=12)
    ax.set_ylabel(f"{p2.name}{unit2}", color='white', fontsize=12)
    ax.set_title("Design Space Exploration", color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color(TEXT_MUTED)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(experiment.objective_name, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.legend(loc='upper right', facecolor=BG_SURFACE, edgecolor=TEXT_MUTED, labelcolor='white')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_convergence_plot(experiment: "Experiment") -> Optional[str]:
    """
    Generate a convergence plot showing optimization progress over trials.
    Returns base64-encoded PNG image or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if len(experiment.trials) < 2:
        return None
    
    # Get objective values in order
    trials_sorted = sorted(experiment.trials, key=lambda t: t.trial_number)
    trial_nums = [t.trial_number for t in trials_sorted if t.objective_value is not None]
    obj_values = [t.objective_value for t in trials_sorted if t.objective_value is not None]
    
    if not obj_values:
        return None
    
    # Calculate best so far
    best_so_far = []
    current_best = obj_values[0]
    for val in obj_values:
        if experiment.minimize:
            current_best = min(current_best, val)
        else:
            current_best = max(current_best, val)
        best_so_far.append(current_best)
    
    # Create figure with theme colors
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_DARK)
    ax.set_facecolor(BG_SURFACE)
    
    # Plot individual points
    ax.scatter(trial_nums, obj_values, c=TEXT_MUTED, s=60, alpha=0.7, label='Individual Runs')
    
    # Plot best so far
    ax.plot(trial_nums, best_so_far, c=PRIMARY, linewidth=2, marker='o', 
            markersize=6, label='Best So Far')
    
    # Styling
    ax.set_xlabel("Run Number", color='white', fontsize=12)
    ax.set_ylabel(experiment.objective_name, color='white', fontsize=12)
    ax.set_title("Optimization Convergence", color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color(TEXT_MUTED)
    
    ax.legend(loc='best', facecolor=BG_SURFACE, edgecolor=TEXT_MUTED, labelcolor='white')
    ax.grid(True, alpha=0.2, color=TEXT_MUTED)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')


def calculate_statistics(experiment: "Experiment") -> Dict[str, Any]:
    """Calculate summary statistics for the experiment."""
    trials = [t for t in experiment.trials if t.objective_value is not None]
    
    if not trials:
        return {"n_trials": 0}
    
    obj_values = [t.objective_value for t in trials]
    
    stats = {
        "n_trials": len(trials),
        "mean": sum(obj_values) / len(obj_values),
        "min": min(obj_values),
        "max": max(obj_values),
        "range": max(obj_values) - min(obj_values),
    }
    
    # Calculate std dev
    if len(obj_values) > 1:
        mean = stats["mean"]
        variance = sum((x - mean) ** 2 for x in obj_values) / (len(obj_values) - 1)
        stats["std"] = variance ** 0.5
    else:
        stats["std"] = 0
    
    # Best trial info
    best = experiment.get_best_trial()
    if best:
        stats["best_value"] = best.objective_value
        stats["best_trial_num"] = best.trial_number
        stats["best_params"] = best.parameters.copy()
    
    return stats


def generate_html_report(experiment: "Experiment") -> str:
    """
    Generate a complete QbD HTML report for the experiment.
    """
    stats = calculate_statistics(experiment)
    best = experiment.get_best_trial()
    
    # Generate plots
    design_space_img = generate_design_space_plot(experiment)
    convergence_img = generate_convergence_plot(experiment)
    
    # Build HTML with centralized theme colors
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QbD Report: {experiment.name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: {BG_DARK};
            color: {TEXT_PRIMARY};
            line-height: 1.6;
            padding: 40px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: {PRIMARY}; font-size: 2em; margin-bottom: 0.5em; }}
        h2 {{ color: {TEXT_SECONDARY}; font-size: 1.3em; margin: 1.5em 0 0.5em; border-bottom: 1px solid #334155; padding-bottom: 0.3em; }}
        h3 {{ color: {TEXT_PRIMARY}; font-size: 1.1em; margin: 1em 0 0.3em; }}
        .meta {{ color: {TEXT_MUTED}; margin-bottom: 2em; }}
        .card {{
            background: {BG_SURFACE};
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }}
        .stat-item {{
            text-align: center;
            padding: 16px;
            background: {BG_DARK};
            border-radius: 8px;
        }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: {PRIMARY}; }}
        .stat-label {{ color: {TEXT_SECONDARY}; font-size: 0.9em; }}
        .best {{ color: #22C55E; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{ background: {BG_DARK}; color: {TEXT_SECONDARY}; font-weight: 600; }}
        tr:hover {{ background: {BG_DARK}; }}
        .optimal-row {{ background: #22C55E20 !important; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 16px 0; }}
        .footer {{ margin-top: 40px; text-align: center; color: {TEXT_MUTED}; font-size: 0.9em; }}
        @media print {{
            body {{ background: white; color: #1e293b; }}
            .card {{ background: #f1f5f9; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 {experiment.name}</h1>
        <p class="meta">
            QbD Report | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | 
            {'Minimizing' if experiment.minimize else 'Maximizing'} {experiment.objective_name}
        </p>
        
        {f'<p>{experiment.description}</p>' if experiment.description else ''}
        
        <h2>📊 Summary Statistics</h2>
        <div class="card">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats.get('n_trials', 0)}</div>
                    <div class="stat-label">Total Runs</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value best">{stats.get('best_value', '-'):.4g if stats.get('best_value') else '-'}</div>
                    <div class="stat-label">Best {experiment.objective_name}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('mean', 0):.4g}</div>
                    <div class="stat-label">Mean</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('std', 0):.4g}</div>
                    <div class="stat-label">Std Dev</div>
                </div>
            </div>
        </div>
        
        <h2>🎯 Optimal Conditions</h2>
        <div class="card">
"""
    
    if best:
        html += "<table><thead><tr><th>Parameter</th><th>Optimal Value</th><th>Range</th></tr></thead><tbody>"
        for p in experiment.parameters:
            val = best.parameters.get(p.name, "-")
            if isinstance(val, float):
                val_str = f"{val:.4g}"
            else:
                val_str = str(val)
            
            if p.unit:
                val_str += f" {p.unit}"
            
            if p.param_type == "categorical":
                range_str = ", ".join(p.categories) if p.categories else "-"
            else:
                range_str = f"{p.low} - {p.high}"
                if p.unit:
                    range_str += f" {p.unit}"
            
            html += f"<tr class='optimal-row'><td>{p.name}</td><td><strong>{val_str}</strong></td><td>{range_str}</td></tr>"
        
        html += "</tbody></table>"
        html += f"<p><strong>Optimal {experiment.objective_name}:</strong> <span class='best'>{best.objective_value:.4g}</span> (Run #{best.trial_number})</p>"
    else:
        html += "<p>No completed runs yet.</p>"
    
    html += "</div>"
    
    # Design Space Plot
    if design_space_img:
        html += """
        <h2>🗺️ Design Space</h2>
        <div class="card">
            <img src="data:image/png;base64,""" + design_space_img + """" alt="Design Space Plot">
            <p style="color: #94A3B8; font-size: 0.9em;">
                The design space shows the explored parameter combinations. 
                The star indicates the optimal conditions found.
            </p>
        </div>
"""
    
    # Convergence Plot  
    if convergence_img:
        html += """
        <h2>📈 Optimization Progress</h2>
        <div class="card">
            <img src="data:image/png;base64,""" + convergence_img + """" alt="Convergence Plot">
            <p style="color: #94A3B8; font-size: 0.9em;">
                The convergence plot shows how the best response improves over successive runs.
            </p>
        </div>
"""
    
    # Experimental Runs Table
    html += """
        <h2>📋 All Experimental Runs</h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Run #</th>
"""
    
    for p in experiment.parameters:
        html += f"<th>{p.name}</th>"
    
    html += f"<th>{experiment.objective_name}</th><th>Notes</th></tr></thead><tbody>"
    
    sorted_trials = sorted(experiment.trials, key=lambda t: t.trial_number)
    best_num = best.trial_number if best else -1
    
    for trial in sorted_trials:
        row_class = "optimal-row" if trial.trial_number == best_num else ""
        html += f"<tr class='{row_class}'><td>{trial.trial_number}</td>"
        
        for p in experiment.parameters:
            val = trial.parameters.get(p.name, "-")
            if isinstance(val, float):
                val_str = f"{val:.4g}"
            else:
                val_str = str(val)
            html += f"<td>{val_str}</td>"
        
        obj_str = f"{trial.objective_value:.4g}" if trial.objective_value is not None else "-"
        html += f"<td>{obj_str}</td><td>{trial.notes or '-'}</td></tr>"
    
    html += """
                </tbody>
            </table>
        </div>
        
        <h2>📐 Design Parameters</h2>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Parameter</th><th>Type</th><th>Range/Levels</th><th>Unit</th></tr>
                </thead>
                <tbody>
"""
    
    for p in experiment.parameters:
        if p.param_type == "categorical":
            range_str = ", ".join(p.categories) if p.categories else "-"
        else:
            range_str = f"{p.low} - {p.high}"
            if p.log_scale:
                range_str += " (log)"
        
        html += f"<tr><td>{p.name}</td><td>{p.param_type.capitalize()}</td><td>{range_str}</td><td>{p.unit or '-'}</td></tr>"
    
    html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by OptiML - Bayesian Optimization for Analytical Development</p>
            <p>© """ + str(datetime.now().year) + f""" OptiML | <a href="https://github.com/optiml" style="color: {PRIMARY};">github.com/optiml</a></p>
        </div>
    </div>
</body>
</html>
"""
    
    return html
