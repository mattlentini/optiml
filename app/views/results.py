"""
Results View - Visualization and analysis of optimization results
"""

import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from core import colors
from core.reports import generate_html_report


def ResultsView(page: ft.Page, session, rail: ft.NavigationRail) -> ft.View:
    """Create the results view."""
    
    if not session.current_experiment:
        return _no_experiment_view(page, rail)
    
    exp = session.current_experiment
    
    # Build parameter lookup for units
    param_lookup = {p.name: p for p in exp.parameters}
    
    if not exp.trials:
        return _no_trials_view(page, rail, exp)
    
    # Create visualizations
    charts = []
    
    # 1. Optimization Progress Chart
    progress_chart = _create_progress_chart(exp)
    if progress_chart:
        charts.append(("Optimization Progress", progress_chart))
    
    # 2. Parameter importance (simple correlation-based)
    if len(exp.trials) >= 5:
        importance_chart = _create_importance_chart(exp)
        if importance_chart:
            charts.append(("Parameter Influence", importance_chart))
    
    # 3. Parameter scatter plots (for all numeric parameters)
    for param in exp.parameters:
        scatter = _create_scatter_chart(exp, param.name)
        if scatter:
            charts.append((f"{param.name} vs {exp.objective_name}", scatter))
    
    # Best result summary
    best = exp.get_best_trial()
    
    # Format best parameters with units
    best_params_display = []
    if best:
        for k, v in best.parameters.items():
            param = param_lookup.get(k)
            unit = param.unit if param and param.unit else ""
            if isinstance(v, float):
                val_str = f"{v:.4g}"
            else:
                val_str = str(v)
            if unit:
                val_str += f" {unit}"
            best_params_display.append((k, val_str))
    
    best_card = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.EMOJI_EVENTS, color=colors.WARNING, size=40),
                ft.Column([
                    ft.Text("Best Result", size=14, color=colors.TEXT_SECONDARY),
                    ft.Text(
                        f"{best.objective_value:.4g}" if best else "-",
                        size=32,
                        weight=ft.FontWeight.BOLD,
                        color=colors.SUCCESS,
                    ),
                ], spacing=2),
            ], spacing=15),
            ft.Divider(color=colors.BG_ELEVATED),
            ft.Text("Optimal Conditions:", color=colors.TEXT_SECONDARY, size=12),
            *[
                ft.Row([
                    ft.Text(f"{k}:", color=colors.TEXT_SECONDARY, width=120),
                    ft.Text(v, color=colors.TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
                ])
                for k, v in best_params_display
            ],
        ], spacing=10),
        padding=25,
        bgcolor=colors.BG_SURFACE,
        border_radius=15,
        border=ft.border.all(2, colors.SUCCESS),
        width=350,
    )
    
    # Stats summary
    completed_trials = [t for t in exp.trials if t.objective_value is not None]
    values = [t.objective_value for t in completed_trials]
    
    stats_card = ft.Container(
        content=ft.Column([
            ft.Text("Statistics", size=18, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
            ft.Divider(color=colors.BG_ELEVATED),
            _stat_row("Total Runs", str(len(completed_trials))),
            _stat_row("Best", f"{min(values) if exp.minimize else max(values):.4g}" if values else "-"),
            _stat_row("Worst", f"{max(values) if exp.minimize else min(values):.4g}" if values else "-"),
            _stat_row("Mean", f"{np.mean(values):.4g}" if values else "-"),
            _stat_row("Std Dev", f"{np.std(values):.4g}" if len(values) > 1 else "-"),
        ], spacing=8),
        padding=25,
        bgcolor=colors.BG_SURFACE,
        border_radius=15,
        width=350,
    )
    
    # Export buttons with file save dialogs
    def export_csv(e):
        """Export trials to CSV with file save dialog."""
        def on_save_csv(e: ft.FilePickerResultEvent):
            if e.path:
                try:
                    filepath = e.path if e.path.endswith('.csv') else f"{e.path}.csv"
                    session.export_trials_csv(filepath)
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"Exported to {filepath}"),
                        bgcolor=colors.SUCCESS,
                    )
                    page.snack_bar.open = True
                    page.update()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"Export failed: {str(ex)}"),
                        bgcolor=colors.ERROR,
                    )
                    page.snack_bar.open = True
                    page.update()
        
        csv_picker = ft.FilePicker(on_result=on_save_csv)
        page.overlay.append(csv_picker)
        page.update()
        csv_picker.save_file(
            file_name=f"{exp.name.replace(' ', '_')}_results.csv",
            allowed_extensions=["csv"],
            dialog_title="Export Runs to CSV",
        )
    
    def save_experiment(e):
        """Save experiment with file save dialog."""
        def on_save_json(e: ft.FilePickerResultEvent):
            if e.path:
                try:
                    filepath = e.path if e.path.endswith('.json') else f"{e.path}.json"
                    session.save_experiment(filepath)
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"Saved to {filepath}"),
                        bgcolor=colors.SUCCESS,
                    )
                    page.snack_bar.open = True
                    page.update()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"Save failed: {str(ex)}"),
                        bgcolor=colors.ERROR,
                    )
                    page.snack_bar.open = True
                    page.update()
        
        json_picker = ft.FilePicker(on_result=on_save_json)
        page.overlay.append(json_picker)
        page.update()
        json_picker.save_file(
            file_name=f"{exp.name.replace(' ', '_')}.json",
            allowed_extensions=["json"],
            dialog_title="Save Study",
        )
    
    def export_qbd_report(e):
        """Generate and save QbD HTML report."""
        def on_save_html(e: ft.FilePickerResultEvent):
            if e.path:
                try:
                    filepath = e.path if e.path.endswith('.html') else f"{e.path}.html"
                    html_content = generate_html_report(exp)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"QbD Report saved to {filepath}"),
                        bgcolor=colors.SUCCESS,
                    )
                    page.snack_bar.open = True
                    page.update()
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"Report generation failed: {str(ex)}"),
                        bgcolor=colors.ERROR,
                    )
                    page.snack_bar.open = True
                    page.update()
        
        html_picker = ft.FilePicker(on_result=on_save_html)
        page.overlay.append(html_picker)
        page.update()
        html_picker.save_file(
            file_name=f"{exp.name.replace(' ', '_')}_QbD_Report.html",
            allowed_extensions=["html"],
            dialog_title="Save QbD Report",
        )
    
    export_buttons = ft.Row([
        ft.ElevatedButton(
            "Export CSV",
            icon=ft.Icons.DOWNLOAD,
            on_click=export_csv,
            style=ft.ButtonStyle(bgcolor=colors.BG_ELEVATED, color=colors.TEXT_PRIMARY),
        ),
        ft.ElevatedButton(
            "Save Study",
            icon=ft.Icons.SAVE,
            on_click=save_experiment,
            style=ft.ButtonStyle(bgcolor=colors.PRIMARY, color=colors.TEXT_PRIMARY),
        ),
        ft.ElevatedButton(
            "QbD Report",
            icon=ft.Icons.DESCRIPTION,
            on_click=export_qbd_report,
            style=ft.ButtonStyle(bgcolor=colors.SUCCESS, color=colors.TEXT_PRIMARY),
        ),
    ], spacing=15)
    
    # Charts display
    chart_tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text=name,
                content=ft.Container(
                    content=chart,
                    padding=20,
                    alignment=ft.alignment.center,
                ),
            )
            for name, chart in charts
        ] if charts else [
            ft.Tab(text="No Charts", content=ft.Text("Not enough data for visualization"))
        ],
        expand=True,
    )
    
    # All trials table
    trials_table = _create_trials_table(exp, param_lookup)
    
    # Main layout
    main_content = ft.Container(
        content=ft.Column([
            # Header
            ft.Row([
                ft.Column([
                    ft.Text("Results & Analysis", size=28, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
                    ft.Text(exp.name, color=colors.TEXT_SECONDARY),
                ]),
                export_buttons,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            ft.Divider(color=colors.BG_ELEVATED),
            
            # Summary cards
            ft.Row([
                best_card,
                stats_card,
            ], spacing=20, wrap=True),
            
            ft.Container(height=20),
            
            # Charts
            ft.Text("Visualizations", size=20, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
            ft.Container(
                content=chart_tabs,
                height=450,
                bgcolor=colors.BG_SURFACE,
                border_radius=15,
            ),
            
            ft.Container(height=20),
            
            # All trials
            ft.Text("All Runs", size=20, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
            ft.Container(
                content=trials_table,
                bgcolor=colors.BG_SURFACE,
                border_radius=15,
                padding=15,
            ),
        ], scroll=ft.ScrollMode.AUTO),
        padding=40,
        expand=True,
    )
    
    return ft.View(
        route="/results",
        controls=[
            ft.Row([
                rail,
                ft.VerticalDivider(width=1, color=colors.BG_ELEVATED),
                main_content,
            ], expand=True),
        ],
        padding=0,
        bgcolor=colors.BG_DARK,
    )


def _no_experiment_view(page, rail):
    """View when no experiment exists."""
    return ft.View(
        route="/results",
        controls=[
            ft.Row([
                rail,
                ft.VerticalDivider(width=1, color=colors.BG_ELEVATED),
                ft.Container(
                    content=ft.Column([
                        ft.Text("📊", size=60),
                        ft.Text("No Results Yet", size=24, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
                        ft.Text("Create a study and run some experiments to see results.", color=colors.TEXT_SECONDARY),
                        ft.Container(height=20),
                        ft.ElevatedButton(
                            "New Study",
                            icon=ft.Icons.ADD,
                            on_click=lambda e: page.go("/new"),
                            style=ft.ButtonStyle(bgcolor=colors.PRIMARY, color=colors.TEXT_PRIMARY),
                        ),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    expand=True,
                    alignment=ft.alignment.center,
                ),
            ], expand=True),
        ],
        padding=0,
        bgcolor=colors.BG_DARK,
    )


def _no_trials_view(page, rail, exp):
    """View when experiment has no trials."""
    return ft.View(
        route="/results",
        controls=[
            ft.Row([
                rail,
                ft.VerticalDivider(width=1, color=colors.BG_ELEVATED),
                ft.Container(
                    content=ft.Column([
                        ft.Text("📊", size=60),
                        ft.Text("No Runs Yet", size=24, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
                        ft.Text(f"Run some experiments in '{exp.name}' to see results.", color=colors.TEXT_SECONDARY),
                        ft.Container(height=20),
                        ft.ElevatedButton(
                            "Start Optimizing",
                            icon=ft.Icons.PLAY_ARROW,
                            on_click=lambda e: page.go("/optimize"),
                            style=ft.ButtonStyle(bgcolor=colors.PRIMARY, color=colors.TEXT_PRIMARY),
                        ),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    expand=True,
                    alignment=ft.alignment.center,
                ),
            ], expand=True),
        ],
        padding=0,
        bgcolor=colors.BG_DARK,
    )


def _stat_row(label: str, value: str) -> ft.Row:
    """Create a statistics row."""
    return ft.Row([
        ft.Text(label, color=colors.TEXT_SECONDARY, width=100),
        ft.Text(value, color=colors.TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)


def _create_progress_chart(exp):
    """Create optimization progress chart."""
    completed = [t for t in exp.trials if t.objective_value is not None]
    if len(completed) < 2:
        return None
    
    trials_nums = [t.trial_number for t in completed]
    values = [t.objective_value for t in completed]
    
    # Calculate running best
    running_best = []
    current_best = values[0]
    for v in values:
        if exp.minimize:
            current_best = min(current_best, v)
        else:
            current_best = max(current_best, v)
        running_best.append(current_best)
    
    # Create figure with dark theme matching new color scheme
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=colors.BG_SURFACE)
    ax.set_facecolor(colors.BG_SURFACE)
    
    ax.scatter(trials_nums, values, color=colors.CHART_PRIMARY, s=50, alpha=0.7, label='Trials')
    ax.plot(trials_nums, running_best, color=colors.CHART_BEST, linewidth=2, label='Best so far')
    
    ax.set_xlabel('Trial Number', color=colors.TEXT_SECONDARY)
    ax.set_ylabel(exp.objective_name, color=colors.TEXT_SECONDARY)
    ax.tick_params(colors=colors.TEXT_SECONDARY)
    ax.legend(facecolor=colors.BG_ELEVATED, edgecolor=colors.BORDER, labelcolor=colors.TEXT_PRIMARY)
    ax.grid(True, alpha=0.2, color=colors.CHART_GRID)
    
    for spine in ax.spines.values():
        spine.set_color(colors.BORDER)
    
    plt.tight_layout()
    chart = MatplotlibChart(fig, expand=True)
    plt.close(fig)  # Prevent memory leak
    
    return chart


def _create_importance_chart(exp):
    """Create parameter importance chart (correlation-based)."""
    completed = [t for t in exp.trials if t.objective_value is not None]
    if len(completed) < 5:
        return None
    
    # Calculate simple correlations
    correlations = {}
    y = np.array([t.objective_value for t in completed])
    
    for param in exp.parameters:
        if param.param_type == 'categorical':
            continue  # Skip categorical for now
        
        x = np.array([t.parameters[param.name] for t in completed])
        if np.std(x) > 0:
            corr = abs(np.corrcoef(x, y)[0, 1])
            correlations[param.name] = corr if not np.isnan(corr) else 0
    
    if not correlations:
        return None
    
    # Sort by importance
    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    names = [p[0] for p in sorted_params]
    values = [p[1] for p in sorted_params]
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=colors.BG_SURFACE)
    ax.set_facecolor(colors.BG_SURFACE)
    
    bars = ax.barh(names, values, color=colors.CHART_PRIMARY)
    ax.set_xlabel('Correlation with Objective', color=colors.TEXT_SECONDARY)
    ax.tick_params(colors=colors.TEXT_SECONDARY)
    ax.set_xlim(0, 1)
    
    for spine in ax.spines.values():
        spine.set_color(colors.BORDER)
    
    plt.tight_layout()
    chart = MatplotlibChart(fig, expand=True)
    plt.close(fig)  # Prevent memory leak
    
    return chart


def _create_scatter_chart(exp, param_name):
    """Create scatter plot for a parameter."""
    completed = [t for t in exp.trials if t.objective_value is not None]
    if len(completed) < 2:
        return None
    
    param = next((p for p in exp.parameters if p.name == param_name), None)
    if not param or param.param_type == 'categorical':
        return None
    
    x = [t.parameters[param_name] for t in completed]
    y = [t.objective_value for t in completed]
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=colors.BG_SURFACE)
    ax.set_facecolor(colors.BG_SURFACE)
    
    ax.scatter(x, y, color=colors.CHART_PRIMARY, s=60, alpha=0.7)
    
    ax.set_xlabel(param_name, color=colors.TEXT_SECONDARY)
    ax.set_ylabel(exp.objective_name, color=colors.TEXT_SECONDARY)
    ax.tick_params(colors=colors.TEXT_SECONDARY)
    ax.grid(True, alpha=0.2, color=colors.CHART_GRID)
    
    for spine in ax.spines.values():
        spine.set_color(colors.BORDER)
    
    plt.tight_layout()
    chart = MatplotlibChart(fig, expand=True)
    plt.close(fig)  # Prevent memory leak
    
    return chart


def _create_trials_table(exp, param_lookup=None):
    """Create trials data table with units."""
    completed = [t for t in exp.trials if t.objective_value is not None]
    
    if not completed:
        return ft.Text("No completed runs", color=colors.TEXT_MUTED)
    
    if param_lookup is None:
        param_lookup = {p.name: p for p in exp.parameters}
    
    # Table columns with units
    columns = [
        ft.DataColumn(ft.Text("Run", color=colors.TEXT_SECONDARY)),
    ]
    for p in exp.parameters:
        unit_str = f" ({p.unit})" if p.unit else ""
        columns.append(ft.DataColumn(ft.Text(f"{p.name}{unit_str}", color=colors.TEXT_SECONDARY)))
    columns.append(ft.DataColumn(ft.Text(exp.objective_name, color=colors.TEXT_SECONDARY)))
    
    # Table rows
    best = exp.get_best_trial()
    rows = []
    
    for trial in sorted(completed, key=lambda t: t.trial_number):
        is_best = best and trial.trial_number == best.trial_number
        
        cells = [ft.DataCell(ft.Text(str(trial.trial_number), color=colors.TEXT_PRIMARY))]
        
        for param in exp.parameters:
            val = trial.parameters.get(param.name, "")
            if isinstance(val, float):
                val = f"{val:.6g}"
            cells.append(ft.DataCell(ft.Text(str(val), color=colors.TEXT_PRIMARY)))
        
        cells.append(
            ft.DataCell(
                ft.Text(
                    f"{trial.objective_value:.6g}",
                    color=colors.SUCCESS if is_best else colors.TEXT_PRIMARY,
                    weight=ft.FontWeight.BOLD if is_best else None,
                )
            )
        )
        
        rows.append(ft.DataRow(cells=cells))
    
    return ft.DataTable(
        columns=columns,
        rows=rows,
        border=ft.border.all(1, colors.BORDER),
        border_radius=10,
        vertical_lines=ft.border.BorderSide(1, colors.BORDER),
        horizontal_lines=ft.border.BorderSide(1, colors.BORDER),
    )
