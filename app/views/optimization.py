"""
Optimization View - Main optimization workflow
"""

import flet as ft
from typing import Dict, Any
from core.colors import (
    PRIMARY, PRIMARY_DARK, BG_DARK, BG_SURFACE, BG_ELEVATED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, SUCCESS, ERROR, WARNING, INFO
)


def OptimizationView(page: ft.Page, session, rail: ft.NavigationRail) -> ft.View:
    """Create the optimization view."""
    
    if not session.current_experiment:
        # No active experiment
        return ft.View(
            route="/optimize",
            controls=[
                ft.Row([
                    rail,
                    ft.VerticalDivider(width=1, color=BG_ELEVATED),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("🔬", size=60),
                            ft.Text("No Active Study", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                            ft.Text("Create a new study to start method development.", color=TEXT_SECONDARY),
                            ft.Container(height=20),
                            ft.ElevatedButton(
                                "New Study",
                                icon=ft.Icons.ADD,
                                on_click=lambda e: page.go("/new"),
                                style=ft.ButtonStyle(bgcolor=PRIMARY, color=TEXT_PRIMARY),
                            ),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        expand=True,
                        alignment=ft.alignment.center,
                    ),
                ], expand=True),
            ],
            padding=0,
            bgcolor=BG_DARK,
        )
    
    exp = session.current_experiment
    
    # Build parameter lookup for units
    param_lookup = {p.name: p for p in exp.parameters}
    
    # State
    suggested_params = [None]
    result_field = ft.TextField(
        label=f"{exp.objective_name} Value",
        hint_text="Enter the measured response",
        width=300,
        border_color=PRIMARY,
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    notes_field = ft.TextField(
        label="Notes (optional)",
        hint_text="Any observations about this run",
        width=400,
        multiline=True,
        min_lines=2,
        max_lines=4,
        border_color=PRIMARY,
    )
    
    # Suggestion display
    suggestion_card = ft.Container(
        content=ft.Column([
            ft.Text("Next Suggested Conditions", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            ft.Text("Click 'Get Suggestion' to get AI-recommended conditions", color=TEXT_MUTED),
        ]),
        padding=30,
        bgcolor=BG_SURFACE,
        border_radius=15,
        width=500,
    )
    
    # Loading state for suggestion button
    suggest_button_ref = ft.Ref[ft.ElevatedButton]()
    
    def get_suggestion(e):
        """Get next suggested parameters."""
        # Show loading state
        suggestion_card.content = ft.Column([
            ft.ProgressRing(width=40, height=40, stroke_width=3, color=PRIMARY),
            ft.Container(height=10),
            ft.Text("Calculating optimal conditions...", color=TEXT_SECONDARY),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        if suggest_button_ref.current:
            suggest_button_ref.current.disabled = True
        page.update()
        
        try:
            params = session.suggest_next()
            suggested_params[0] = params
            
            # Update suggestion card with units
            param_rows = []
            for name, value in params.items():
                param = param_lookup.get(name)
                unit = param.unit if param and param.unit else ""
                
                if isinstance(value, float):
                    display_value = f"{value:.4g}"
                else:
                    display_value = str(value)
                
                if unit:
                    display_value = f"{display_value} {unit}"
                
                param_rows.append(
                    ft.Container(
                        content=ft.Row([
                            ft.Text(name, weight=ft.FontWeight.BOLD, color=TEXT_SECONDARY, width=150),
                            ft.Text(display_value, size=18, color=PRIMARY, weight=ft.FontWeight.BOLD),
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=15,
                        bgcolor=BG_ELEVATED,
                        border_radius=8,
                    )
                )
            
            suggestion_card.content = ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.LIGHTBULB, color=WARNING, size=24),
                    ft.Text("Suggested Conditions", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ], spacing=10),
                ft.Text("Try these conditions in your next run:", color=TEXT_SECONDARY, size=14),
                ft.Container(height=10),
                *param_rows,
            ], spacing=8)
            
        except Exception as ex:
            suggestion_card.content = ft.Column([
                ft.Text("Next Suggested Conditions", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ft.Text("Click 'Get Suggestion' to get AI-recommended conditions", color=TEXT_MUTED),
            ])
            page.snack_bar = ft.SnackBar(
                content=ft.Text(f"Error: {str(ex)}"),
                bgcolor=ERROR,
            )
            page.snack_bar.open = True
        finally:
            if suggest_button_ref.current:
                suggest_button_ref.current.disabled = False
            page.update()
    
    def record_result(e):
        """Record the experiment result."""
        if not suggested_params[0]:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Get a suggestion first"),
                bgcolor=WARNING,
            )
            page.snack_bar.open = True
            page.update()
            return
        
        if not result_field.value:
            result_field.error_text = "Response value is required"
            page.update()
            return
        
        try:
            value = float(result_field.value)
        except ValueError:
            result_field.error_text = "Enter a valid number"
            page.update()
            return
        
        result_field.error_text = None
        
        # Record the trial
        trial = session.record_result(value, notes_field.value or "")
        suggested_params[0] = None
        
        # Clear fields
        result_field.value = ""
        notes_field.value = ""
        
        # Reset suggestion card
        suggestion_card.content = ft.Column([
            ft.Icon(ft.Icons.CHECK_CIRCLE, color=SUCCESS, size=48),
            ft.Text(f"Run {trial.trial_number} Recorded!", size=18, weight=ft.FontWeight.BOLD, color=SUCCESS),
            ft.Text("Click 'Get Suggestion' for the next conditions", color=TEXT_SECONDARY),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        
        # Update trials list
        _update_trials_list()
        page.update()
    
    # Trials history
    trials_list = ft.Column([], spacing=8, scroll=ft.ScrollMode.AUTO)
    
    def _update_trials_list():
        """Update the trials history display."""
        trials_list.controls.clear()
        
        if not exp.trials:
            trials_list.controls.append(
                ft.Text("No runs yet", color=TEXT_MUTED, italic=True)
            )
            return
        
        # Sort by trial number descending (most recent first)
        sorted_trials = sorted(exp.trials, key=lambda t: t.trial_number, reverse=True)
        
        best = exp.get_best_trial()
        
        for trial in sorted_trials[:10]:  # Show last 10
            is_best = best and trial.trial_number == best.trial_number
            
            # Format parameters with units
            param_parts = []
            for k, v in trial.parameters.items():
                param = param_lookup.get(k)
                unit = param.unit if param and param.unit else ""
                if isinstance(v, float):
                    val_str = f"{k}={v:.3g}"
                else:
                    val_str = f"{k}={v}"
                if unit:
                    val_str += f" {unit}"
                param_parts.append(val_str)
            param_str = ", ".join(param_parts)
            
            trials_list.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Row([
                                ft.Text(f"Run {trial.trial_number}", weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                                ft.Container(
                                    content=ft.Text("BEST", size=10, color=TEXT_PRIMARY),
                                    bgcolor=SUCCESS,
                                    padding=ft.padding.symmetric(horizontal=8, vertical=2),
                                    border_radius=4,
                                    visible=is_best,
                                ),
                            ], spacing=10),
                            ft.Text(param_str, size=12, color=TEXT_SECONDARY),
                        ], spacing=2, expand=True),
                        ft.Text(
                            f"{trial.objective_value:.4g}" if trial.objective_value else "-",
                            size=18,
                            weight=ft.FontWeight.BOLD,
                            color=PRIMARY if is_best else TEXT_PRIMARY,
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=15,
                    bgcolor=SUCCESS + "20" if is_best else BG_ELEVATED,
                    border_radius=10,
                    border=ft.border.all(2, SUCCESS) if is_best else None,
                )
            )
    
    _update_trials_list()
    
    # Stats card
    def _get_stats_card():
        """Create stats summary card."""
        total = len(exp.trials)
        best = exp.get_best_trial()
        
        return ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text(str(total), size=32, weight=ft.FontWeight.BOLD, color=PRIMARY),
                    ft.Text("Runs", color=TEXT_SECONDARY),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                ft.VerticalDivider(color=BG_ELEVATED),
                ft.Column([
                    ft.Text(
                        f"{best.objective_value:.4g}" if best else "-",
                        size=32,
                        weight=ft.FontWeight.BOLD,
                        color=SUCCESS,
                    ),
                    ft.Text(f"Best {exp.objective_name}", color=TEXT_SECONDARY),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            ], alignment=ft.MainAxisAlignment.SPACE_AROUND),
            padding=20,
            bgcolor=BG_SURFACE,
            border_radius=15,
            width=400,
        )
    
    # Main layout
    main_content = ft.Container(
        content=ft.Column([
            # Header
            ft.Row([
                ft.Column([
                    ft.Text(exp.name, size=28, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                    ft.Text(
                        f"{'Minimizing' if exp.minimize else 'Maximizing'} {exp.objective_name}",
                        color=TEXT_SECONDARY,
                    ),
                ]),
                ft.ElevatedButton(
                    "View Results",
                    icon=ft.Icons.ANALYTICS,
                    on_click=lambda e: page.go("/results"),
                    style=ft.ButtonStyle(bgcolor=BG_ELEVATED, color=TEXT_PRIMARY),
                ),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            ft.Divider(color=BG_ELEVATED),
            
            # Main content
            ft.Row([
                # Left column - Optimization workflow
                ft.Column([
                    ft.Text("Optimization Workflow", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                    ft.Container(height=10),
                    
                    # Step 1: Get suggestion
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Container(
                                    content=ft.Text("1", color=TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
                                    width=30, height=30, bgcolor=PRIMARY, border_radius=15,
                                    alignment=ft.alignment.center,
                                ),
                                ft.Text("Get AI Suggestion", weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                            ], spacing=10),
                            ft.ElevatedButton(
                                "Get Suggestion",
                                icon=ft.Icons.AUTO_AWESOME,
                                on_click=get_suggestion,
                                ref=suggest_button_ref,
                                style=ft.ButtonStyle(bgcolor=PRIMARY, color=TEXT_PRIMARY, padding=15),
                            ),
                        ], spacing=15),
                        padding=20,
                        bgcolor=BG_SURFACE,
                        border_radius=15,
                    ),
                    
                    ft.Container(height=10),
                    suggestion_card,
                    ft.Container(height=10),
                    
                    # Step 2: Record result
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Container(
                                    content=ft.Text("2", color=TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
                                    width=30, height=30, bgcolor=PRIMARY_DARK, border_radius=15,
                                    alignment=ft.alignment.center,
                                ),
                                ft.Text("Record Response", weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                            ], spacing=10),
                            ft.Text("Run the experiment with suggested conditions, then enter the response:", color=TEXT_SECONDARY, size=14),
                            result_field,
                            notes_field,
                            ft.ElevatedButton(
                                "Record Result",
                                icon=ft.Icons.SAVE,
                                on_click=record_result,
                                style=ft.ButtonStyle(bgcolor=PRIMARY_DARK, color=TEXT_PRIMARY, padding=15),
                            ),
                        ], spacing=10),
                        padding=20,
                        bgcolor=BG_SURFACE,
                        border_radius=15,
                    ),
                ], expand=True, scroll=ft.ScrollMode.AUTO),
                
                # Right column - History
                ft.Container(width=20),
                ft.Column([
                    _get_stats_card(),
                    ft.Container(height=20),
                    ft.Text("Recent Runs", size=18, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                    ft.Container(height=10),
                    ft.Container(
                        content=trials_list,
                        height=400,
                        width=400,
                    ),
                ], horizontal_alignment=ft.CrossAxisAlignment.START),
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START),
        ], scroll=ft.ScrollMode.AUTO),
        padding=40,
        expand=True,
    )
    
    return ft.View(
        route="/optimize",
        controls=[
            ft.Row([
                rail,
                ft.VerticalDivider(width=1, color=BG_ELEVATED),
                main_content,
            ], expand=True),
        ],
        padding=0,
        bgcolor=BG_DARK,
    )
