"""
Home View - Landing page for OptiML
"""

import flet as ft
import os
from core import colors

# Get assets directory path
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def HomeView(page: ft.Page, session, rail: ft.NavigationRail) -> ft.View:
    """Create the home view."""
    
    def on_new_experiment(e):
        page.go("/new")
    
    def on_load_experiment(e):
        """Open file picker to load an experiment."""
        def on_file_picked(e: ft.FilePickerResultEvent):
            if e.files and len(e.files) > 0:
                filepath = e.files[0].path
                try:
                    session.load_experiment(filepath)
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"Loaded: {session.current_experiment.name}"),
                        bgcolor=colors.SUCCESS,
                    )
                    page.snack_bar.open = True
                    page.update()
                    page.go("/optimize")
                except Exception as ex:
                    page.snack_bar = ft.SnackBar(
                        content=ft.Text(f"Failed to load: {str(ex)}"),
                        bgcolor=colors.ERROR,
                    )
                    page.snack_bar.open = True
                    page.update()
        
        file_picker = ft.FilePicker(on_result=on_file_picked)
        page.overlay.append(file_picker)
        page.update()
        file_picker.pick_files(
            allowed_extensions=["json"],
            dialog_title="Load Experiment",
        )
    
    # Logo image (includes app name)
    logo_display = ft.Image(
        src=os.path.join(ASSETS_DIR, "optiml_logo.svg"),
        width=180,
        height=180,
        fit=ft.ImageFit.CONTAIN,
    )
    
    # Hero section
    hero = ft.Container(
        content=ft.Column([
            logo_display,
            ft.Container(height=20),
            ft.Text(
                "Bayesian Optimization Made Simple",
                size=24,
                color=colors.TEXT_SECONDARY,
            ),
            ft.Container(height=20),
            ft.Text(
                "The free, open-source tool for optimizing experiments.\n"
                "No coding required. Just define your parameters and let AI guide your search.",
                size=16,
                color=colors.TEXT_MUTED,
                text_align=ft.TextAlign.CENTER,
            ),
            ft.Container(height=40),
            ft.Row([
                ft.ElevatedButton(
                    "New Experiment",
                    icon=ft.Icons.ADD,
                    on_click=on_new_experiment,
                    style=ft.ButtonStyle(
                        bgcolor=colors.PRIMARY,
                        color=colors.TEXT_PRIMARY,
                        padding=20,
                    ),
                ),
                ft.OutlinedButton(
                    "Load Experiment",
                    icon=ft.Icons.FOLDER_OPEN,
                    on_click=on_load_experiment,
                    style=ft.ButtonStyle(
                        color=colors.PRIMARY,
                        padding=20,
                    ),
                ),
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=60,
        expand=True,
    )
    
    # Features section
    features = ft.Container(
        content=ft.Column([
            ft.Text("Why OptiML?", size=28, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
            ft.Container(height=30),
            ft.Row([
                _feature_card(
                    ft.Icons.SCIENCE_OUTLINED,
                    "Designed for Scientists",
                    "Built for lab experiments, manufacturing, and research. No programming knowledge needed.",
                ),
                _feature_card(
                    ft.Icons.AUTO_AWESOME,
                    "AI-Powered",
                    "Uses Bayesian optimization to intelligently explore your parameter space with fewer experiments.",
                ),
                _feature_card(
                    ft.Icons.INSIGHTS,
                    "Visual Insights",
                    "Interactive charts show your optimization progress and help understand parameter relationships.",
                ),
                _feature_card(
                    ft.Icons.SAVE_OUTLINED,
                    "Save & Resume",
                    "Save experiments to continue later. Export results to CSV for further analysis.",
                ),
            ], alignment=ft.MainAxisAlignment.CENTER, wrap=True, spacing=20),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=40,
        bgcolor=colors.BG_SURFACE,
        border_radius=20,
        margin=ft.margin.all(40),
    )
    
    # How it works
    how_it_works = ft.Container(
        content=ft.Column([
            ft.Text("How It Works", size=28, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
            ft.Container(height=30),
            ft.Row([
                _step_card("1", "Define Parameters", "Set up your optimization variables - continuous, integer, or categorical."),
                ft.Icon(ft.Icons.ARROW_FORWARD, color=colors.PRIMARY, size=30),
                _step_card("2", "Run Suggestions", "Get AI-recommended parameter values to try in your experiment."),
                ft.Icon(ft.Icons.ARROW_FORWARD, color=colors.PRIMARY, size=30),
                _step_card("3", "Record Results", "Enter the outcome of each experiment."),
                ft.Icon(ft.Icons.ARROW_FORWARD, color=colors.PRIMARY, size=30),
                _step_card("4", "Find Optimum", "After a few iterations, discover your optimal settings."),
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=40,
        margin=ft.margin.only(left=40, right=40, bottom=40),
    )
    
    # Recent experiments from database
    def load_experiment_from_db(exp_id: int):
        """Load an experiment from the database."""
        def handler(e):
            try:
                session.load_experiment_from_db(exp_id)
                page.snack_bar = ft.SnackBar(
                    content=ft.Text(f"Loaded: {session.current_experiment.name}"),
                    bgcolor=colors.SUCCESS,
                )
                page.snack_bar.open = True
                page.update()
                page.go("/optimize")
            except Exception as ex:
                page.snack_bar = ft.SnackBar(
                    content=ft.Text(f"Failed to load: {str(ex)}"),
                    bgcolor=colors.ERROR,
                )
                page.snack_bar.open = True
                page.update()
        return handler
    
    def create_recent_experiments_section():
        """Create the recent experiments section."""
        recent_exps = session.list_experiments_from_db()[:5]  # Last 5
        
        if not recent_exps:
            return ft.Container()  # Empty if no experiments
        
        exp_cards = []
        for exp in recent_exps:
            from datetime import datetime
            # Parse and format date
            try:
                dt = datetime.fromisoformat(exp['updated_at'])
                date_str = dt.strftime("%b %d, %Y")
            except:
                date_str = exp['updated_at'][:10]
            
            exp_cards.append(
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SCIENCE, color=colors.PRIMARY, size=20),
                            ft.Text(
                                exp['name'], 
                                weight=ft.FontWeight.BOLD, 
                                color=colors.TEXT_PRIMARY,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ], spacing=10),
                        ft.Text(
                            f"{exp['trial_count']} runs • {date_str}",
                            size=12, color=colors.TEXT_MUTED
                        ),
                    ], spacing=4),
                    width=220,
                    padding=15,
                    bgcolor=colors.BG_ELEVATED,
                    border_radius=10,
                    on_click=load_experiment_from_db(exp['id']),
                    ink=True,
                )
            )
        
        return ft.Container(
            content=ft.Column([
                ft.Text("Recent Studies", size=20, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
                ft.Container(height=15),
                ft.Row(exp_cards, wrap=True, spacing=15),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.only(left=40, right=40, bottom=20),
        )
    
    recent_section = create_recent_experiments_section()
    
    # Main content
    content = ft.Container(
        content=ft.Column([
            hero,
            recent_section,
            features,
            how_it_works,
        ], scroll=ft.ScrollMode.AUTO, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        expand=True,
    )
    
    return ft.View(
        route="/",
        controls=[
            ft.Row([
                rail,
                ft.VerticalDivider(width=1, color=colors.BORDER),
                content,
            ], expand=True),
        ],
        padding=0,
        bgcolor=colors.BG_DARK,
    )


def _feature_card(icon, title: str, description: str) -> ft.Container:
    """Create a feature card."""
    return ft.Container(
        content=ft.Column([
            ft.Icon(icon, size=40, color=colors.PRIMARY),
            ft.Text(title, size=18, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
            ft.Text(description, size=14, color=colors.TEXT_SECONDARY, text_align=ft.TextAlign.CENTER),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
        width=220,
        height=180,
        padding=20,
        bgcolor=colors.BG_ELEVATED,
        border_radius=15,
    )


def _step_card(number: str, title: str, description: str) -> ft.Container:
    """Create a step card."""
    return ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Text(number, size=24, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
                width=50,
                height=50,
                bgcolor=colors.PRIMARY,
                border_radius=25,
                alignment=ft.alignment.center,
            ),
            ft.Text(title, size=16, weight=ft.FontWeight.BOLD, color=colors.TEXT_PRIMARY),
            ft.Text(description, size=12, color=colors.TEXT_SECONDARY, text_align=ft.TextAlign.CENTER),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
        width=180,
        padding=15,
    )
