"""
OptiML - Desktop Application
============================
A free, open-source Bayesian optimization tool for scientists and engineers.
No coding required.
"""

import flet as ft
import os
from views.home import HomeView
from views.new_experiment import NewExperimentView
from views.optimization import OptimizationView
from views.results import ResultsView
from views.notebook import NotebookView
from core.session import Session
from core import colors

# Get the app directory for asset paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_DIR, "assets")


def main(page: ft.Page):
    """Main application entry point."""
    
    # Page configuration
    page.title = "OptiML - Bayesian Optimization"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.bgcolor = colors.BG_DARK
    
    # Custom theme with science-inspired color palette
    page.theme = ft.Theme(
        color_scheme=colors.get_flet_color_scheme(),
        font_family="Inter",
    )
    
    # Session state
    session = Session()
    
    def route_change(e):
        """Handle route changes."""
        page.views.clear()
        
        # Navigation rail (sidebar)
        rail = create_nav_rail(page, session)
        
        if page.route == "/" or page.route == "":
            page.views.append(HomeView(page, session, rail))
        elif page.route == "/new":
            page.views.append(NewExperimentView(page, session, rail))
        elif page.route == "/optimize":
            page.views.append(OptimizationView(page, session, rail))
        elif page.route == "/results":
            page.views.append(ResultsView(page, session, rail))
        elif page.route == "/notebook":
            page.views.append(NotebookView(page, session, rail))
        
        page.update()
    
    def view_pop(e):
        """Handle view pop."""
        page.views.pop()
        if page.views:
            top_view = page.views[-1]
            page.go(top_view.route)
    
    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go(page.route or "/")


def create_nav_rail(page: ft.Page, session) -> ft.NavigationRail:
    """Create the navigation sidebar."""
    
    def on_nav_change(e):
        routes = ["/", "/new", "/optimize", "/notebook", "/results"]
        if e.control.selected_index < len(routes):
            page.go(routes[e.control.selected_index])
    
    # Determine selected index based on current route
    route_index = {"/": 0, "/new": 1, "/optimize": 2, "/notebook": 3, "/results": 4}
    selected = route_index.get(page.route, 0)
    
    # Use logo SVG
    logo_path = os.path.join(ASSETS_DIR, "optiml_logo.svg")
    logo_image = ft.Image(
        src=logo_path,
        width=70,
        height=70,
        fit=ft.ImageFit.CONTAIN,
    )
    
    return ft.NavigationRail(
        selected_index=selected,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=200,
        bgcolor=colors.BG_DARK,  # Match logo background color
        leading=ft.Container(
            content=logo_image,
            padding=ft.padding.only(top=20, bottom=10),
        ),
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.HOME_OUTLINED,
                selected_icon=ft.Icons.HOME,
                label="Home",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.ADD_CIRCLE_OUTLINE,
                selected_icon=ft.Icons.ADD_CIRCLE,
                label="New",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
                selected_icon=ft.Icons.PLAY_CIRCLE,
                label="Optimize",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.MENU_BOOK_OUTLINED,
                selected_icon=ft.Icons.MENU_BOOK,
                label="Notebook",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.ANALYTICS_OUTLINED,
                selected_icon=ft.Icons.ANALYTICS,
                label="Results",
            ),
        ],
        on_change=on_nav_change,
    )


if __name__ == "__main__":
    ft.app(target=main)
