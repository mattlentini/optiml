"""
New Experiment View - Wizard for creating a new method development study
"""

import flet as ft
from core.session import Parameter
from core.colors import (
    PRIMARY, PRIMARY_DARK, BG_DARK, BG_SURFACE, BG_ELEVATED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, SUCCESS, ERROR, WARNING
)
from core.templates import TEMPLATES_BY_CATEGORY, get_template


def NewExperimentView(page: ft.Page, session, rail: ft.NavigationRail) -> ft.View:
    """Create the new experiment wizard view."""
    
    # State
    current_step = [0]  # Using list to allow mutation in nested functions
    parameters = []
    selected_template = [None]  # Track selected template
    
    # Form fields
    name_field = ft.TextField(
        label="Study Name",
        hint_text="e.g., HPLC Method Optimization",
        width=400,
        border_color=PRIMARY,
    )
    
    description_field = ft.TextField(
        label="Description (optional)",
        hint_text="What method are you developing?",
        width=400,
        multiline=True,
        min_lines=2,
        max_lines=4,
        border_color=PRIMARY,
    )
    
    objective_name_field = ft.TextField(
        label="Primary Response",
        hint_text="e.g., Resolution, Recovery, Purity",
        value="Response",
        width=400,
        border_color=PRIMARY,
    )
    
    minimize_toggle = ft.Switch(
        label="Minimize response",
        value=True,
        active_color=PRIMARY,
    )
    
    # Parameter form fields with unit support
    param_name_field = ft.TextField(
        label="Parameter Name",
        hint_text="e.g., Column Temperature",
        width=250,
        border_color=PRIMARY,
    )
    
    param_unit_field = ft.TextField(
        label="Unit",
        hint_text="e.g., °C",
        width=80,
        border_color=PRIMARY,
    )
    
    param_type_dropdown = ft.Dropdown(
        label="Type",
        width=180,
        options=[
            ft.dropdown.Option("real", "Continuous (Real)"),
            ft.dropdown.Option("integer", "Integer"),
            ft.dropdown.Option("categorical", "Categorical"),
        ],
        value="real",
        border_color=PRIMARY,
    )
    
    param_low_field = ft.TextField(
        label="Minimum",
        hint_text="0.0",
        width=120,
        border_color=PRIMARY,
    )
    
    param_high_field = ft.TextField(
        label="Maximum",
        hint_text="100.0",
        width=120,
        border_color=PRIMARY,
    )
    
    # Constraint fields (optional limits on response)
    param_constraint_min = ft.TextField(
        label="Constraint Min",
        hint_text="Optional",
        width=100,
        border_color=PRIMARY,
    )
    
    param_constraint_max = ft.TextField(
        label="Constraint Max",
        hint_text="Optional",
        width=100,
        border_color=PRIMARY,
    )
    
    param_log_scale = ft.Checkbox(
        label="Log scale",
        value=False,
    )
    
    param_categories_field = ft.TextField(
        label="Categories (comma-separated)",
        hint_text="e.g., low, medium, high",
        width=400,
        border_color=PRIMARY,
        visible=False,
    )
    
    # Bulk entry field
    bulk_entry_field = ft.TextField(
        label="Bulk Parameter Entry",
        hint_text="Enter one parameter per line:\nColumn Temp (°C), 25, 45\npH, 6.5, 8.5\nBuffer, Phosphate, Tris, Acetate",
        width=500,
        multiline=True,
        min_lines=6,
        max_lines=12,
        border_color=PRIMARY,
    )
    
    bulk_error_text = ft.Text("", color=ERROR, size=12, visible=False)
    
    # Entry mode state
    entry_mode = [0]  # 0 = single, 1 = bulk
    
    # Parameters list display
    params_list = ft.Column([], spacing=10)
    
    def update_param_fields(e):
        """Update visibility based on parameter type."""
        is_categorical = param_type_dropdown.value == "categorical"
        param_low_field.visible = not is_categorical
        param_high_field.visible = not is_categorical
        param_unit_field.visible = not is_categorical
        param_constraint_min.visible = not is_categorical
        param_constraint_max.visible = not is_categorical
        param_log_scale.visible = param_type_dropdown.value == "real"
        param_categories_field.visible = is_categorical
        page.update()
    
    param_type_dropdown.on_change = update_param_fields
    
    def add_parameter(e):
        """Add a parameter to the list."""
        if not param_name_field.value:
            param_name_field.error_text = "Name is required"
            page.update()
            return
        
        # Check for duplicate names
        if any(p.name == param_name_field.value for p in parameters):
            param_name_field.error_text = "Parameter name already exists"
            page.update()
            return
        
        param = Parameter(
            name=param_name_field.value,
            param_type=param_type_dropdown.value,
            unit=param_unit_field.value if param_unit_field.visible else "",
        )
        
        if param_type_dropdown.value == "categorical":
            if not param_categories_field.value:
                param_categories_field.error_text = "Categories are required"
                page.update()
                return
            categories = [c.strip() for c in param_categories_field.value.split(",") if c.strip()]
            if len(categories) < 2:
                param_categories_field.error_text = "Need at least 2 categories"
                page.update()
                return
            param.categories = categories
        else:
            try:
                param.low = float(param_low_field.value)
                param.high = float(param_high_field.value)
            except ValueError:
                param_low_field.error_text = "Enter valid numbers"
                param_high_field.error_text = "Enter valid numbers"
                page.update()
                return
            
            # Validate low < high
            if param.low >= param.high:
                param_low_field.error_text = "Min must be less than Max"
                param_high_field.error_text = "Max must be greater than Min"
                page.update()
                return
            
            # Validate log scale requires positive values
            if param_type_dropdown.value == "real":
                param.log_scale = param_log_scale.value
                if param.log_scale and param.low <= 0:
                    param_low_field.error_text = "Log scale requires positive values"
                    page.update()
                    return
            
            # Parse optional constraints
            if param_constraint_min.value:
                try:
                    param.constraint_min = float(param_constraint_min.value)
                except ValueError:
                    param_constraint_min.error_text = "Enter valid number"
                    page.update()
                    return
            
            if param_constraint_max.value:
                try:
                    param.constraint_max = float(param_constraint_max.value)
                except ValueError:
                    param_constraint_max.error_text = "Enter valid number"
                    page.update()
                    return
        
        parameters.append(param)
        _update_params_display()
        
        # Clear fields
        param_name_field.value = ""
        param_name_field.error_text = None
        param_unit_field.value = ""
        param_low_field.value = ""
        param_low_field.error_text = None
        param_high_field.value = ""
        param_high_field.error_text = None
        param_constraint_min.value = ""
        param_constraint_min.error_text = None
        param_constraint_max.value = ""
        param_constraint_max.error_text = None
        param_categories_field.value = ""
        param_categories_field.error_text = None
        param_log_scale.value = False
        page.update()
    
    def parse_bulk_parameters(e):
        """Parse and add parameters from bulk entry."""
        if not bulk_entry_field.value or not bulk_entry_field.value.strip():
            bulk_error_text.value = "Enter at least one parameter"
            bulk_error_text.visible = True
            page.update()
            return
        
        lines = bulk_entry_field.value.strip().split("\n")
        new_params = []
        errors = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                errors.append(f"Line {i}: Need at least name and 2 values")
                continue
            
            name = parts[0]
            
            # Check if it's categorical (non-numeric values after name)
            try:
                low = float(parts[1])
                high = float(parts[2])
                # It's numeric - treat as real/continuous
                param = Parameter(
                    name=name,
                    param_type="real",
                    low=low,
                    high=high,
                    log_scale=False,
                )
                new_params.append(param)
            except ValueError:
                # Non-numeric - treat as categorical
                categories = parts[1:]
                param = Parameter(
                    name=name,
                    param_type="categorical",
                    categories=categories,
                )
                new_params.append(param)
        
        if errors:
            bulk_error_text.value = "; ".join(errors)
            bulk_error_text.visible = True
            page.update()
            return
        
        if not new_params:
            bulk_error_text.value = "No valid parameters found"
            bulk_error_text.visible = True
            page.update()
            return
        
        # Add all parameters
        parameters.extend(new_params)
        _update_params_display()
        
        # Clear and show success
        bulk_entry_field.value = ""
        bulk_error_text.value = f"✓ Added {len(new_params)} parameter(s)"
        bulk_error_text.color = "#48bb78"
        bulk_error_text.visible = True
        page.update()
        
        # Reset color after a moment
        bulk_error_text.color = "#e53e3e"
    
    def remove_parameter(idx):
        """Remove a parameter from the list."""
        def handler(e):
            parameters.pop(idx)
            _update_params_display()
            page.update()
        return handler
    
    def clear_all_parameters(e):
        """Remove all parameters."""
        parameters.clear()
        _update_params_display()
        page.update()
    
    def _update_params_display():
        """Update the parameters list display."""
        params_list.controls.clear()
        
        if not parameters:
            params_list.controls.append(
                ft.Text("No parameters added yet", color=TEXT_MUTED, italic=True)
            )
        else:
            # Add clear all button at top
            params_list.controls.append(
                ft.Row([
                    ft.Text(f"{len(parameters)} parameter(s)", color=TEXT_SECONDARY, size=12),
                    ft.TextButton(
                        "Clear All",
                        icon=ft.Icons.DELETE_SWEEP,
                        on_click=clear_all_parameters,
                        style=ft.ButtonStyle(color=ERROR),
                    ),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            )
            
            for i, p in enumerate(parameters):
                if p.param_type == "categorical":
                    range_text = f"Options: {', '.join(p.categories)}"
                else:
                    scale = " (log)" if p.log_scale else ""
                    unit_str = f" {p.unit}" if p.unit else ""
                    range_text = f"Range: {p.low}{unit_str} - {p.high}{unit_str}{scale}"
                
                # Show constraint if set
                constraint_text = ""
                if p.constraint_min is not None or p.constraint_max is not None:
                    c_parts = []
                    if p.constraint_min is not None:
                        c_parts.append(f"≥{p.constraint_min}")
                    if p.constraint_max is not None:
                        c_parts.append(f"≤{p.constraint_max}")
                    constraint_text = f" | Constraint: {', '.join(c_parts)}"
                
                params_list.controls.append(
                    ft.Container(
                        content=ft.Row([
                            ft.Column([
                                ft.Text(p.name, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                                ft.Text(
                                    f"{p.param_type.capitalize()} | {range_text}{constraint_text}", 
                                    size=12, color=TEXT_SECONDARY
                                ),
                            ], spacing=2),
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE,
                                icon_color=ERROR,
                                on_click=remove_parameter(i),
                            ),
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=15,
                        bgcolor=BG_ELEVATED,
                        border_radius=10,
                    )
                )
    
    _update_params_display()
    
    # Template selection helpers
    def apply_template(template_id):
        """Apply a method template to populate parameters."""
        template = get_template(template_id)
        if not template:
            return
        
        selected_template[0] = template
        
        # Populate study details
        name_field.value = f"{template.name} Optimization"
        description_field.value = template.description
        
        # Set response from template
        if template.responses:
            primary = template.responses[0]
            objective_name_field.value = primary.name
            minimize_toggle.value = primary.minimize
        
        # Convert template parameters to Parameter objects
        parameters.clear()
        for pt in template.parameters:
            param = Parameter(
                name=pt.name,
                param_type=pt.param_type,
                low=pt.low,
                high=pt.high,
                log_scale=pt.log_scale,
                categories=pt.categories,
                unit=pt.unit,
                description=pt.description,
            )
            parameters.append(param)
        
        _update_params_display()
        page.update()
    
    def create_template_card(template):
        """Create a card for a template."""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SCIENCE, color=PRIMARY, size=20),
                    ft.Text(template.name, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY, size=14),
                ], spacing=8),
                ft.Text(
                    template.description[:80] + "..." if len(template.description) > 80 else template.description,
                    color=TEXT_SECONDARY, size=11
                ),
                ft.Text(f"{len(template.parameters)} parameters", color=TEXT_MUTED, size=10),
            ], spacing=4),
            padding=12,
            bgcolor=BG_ELEVATED,
            border_radius=10,
            border=ft.border.all(1, BG_SURFACE),
            on_click=lambda e, t=template: apply_template(t.id),
            ink=True,
        )
    
    # Step content - Template Selection (Step 0)
    def step_0_content():
        """Template selection step."""
        category_tabs = []
        
        for category, templates in TEMPLATES_BY_CATEGORY.items():
            template_cards = ft.Column([
                ft.Row(
                    controls=[create_template_card(t) for t in templates[:3]],
                    wrap=True,
                    spacing=10,
                ),
                ft.Row(
                    controls=[create_template_card(t) for t in templates[3:6]],
                    wrap=True,
                    spacing=10,
                ) if len(templates) > 3 else ft.Container(),
            ], spacing=10)
            
            category_tabs.append(
                ft.Tab(
                    text=category,
                    content=ft.Container(
                        content=template_cards,
                        padding=20,
                    ),
                )
            )
        
        return ft.Column([
            ft.Text("Select a Method Template", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            ft.Text(
                "Start with a pre-configured template for your analytical method, or skip to create a custom study.",
                color=TEXT_SECONDARY
            ),
            ft.Container(height=15),
            
            ft.Container(
                content=ft.Tabs(
                    selected_index=0,
                    animation_duration=200,
                    expand=True,
                    tabs=category_tabs,
                ),
                bgcolor=BG_SURFACE,
                border_radius=15,
                height=320,
            ),
            
            ft.Container(height=15),
            ft.Row([
                ft.Text(
                    f"Selected: {selected_template[0].name}" if selected_template[0] else "No template selected",
                    color=SUCCESS if selected_template[0] else TEXT_MUTED,
                    size=14,
                    weight=ft.FontWeight.BOLD if selected_template[0] else ft.FontWeight.NORMAL,
                ),
                ft.TextButton(
                    "Skip - Start from scratch",
                    on_click=lambda e: skip_template(),
                ),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ], spacing=5)
    
    def skip_template():
        """Skip template selection and go to details."""
        selected_template[0] = None
        current_step[0] = 1
        update_content()
        page.update()
    
    # Step content
    def step_1_content():
        """Experiment info step."""
        return ft.Column([
            ft.Text("Step 1: Study Details", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            ft.Text("Define your method development study.", color=TEXT_SECONDARY),
            ft.Container(height=30),
            name_field,
            ft.Container(height=15),
            description_field,
            ft.Container(height=15),
            objective_name_field,
            ft.Container(height=15),
            ft.Row([
                minimize_toggle,
                ft.Text("(e.g., minimize run time, maximize resolution)", color=TEXT_MUTED, size=12),
            ]),
        ], spacing=5)
    
    def step_2_content():
        """Parameters definition step."""
        
        # Single entry form
        single_entry = ft.Container(
            content=ft.Column([
                ft.Text("Add One Parameter", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ft.Row([param_name_field, param_unit_field, param_type_dropdown], spacing=15),
                ft.Row([param_low_field, param_high_field, param_constraint_min, param_constraint_max], spacing=15),
                ft.Row([param_log_scale], spacing=20),
                param_categories_field,
                ft.ElevatedButton(
                    "Add Parameter",
                    icon=ft.Icons.ADD,
                    on_click=add_parameter,
                    style=ft.ButtonStyle(bgcolor=PRIMARY, color=TEXT_PRIMARY),
                ),
            ], spacing=15),
            padding=20,
        )
        
        # Bulk entry form
        bulk_entry = ft.Container(
            content=ft.Column([
                ft.Text("Add Multiple Parameters", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ft.Text(
                    "Format: Name (unit), Min, Max  OR  Name, Option1, Option2, ...",
                    color=TEXT_SECONDARY,
                    size=12,
                ),
                ft.Container(height=10),
                bulk_entry_field,
                bulk_error_text,
                ft.Container(height=10),
                ft.Row([
                    ft.ElevatedButton(
                        "Add All Parameters",
                        icon=ft.Icons.PLAYLIST_ADD,
                        on_click=parse_bulk_parameters,
                        style=ft.ButtonStyle(bgcolor=PRIMARY, color=TEXT_PRIMARY),
                    ),
                    ft.TextButton(
                        "Load Example",
                        on_click=lambda e: _load_example(),
                    ),
                ], spacing=15),
            ], spacing=8),
            padding=20,
        )
        
        def _load_example():
            bulk_entry_field.value = """Column Temperature (°C), 25, 45
Flow Rate (mL/min), 0.8, 1.5
Buffer pH, 6.5, 8.0
Organic Modifier, ACN, MeOH, IPA
Gradient Time (min), 10, 30"""
            bulk_error_text.visible = False
            page.update()
        
        return ft.Column([
            ft.Text("Step 2: Define Method Parameters", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            ft.Text("Add the critical process parameters (CPPs) you want to optimize.", color=TEXT_SECONDARY),
            ft.Container(height=20),
            
            # Tabs for entry mode
            ft.Container(
                content=ft.Tabs(
                    selected_index=entry_mode[0],
                    animation_duration=200,
                    expand=True,
                    tabs=[
                        ft.Tab(
                            text="Single Entry",
                            icon=ft.Icons.ADD_CIRCLE_OUTLINE,
                            content=single_entry,
                        ),
                        ft.Tab(
                            text="Bulk Entry",
                            icon=ft.Icons.LIST,
                            content=bulk_entry,
                        ),
                    ],
                    on_change=lambda e: entry_mode.__setitem__(0, e.control.selected_index),
                ),
                bgcolor=BG_SURFACE,
                border_radius=15,
                height=350,
            ),
            
            ft.Container(height=20),
            ft.Row([
                ft.Text("Parameters", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ft.Text(f"({len(parameters)} added)", color=TEXT_SECONDARY, size=14) if parameters else ft.Container(),
            ], spacing=10),
            params_list,
        ], spacing=5, scroll=ft.ScrollMode.AUTO)
    
    def step_3_content():
        """Review and confirm step."""
        exp = session.current_experiment
        
        param_items = []
        for p in parameters:
            if p.param_type == "categorical":
                detail = f"Categories: {', '.join(p.categories)}"
            else:
                scale = " (log scale)" if p.log_scale else ""
                unit_str = f" {p.unit}" if p.unit else ""
                detail = f"Range: {p.low}{unit_str} to {p.high}{unit_str}{scale}"
            
            param_items.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text(p.name, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                        ft.Text(f"{p.param_type.capitalize()} | {detail}", size=12, color=TEXT_SECONDARY),
                    ], spacing=2),
                    padding=10,
                    bgcolor=BG_ELEVATED,
                    border_radius=8,
                )
            )
        
        return ft.Column([
            ft.Text("Step 3: Review & Start", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            ft.Text("Review your study settings before starting optimization.", color=TEXT_SECONDARY),
            ft.Container(height=20),
            
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text("Study Name:", color=TEXT_SECONDARY, width=120),
                        ft.Text(name_field.value or "Untitled", weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                    ]),
                    ft.Row([
                        ft.Text("Response:", color=TEXT_SECONDARY, width=120),
                        ft.Text(
                            f"{'Minimize' if minimize_toggle.value else 'Maximize'} {objective_name_field.value}",
                            color=TEXT_PRIMARY,
                        ),
                    ]),
                    ft.Row([
                        ft.Text("Parameters:", color=TEXT_SECONDARY, width=120),
                        ft.Text(f"{len(parameters)} defined", color=TEXT_PRIMARY),
                    ]),
                    ft.Row([
                        ft.Text("Template:", color=TEXT_SECONDARY, width=120),
                        ft.Text(
                            selected_template[0].name if selected_template[0] else "Custom",
                            color=TEXT_PRIMARY,
                        ),
                    ]),
                    ft.Divider(color=BG_ELEVATED),
                    *param_items,
                ], spacing=10),
                padding=20,
                bgcolor=BG_SURFACE,
                border_radius=15,
            ),
        ], spacing=5)
    
    # Navigation
    step_indicator = ft.Row([], alignment=ft.MainAxisAlignment.CENTER, spacing=10)
    content_container = ft.Container(expand=True)
    
    def update_step_indicator():
        """Update step indicator dots."""
        step_indicator.controls.clear()
        step_names = ["Template", "Details", "Parameters", "Review"]
        for i in range(4):
            step_indicator.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Container(
                            width=12 if i == current_step[0] else 8,
                            height=12 if i == current_step[0] else 8,
                            bgcolor=PRIMARY if i <= current_step[0] else BG_ELEVATED,
                            border_radius=6,
                        ),
                        ft.Text(
                            step_names[i], 
                            size=10, 
                            color=TEXT_PRIMARY if i == current_step[0] else TEXT_MUTED
                        ) if i == current_step[0] else ft.Container(),
                    ], spacing=5),
                )
            )
    
    def update_content():
        """Update displayed content based on current step."""
        if current_step[0] == 0:
            content_container.content = step_0_content()
        elif current_step[0] == 1:
            content_container.content = step_1_content()
        elif current_step[0] == 2:
            content_container.content = step_2_content()
        else:
            content_container.content = step_3_content()
        update_step_indicator()
    
    def next_step(e):
        """Go to next step."""
        if current_step[0] == 0:
            # Template selection - always allow to proceed
            pass
        elif current_step[0] == 1:
            if not name_field.value:
                name_field.error_text = "Study name is required"
                page.update()
                return
            name_field.error_text = None
        elif current_step[0] == 2:
            if not parameters:
                page.snack_bar = ft.SnackBar(
                    content=ft.Text("Add at least one parameter"),
                    bgcolor=ERROR,
                )
                page.snack_bar.open = True
                page.update()
                return
        
        if current_step[0] < 3:
            current_step[0] += 1
            update_content()
            page.update()
        else:
            # Finish - create experiment
            exp = session.new_experiment(
                name=name_field.value,
                description=description_field.value,
            )
            exp.objective_name = objective_name_field.value
            exp.minimize = minimize_toggle.value
            exp.parameters = parameters.copy()
            
            # Save parameters to database
            session.add_parameters_to_db()
            # Update experiment settings in database
            session.save_current_to_db()
            
            page.go("/optimize")
    
    def prev_step(e):
        """Go to previous step."""
        if current_step[0] > 0:
            current_step[0] -= 1
            update_content()
            page.update()
    
    # Navigation buttons
    prev_button = ft.TextButton(
        "Back",
        icon=ft.Icons.ARROW_BACK,
        on_click=prev_step,
    )
    
    next_button = ft.ElevatedButton(
        "Next",
        icon=ft.Icons.ARROW_FORWARD,
        on_click=next_step,
        style=ft.ButtonStyle(bgcolor=PRIMARY, color=TEXT_PRIMARY),
    )
    
    update_content()
    update_step_indicator()
    
    # Main layout
    main_content = ft.Container(
        content=ft.Column([
            step_indicator,
            ft.Container(height=20),
            content_container,
            ft.Container(height=20),
            ft.Row([
                prev_button,
                next_button,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ]),
        padding=40,
        expand=True,
    )
    
    return ft.View(
        route="/new",
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
