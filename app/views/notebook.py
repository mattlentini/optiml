"""
Notebook View - Lab journal and experiment notes
"""

import flet as ft
from datetime import datetime
from core.colors import (
    PRIMARY, PRIMARY_LIGHT, PRIMARY_DARK, BG_DARK, BG_SURFACE, BG_ELEVATED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, SUCCESS, ERROR, WARNING, INFO
)
from core.session import NotebookEntry


# Entry type icons and colors
ENTRY_TYPES = {
    "note": {"icon": ft.Icons.NOTE_OUTLINED, "color": TEXT_SECONDARY, "label": "Note"},
    "observation": {"icon": ft.Icons.VISIBILITY_OUTLINED, "color": PRIMARY, "label": "Observation"},
    "issue": {"icon": ft.Icons.WARNING_OUTLINED, "color": WARNING, "label": "Issue"},
    "decision": {"icon": ft.Icons.LIGHTBULB_OUTLINED, "color": SUCCESS, "label": "Decision"},
    "milestone": {"icon": ft.Icons.FLAG_OUTLINED, "color": PRIMARY_LIGHT, "label": "Milestone"},
}


def NotebookView(page: ft.Page, session, rail: ft.NavigationRail) -> ft.View:
    """Create the notebook/lab journal view."""
    
    if not session.current_experiment:
        return ft.View(
            route="/notebook",
            controls=[
                ft.Row([
                    rail,
                    ft.VerticalDivider(width=1, color=BG_ELEVATED),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("📓", size=60),
                            ft.Text("No Active Study", size=24, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                            ft.Text("Create or load a study to use the notebook.", color=TEXT_SECONDARY),
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
    entries_list = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10, expand=True)
    
    # Entry form fields
    title_field = ft.TextField(
        label="Title",
        hint_text="Brief title for this entry",
        border_color=PRIMARY,
        expand=True,
    )
    
    content_field = ft.TextField(
        label="Content",
        hint_text="Detailed notes, observations, or thoughts...",
        multiline=True,
        min_lines=4,
        max_lines=10,
        border_color=PRIMARY,
        expand=True,
    )
    
    entry_type_dropdown = ft.Dropdown(
        label="Entry Type",
        value="note",
        width=180,
        options=[
            ft.dropdown.Option(key=key, text=info["label"])
            for key, info in ENTRY_TYPES.items()
        ],
        border_color=PRIMARY,
    )
    
    # Trial selection for linking
    trial_options = [ft.dropdown.Option(key="", text="None")]
    for t in exp.trials:
        trial_options.append(ft.dropdown.Option(
            key=str(t.trial_number),
            text=f"Run {t.trial_number}"
        ))
    
    related_trial_dropdown = ft.Dropdown(
        label="Related Run",
        value="",
        width=150,
        options=trial_options,
        border_color=PRIMARY,
    )
    
    tags_field = ft.TextField(
        label="Tags",
        hint_text="Comma-separated tags",
        width=200,
        border_color=PRIMARY,
    )
    
    def refresh_entries():
        """Refresh the entries list."""
        entries_list.controls.clear()
        
        if not exp.notebook_entries:
            entries_list.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Text("📝", size=40),
                        ft.Text("No entries yet", size=16, color=TEXT_SECONDARY),
                        ft.Text("Add your first note above!", size=14, color=TEXT_MUTED),
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                    padding=40,
                    alignment=ft.alignment.center,
                )
            )
        else:
            # Show entries in reverse chronological order
            for entry in reversed(exp.notebook_entries):
                entries_list.controls.append(create_entry_card(entry))
        
        page.update()
    
    def create_entry_card(entry: NotebookEntry) -> ft.Container:
        """Create a card for a notebook entry."""
        entry_info = ENTRY_TYPES.get(entry.entry_type, ENTRY_TYPES["note"])
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(entry.timestamp)
            time_str = dt.strftime("%b %d, %Y at %I:%M %p")
        except:
            time_str = entry.timestamp
        
        # Build tags row
        tags_row = ft.Row(spacing=5, wrap=True)
        for tag in entry.tags:
            tags_row.controls.append(
                ft.Container(
                    content=ft.Text(tag, size=11, color=PRIMARY_LIGHT),
                    bgcolor=PRIMARY + "20",
                    border_radius=10,
                    padding=ft.padding.symmetric(horizontal=8, vertical=2),
                )
            )
        
        # Related trial badge
        trial_badge = None
        if entry.related_trial:
            trial_badge = ft.Container(
                content=ft.Text(f"Run {entry.related_trial}", size=11, color=SUCCESS),
                bgcolor=SUCCESS + "20",
                border_radius=10,
                padding=ft.padding.symmetric(horizontal=8, vertical=2),
            )
        
        def delete_entry(e):
            exp.notebook_entries = [en for en in exp.notebook_entries if en.entry_id != entry.entry_id]
            refresh_entries()
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Entry deleted"),
                bgcolor=BG_ELEVATED,
            )
            page.snack_bar.open = True
            page.update()
        
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(entry_info["icon"], color=entry_info["color"], size=20),
                    ft.Text(entry.title or "Untitled", weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY, expand=True),
                    trial_badge if trial_badge else ft.Container(),
                    ft.Text(time_str, size=12, color=TEXT_MUTED),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        icon_color=TEXT_MUTED,
                        icon_size=18,
                        tooltip="Delete entry",
                        on_click=delete_entry,
                    ),
                ], alignment=ft.MainAxisAlignment.START),
                ft.Text(entry.content, color=TEXT_SECONDARY, size=14),
                tags_row if entry.tags else ft.Container(),
            ], spacing=8),
            bgcolor=BG_SURFACE,
            border_radius=8,
            padding=16,
        )
    
    def add_entry(e):
        """Add a new notebook entry."""
        if not content_field.value and not title_field.value:
            page.snack_bar = ft.SnackBar(
                content=ft.Text("Please enter a title or content"),
                bgcolor=ERROR,
            )
            page.snack_bar.open = True
            page.update()
            return
        
        # Parse tags
        tags = []
        if tags_field.value:
            tags = [t.strip() for t in tags_field.value.split(",") if t.strip()]
        
        # Parse related trial
        related_trial = None
        if related_trial_dropdown.value:
            try:
                related_trial = int(related_trial_dropdown.value)
            except:
                pass
        
        # Add entry
        exp.add_notebook_entry(
            title=title_field.value or "",
            content=content_field.value or "",
            entry_type=entry_type_dropdown.value or "note",
            related_trial=related_trial,
            tags=tags,
        )
        
        # Clear form
        title_field.value = ""
        content_field.value = ""
        tags_field.value = ""
        related_trial_dropdown.value = ""
        entry_type_dropdown.value = "note"
        
        refresh_entries()
        
        page.snack_bar = ft.SnackBar(
            content=ft.Text("Entry added! ✓"),
            bgcolor=SUCCESS,
        )
        page.snack_bar.open = True
        page.update()
    
    # Build the view
    header = ft.Container(
        content=ft.Row([
            ft.Column([
                ft.Text("📓 Lab Notebook", size=28, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
                ft.Text(f"Study: {exp.name}", size=14, color=TEXT_SECONDARY),
            ], spacing=2),
            ft.Container(expand=True),
            ft.Text(f"{len(exp.notebook_entries)} entries", color=TEXT_MUTED),
        ]),
        padding=ft.padding.only(left=30, right=30, top=20, bottom=10),
    )
    
    # Entry form
    entry_form = ft.Container(
        content=ft.Column([
            ft.Text("New Entry", size=16, weight=ft.FontWeight.BOLD, color=TEXT_PRIMARY),
            ft.Row([
                title_field,
            ]),
            content_field,
            ft.Row([
                entry_type_dropdown,
                related_trial_dropdown,
                tags_field,
                ft.Container(expand=True),
                ft.ElevatedButton(
                    "Add Entry",
                    icon=ft.Icons.ADD,
                    on_click=add_entry,
                    style=ft.ButtonStyle(bgcolor=PRIMARY, color=TEXT_PRIMARY),
                ),
            ], spacing=10),
        ], spacing=12),
        bgcolor=BG_SURFACE,
        border_radius=12,
        padding=20,
        margin=ft.margin.only(left=30, right=30, bottom=20),
    )
    
    # Entries container
    entries_container = ft.Container(
        content=entries_list,
        expand=True,
        padding=ft.padding.only(left=30, right=30, bottom=20),
    )
    
    # Initial load
    refresh_entries()
    
    return ft.View(
        route="/notebook",
        controls=[
            ft.Row([
                rail,
                ft.VerticalDivider(width=1, color=BG_ELEVATED),
                ft.Column([
                    header,
                    entry_form,
                    ft.Text("Journal Entries", size=16, weight=ft.FontWeight.BOLD, 
                            color=TEXT_PRIMARY, 
                            style=ft.TextStyle(italic=False)),
                    entries_container,
                ], expand=True, spacing=0),
            ], expand=True),
        ],
        padding=0,
        bgcolor=BG_DARK,
    )
