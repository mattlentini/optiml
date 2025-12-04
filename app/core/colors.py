"""
OptiML Color Scheme
====================
A clean, minimal science-inspired color palette matching the OptiML logo.

Design Philosophy:
- Blue gradient from logo (light to dark blue)
- Dark navy background from logo
- Light text for high contrast
- Muted accents keep focus on data
"""

# === PRIMARY COLORS (extracted from logo) ===
PRIMARY = "#2951AA"          # Main blue from logo (mid-blue)
PRIMARY_LIGHT = "#8AAAE9"    # Light blue from logo gradient
PRIMARY_DARK = "#224490"     # Dark blue from logo
ACCENT = "#2A54B0"           # Accent blue

# === BACKGROUND COLORS (from logo background) ===
BG_DARK = "#060F1F"          # Deep navy (exact from logo)
BG_SURFACE = "#0A1628"       # Cards, panels (slightly lighter)
BG_ELEVATED = "#101D30"      # Elevated elements, inputs

# === TEXT COLORS (from logo text) ===
TEXT_PRIMARY = "#E9F0FB"     # Primary text (exact from logo)
TEXT_SECONDARY = "#94A3B8"   # Secondary text (slate-400)
TEXT_MUTED = "#64748B"       # Muted, hints (slate-500)

# === SEMANTIC COLORS ===
SUCCESS = "#22C55E"          # Green - best result, success
SUCCESS_BG = "#22C55E20"     # Success background tint
WARNING = "#F59E0B"          # Amber - suggestions, attention
WARNING_BG = "#F59E0B20"     # Warning background tint
ERROR = "#EF4444"            # Red - errors, delete
ERROR_BG = "#EF444420"       # Error background tint
INFO = "#2951AA"             # Blue - information (matches primary)

# === CHART COLORS (from logo gradient) ===
CHART_PRIMARY = "#2951AA"    # Primary data series (logo blue)
CHART_SECONDARY = "#8AAAE9"  # Secondary series (light blue)
CHART_TERTIARY = "#6589D2"   # Tertiary series (mid blue)
CHART_POINTS = "#64748B"     # Data points
CHART_BEST = "#22C55E"       # Best/optimal line
CHART_GRID = "#101D30"       # Grid lines

# === BORDER COLORS ===
BORDER = "#101D30"           # Default borders
BORDER_FOCUS = "#2951AA"     # Focused input borders

# === GRADIENTS (from logo) ===
GRADIENT_PRIMARY = ("#8AAAE9", "#224490")  # Light to dark blue (logo gradient)
GRADIENT_ACCENT = ("#2951AA", "#2A54B0")   # Logo blue gradient


def get_flet_color_scheme():
    """Return a Flet ColorScheme object with the OptiML palette."""
    import flet as ft
    return ft.ColorScheme(
        primary=PRIMARY,
        secondary=CHART_SECONDARY,
        surface=BG_SURFACE,
        background=BG_DARK,
        on_primary=TEXT_PRIMARY,
        on_secondary=TEXT_PRIMARY,
        on_surface=TEXT_PRIMARY,
        on_background=TEXT_SECONDARY,
        error=ERROR,
    )
