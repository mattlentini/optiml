"""
Method Development Templates for Analytical Development
========================================================
Pre-built templates for common analytical methods in biotechnology.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ParameterTemplate:
    """Template for a method parameter."""
    name: str
    param_type: str  # 'real', 'integer', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    unit: Optional[str] = None
    log_scale: bool = False
    categories: Optional[List[str]] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "param_type": self.param_type,
            "low": self.low,
            "high": self.high,
            "unit": self.unit,
            "log_scale": self.log_scale,
            "categories": self.categories,
            "description": self.description,
        }


@dataclass
class ResponseTemplate:
    """Template for a response/objective."""
    name: str
    minimize: bool
    unit: Optional[str] = None
    description: str = ""
    target_value: Optional[float] = None  # For target-based optimization


@dataclass 
class MethodTemplate:
    """Template for a complete method development study."""
    id: str
    name: str
    category: str
    description: str
    parameters: List[ParameterTemplate]
    responses: List[ResponseTemplate]
    icon: str = "science"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "responses": [{"name": r.name, "minimize": r.minimize, "unit": r.unit} for r in self.responses],
            "tags": self.tags,
        }


# =============================================================================
# CHROMATOGRAPHY TEMPLATES
# =============================================================================

HPLC_METHOD = MethodTemplate(
    id="hplc_method",
    name="HPLC/UPLC Method",
    category="Chromatography",
    description="Optimize reversed-phase or ion-exchange chromatography methods for protein or small molecule analysis.",
    icon="water_drop",
    tags=["chromatography", "separation", "purity"],
    parameters=[
        ParameterTemplate("Organic Modifier %", "real", 5, 95, "%", description="Acetonitrile or methanol percentage"),
        ParameterTemplate("Gradient Time", "real", 5, 60, "min", description="Total gradient duration"),
        ParameterTemplate("Flow Rate", "real", 0.2, 2.0, "mL/min", description="Mobile phase flow rate"),
        ParameterTemplate("Column Temperature", "real", 20, 60, "°C", description="Column oven temperature"),
        ParameterTemplate("Buffer pH", "real", 2.0, 8.0, None, description="Mobile phase pH"),
        ParameterTemplate("Injection Volume", "real", 1, 50, "µL", description="Sample injection volume"),
    ],
    responses=[
        ResponseTemplate("Resolution", minimize=False, description="Peak resolution (Rs)"),
        ResponseTemplate("Run Time", minimize=True, unit="min", description="Total analysis time"),
        ResponseTemplate("Peak Symmetry", minimize=False, description="USP tailing factor (target: 1.0)"),
    ],
)

SEC_METHOD = MethodTemplate(
    id="sec_method",
    name="Size Exclusion Chromatography",
    category="Chromatography",
    description="Optimize SEC methods for aggregate and fragment analysis of proteins.",
    icon="donut_large",
    tags=["chromatography", "aggregation", "size"],
    parameters=[
        ParameterTemplate("Flow Rate", "real", 0.3, 1.0, "mL/min", description="Isocratic flow rate"),
        ParameterTemplate("Column Temperature", "real", 15, 35, "°C", description="Column temperature"),
        ParameterTemplate("Salt Concentration", "real", 100, 500, "mM", description="Buffer ionic strength"),
        ParameterTemplate("Buffer pH", "real", 6.0, 8.0, None, description="Mobile phase pH"),
        ParameterTemplate("Injection Volume", "real", 5, 100, "µL", description="Sample volume"),
        ParameterTemplate("Sample Concentration", "real", 0.5, 10, "mg/mL", description="Protein concentration"),
    ],
    responses=[
        ResponseTemplate("Monomer Peak Resolution", minimize=False, description="Resolution from aggregates"),
        ResponseTemplate("Recovery %", minimize=False, unit="%", description="Mass balance recovery"),
        ResponseTemplate("Run Time", minimize=True, unit="min"),
    ],
)

IEX_METHOD = MethodTemplate(
    id="iex_method",
    name="Ion Exchange Chromatography",
    category="Chromatography",
    description="Optimize cation or anion exchange methods for charge variant analysis.",
    icon="swap_vert",
    tags=["chromatography", "charge", "variants"],
    parameters=[
        ParameterTemplate("Starting Salt", "real", 0, 100, "mM", description="Initial salt concentration"),
        ParameterTemplate("Ending Salt", "real", 200, 1000, "mM", description="Final salt concentration"),
        ParameterTemplate("Gradient Time", "real", 10, 60, "min", description="Salt gradient duration"),
        ParameterTemplate("Flow Rate", "real", 0.5, 2.0, "mL/min"),
        ParameterTemplate("Column Temperature", "real", 20, 40, "°C"),
        ParameterTemplate("Buffer pH", "real", 5.0, 9.0, None, description="Optimized for pI"),
    ],
    responses=[
        ResponseTemplate("Main Peak %", minimize=False, unit="%", description="Relative area of main species"),
        ResponseTemplate("Resolution", minimize=False, description="Acidic/basic variant resolution"),
        ResponseTemplate("Run Time", minimize=True, unit="min"),
    ],
)

# =============================================================================
# MASS SPECTROMETRY TEMPLATES
# =============================================================================

MS_INTACT = MethodTemplate(
    id="ms_intact",
    name="Intact Mass Spectrometry",
    category="Mass Spectrometry",
    description="Optimize ESI-MS conditions for intact protein mass analysis.",
    icon="schema",
    tags=["mass spec", "intact", "molecular weight"],
    parameters=[
        ParameterTemplate("Capillary Voltage", "real", 2.0, 4.5, "kV", description="ESI capillary voltage"),
        ParameterTemplate("Cone Voltage", "real", 20, 150, "V", description="Sampling cone voltage"),
        ParameterTemplate("Source Temperature", "real", 80, 150, "°C"),
        ParameterTemplate("Desolvation Temperature", "real", 200, 500, "°C"),
        ParameterTemplate("Desolvation Gas Flow", "real", 500, 1200, "L/hr"),
        ParameterTemplate("Cone Gas Flow", "real", 0, 100, "L/hr"),
    ],
    responses=[
        ResponseTemplate("Signal Intensity", minimize=False, unit="counts", description="Peak intensity"),
        ResponseTemplate("Mass Accuracy", minimize=True, unit="ppm", description="Mass error"),
        ResponseTemplate("Peak Width", minimize=True, unit="Da", description="Deconvoluted peak width"),
    ],
)

MS_PEPTIDE_MAP = MethodTemplate(
    id="ms_peptide_map",
    name="Peptide Mapping LC-MS/MS",
    category="Mass Spectrometry",
    description="Optimize LC-MS/MS conditions for peptide mapping and PTM analysis.",
    icon="grid_on",
    tags=["mass spec", "peptide", "PTM", "sequence"],
    parameters=[
        ParameterTemplate("Collision Energy", "real", 15, 45, "eV", description="CID/HCD collision energy"),
        ParameterTemplate("MS2 Resolution", "categorical", categories=["15K", "30K", "60K", "120K"]),
        ParameterTemplate("AGC Target", "real", 1e4, 1e6, None, log_scale=True, description="Ion target"),
        ParameterTemplate("Max Injection Time", "real", 20, 200, "ms"),
        ParameterTemplate("Dynamic Exclusion", "real", 10, 60, "s"),
        ParameterTemplate("Isolation Window", "real", 1.0, 3.0, "m/z"),
    ],
    responses=[
        ResponseTemplate("Sequence Coverage", minimize=False, unit="%"),
        ResponseTemplate("PTM Site Localization", minimize=False, unit="%", description="Localization confidence"),
        ResponseTemplate("# Unique Peptides", minimize=False),
    ],
)

# =============================================================================
# IMMUNOASSAY TEMPLATES
# =============================================================================

ELISA_METHOD = MethodTemplate(
    id="elisa_method",
    name="ELISA Optimization",
    category="Immunoassays",
    description="Optimize sandwich or competitive ELISA for protein quantification.",
    icon="science",
    tags=["immunoassay", "ELISA", "quantification"],
    parameters=[
        ParameterTemplate("Capture Ab Concentration", "real", 0.5, 10, "µg/mL", description="Coating concentration"),
        ParameterTemplate("Coating Time", "real", 1, 24, "hr", description="Plate coating duration"),
        ParameterTemplate("Blocking Time", "real", 0.5, 4, "hr"),
        ParameterTemplate("Detection Ab Dilution", "real", 100, 10000, "fold", log_scale=True),
        ParameterTemplate("Sample Incubation Time", "real", 0.5, 4, "hr"),
        ParameterTemplate("Substrate Development Time", "real", 5, 30, "min"),
    ],
    responses=[
        ResponseTemplate("LLOQ", minimize=True, unit="ng/mL", description="Lower limit of quantification"),
        ResponseTemplate("Dynamic Range", minimize=False, description="Log range of quantification"),
        ResponseTemplate("Intra-assay CV%", minimize=True, unit="%", description="Precision"),
        ResponseTemplate("Signal/Background", minimize=False, description="S/B ratio at mid-curve"),
    ],
)

MSD_METHOD = MethodTemplate(
    id="msd_method",
    name="MSD Electrochemiluminescence",
    category="Immunoassays",
    description="Optimize Meso Scale Discovery ECL immunoassays.",
    icon="lightbulb",
    tags=["immunoassay", "MSD", "ECL", "multiplex"],
    parameters=[
        ParameterTemplate("Capture Ab Concentration", "real", 1, 20, "µg/mL"),
        ParameterTemplate("Detection Ab Concentration", "real", 0.1, 5, "µg/mL"),
        ParameterTemplate("Sample Dilution", "real", 2, 100, "fold", log_scale=True),
        ParameterTemplate("Incubation Time", "real", 1, 4, "hr"),
        ParameterTemplate("Shaking Speed", "integer", 300, 700, "rpm"),
        ParameterTemplate("Read Buffer Volume", "real", 100, 200, "µL"),
    ],
    responses=[
        ResponseTemplate("LLOQ", minimize=True, unit="pg/mL"),
        ResponseTemplate("Signal/Noise", minimize=False),
        ResponseTemplate("CV%", minimize=True, unit="%"),
    ],
)

# =============================================================================
# CELL-BASED ASSAY TEMPLATES
# =============================================================================

CELL_POTENCY = MethodTemplate(
    id="cell_potency",
    name="Cell-Based Potency Assay",
    category="Cell-Based Assays",
    description="Optimize reporter gene or proliferation-based potency assays.",
    icon="biotech",
    tags=["potency", "cell", "bioassay"],
    parameters=[
        ParameterTemplate("Cell Density", "real", 5000, 50000, "cells/well"),
        ParameterTemplate("Pre-incubation Time", "real", 2, 24, "hr", description="Cell attachment/rest"),
        ParameterTemplate("Treatment Time", "real", 4, 72, "hr", description="Drug exposure duration"),
        ParameterTemplate("Serum %", "real", 0, 10, "%", description="FBS in treatment media"),
        ParameterTemplate("Sample Dilution Series", "integer", 6, 12, "points"),
        ParameterTemplate("Starting Concentration", "real", 0.1, 100, "µg/mL", log_scale=True),
    ],
    responses=[
        ResponseTemplate("EC50", minimize=True, unit="ng/mL", description="Half-maximal effective concentration"),
        ResponseTemplate("Signal Window", minimize=False, description="Max/min response ratio"),
        ResponseTemplate("Z-factor", minimize=False, description="Assay quality metric (target >0.5)"),
        ResponseTemplate("Hill Slope", minimize=False, description="Target: 0.8-1.2"),
    ],
)

# =============================================================================
# qPCR TEMPLATES
# =============================================================================

QPCR_METHOD = MethodTemplate(
    id="qpcr_method",
    name="qPCR Optimization",
    category="Molecular Biology",
    description="Optimize quantitative PCR conditions for gene expression or copy number analysis.",
    icon="dna",
    tags=["qPCR", "PCR", "gene expression"],
    parameters=[
        ParameterTemplate("Primer Concentration", "real", 100, 900, "nM"),
        ParameterTemplate("Probe Concentration", "real", 50, 400, "nM"),
        ParameterTemplate("Annealing Temperature", "real", 55, 68, "°C"),
        ParameterTemplate("Extension Time", "real", 15, 60, "s"),
        ParameterTemplate("MgCl2 Concentration", "real", 1.5, 4.0, "mM"),
        ParameterTemplate("Template Amount", "real", 1, 100, "ng", log_scale=True),
    ],
    responses=[
        ResponseTemplate("Ct Value", minimize=True, description="Threshold cycle (for same input)"),
        ResponseTemplate("Efficiency %", minimize=False, unit="%", target_value=100),
        ResponseTemplate("R² Value", minimize=False, target_value=0.99),
        ResponseTemplate("NTC Ct", minimize=False, description="No-template control (higher is better)"),
    ],
)

# =============================================================================
# SPECTROSCOPY TEMPLATES
# =============================================================================

DLS_METHOD = MethodTemplate(
    id="dls_method",
    name="Dynamic Light Scattering",
    category="Biophysical",
    description="Optimize DLS conditions for particle size and polydispersity measurement.",
    icon="blur_on",
    tags=["DLS", "particle size", "aggregation"],
    parameters=[
        ParameterTemplate("Protein Concentration", "real", 0.1, 10, "mg/mL"),
        ParameterTemplate("Measurement Temperature", "real", 15, 40, "°C"),
        ParameterTemplate("Equilibration Time", "real", 1, 10, "min"),
        ParameterTemplate("Number of Runs", "integer", 3, 20, None),
        ParameterTemplate("Run Duration", "real", 5, 30, "s"),
        ParameterTemplate("Attenuator Position", "integer", 1, 11, None, description="Auto or fixed"),
    ],
    responses=[
        ResponseTemplate("Z-Average", minimize=False, unit="nm", description="Mean hydrodynamic diameter"),
        ResponseTemplate("Polydispersity Index", minimize=True, description="PDI (target <0.2)"),
        ResponseTemplate("Derived Count Rate", minimize=False, unit="kcps", description="Signal intensity"),
    ],
)

DSF_METHOD = MethodTemplate(
    id="dsf_method",
    name="Differential Scanning Fluorimetry",
    category="Biophysical",
    description="Optimize thermal shift assay conditions for protein stability screening.",
    icon="thermostat",
    tags=["DSF", "thermal stability", "Tm"],
    parameters=[
        ParameterTemplate("Protein Concentration", "real", 0.1, 2.0, "mg/mL"),
        ParameterTemplate("Dye Concentration", "real", 1, 20, "X", description="SYPRO Orange dilution"),
        ParameterTemplate("Heating Rate", "real", 0.5, 4.0, "°C/min"),
        ParameterTemplate("Start Temperature", "real", 20, 30, "°C"),
        ParameterTemplate("End Temperature", "real", 80, 99, "°C"),
        ParameterTemplate("Buffer pH", "real", 5.0, 8.5, None),
    ],
    responses=[
        ResponseTemplate("Tm", minimize=False, unit="°C", description="Melting temperature"),
        ResponseTemplate("ΔTm", minimize=False, unit="°C", description="Tm shift vs reference"),
        ResponseTemplate("Transition Sharpness", minimize=False, description="Cooperative unfolding"),
    ],
)

# =============================================================================
# CAPILLARY ELECTROPHORESIS
# =============================================================================

CE_SDS_METHOD = MethodTemplate(
    id="ce_sds_method",
    name="CE-SDS Purity Analysis",
    category="Electrophoresis",
    description="Optimize capillary electrophoresis SDS conditions for purity and fragment analysis.",
    icon="straighten",
    tags=["CE-SDS", "purity", "electrophoresis"],
    parameters=[
        ParameterTemplate("Separation Voltage", "real", 10, 25, "kV"),
        ParameterTemplate("Capillary Temperature", "real", 20, 30, "°C"),
        ParameterTemplate("Injection Time", "real", 5, 30, "s"),
        ParameterTemplate("Injection Voltage", "real", 3, 10, "kV"),
        ParameterTemplate("Sample Concentration", "real", 0.5, 5, "mg/mL"),
        ParameterTemplate("SDS Incubation Temp", "real", 60, 80, "°C"),
    ],
    responses=[
        ResponseTemplate("Main Peak %", minimize=False, unit="%"),
        ResponseTemplate("Resolution", minimize=False, description="HC/LC or fragment resolution"),
        ResponseTemplate("Migration Time CV%", minimize=True, unit="%"),
    ],
)

CIEF_METHOD = MethodTemplate(
    id="cief_method",
    name="cIEF Charge Heterogeneity",
    category="Electrophoresis",
    description="Optimize capillary isoelectric focusing for charge variant analysis.",
    icon="align_horizontal_center",
    tags=["cIEF", "charge variants", "isoelectric"],
    parameters=[
        ParameterTemplate("Sample Concentration", "real", 0.2, 2.0, "mg/mL"),
        ParameterTemplate("Ampholyte %", "real", 2, 6, "%", description="Carrier ampholyte concentration"),
        ParameterTemplate("Focusing Time", "real", 5, 20, "min"),
        ParameterTemplate("Focusing Voltage", "real", 15, 30, "kV"),
        ParameterTemplate("pI Marker 1", "real", 4.0, 6.0, None, description="Acidic marker pI"),
        ParameterTemplate("pI Marker 2", "real", 8.0, 10.0, None, description="Basic marker pI"),
    ],
    responses=[
        ResponseTemplate("Main Peak %", minimize=False, unit="%"),
        ResponseTemplate("Acidic Variants %", minimize=True, unit="%"),
        ResponseTemplate("Basic Variants %", minimize=True, unit="%"),
        ResponseTemplate("pI Accuracy", minimize=True, description="pI vs expected"),
    ],
)

# =============================================================================
# STABILITY TEMPLATES
# =============================================================================

FORMULATION_SCREEN = MethodTemplate(
    id="formulation_screen",
    name="Formulation Screening",
    category="Stability",
    description="Optimize buffer and excipient conditions for protein stability.",
    icon="science",
    tags=["formulation", "stability", "excipients"],
    parameters=[
        ParameterTemplate("Buffer pH", "real", 5.0, 8.0, None),
        ParameterTemplate("Buffer Concentration", "real", 10, 50, "mM"),
        ParameterTemplate("NaCl Concentration", "real", 0, 300, "mM"),
        ParameterTemplate("Sucrose %", "real", 0, 10, "%"),
        ParameterTemplate("Polysorbate 80", "real", 0, 0.1, "%"),
        ParameterTemplate("Protein Concentration", "real", 1, 150, "mg/mL"),
    ],
    responses=[
        ResponseTemplate("Tm", minimize=False, unit="°C", description="Thermal stability"),
        ResponseTemplate("Aggregation %", minimize=True, unit="%", description="By SEC at T=0"),
        ResponseTemplate("Viscosity", minimize=True, unit="cP", description="At target concentration"),
        ResponseTemplate("Opalescence", minimize=True, unit="NTU"),
    ],
)


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

ALL_TEMPLATES = [
    # Chromatography
    HPLC_METHOD,
    SEC_METHOD,
    IEX_METHOD,
    # Mass Spectrometry
    MS_INTACT,
    MS_PEPTIDE_MAP,
    # Immunoassays
    ELISA_METHOD,
    MSD_METHOD,
    # Cell-Based
    CELL_POTENCY,
    # Molecular Biology
    QPCR_METHOD,
    # Biophysical
    DLS_METHOD,
    DSF_METHOD,
    # Electrophoresis
    CE_SDS_METHOD,
    CIEF_METHOD,
    # Stability
    FORMULATION_SCREEN,
]

TEMPLATES_BY_ID = {t.id: t for t in ALL_TEMPLATES}

TEMPLATES_BY_CATEGORY = {}
for template in ALL_TEMPLATES:
    if template.category not in TEMPLATES_BY_CATEGORY:
        TEMPLATES_BY_CATEGORY[template.category] = []
    TEMPLATES_BY_CATEGORY[template.category].append(template)


def get_template(template_id: str) -> MethodTemplate:
    """Get a template by ID."""
    return TEMPLATES_BY_ID.get(template_id)


def get_all_categories() -> List[str]:
    """Get list of all template categories."""
    return list(TEMPLATES_BY_CATEGORY.keys())


def get_templates_for_category(category: str) -> List[MethodTemplate]:
    """Get all templates for a category."""
    return TEMPLATES_BY_CATEGORY.get(category, [])
