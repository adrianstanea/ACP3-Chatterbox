"""Configuration and constants for analysis."""

from pathlib import Path
from typing import Dict, List

# Analysis conditions (8 total)
CONDITIONS = [
    "bas_exag0.3_cfg0.3",
    "bas_exag0.3_cfg0.5",
    "bas_exag0.7_cfg0.3",
    "bas_exag0.7_cfg0.5",
    "sgs_exag0.3_cfg0.3",
    "sgs_exag0.3_cfg0.5",
    "sgs_exag0.7_cfg0.3",
    "sgs_exag0.7_cfg0.5",
]

# Condition groupings
SYNTHESIZERS = {
    "bas": [
        "bas_exag0.3_cfg0.3",
        "bas_exag0.3_cfg0.5",
        "bas_exag0.7_cfg0.3",
        "bas_exag0.7_cfg0.5",
    ],
    "sgs": [
        "sgs_exag0.3_cfg0.3",
        "sgs_exag0.3_cfg0.5",
        "sgs_exag0.7_cfg0.3",
        "sgs_exag0.7_cfg0.5",
    ],
}

# Metrics names
METRICS = ["CER", "WER", "UTMOS_Synthesized", "Speaker_Cosine_Similarity"]

# Output directories structure
OUTPUT_STRUCTURE = {
    "figures": [
        "dataset_characterization",
        "training_dynamics",
        "evaluation_results",
        "ablation_analysis",
    ],
    "tables": [],
    "data": [],
}

# Color schemes for academic plots
COLORS_ACADEMIC = {
    "bas": "#1f77b4",
    "sgs": "#ff7f0e",
}

COLORS_METRICS = {
    "CER": "#d62728",
    "WER": "#2ca02c",
    "UTMOS_Synthesized": "#9467bd",
    "Speaker_Cosine_Similarity": "#8c564b",
}
