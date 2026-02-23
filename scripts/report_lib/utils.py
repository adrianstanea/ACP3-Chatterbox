"""Utility functions for report generation."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_output_dirs(output_dir: Path, formats: list) -> Dict[str, Path]:
    """Create output directory structure based on requested formats."""
    dirs = {"root": output_dir}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dirs["figures"] = output_dir / "figures"
    dirs["figures"].mkdir(exist_ok=True)

    for subdir in [
        "dataset_characterization",
        "training_dynamics",
        "evaluation_results",
        "ablation_analysis",
    ]:
        (dirs["figures"] / subdir).mkdir(exist_ok=True)

    dirs["tables"] = output_dir / "tables"
    dirs["tables"].mkdir(exist_ok=True)

    dirs["data"] = output_dir / "data"
    dirs["data"].mkdir(exist_ok=True)

    return dirs


def save_json(data: Any, path: Path, pretty: bool = True) -> None:
    """Save data to JSON with error handling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2 if pretty else None, default=str)


def load_json(path: Path) -> Any:
    """Load JSON with error handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)
