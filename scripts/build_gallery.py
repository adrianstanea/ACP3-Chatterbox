#!/usr/bin/env python3
"""
GitHub Pages Gallery Generator for Romanian Chatterbox TTS Evaluation

Creates a clean, academic-style static HTML site for audio comparison across
different parameter conditions (speaker, exaggeration, CFG weight).

Output Structure:
    docs/
    ├── index.html           # Overview with experiment info
    ├── results.html         # Metrics summary table
    ├── {condition}.html     # One page per condition with audio samples
    ├── audio/{condition}/   # Audio files (copied, limited to max-samples)
    └── style.css            # Minimal academic CSS

Usage:
    python scripts/build_gallery.py \\
        --experiment-dir experiments/2026-02-22 \\
        --output-dir docs \\
        --max-samples 10

Author: Adrian Stanea / ACP3
License: Apache 2.0
"""

import argparse
import csv
import html
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate static HTML gallery for TTS audio comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory containing audio conditions and config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs"),
        help="Directory to output HTML files and audio",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of samples per condition to include",
    )
    parser.add_argument(
        "--selected-samples",
        type=Path,
        default=None,
        help="Path to selected_samples.json from select_best_samples.py. "
             "When provided, only these best samples are included in the gallery.",
    )

    return parser.parse_args()


# ==============================================================================
# CSS Styles
# ==============================================================================

ACADEMIC_CSS = """/* Academic-style CSS for TTS Audio Gallery */

:root {
    --color-bg: #fafafa;
    --color-surface: #ffffff;
    --color-text: #333333;
    --color-text-muted: #666666;
    --color-border: #e0e0e0;
    --color-accent: #2c5282;
    --color-accent-hover: #1a365d;
    --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
    --font-mono: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono",
                 Consolas, monospace;
}

* {
    box-sizing: border-box;
}

body {
    font-family: var(--font-family);
    font-size: 16px;
    line-height: 1.6;
    color: var(--color-text);
    background-color: var(--color-bg);
    margin: 0;
    padding: 0;
}

.container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

/* Typography */
h1 {
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    color: var(--color-text);
}

h2 {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 2rem 0 1rem 0;
    color: var(--color-text);
    border-bottom: 1px solid var(--color-border);
    padding-bottom: 0.5rem;
}

h3 {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.5rem 0 0.5rem 0;
    color: var(--color-text);
}

p {
    margin: 0.5rem 0;
}

.subtitle {
    color: var(--color-text-muted);
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

/* Navigation */
nav {
    background-color: var(--color-surface);
    border-bottom: 1px solid var(--color-border);
    padding: 1rem 0;
    margin-bottom: 2rem;
}

nav .container {
    display: flex;
    align-items: center;
    gap: 2rem;
    padding-top: 0;
    padding-bottom: 0;
}

nav a {
    color: var(--color-text-muted);
    text-decoration: none;
    font-size: 0.9rem;
}

nav a:hover,
nav a.active {
    color: var(--color-accent);
}

nav .brand {
    font-weight: 600;
    color: var(--color-text);
}

/* Links */
a {
    color: var(--color-accent);
    text-decoration: none;
}

a:hover {
    color: var(--color-accent-hover);
    text-decoration: underline;
}

/* Cards */
.card {
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Lists */
ul.condition-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

ul.condition-list li {
    padding: 0.75rem 0;
    border-bottom: 1px solid var(--color-border);
}

ul.condition-list li:last-child {
    border-bottom: none;
}

ul.condition-list .condition-name {
    font-weight: 500;
}

ul.condition-list .condition-meta {
    color: var(--color-text-muted);
    font-size: 0.875rem;
    margin-top: 0.25rem;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin: 1rem 0;
}

th, td {
    text-align: left;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--color-border);
}

th {
    background-color: var(--color-bg);
    font-weight: 600;
    color: var(--color-text);
}

tr:hover {
    background-color: var(--color-bg);
}

.number {
    font-family: var(--font-mono);
    text-align: right;
}

/* Audio Samples */
.sample {
    background-color: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}

.sample-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
}

.sample-id {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--color-text-muted);
}

.sample-text {
    font-size: 0.95rem;
    color: var(--color-text);
    margin-bottom: 0.75rem;
    padding: 0.5rem 0.75rem;
    background-color: var(--color-bg);
    border-radius: 3px;
    font-style: italic;
}

audio {
    width: 100%;
    height: 40px;
}

/* Parameters */
.params {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}

.param {
    display: flex;
    flex-direction: column;
}

.param-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-muted);
}

.param-value {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 500;
}

/* Footer */
footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--color-border);
    color: var(--color-text-muted);
    font-size: 0.85rem;
}

/* Responsive */
@media (max-width: 600px) {
    .container {
        padding: 1rem;
    }

    nav .container {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.5rem;
    }

    .params {
        flex-direction: column;
        gap: 0.75rem;
    }

    th, td {
        padding: 0.5rem;
    }
}
"""


# ==============================================================================
# HTML Templates
# ==============================================================================

def html_head(title: str, extra_css: str = "") -> str:
    """Generate HTML head section."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <link rel="stylesheet" href="style.css">
    {extra_css}
</head>
"""


def html_nav(active_page: str = "index") -> str:
    """Generate navigation bar."""
    pages = [
        ("index", "index.html", "Overview"),
        ("results", "results.html", "Results"),
    ]

    links = []
    for page_id, href, label in pages:
        active_class = ' class="active"' if page_id == active_page else ""
        links.append(f'<a href="{href}"{active_class}>{label}</a>')

    return f"""<body>
<nav>
    <div class="container">
        <span class="brand">Romanian TTS Evaluation</span>
        {" ".join(links)}
    </div>
</nav>
"""


def html_footer() -> str:
    """Generate footer."""
    return """<footer>
    <p>Generated by Romanian Chatterbox TTS Evaluation Pipeline</p>
</footer>
</div>
</body>
</html>
"""


# ==============================================================================
# Data Loading
# ==============================================================================

def load_experiment_config(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    """Load experiment configuration from JSON."""
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results_summary(experiment_dir: Path) -> Optional[List[Dict[str, str]]]:
    """Load results summary from CSV."""
    results_path = experiment_dir / "results_summary.csv"
    if not results_path.exists():
        return None

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results


def find_conditions(experiment_dir: Path) -> List[str]:
    """Find all condition directories in the experiment."""
    conditions = []
    for item in experiment_dir.iterdir():
        if item.is_dir() and "_exag" in item.name and "_cfg" in item.name:
            conditions.append(item.name)
    return sorted(conditions)


def parse_condition_name(condition: str) -> Dict[str, str]:
    """Parse condition name into components."""
    # Format: {speaker}_exag{exag}_cfg{cfg}
    parts = condition.split("_")
    result = {"speaker": parts[0].upper()}

    for part in parts[1:]:
        if part.startswith("exag"):
            result["exaggeration"] = part[4:]
        elif part.startswith("cfg"):
            result["cfg_weight"] = part[3:]

    return result


def load_filelist(experiment_dir: Path, condition: str) -> List[Tuple[str, str]]:
    """Load filelist for a condition."""
    filelist_path = experiment_dir / "filelists" / f"{condition}.csv"
    if not filelist_path.exists():
        return []

    samples = []
    with open(filelist_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                sample_id, transcript = row[0], row[1]
                samples.append((sample_id, transcript))

    return samples


def load_selected_samples(selected_path: Path) -> Optional[Dict[str, List[Dict]]]:
    """Load selected samples from select_best_samples.py output."""
    if not selected_path.exists():
        return None

    with open(selected_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("selected", None)


def get_samples_for_condition(
    experiment_dir: Path,
    condition: str,
    selected_samples: Optional[Dict[str, List[Dict]]],
    max_samples: int,
) -> List[Tuple[str, str, Optional[Dict]]]:
    """
    Get samples for a condition, optionally filtering to selected best samples.

    Returns list of (sample_id, transcript, metrics_dict) tuples.
    """
    if selected_samples and condition in selected_samples:
        # Use pre-selected best samples
        samples_data = selected_samples[condition]
        samples = []
        for s in samples_data[:max_samples]:
            audio_file = s.get("audio_file", "")
            # Remove .wav extension for sample_id if present
            sample_id = audio_file.replace(".wav", "") if audio_file.endswith(".wav") else audio_file
            transcript = s.get("original_transcription", "")
            metrics = {k: v for k, v in s.items() if k not in ["audio_file", "original_transcription", "synthesized_transcription"]}
            samples.append((sample_id, transcript, metrics))
        return samples
    else:
        # Fall back to filelist
        raw_samples = load_filelist(experiment_dir, condition)
        samples = []
        for sample_id, transcript in raw_samples[:max_samples]:
            # Remove .wav extension if present
            if sample_id.endswith(".wav"):
                sample_id = sample_id[:-4]
            samples.append((sample_id, transcript, None))
        return samples


# ==============================================================================
# Page Generators
# ==============================================================================

def generate_index_page(
    config: Optional[Dict[str, Any]],
    conditions: List[str],
    output_dir: Path,
) -> str:
    """Generate the index.html page."""
    html_parts = [
        html_head("Romanian TTS Evaluation - Overview"),
        html_nav("index"),
        '<div class="container">',
        "<h1>Romanian Chatterbox TTS Evaluation</h1>",
    ]

    # Experiment info
    if config:
        timestamp = config.get("timestamp", "Unknown")
        stats = config.get("statistics", {})
        params = config.get("parameters", {})

        html_parts.append(f'<p class="subtitle">Experiment: {timestamp[:10] if len(timestamp) >= 10 else timestamp}</p>')

        html_parts.append("<h2>Experiment Parameters</h2>")
        html_parts.append('<div class="card">')
        html_parts.append('<div class="params">')

        # Display parameters
        param_items = [
            ("Speakers", ", ".join(s.upper() for s in params.get("speakers", []))),
            ("Exaggeration", ", ".join(str(e) for e in params.get("exaggerations", []))),
            ("CFG Weight", ", ".join(str(c) for c in params.get("cfg_weights", []))),
            ("Samples/Condition", str(params.get("samples_per_condition", "N/A"))),
            ("Seed", str(params.get("seed", "N/A"))),
        ]

        for label, value in param_items:
            html_parts.append(f"""
            <div class="param">
                <span class="param-label">{label}</span>
                <span class="param-value">{html.escape(value)}</span>
            </div>""")

        html_parts.append("</div>")  # .params
        html_parts.append("</div>")  # .card

        # Statistics
        total_samples = stats.get("total_samples", 0)
        n_conditions = stats.get("conditions", len(conditions))
        html_parts.append(f"<p>Total: {total_samples} samples across {n_conditions} conditions</p>")
    else:
        html_parts.append('<p class="subtitle">Experiment configuration not found</p>')

    # Condition list
    html_parts.append("<h2>Conditions</h2>")
    html_parts.append('<ul class="condition-list">')

    for condition in conditions:
        params = parse_condition_name(condition)
        html_parts.append(f"""
        <li>
            <a href="{condition}.html" class="condition-name">{condition}</a>
            <div class="condition-meta">
                Speaker: {params.get('speaker', 'N/A')} |
                Exaggeration: {params.get('exaggeration', 'N/A')} |
                CFG: {params.get('cfg_weight', 'N/A')}
            </div>
        </li>""")

    html_parts.append("</ul>")
    html_parts.append(html_footer())

    return "".join(html_parts)


def generate_results_page(
    results: Optional[List[Dict[str, str]]],
    conditions: List[str],
) -> str:
    """Generate the results.html page."""
    html_parts = [
        html_head("Romanian TTS Evaluation - Results"),
        html_nav("results"),
        '<div class="container">',
        "<h1>Evaluation Results</h1>",
        '<p class="subtitle">Aggregated metrics (mean +/- std) per condition</p>',
    ]

    if not results:
        html_parts.append("<p>No results available. Run the metrics pipeline first.</p>")
        html_parts.append(html_footer())
        return "".join(html_parts)

    # Define metric display info
    metrics_info = [
        ("CER", "CER", "Character Error Rate (lower is better)"),
        ("WER", "WER", "Word Error Rate (lower is better)"),
        ("UTMOS_Synthesized", "UTMOS", "Perceived quality score (higher is better)"),
        ("Log_F0_RMSE", "F0 RMSE", "Pitch tracking error (lower is better)"),
        ("MCD", "MCD", "Mel Cepstral Distortion (lower is better)"),
        ("Speaker_Cosine_Similarity", "Speaker Sim", "Voice similarity (higher is better)"),
    ]

    html_parts.append("<h2>Metrics Summary</h2>")
    html_parts.append("<table>")
    html_parts.append("<thead><tr>")
    html_parts.append("<th>Condition</th>")

    for _, short_name, description in metrics_info:
        html_parts.append(f'<th class="number" title="{html.escape(description)}">{short_name}</th>')

    html_parts.append("</tr></thead>")
    html_parts.append("<tbody>")

    for row in results:
        condition = row.get("condition", "Unknown")
        html_parts.append("<tr>")
        html_parts.append(f'<td><a href="{condition}.html">{condition}</a></td>')

        for metric_key, _, _ in metrics_info:
            mean_key = f"{metric_key}_mean"
            std_key = f"{metric_key}_std"

            mean_val = row.get(mean_key, "N/A")
            std_val = row.get(std_key, "N/A")

            try:
                mean_float = float(mean_val)
                std_float = float(std_val)
                display = f"{mean_float:.3f} +/- {std_float:.3f}"
            except (ValueError, TypeError):
                display = "N/A"

            html_parts.append(f'<td class="number">{display}</td>')

        html_parts.append("</tr>")

    html_parts.append("</tbody>")
    html_parts.append("</table>")

    # Legend
    html_parts.append("<h2>Metric Descriptions</h2>")
    html_parts.append('<div class="card">')
    html_parts.append("<dl>")

    for metric_key, short_name, description in metrics_info:
        html_parts.append(f"<dt><strong>{short_name}</strong> ({metric_key})</dt>")
        html_parts.append(f"<dd>{description}</dd>")

    html_parts.append("</dl>")
    html_parts.append("</div>")

    html_parts.append(html_footer())
    return "".join(html_parts)


def generate_condition_page(
    experiment_dir: Path,
    output_dir: Path,
    condition: str,
    max_samples: int,
    selected_samples: Optional[Dict[str, List[Dict]]] = None,
) -> Tuple[str, int]:
    """
    Generate a condition page with audio samples.

    Returns:
        Tuple of (html_content, number_of_samples_copied)
    """
    params = parse_condition_name(condition)
    samples = get_samples_for_condition(experiment_dir, condition, selected_samples, max_samples)

    html_parts = [
        html_head(f"Condition: {condition}"),
        html_nav(""),
        '<div class="container">',
        f"<h1>{condition}</h1>",
    ]

    # Parameters
    html_parts.append('<div class="card">')
    html_parts.append('<div class="params">')

    for label, key in [("Speaker", "speaker"), ("Exaggeration", "exaggeration"), ("CFG Weight", "cfg_weight")]:
        value = params.get(key, "N/A")
        html_parts.append(f"""
        <div class="param">
            <span class="param-label">{label}</span>
            <span class="param-value">{html.escape(str(value))}</span>
        </div>""")

    html_parts.append("</div>")
    html_parts.append("</div>")

    # Audio samples
    html_parts.append(f"<h2>Audio Samples ({len(samples)})</h2>")

    # Create audio output directory
    audio_out_dir = output_dir / "audio" / condition
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    samples_copied = 0
    condition_audio_dir = experiment_dir / condition

    for sample_id, transcript, metrics in samples:
        audio_src = condition_audio_dir / f"{sample_id}.wav"
        audio_dest = audio_out_dir / f"{sample_id}.wav"
        audio_href = f"audio/{condition}/{sample_id}.wav"

        # Copy audio file
        if audio_src.exists():
            shutil.copy2(audio_src, audio_dest)
            samples_copied += 1
            audio_available = True
        else:
            audio_available = False

        html_parts.append('<div class="sample">')
        html_parts.append('<div class="sample-header">')
        html_parts.append(f'<span class="sample-id">{html.escape(sample_id)}</span>')

        # Display metrics if available (from selected samples)
        if metrics:
            metrics_str = []
            if "UTMOS_Synthesized" in metrics:
                metrics_str.append(f"UTMOS: {metrics['UTMOS_Synthesized']:.2f}")
            if "CER" in metrics:
                metrics_str.append(f"CER: {metrics['CER']:.3f}")
            if "Speaker_Cosine_Similarity" in metrics:
                metrics_str.append(f"Spk: {metrics['Speaker_Cosine_Similarity']:.2f}")
            if metrics_str:
                html_parts.append(f'<span class="sample-metrics" style="color: var(--color-text-muted); font-size: 0.8rem;">{" | ".join(metrics_str)}</span>')

        html_parts.append("</div>")
        html_parts.append(f'<div class="sample-text">{html.escape(transcript)}</div>')

        if audio_available:
            html_parts.append(f'<audio controls preload="none" src="{audio_href}"></audio>')
        else:
            html_parts.append('<p style="color: var(--color-text-muted);">Audio file not found</p>')

        html_parts.append("</div>")

    # Navigation
    html_parts.append("<p><a href=\"index.html\">&larr; Back to Overview</a></p>")
    html_parts.append(html_footer())

    return "".join(html_parts), samples_copied


# ==============================================================================
# Main
# ==============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()

    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return 1

    print("=" * 60)
    print("GitHub Pages Gallery Generator")
    print("=" * 60)
    print(f"Experiment: {experiment_dir}")
    print(f"Output: {output_dir}")
    print(f"Max samples per condition: {args.max_samples}")

    # Load selected samples if provided
    selected_samples = None
    if args.selected_samples:
        selected_samples = load_selected_samples(args.selected_samples)
        if selected_samples:
            total_selected = sum(len(v) for v in selected_samples.values())
            print(f"Using selected samples: {total_selected} samples from {len(selected_samples)} conditions")
        else:
            print(f"Warning: Could not load selected samples from {args.selected_samples}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    config = load_experiment_config(experiment_dir)
    results = load_results_summary(experiment_dir)
    conditions = find_conditions(experiment_dir)

    if not conditions:
        print("\nWarning: No condition directories found")
        print("Expected directories matching pattern: {speaker}_exag{x}_cfg{y}")

    print(f"\nFound {len(conditions)} conditions")
    if config:
        print("Experiment config: loaded")
    else:
        print("Experiment config: not found (will continue without)")
    if results:
        print(f"Results summary: {len(results)} rows")
    else:
        print("Results summary: not found (will continue without)")

    # Write CSS
    css_path = output_dir / "style.css"
    with open(css_path, "w", encoding="utf-8") as f:
        f.write(ACADEMIC_CSS)
    print(f"\nGenerated: {css_path.name}")

    # Generate index page
    index_html = generate_index_page(config, conditions, output_dir)
    index_path = output_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"Generated: {index_path.name}")

    # Generate results page
    results_html = generate_results_page(results, conditions)
    results_path = output_dir / "results.html"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(results_html)
    print(f"Generated: {results_path.name}")

    # Generate condition pages
    total_audio_copied = 0
    for condition in conditions:
        condition_html, n_copied = generate_condition_page(
            experiment_dir, output_dir, condition, args.max_samples, selected_samples
        )
        condition_path = output_dir / f"{condition}.html"
        with open(condition_path, "w", encoding="utf-8") as f:
            f.write(condition_html)
        print(f"Generated: {condition_path.name} ({n_copied} audio files)")
        total_audio_copied += n_copied

    # Summary
    print("\n" + "=" * 60)
    print("Gallery Generation Complete")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - style.css")
    print(f"  - index.html")
    print(f"  - results.html")
    print(f"  - {len(conditions)} condition pages")
    print(f"  - {total_audio_copied} audio files")

    print(f"\nTo serve locally:")
    print(f"  cd {output_dir}")
    print(f"  python -m http.server 8000")
    print(f"  # Open http://localhost:8000")

    print(f"\nTo deploy to GitHub Pages:")
    print(f"  1. Push docs/ to your repository")
    print(f"  2. Settings > Pages > Source: Deploy from branch")
    print(f"  3. Select: main branch, /docs folder")

    return 0


if __name__ == "__main__":
    sys.exit(main())
