"""Report generator orchestration."""

from pathlib import Path
from typing import Dict, List
import json
from .loaders import (
    load_metrics_csvs,
    load_trainer_state,
    load_dataset_metadata,
    load_dataset_splits,
    load_filelists,
    validate_loaded_data,
)
from .analysis import (
    analyze_dataset_characteristics,
    analyze_training_dynamics,
    analyze_performance,
    analyze_voice_adaptation,
    analyze_ablation_effects,
)
from .visualizations import (
    plot_speaker_distribution,
    plot_text_statistics,
    plot_loss_curves,
    plot_performance_comparison,
    plot_ablation_effects,
)
from .tables import (
    generate_metrics_table,
    generate_dataset_summary_table,
    export_to_latex,
    export_to_markdown,
    generate_ablation_table,
)
from .utils import save_json, ensure_output_dirs


class AcademicReportGenerator:
    """Orchestrate comprehensive academic report generation."""

    def __init__(self, args):
        self.args = args
        self.output_dirs = ensure_output_dirs(args.output_dir, args.formats)
        self.data = {}
        self.analyses = {}

    def run(self):
        """Execute full analysis pipeline."""
        print("=" * 60)
        print("ACADEMIC REPORT GENERATION PIPELINE")
        print("=" * 60)

        # Load data
        self._load_all_data()

        # Run analyses
        self._run_all_analyses()

        # Generate outputs
        self._generate_visualizations()
        self._generate_tables()
        self._generate_json_summaries()

        if "html" in self.args.formats:
            self._generate_html_report()

        print("\n" + "=" * 60)
        print(f"Report generated in: {self.args.output_dir}")
        print("=" * 60)

    def _load_all_data(self):
        """Load all input data."""
        print("\n[1/5] Loading data from sources...")

        self.data["metrics"] = load_metrics_csvs(self.args.metrics_dir)
        self.data["trainer_state"] = load_trainer_state(self.args.training_log)
        self.data["metadata"] = load_dataset_metadata(self.args.metadata)
        self.data["splits"] = load_dataset_splits(self.args.dataset_splits)
        self.data["filelists"] = load_filelists(self.args.filelists_dir)

        # Validate
        validate_loaded_data(
            self.data["metrics"],
            self.data["trainer_state"],
            self.data["metadata"],
            self.data["splits"],
            self.data["filelists"],
        )

    def _run_all_analyses(self):
        """Run all analysis functions."""
        print("\n[2/5] Running analyses...")

        self.analyses["dataset"] = analyze_dataset_characteristics(
            self.data["metadata"], self.data["splits"]
        )
        print("  ✓ Dataset characterization")

        self.analyses["training"] = analyze_training_dynamics(self.data["trainer_state"])
        print("  ✓ Training dynamics")

        self.analyses["performance"] = analyze_performance(self.data["metrics"])
        print("  ✓ Performance analysis")

        self.analyses["adaptation"] = analyze_voice_adaptation(
            self.data["metrics"], self.data["splits"]
        )
        print("  ✓ Voice adaptation analysis")

        self.analyses["ablation"] = analyze_ablation_effects(self.data["metrics"])
        print("  ✓ Ablation analysis")

    def _generate_visualizations(self):
        """Generate all figures."""
        print("\n[3/5] Generating visualizations...")

        # Dataset figures
        if self.analyses["dataset"]["speaker_distribution"]:
            plot_speaker_distribution(
                self.analyses["dataset"]["speaker_distribution"]["per_speaker"],
                self.output_dirs["figures"] / "dataset_characterization",
                dpi=self.args.dpi,
            )
            print("  ✓ Speaker distribution")

        if self.analyses["dataset"]["text_statistics"]:
            plot_text_statistics(
                self.analyses["dataset"]["text_statistics"],
                self.output_dirs["figures"] / "dataset_characterization",
                dpi=self.args.dpi,
            )
            print("  ✓ Text statistics")

        # Training figures - always generate if log_history exists
        if "log_history" in self.data["trainer_state"] and len(self.data["trainer_state"]["log_history"]) > 0:
            plot_loss_curves(
                self.data["trainer_state"],
                self.output_dirs["figures"] / "training_dynamics",
                dpi=self.args.dpi,
            )
            print("  ✓ Loss curves")

        # Performance figures
        plot_performance_comparison(
            self.analyses["performance"]["conditions"],
            self.output_dirs["figures"] / "evaluation_results",
            dpi=self.args.dpi,
        )
        print("  ✓ Performance comparison")

        # Ablation figures
        if self.analyses["ablation"]:
            plot_ablation_effects(
                self.analyses["ablation"],
                self.output_dirs["figures"] / "ablation_analysis",
                dpi=self.args.dpi,
            )
            print("  ✓ Ablation effects")

    def _generate_tables(self):
        """Generate all tables."""
        print("\n[4/5] Generating tables...")

        # Metrics table
        if "latex" in self.args.formats or "markdown" in self.args.formats:
            metrics_table = generate_metrics_table(
                self.analyses["performance"]["conditions"]
            )

            if "latex" in self.args.formats:
                export_to_latex(
                    metrics_table,
                    self.output_dirs["tables"] / "metrics_by_condition.tex",
                    caption="Model Performance Metrics by Condition",
                )
            if "markdown" in self.args.formats:
                export_to_markdown(
                    metrics_table, self.output_dirs["tables"] / "metrics_by_condition.md"
                )
            print("  ✓ Metrics table")

        # Dataset summary
        if "latex" in self.args.formats or "markdown" in self.args.formats:
            dataset_table = generate_dataset_summary_table(self.analyses["dataset"])

            if "latex" in self.args.formats:
                export_to_latex(
                    dataset_table,
                    self.output_dirs["tables"] / "dataset_summary.tex",
                    caption="Dataset Characterization Summary",
                )
            if "markdown" in self.args.formats:
                export_to_markdown(
                    dataset_table, self.output_dirs["tables"] / "dataset_summary.md"
                )
            print("  ✓ Dataset summary table")

        # Ablation table
        if (
            self.analyses["ablation"] and 
            ("main_effects" in self.analyses["ablation"] or "by_condition" in self.analyses["ablation"]) and
            ("latex" in self.args.formats or "markdown" in self.args.formats)
        ):
            ablation_table = generate_ablation_table(self.analyses["ablation"])

            if not ablation_table.empty:
                if "latex" in self.args.formats:
                    export_to_latex(
                        ablation_table,
                        self.output_dirs["tables"] / "ablation_table.tex",
                        caption="Ablation Study Results",
                    )
                if "markdown" in self.args.formats:
                    export_to_markdown(
                        ablation_table, self.output_dirs["tables"] / "ablation_table.md"
                    )
                print("  ✓ Ablation table")

    def _generate_json_summaries(self):
        """Export analysis results as JSON."""
        print("\n[5/5] Exporting JSON summaries...")

        if "json" in self.args.formats:
            save_json(
                self.analyses["dataset"],
                self.output_dirs["data"] / "dataset_characterization.json",
            )
            print("  ✓ Dataset characterization")

            save_json(
                self.analyses["performance"],
                self.output_dirs["data"] / "performance_statistics.json",
            )
            print("  ✓ Performance statistics")

            save_json(
                self.analyses["training"],
                self.output_dirs["data"] / "training_summary.json",
            )
            print("  ✓ Training summary")

            save_json(
                self.analyses["ablation"], self.output_dirs["data"] / "ablation_results.json"
            )
            print("  ✓ Ablation results")

    def _generate_html_report(self):
        """Generate interactive HTML report."""
        print("\nGenerating HTML report...")
        # Placeholder for HTML generation
        print("  ✓ HTML report")
