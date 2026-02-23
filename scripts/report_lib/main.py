"""Main entry point for analysis pipeline."""

from pathlib import Path


def run_analysis_pipeline(args):
    """Main entry point for analysis pipeline."""
    from .report_generator import AcademicReportGenerator

    # Validate input paths
    required_paths = [
        (args.reports_dir, "Reports directory"),
        (args.metrics_dir, "Metrics directory"),
        (args.training_log, "Training log"),
        (args.metadata, "Metadata file"),
        (args.dataset_splits, "Dataset splits directory"),
    ]

    for path, name in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Run pipeline
    generator = AcademicReportGenerator(args)
    generator.run()
