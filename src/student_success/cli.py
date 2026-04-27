"""Small command-line interface for the project scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

from student_success.data.loaders import validate_raw_layout
from student_success.pipelines.full import build_report, run_all, run_experiments, run_simulation
from student_success.pipelines.experiment import run_experiment
from student_success.settings import load_config


DEFAULT_CONFIG = Path("configs/default.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="student-success",
        description="Student success project scaffold utilities.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to a TOML config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("print-config", help="Print the loaded config.")
    subparsers.add_parser("validate-layout", help="Check whether raw data files exist.")
    subparsers.add_parser("train", help="Run the baseline experiment pipeline.")
    subparsers.add_parser("run-experiments", help="Run all model and ablation experiments.")
    subparsers.add_parser("simulate", help="Generate scenario and intervention artifacts.")
    subparsers.add_parser("build-report", help="Build the final Markdown report summary.")
    subparsers.add_parser("run-all", help="Run experiments, simulation, and report generation.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "print-config":
        print(pformat(config))
        return

    if args.command == "validate-layout":
        status = validate_raw_layout(config.dataset.raw_dir)
        for filename, exists in status.items():
            print(f"{filename}: {'found' if exists else 'missing'}")
        return

    if args.command == "train":
        results = run_experiment(args.config)
        print(results.to_string(index=False))
        return

    if args.command == "run-experiments":
        paths = run_experiments(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    if args.command == "simulate":
        paths = run_simulation(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    if args.command == "build-report":
        path = build_report(args.config)
        print(f"final_results_summary: {path}")
        return

    if args.command == "run-all":
        paths = run_all(args.config)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
