#!/usr/bin/env python3
"""
CLI script to score results and generate reports.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

from src.schemas import ExperimentConfig
from src.scoring import load_results_from_jsonl, compute_metrics
from src.report import generate_report, generate_summary_csv

app = typer.Typer(help="Score experiment results and generate reports.")
console = Console()


@app.command()
def main(
    run_dir: Path = typer.Option(
        ...,
        "--run_dir",
        "-r",
        help="Path to run directory containing results.jsonl.",
    ),
):
    """
    Score experiment results and generate report.md and metrics.json.
    """
    # Validate run directory
    if not run_dir.exists():
        console.print(f"[red]Error: Run directory not found: {run_dir}[/red]")
        raise typer.Exit(1)
    
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        console.print(f"[red]Error: results.jsonl not found in {run_dir}[/red]")
        raise typer.Exit(1)
    
    config_path = run_dir / "config.json"
    if not config_path.exists():
        console.print(f"[red]Error: config.json not found in {run_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Scoring Results from {run_dir}[/bold]")
    console.print()
    
    # Load config
    console.print("Loading configuration...")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config = ExperimentConfig.model_validate(config_data)
    console.print(f"  Model: {config.model}")
    console.print(f"  Prompt Variant: {config.prompt_variant}")
    console.print(f"  Seed: {config.seed}")
    
    # Load results
    console.print("Loading results...")
    results = load_results_from_jsonl(results_path)
    console.print(f"  Loaded {len(results)} case results.")
    
    # Compute metrics
    console.print("Computing metrics...")
    metrics = compute_metrics(results)
    
    # Display summary
    console.print()
    console.print("[bold]Summary Metrics[/bold]")
    console.print(f"  Overall Accuracy: {metrics.accuracy_overall:.1%}")
    console.print(f"  Tool Call Compliance: {metrics.tool_call_compliance:.1%}")
    console.print(f"  Unclear Rate: {metrics.unclear_rate:.1%}")
    console.print(f"  Successful Cases: {metrics.successful_cases} / {metrics.total_cases}")
    
    if metrics.injection_susceptibility is not None:
        console.print(f"  Injection Susceptibility: {metrics.injection_susceptibility:.1%}")
    
    console.print()
    console.print("[bold]Accuracy by Trap Type[/bold]")
    for trap_type, accuracy in sorted(metrics.accuracy_by_trap.items()):
        console.print(f"  {trap_type}: {accuracy:.1%}")
    
    # Save metrics.json
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        f.write(metrics.model_dump_json(indent=2))
    console.print()
    console.print(f"Saved metrics to: {metrics_path}")
    
    # Generate summary CSV
    csv_path = run_dir / "summary.csv"
    generate_summary_csv(results, csv_path)
    console.print(f"Saved summary to: {csv_path}")
    
    # Generate report
    report_path = run_dir / "report.md"
    generate_report(config, metrics, results, report_path)
    console.print(f"Saved report to: {report_path}")
    
    console.print()
    console.print("[bold green]Scoring complete![/bold green]")


if __name__ == "__main__":
    app()
