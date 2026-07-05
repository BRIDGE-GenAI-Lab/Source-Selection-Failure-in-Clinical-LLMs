#!/usr/bin/env python3
"""
CLI script to run the guideline tool trustworthiness experiment.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv

from src.load_cases import load_cases_limited
from src.runner import ExperimentRunner
from src.utils import timestamp_for_folder

# Load .env file if present
load_dotenv()

app = typer.Typer(help="Run the safe guideline tooling evaluation experiment.")
console = Console()


@app.command()
def main(
    input_file: Path = typer.Option(
        Path("data/100cases_with_tools.json"),
        "--input",
        "-i",
        help="Path to input JSON file with cases.",
    ),
    model: str = typer.Option(
        "gpt-5-nano",
        "--model",
        "-m",
        help="OpenAI model ID to use.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    outdir: Path = typer.Option(
        None,
        "--outdir",
        "-o",
        help="Output directory (auto-generated if not specified).",
    ),
    max_cases: int = typer.Option(
        None,
        "--max_cases",
        "-n",
        help="Maximum number of cases to process (all if not specified).",
    ),
    prompt_variant: str = typer.Option(
        "baseline",
        "--prompt_variant",
        "-p",
        help="Prompt variant: 'baseline' or 'defended'.",
    ),
    require_both_tools: bool = typer.Option(
        True,
        "--require_both_tools/--no-require_both_tools",
        help="Require the model to call both tools.",
    ),
):
    """
    Run the safe guideline tooling evaluation experiment.
    
    This evaluates whether an LLM can distinguish between trustworthy and
    sham/poisoned guideline tools.
    """
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set.[/red]")
        console.print("Set it with: export OPENAI_API_KEY='your-key-here'")
        console.print("Or create a .env file with: OPENAI_API_KEY=your-key-here")
        raise typer.Exit(1)
    
    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Validate prompt variant
    if prompt_variant not in ("baseline", "defended"):
        console.print(f"[red]Error: Invalid prompt_variant: {prompt_variant}[/red]")
        console.print("Must be 'baseline' or 'defended'.")
        raise typer.Exit(1)
    
    # Setup output directory
    if outdir is None:
        outdir = Path("results") / f"run_{timestamp_for_folder()}"
    
    console.print(f"[bold]Safe Guideline Tooling Evaluation[/bold]")
    console.print(f"Model: {model}")
    console.print(f"Seed: {seed}")
    console.print(f"Prompt Variant: {prompt_variant}")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {outdir}")
    console.print()
    
    # Load cases
    console.print("Loading cases...")
    try:
        cases = load_cases_limited(input_file, max_cases)
        console.print(f"Loaded {len(cases)} cases.")
    except Exception as e:
        console.print(f"[red]Error loading cases: {e}[/red]")
        raise typer.Exit(1)
    
    # Initialize runner
    console.print("Initializing experiment runner...")
    try:
        runner = ExperimentRunner(
            model=model,
            seed=seed,
            prompt_variant=prompt_variant,
            require_both_tools=require_both_tools,
            output_dir=outdir,
        )
    except Exception as e:
        console.print(f"[red]Error initializing runner: {e}[/red]")
        raise typer.Exit(1)
    
    # Run experiment with progress bar
    console.print()
    console.print("[bold]Running experiment...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing cases", total=len(cases))
        
        def update_progress(i, total, case_id):
            progress.update(task, completed=i, description=f"Processing {case_id}")
        
        results = runner.run_experiment(
            cases=cases,
            input_file=str(input_file),
            progress_callback=update_progress,
        )
        
        progress.update(task, completed=len(cases))
    
    # Summary
    console.print()
    console.print("[bold]Experiment Complete[/bold]")
    console.print(f"Total cases: {len(results)}")
    
    successful = sum(1 for r in results if r.model_decision and not r.errors)
    failed = len(results) - successful
    correct = sum(1 for r in results if r.selected_tool_correct is True)
    
    console.print(f"Successful: {successful}")
    console.print(f"Failed: {failed}")
    console.print(f"Correct selections: {correct}")
    
    if successful > 0:
        accuracy = correct / successful
        console.print(f"[bold]Accuracy: {accuracy:.1%}[/bold]")
    
    console.print()
    console.print(f"Results saved to: {outdir}")
    console.print(f"Run scoring with: python scripts/score_and_report.py --run_dir {outdir}")


if __name__ == "__main__":
    app()
