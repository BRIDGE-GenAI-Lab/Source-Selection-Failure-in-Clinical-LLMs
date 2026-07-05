"""
Report generation module.
"""

import json
from pathlib import Path
from datetime import datetime

from .schemas import CaseResult, ExperimentConfig, Metrics
from .scoring import analyze_failures, confidence_stats


def generate_report(
    config: ExperimentConfig,
    metrics: Metrics,
    results: list[CaseResult],
    output_path: Path,
) -> None:
    """
    Generate a markdown report.
    
    Args:
        config: Experiment configuration.
        metrics: Computed metrics.
        results: List of case results.
        output_path: Path to write report.md.
    """
    lines = []
    
    # Header
    lines.append("# Safe Guideline Tooling Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append("")
    lines.append("> ⚠️ **RESEARCH SIMULATION ONLY — NOT MEDICAL ADVICE**")
    lines.append("")
    
    # Configuration
    lines.append("## Experiment Configuration")
    lines.append("")
    lines.append(f"- **Model:** `{config.model}`")
    lines.append(f"- **Seed:** {config.seed}")
    lines.append(f"- **Prompt Variant:** {config.prompt_variant}")
    lines.append(f"- **Require Both Tools:** {config.require_both_tools}")
    lines.append(f"- **Total Cases:** {config.max_cases}")
    lines.append(f"- **Input File Hash:** `{config.input_file_hash[:16]}...`")
    if config.git_commit:
        lines.append(f"- **Git Commit:** `{config.git_commit}`")
    lines.append("")
    
    # Summary Metrics
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Overall Accuracy | {metrics.accuracy_overall:.1%} |")
    lines.append(f"| Tool Call Compliance | {metrics.tool_call_compliance:.1%} |")
    lines.append(f"| Unclear Rate | {metrics.unclear_rate:.1%} |")
    lines.append(f"| Successful Cases | {metrics.successful_cases} / {metrics.total_cases} |")
    lines.append(f"| Failed Cases | {metrics.failed_cases} |")
    if metrics.injection_susceptibility is not None:
        lines.append(f"| Injection Susceptibility | {metrics.injection_susceptibility:.1%} |")
    lines.append("")
    
    # Accuracy by Trap Type
    lines.append("## Accuracy by Sham Trap Type")
    lines.append("")
    lines.append("| Trap Type | Accuracy | n |")
    lines.append("|-----------|----------|---|")
    
    # Count cases per trap type
    trap_counts = {}
    for r in results:
        trap_counts[r.sham_trap_type] = trap_counts.get(r.sham_trap_type, 0) + 1
    
    for trap_type, accuracy in sorted(metrics.accuracy_by_trap.items()):
        count = trap_counts.get(trap_type, 0)
        lines.append(f"| {trap_type} | {accuracy:.1%} | {count} |")
    lines.append("")
    
    # Confidence Distribution
    lines.append("## Confidence Score Distribution")
    lines.append("")
    conf_stats = confidence_stats(results)
    if conf_stats["mean"] is not None:
        lines.append(f"- **Mean:** {conf_stats['mean']:.3f}")
        lines.append(f"- **Std Dev:** {conf_stats['std']:.3f}")
        lines.append(f"- **Min:** {conf_stats['min']:.3f}")
        lines.append(f"- **Max:** {conf_stats['max']:.3f}")
        lines.append(f"- **Median (Q50):** {conf_stats['q50']:.3f}")
        lines.append(f"- **Q25:** {conf_stats['q25']:.3f}")
        lines.append(f"- **Q75:** {conf_stats['q75']:.3f}")
    else:
        lines.append("No confidence data available.")
    lines.append("")
    
    # Top Failures
    lines.append("## Top Failure Cases")
    lines.append("")
    failures = analyze_failures(results, top_n=10)
    
    if failures:
        for i, f in enumerate(failures, 1):
            lines.append(f"### {i}. {f['case_id']}")
            lines.append("")
            lines.append(f"- **Trap Type:** {f['trap_type']}")
            lines.append(f"- **Selected:** {f['selected_tool']} (Correct: {f['correct_tool']})")
            if f['errors']:
                lines.append(f"- **Errors:** {', '.join(f['errors'][:2])}")
            if f['rationale_excerpt']:
                lines.append(f"- **Rationale Excerpt:** \"{f['rationale_excerpt']}...\"")
            if f['safety_notes']:
                lines.append(f"- **Safety Notes:** {f['safety_notes']}")
            lines.append("")
    else:
        lines.append("No failures to report.")
        lines.append("")
    
    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("1. **Synthetic Data:** All cases are synthetic and may not represent real clinical complexity.")
    lines.append("2. **Single Model:** Results are specific to the tested model and may not generalize.")
    lines.append("3. **Prompt Sensitivity:** Model behavior may vary with different prompt formulations.")
    lines.append("4. **Trap Detection Heuristics:** Injection susceptibility is measured via keyword matching, which may miss subtle influences.")
    lines.append("5. **No Clinical Validation:** This is a research simulation; outputs should never be used for actual medical decisions.")
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*This report was automatically generated by safe-guideline-tooling-eval.*")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def generate_summary_csv(results: list[CaseResult], output_path: Path) -> None:
    """
    Generate a summary CSV of all cases.
    
    Args:
        results: List of case results.
        output_path: Path to write summary.csv.
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "case_id",
            "trap_type",
            "mapping_a",
            "mapping_b",
            "selected_tool",
            "correct",
            "confidence",
            "tools_called",
            "has_errors",
            "latency_ms",
        ])
        
        # Data rows
        for r in results:
            writer.writerow([
                r.case_id,
                r.sham_trap_type,
                r.mapping.A,
                r.mapping.B,
                r.model_decision.selected_tool if r.model_decision else None,
                r.selected_tool_correct,
                r.model_decision.confidence if r.model_decision else None,
                ",".join(r.model_decision.tool_calls_made) if r.model_decision else "",
                len(r.errors) > 0,
                r.latency_ms,
            ])
