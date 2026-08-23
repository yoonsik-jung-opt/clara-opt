"""CLARA command-line interface.

Pipeline: LP Parser → Solve Engine → Explainer → Output
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from clara.io.lp_parser import LPParseError, read_lp
from clara.explain.explainer import Explainer
from clara.explain.types import DetailLevel


@click.group()
@click.version_option(version="0.3.0", prog_name="clara")
def main():
    """CLARA — Classical LP Analysis for Reoptimization and Attribution.

    A white-box optimization solver that explains why solutions are optimal,
    when reoptimization is needed, and what changed.
    """
    pass


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--engine", type=click.Choice(["highs"]),
              default="highs", help="Solve engine (HiGHS).")
@click.option("--level", type=click.Choice(["brief", "detailed"]),
              default="detailed", help="Explanation detail level.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]),
              default="text", help="Output format.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write output to file.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress header.")
def explain(file, engine, level, fmt, output, quiet):
    """Solve an LP and explain the solution."""
    problem = _parse_file(file)
    state = _solve(problem, engine)

    if not state.is_optimal:
        click.secho(f"Problem is {state.status.name}.", fg="yellow", err=True)
        sys.exit(1)

    detail = DetailLevel.BRIEF if level == "brief" else DetailLevel.DETAILED
    explainer = Explainer()
    report = explainer.explain(state, level=detail, problem=problem)

    if fmt == "json":
        text = report.to_json()
    else:
        text = report.to_text(level=detail)
        if quiet:
            text = _strip_header(text)

    _write_output(text, output)


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--engine", type=click.Choice(["highs"]),
              default="highs")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]),
              default="text")
@click.option("--output", "-o", type=click.Path(), default=None)
def solve(file, engine, fmt, output):
    """Solve an LP without explanation (quick mode)."""
    problem = _parse_file(file)
    state = _solve(problem, engine)

    if fmt == "json":
        text = state.to_json()
    else:
        text = _format_solve_text(state)

    _write_output(text, output)


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def info(file):
    """Show problem statistics without solving."""
    problem = _parse_file(file)
    nonzeros = int((problem.A != 0).sum())
    lines = [
        f"Problem: {problem.name}",
        f"Variables: {problem.num_variables}",
        f"Constraints: {problem.num_constraints}",
        f"Nonzeros: {nonzeros}",
    ]
    click.echo("\n".join(lines))


@main.command()
@click.argument("old_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("new_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--engine", type=click.Choice(["highs"]),
              default="highs")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]),
              default="text")
@click.option("--output", "-o", type=click.Path(), default=None)
def diff(old_file, new_file, engine, fmt, output):
    """Compare two LP problems and explain what changed.

    Runs the full reoptimization pipeline:
    detect -> analyze -> reoptimize -> diff report.
    """
    from clara.reopt.detector import ChangeDetector
    from clara.reopt.analyzer import ImpactAnalyzer
    from clara.reopt.reoptimizer import Reoptimizer
    from clara.reopt.diff_report import DiffReporter

    old_problem = _parse_file(old_file)
    new_problem = _parse_file(new_file)

    old_state = _solve(old_problem, engine)
    if not old_state.is_optimal:
        click.secho(f"Old problem is {old_state.status.name}.", fg="red", err=True)
        sys.exit(1)

    change = ChangeDetector().detect(old_problem, new_problem)
    if change is None:
        click.echo("No parameter changes detected. Solutions are identical.")
        return

    decision = ImpactAnalyzer().analyze(old_state, change, old_problem)
    result = Reoptimizer().reoptimize(old_state, new_problem, change, decision, old_problem=old_problem)
    report = DiffReporter().diff(old_state, result.new_state, change, result)

    if fmt == "json":
        text = report.to_json()
    else:
        text = report.to_text()

    _write_output(text, output)


@main.command(name="what-if", hidden=True)
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def what_if(file):
    """[Phase 1.5] Interactive what-if analysis."""
    click.secho("what-if is not yet implemented (Phase 1.5).", fg="yellow")


@main.command(hidden=True)
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--params", type=click.Path(exists=True))
def watch(file, params):
    """[Phase 2] Stream parameter changes and reoptimize."""
    click.secho("watch is not yet implemented (Phase 2).", fg="yellow")


# ============================================================
# Helpers
# ============================================================

def _parse_file(filepath: str):
    """Parse LP or MPS file with user-friendly error handling."""
    p = Path(filepath)
    if p.suffix.lower() == ".mps":
        from clara.io.mps_parser import read_mps, MPSParseError
        try:
            return read_mps(filepath)
        except MPSParseError as e:
            click.secho(f"MPS parse error: {e}", fg="red", err=True)
            sys.exit(1)
    try:
        return read_lp(filepath)
    except LPParseError as e:
        click.secho(f"Parse error: {e}", fg="red", err=True)
        sys.exit(1)


def _solve(problem, engine_name: str):
    """Solve with the HiGHS backend (CLARA's single solver backend)."""
    if problem.has_integers:
        click.secho(
            "Integer problems are not supported: the internal B&B engine "
            "was removed in v0.3.0. Solve the LP relaxation instead "
            "(problem.as_lp()).",
            fg="red", err=True,
        )
        sys.exit(1)

    from clara.engine.highs_backend import HiGHSBackend
    return HiGHSBackend().solve(problem)


def _format_solve_text(state) -> str:
    """Minimal text output for `clara solve`."""
    lines = [f"Optimal value: {state.optimal_value:.4f}"]
    for v in state.variables:
        lines.append(f"  {v.name} = {v.value:.4f}")
    lines.append(
        f"Solved in {state.iteration_count} iterations, "
        f"{state.solve_time_seconds:.3f}s ({state.engine.name})"
    )
    return "\n".join(lines)


def _strip_header(text: str) -> str:
    """Remove header section (everything before double newline after header)."""
    # Header is between two ═ lines, skip past the second one
    lines = text.split("\n")
    sep_count = 0
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("\u2550"):
            sep_count += 1
            if sep_count == 2:
                start = i + 1
                break
    # Skip blank line after header
    while start < len(lines) and not lines[start].strip():
        start += 1
    return "\n".join(lines[start:])


def _write_output(text: str, output_path: Optional[str]) -> None:
    """Write to file or stdout."""
    if output_path:
        try:
            Path(output_path).write_text(text)
            click.secho(f"Written to {output_path}", fg="green", err=True)
        except IOError as e:
            click.secho(f"Write error: {e}", fg="red", err=True)
            sys.exit(1)
    else:
        click.echo(text)
