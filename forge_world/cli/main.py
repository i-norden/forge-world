"""CLI entry point for forge-world.

Commands:
    forge bench          Run benchmark, print report
    forge lock           Lock current outputs as baseline
    forge check          Run benchmark, check for regressions
    forge analyze        Print structured AI agent context
    forge diff           Compact delta between current and baseline
    forge cache clear    Manage disk cache
    forge evolve         Autonomous bench-modify-bench loop

Module discovery (in order):
    1. --module / -m flag
    2. forge-world.toml [forge-world].module in cwd
    3. Error
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _load_config_toml() -> dict[str, Any]:
    """Load forge-world.toml from cwd if it exists."""
    toml_path = Path.cwd() / "forge-world.toml"
    if not toml_path.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def _resolve_module(module_flag: str | None) -> str:
    """Resolve the module to load: flag > forge-world.toml > error."""
    if module_flag:
        return module_flag
    config = _load_config_toml()
    fw_section = config.get("forge-world", {})
    if "module" in fw_section:
        return fw_section["module"]
    console.print(
        "[red]No module specified. Use --module / -m or create a forge-world.toml with:\n"
        "[forge-world]\n"
        'module = "your_project.forge_world"[/red]'
    )
    sys.exit(1)


def _load_module(module_name: str):
    """Dynamically import a module by dotted path."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        console.print(f"[red]Failed to load module '{module_name}': {exc}[/red]")
        sys.exit(1)


def _build_runner(mod, *, quiet: bool = False, analysis_cache=None):
    """Build a BenchmarkRunner from a module exporting factory functions."""
    from forge_world.core.runner import BenchmarkRunner

    pipeline = mod.create_pipeline()
    aggregator = mod.create_aggregator()
    dataset = mod.create_dataset()
    rules = mod.create_rules()

    if quiet:
        return BenchmarkRunner(
            pipeline=pipeline, aggregator=aggregator, dataset=dataset, rules=rules,
            analysis_cache=analysis_cache,
        )

    def on_item_start(item, idx, total, cached):
        status = "[dim]cached[/dim]" if cached else f"[cyan]{item.category}[/cyan]"
        console.print(
            f"  [{idx + 1}/{total}] {item.id} ({status})",
            highlight=False,
        )

    def on_item_complete(result):
        mark = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(
            f"         {mark}  risk={result.risk_level}  "
            f"methods={','.join(result.methods_flagged) or 'none'}",
            highlight=False,
        )

    def on_seed_start(seed, kind, seed_idx, total_seeds):
        console.print()
        console.rule(f"Seed {seed} ({kind}) [{seed_idx + 1}/{total_seeds}]")

    def on_seed_complete(seed, kind, seed_idx, total_seeds, report):
        console.print(
            f"  [bold]Seed {seed} done:[/bold] "
            f"{report.pass_count}/{report.total_count} passed, "
            f"sensitivity={report.sensitivity:.4f}, fpr={report.fpr:.4f}"
        )

    return BenchmarkRunner(
        pipeline=pipeline,
        aggregator=aggregator,
        dataset=dataset,
        rules=rules,
        on_item_start=on_item_start,
        on_item_complete=on_item_complete,
        on_seed_start=on_seed_start,
        on_seed_complete=on_seed_complete,
        analysis_cache=analysis_cache,
    )


def _parse_seed_strategy(seeds: str | None, exploration_seeds: int):
    """Parse --seeds flag into a SeedStrategy."""
    from forge_world.core.runner import SeedStrategy

    if seeds:
        stable = [int(s.strip()) for s in seeds.split(",")]
    else:
        stable = [42]

    return SeedStrategy(stable_seeds=stable, n_exploration_seeds=exploration_seeds)


def _resolve_sample_size(cli_value: int | None) -> int | None:
    """Resolve sample size: CLI flag > forge-world.toml > None."""
    if cli_value is not None:
        return cli_value
    config = _load_config_toml()
    fw_section = config.get("forge-world", {})
    if "sample_size" in fw_section:
        return int(fw_section["sample_size"])
    return None


def _build_analysis_cache(cache_dir: str, no_cache: bool):
    """Build AnalysisCache if caching is enabled."""
    if no_cache:
        return None
    from forge_world.core.cache import AnalysisCache
    return AnalysisCache(cache_dir)


def _print_report_table(report):
    """Print a rich table summary of a BenchmarkReport."""
    table = Table(title=f"Benchmark Results ({report.pass_count}/{report.total_count})")
    table.add_column("Category", style="bold")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Rate", justify="right")

    for cat in report.category_metrics:
        rate = f"{cat.pass_rate:.0%}"
        style = "green" if cat.pass_rate == 1.0 else "yellow" if cat.pass_rate >= 0.5 else "red"
        table.add_row(cat.category, str(cat.passed), str(cat.total), f"[{style}]{rate}[/{style}]")

    console.print()
    console.print(table)
    console.print()

    console.print(f"[bold]Sensitivity:[/bold] {report.sensitivity:.4f}")
    console.print(f"[bold]Specificity:[/bold] {report.specificity:.4f}")
    console.print(f"[bold]FPR:[/bold] {report.fpr:.4f}")
    console.print(f"[bold]F1:[/bold] {report.f1:.4f}")

    if report.fpr > 0:
        console.print(
            f"[red bold]WARNING: {report.confusion_matrix.false_positives} "
            f"false positive(s)![/red bold]"
        )
    console.print()


def _print_multi_report_table(multi_report):
    """Print a rich summary of a MultiBenchmarkReport."""
    am = multi_report.aggregate_metrics

    console.print()
    console.print(f"[bold]Multi-seed benchmark ({len(multi_report.seed_reports)} seeds)[/bold]")
    console.print()

    # Per-seed summary table
    table = Table(title="Per-Seed Results")
    table.add_column("Seed", style="bold")
    table.add_column("Kind")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Sensitivity", justify="right")
    table.add_column("FPR", justify="right")
    table.add_column("F1", justify="right")

    for sr in multi_report.seed_reports:
        r = sr.report
        fpr_style = "red" if r.fpr > 0 else "green"
        table.add_row(
            str(sr.seed),
            sr.seed_kind,
            f"{r.pass_count}/{r.total_count}",
            f"{r.sensitivity:.4f}",
            f"[{fpr_style}]{r.fpr:.4f}[/{fpr_style}]",
            f"{r.f1:.4f}",
        )

    console.print(table)
    console.print()

    # Aggregate summary
    console.print(f"[bold]Mean pass rate:[/bold] {am.mean_pass_rate:.1%}")
    console.print(f"[bold]Range:[/bold] {am.min_pass_rate:.1%} - {am.max_pass_rate:.1%}")
    console.print(f"[bold]Mean sensitivity:[/bold] {am.mean_sensitivity:.4f}")
    console.print(f"[bold]Worst-case FPR:[/bold] {am.worst_case_fpr:.4f}")
    console.print(f"[bold]Mean F1:[/bold] {am.mean_f1:.4f}")

    if am.worst_case_fpr > 0:
        console.print("[red bold]WARNING: False positives detected on at least one seed![/red bold]")

    # Unstable items
    unstable = [
        (iid, stab) for iid, stab in am.item_stability.items()
        if 0 < stab < 1
    ]
    if unstable:
        console.print()
        console.print(f"[yellow]{len(unstable)} unstable item(s) across seeds:[/yellow]")
        for iid, stab in sorted(unstable, key=lambda x: x[1]):
            console.print(f"  - {iid}: passes {stab:.0%} of seeds")

    console.print()


def _print_failures(report):
    """Print failing items."""
    failures = [r for r in report.item_results if not r.passed]
    if not failures:
        console.print("[green]No failures![/green]")
        return

    table = Table(title=f"Failures ({len(failures)} items)")
    table.add_column("Item", style="bold")
    table.add_column("Category")
    table.add_column("Expected")
    table.add_column("Risk")
    table.add_column("Confidence", justify="right")
    table.add_column("Methods")

    for r in failures:
        table.add_row(
            r.item_id,
            r.category,
            r.expected_label,
            r.risk_level,
            f"{r.confidence:.2f}",
            ", ".join(r.methods_flagged),
        )

    console.print(table)
    console.print()


def _load_diagnostics(mod, report):
    """Attempt to load diagnostics from the module if create_diagnostics() exists."""
    if not hasattr(mod, "create_diagnostics"):
        return None
    try:
        diagnostics_impl = mod.create_diagnostics()
        # Get failing items from the report
        if hasattr(report, "stable_reports") and report.stable_reports:
            failing = [r for r in report.stable_reports[0].report.item_results if not r.passed]
        elif hasattr(report, "item_results"):
            failing = [r for r in report.item_results if not r.passed]
        else:
            return None
        clusters = diagnostics_impl.diagnose(failing)
        return [c.to_dict() for c in clusters]
    except Exception:
        return None


# --- CLI group and commands ---


@click.group()
@click.option("--module", "-m", default=None, help="Module exporting pipeline factories (e.g. snoopy.forge_world)")
@click.option("--snapshots-dir", default=".forge-world/snapshots", help="Directory for snapshots")
@click.option("--cache-dir", default=".forge-world/cache", help="Directory for analysis cache")
@click.option("--no-cache", is_flag=True, help="Disable disk analysis cache")
@click.option("--quiet", "-q", is_flag=True, help="Suppress per-item progress output (for agent/CI use)")
@click.pass_context
def cli(ctx, module: str | None, snapshots_dir: str, cache_dir: str, no_cache: bool, quiet: bool):
    """forge-world: benchmark-modify-benchmark harness."""
    ctx.ensure_object(dict)
    ctx.obj["module_name"] = module  # Resolved lazily per command
    ctx.obj["snapshots_dir"] = snapshots_dir
    ctx.obj["cache_dir"] = cache_dir
    ctx.obj["no_cache"] = no_cache
    ctx.obj["quiet"] = quiet


@cli.command()
@click.option("--json-output", "json_out", is_flag=True, help="Machine-readable JSON output")
@click.option("--show-failures", is_flag=True, help="Show detailed failure info")
@click.option("--seeds", default=None, help="Stable seeds, comma-separated (e.g. 42,137,256)")
@click.option("--exploration-seeds", default=1, type=int, help="Number of exploration seeds")
@click.option("--single-seed", default=None, type=int, help="Legacy single-seed mode")
@click.option("--sample-size", "-M", default=None, type=int, help="Number of seeded-random items per seed")
@click.option("--tier", "-t", default=None, help="Run only items in this tier (e.g. smoke, standard, full)")
@click.option("--only-failures", is_flag=True, help="Re-run only items that failed in baseline")
@click.option("--then-full", is_flag=True, help="After --only-failures, also run the full suite")
@click.option("--failure-baseline", default="baseline", help="Snapshot to read failures from (default: baseline)")
@click.pass_context
def bench(
    ctx,
    json_out: bool,
    show_failures: bool,
    seeds: str | None,
    exploration_seeds: int,
    single_seed: int | None,
    sample_size: int | None,
    tier: str | None,
    only_failures: bool,
    then_full: bool,
    failure_baseline: str,
):
    """Run benchmark, print report."""
    quiet = ctx.obj["quiet"] or json_out
    module_name = _resolve_module(ctx.obj["module_name"])
    mod = _load_module(module_name)
    analysis_cache = _build_analysis_cache(ctx.obj["cache_dir"], ctx.obj["no_cache"])
    runner = _build_runner(mod, quiet=quiet, analysis_cache=analysis_cache)
    sample_size = _resolve_sample_size(sample_size)

    # Resolve item_filter for --only-failures
    item_filter: set[str] | None = None
    if only_failures:
        from forge_world.core.snapshots import SnapshotManager
        sm = SnapshotManager(ctx.obj["snapshots_dir"])
        try:
            item_filter = sm.get_failed_item_ids(failure_baseline)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        if not quiet:
            console.print(f"[bold]Re-running {len(item_filter)} failed items from '{failure_baseline}'[/bold]")

    if single_seed is not None:
        # Legacy single-seed mode
        if not quiet:
            console.print(f"[bold]Running single-seed benchmark (seed={single_seed})...[/bold]")
        report = runner.run(
            seed=single_seed, sample_size=sample_size, tier=tier, item_filter=item_filter,
        )
        if json_out:
            click.echo(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            _print_report_table(report)
            if show_failures:
                _print_failures(report)

        # --then-full: re-run without filter (benefits from cache populated above)
        if then_full and item_filter is not None:
            if not quiet:
                console.print("[bold]Running full suite...[/bold]")
            full_report = runner.run(seed=single_seed, sample_size=sample_size, tier=tier)
            if json_out:
                click.echo(json.dumps(full_report.to_dict(), indent=2, default=str))
            else:
                _print_report_table(full_report)
    else:
        # Multi-seed mode (default)
        strategy = _parse_seed_strategy(seeds, exploration_seeds)
        if not quiet:
            console.print(
                f"[bold]Running multi-seed benchmark "
                f"({len(strategy.stable_seeds)} stable + {strategy.n_exploration_seeds} exploration)...[/bold]"
            )
        multi_report = runner.run_multi(
            strategy, sample_size=sample_size, tier=tier, item_filter=item_filter,
        )
        if json_out:
            click.echo(json.dumps(multi_report.to_dict(), indent=2, default=str))
        else:
            _print_multi_report_table(multi_report)
            if show_failures and multi_report.stable_reports:
                _print_failures(multi_report.stable_reports[0].report)

        if then_full and item_filter is not None:
            if not quiet:
                console.print("[bold]Running full suite...[/bold]")
            full_multi = runner.run_multi(strategy, sample_size=sample_size, tier=tier)
            if json_out:
                click.echo(json.dumps(full_multi.to_dict(), indent=2, default=str))
            else:
                _print_multi_report_table(full_multi)

    if analysis_cache is not None and not quiet:
        stats = analysis_cache.stats
        if stats["hits"] > 0:
            console.print(
                f"[dim]Cache: {stats['hits']} hits, {stats['misses']} misses[/dim]"
            )


@cli.command()
@click.option("--name", default="baseline", help="Snapshot name")
@click.option("--seeds", default=None, help="Stable seeds, comma-separated")
@click.option("--exploration-seeds", default=1, type=int, help="Number of exploration seeds")
@click.option("--single-seed", default=None, type=int, help="Legacy single-seed mode")
@click.option("--sample-size", "-M", default=None, type=int, help="Number of seeded-random items per seed")
@click.option("--tier", "-t", default=None, help="Run only items in this tier")
@click.pass_context
def lock(
    ctx,
    name: str,
    seeds: str | None,
    exploration_seeds: int,
    single_seed: int | None,
    sample_size: int | None,
    tier: str | None,
):
    """Lock current outputs as baseline snapshot."""
    from forge_world.core.snapshots import SnapshotManager

    quiet = ctx.obj["quiet"]
    module_name = _resolve_module(ctx.obj["module_name"])
    mod = _load_module(module_name)
    analysis_cache = _build_analysis_cache(ctx.obj["cache_dir"], ctx.obj["no_cache"])
    runner = _build_runner(mod, quiet=quiet, analysis_cache=analysis_cache)
    sample_size = _resolve_sample_size(sample_size)

    sm = SnapshotManager(ctx.obj["snapshots_dir"])

    if single_seed is not None:
        if not quiet:
            console.print(f"[bold]Running single-seed benchmark (seed={single_seed})...[/bold]")
        report = runner.run(seed=single_seed, sample_size=sample_size, tier=tier)
        if not quiet:
            _print_report_table(report)
        snapshot = sm.lock(report, name=name)
        console.print(
            f"[green]Locked snapshot '{name}' at {snapshot.timestamp} "
            f"({report.pass_count}/{report.total_count})[/green]"
        )
    else:
        strategy = _parse_seed_strategy(seeds, exploration_seeds)
        if not quiet:
            console.print(
                f"[bold]Running multi-seed benchmark "
                f"({len(strategy.stable_seeds)} stable + {strategy.n_exploration_seeds} exploration)...[/bold]"
            )
        multi_report = runner.run_multi(strategy, sample_size=sample_size, tier=tier)
        if not quiet:
            _print_multi_report_table(multi_report)
        snapshot = sm.lock(multi_report, name=name)
        console.print(
            f"[green]Locked multi-seed snapshot '{name}' at {snapshot.timestamp}[/green]"
        )


@cli.command()
@click.option("--baseline", default="baseline", help="Baseline snapshot name")
@click.option("--json-output", "json_out", is_flag=True, help="Machine-readable JSON output")
@click.option("--seeds", default=None, help="Stable seeds, comma-separated")
@click.option("--exploration-seeds", default=1, type=int, help="Number of exploration seeds")
@click.option("--single-seed", default=None, type=int, help="Legacy single-seed mode")
@click.option("--sample-size", "-M", default=None, type=int, help="Number of seeded-random items per seed")
@click.option("--tier", "-t", default=None, help="Run only items in this tier")
@click.option("--only-failures", is_flag=True, help="Re-run only items that failed in baseline")
@click.option("--failure-baseline", default="baseline", help="Snapshot to read failures from")
@click.pass_context
def check(
    ctx,
    baseline: str,
    json_out: bool,
    seeds: str | None,
    exploration_seeds: int,
    single_seed: int | None,
    sample_size: int | None,
    tier: str | None,
    only_failures: bool,
    failure_baseline: str,
):
    """Run benchmark, check for regressions against baseline."""
    from forge_world.core.snapshots import SnapshotManager

    quiet = ctx.obj["quiet"] or json_out
    module_name = _resolve_module(ctx.obj["module_name"])
    mod = _load_module(module_name)
    analysis_cache = _build_analysis_cache(ctx.obj["cache_dir"], ctx.obj["no_cache"])
    runner = _build_runner(mod, quiet=quiet, analysis_cache=analysis_cache)
    sample_size = _resolve_sample_size(sample_size)

    sm = SnapshotManager(ctx.obj["snapshots_dir"])

    # Resolve item_filter for --only-failures
    item_filter: set[str] | None = None
    if only_failures:
        try:
            item_filter = sm.get_failed_item_ids(failure_baseline)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)

    if single_seed is not None:
        if not quiet:
            console.print(f"[bold]Running single-seed benchmark (seed={single_seed})...[/bold]")
        report = runner.run(
            seed=single_seed, sample_size=sample_size, tier=tier, item_filter=item_filter,
        )
    else:
        strategy = _parse_seed_strategy(seeds, exploration_seeds)
        if not quiet:
            console.print(
                f"[bold]Running multi-seed benchmark "
                f"({len(strategy.stable_seeds)} stable + {strategy.n_exploration_seeds} exploration)...[/bold]"
            )
        report = runner.run_multi(
            strategy, sample_size=sample_size, tier=tier, item_filter=item_filter,
        )

    try:
        regression = sm.check(report, baseline_name=baseline)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if json_out:
        summary = report.summary() if hasattr(report, "summary") else {}
        output = {
            "report": summary,
            "regression": regression.to_dict(),
        }
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        from forge_world.core.runner import MultiBenchmarkReport

        if isinstance(report, MultiBenchmarkReport):
            _print_multi_report_table(report)
        else:
            _print_report_table(report)

        console.print(f"[bold]Baseline:[/bold] {regression.baseline_pass_rate}")
        console.print(f"[bold]Current:[/bold] {regression.current_pass_rate}")
        console.print()

        if regression.has_new_false_positives:
            console.print(
                f"[red bold]CRITICAL: {regression.new_false_positives} "
                f"new false positive(s)![/red bold]"
            )

        if regression.has_exploration_regression:
            details = regression.exploration_details
            console.print(
                f"[red bold]EXPLORATION REGRESSION: "
                f"pass_rate={details.get('current_exploration_pass_rate', '?')} "
                f"< threshold={details.get('threshold', '?')}[/red bold]"
            )

        if regression.has_regressions:
            console.print(
                f"[red]{regression.items_regressed} item(s) regressed[/red]"
            )
            for r in regression.regressions:
                seed_info = f" [seed={r.seed}]" if r.seed is not None else ""
                console.print(
                    f"  - {r.item_id} ({r.category}): "
                    f"{r.baseline_risk} -> {r.current_risk}{seed_info}"
                )
        else:
            console.print("[green]No regressions![/green]")

        if regression.items_improved > 0:
            console.print(
                f"[green]{regression.items_improved} item(s) improved[/green]"
            )

        if regression.unstable_items:
            console.print()
            console.print(
                f"[yellow]{len(regression.unstable_items)} unstable item(s) across seeds[/yellow]"
            )

    if regression.has_new_false_positives:
        sys.exit(1)


@cli.command()
@click.option("--baseline", default=None, help="Compare against baseline snapshot")
@click.option("--json-output", "json_out", is_flag=True, help="Machine-readable JSON output")
@click.option("--seeds", default=None, help="Stable seeds, comma-separated")
@click.option("--exploration-seeds", default=1, type=int, help="Number of exploration seeds")
@click.option("--single-seed", default=None, type=int, help="Legacy single-seed mode")
@click.option("--sample-size", "-M", default=None, type=int, help="Number of seeded-random items per seed")
@click.option("--tier", "-t", default=None, help="Run only items in this tier")
@click.pass_context
def analyze(
    ctx,
    baseline: str | None,
    json_out: bool,
    seeds: str | None,
    exploration_seeds: int,
    single_seed: int | None,
    sample_size: int | None,
    tier: str | None,
):
    """Print structured AI agent context."""
    from forge_world.core.agent_interface import build_evolution_context
    from forge_world.core.snapshots import SnapshotManager

    quiet = ctx.obj["quiet"] or json_out
    module_name = _resolve_module(ctx.obj["module_name"])
    mod = _load_module(module_name)
    analysis_cache = _build_analysis_cache(ctx.obj["cache_dir"], ctx.obj["no_cache"])
    runner = _build_runner(mod, quiet=quiet, analysis_cache=analysis_cache)
    sample_size = _resolve_sample_size(sample_size)

    if single_seed is not None:
        if not quiet:
            console.print(f"[bold]Running single-seed benchmark (seed={single_seed})...[/bold]")
        report = runner.run(seed=single_seed, sample_size=sample_size, tier=tier)
    else:
        strategy = _parse_seed_strategy(seeds, exploration_seeds)
        if not quiet:
            console.print(
                f"[bold]Running multi-seed benchmark "
                f"({len(strategy.stable_seeds)} stable + {strategy.n_exploration_seeds} exploration)...[/bold]"
            )
        report = runner.run_multi(strategy, sample_size=sample_size, tier=tier)

    regression = None
    if baseline:
        sm = SnapshotManager(ctx.obj["snapshots_dir"])
        try:
            regression = sm.check(report, baseline_name=baseline)
        except FileNotFoundError:
            console.print(
                f"[yellow]Baseline '{baseline}' not found, skipping regression check[/yellow]"
            )

    config_schema = {}
    try:
        config_schema = runner.pipeline.get_config_schema()
    except Exception:
        pass

    # Load diagnostics if available
    diagnostics_data = _load_diagnostics(mod, report)

    context = build_evolution_context(
        report=report,
        regression=regression,
        pipeline_config_schema=config_schema,
        hard_constraints=[{"metric": "fpr", "op": "<=", "value": 0}],
        optimization_target={"metric": "sensitivity", "direction": "max"},
        sample_size=sample_size,
        diagnostics=diagnostics_data,
    )

    if json_out:
        click.echo(json.dumps(context.to_dict(), indent=2, default=str))
    else:
        click.echo(context.to_prompt_context())


@cli.command("diff")
@click.option("--baseline", default="baseline", help="Baseline snapshot name")
@click.option("--json-output", "json_out", is_flag=True, help="Machine-readable JSON output")
@click.option("--seeds", default=None, help="Stable seeds, comma-separated")
@click.option("--exploration-seeds", default=1, type=int, help="Number of exploration seeds")
@click.option("--single-seed", default=None, type=int, help="Legacy single-seed mode")
@click.option("--sample-size", "-M", default=None, type=int, help="Number of seeded-random items per seed")
@click.option("--tier", "-t", default=None, help="Run only items in this tier")
@click.pass_context
def diff_cmd(
    ctx,
    baseline: str,
    json_out: bool,
    seeds: str | None,
    exploration_seeds: int,
    single_seed: int | None,
    sample_size: int | None,
    tier: str | None,
):
    """Compact delta between current run and baseline."""
    from forge_world.core.diff import compute_diff
    from forge_world.core.snapshots import SnapshotManager

    quiet = ctx.obj["quiet"] or json_out
    module_name = _resolve_module(ctx.obj["module_name"])
    mod = _load_module(module_name)
    analysis_cache = _build_analysis_cache(ctx.obj["cache_dir"], ctx.obj["no_cache"])
    runner = _build_runner(mod, quiet=quiet, analysis_cache=analysis_cache)
    sample_size = _resolve_sample_size(sample_size)

    sm = SnapshotManager(ctx.obj["snapshots_dir"])

    if single_seed is not None:
        if not quiet:
            console.print(f"[bold]Running single-seed benchmark (seed={single_seed})...[/bold]")
        report = runner.run(seed=single_seed, sample_size=sample_size, tier=tier)
    else:
        strategy = _parse_seed_strategy(seeds, exploration_seeds)
        if not quiet:
            console.print(
                f"[bold]Running multi-seed benchmark "
                f"({len(strategy.stable_seeds)} stable + {strategy.n_exploration_seeds} exploration)...[/bold]"
            )
        report = runner.run_multi(strategy, sample_size=sample_size, tier=tier)

    snapshot = sm.load(baseline)
    if snapshot is None:
        console.print(f"[red]Baseline '{baseline}' not found. Run 'forge lock' first.[/red]")
        sys.exit(1)

    try:
        regression = sm.check(report, baseline_name=baseline)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    diff_report = compute_diff(report, snapshot, regression)

    if json_out:
        click.echo(json.dumps(diff_report.to_dict(), indent=2, default=str))
    else:
        click.echo(diff_report.to_markdown())


@cli.group("cache")
def cache_group():
    """Manage analysis disk cache."""
    pass


@cache_group.command("clear")
@click.option("--config-hash", default=None, help="Only clear entries for this config hash")
@click.pass_context
def cache_clear(ctx, config_hash: str | None):
    """Clear the analysis cache."""
    from forge_world.core.cache import AnalysisCache

    cache_dir = ctx.obj["cache_dir"]
    cache = AnalysisCache(cache_dir)
    count = cache.clear(config_hash)
    console.print(f"[green]Cleared {count} cached entries.[/green]")


@cli.command()
@click.option("--agent-command", default=None, help="Command to invoke the agent (use {context_file} placeholder)")
@click.option("--max-iterations", default=None, type=int, help="Maximum evolution iterations")
@click.option("--patience", default=3, type=int, help="Stop after N rounds without improvement")
@click.option("--baseline", default="baseline", help="Baseline snapshot name")
@click.option("--single-seed", default=None, type=int, help="Single-seed mode")
@click.option("--seeds", default=None, help="Stable seeds, comma-separated")
@click.option("--exploration-seeds", default=1, type=int, help="Number of exploration seeds")
@click.option("--sample-size", "-M", default=None, type=int, help="Number of seeded-random items per seed")
@click.option("--tier", "-t", default=None, help="Run only items in this tier")
@click.option("--json-output", "json_out", is_flag=True, help="Machine-readable JSON output")
@click.pass_context
def evolve(
    ctx,
    agent_command: str | None,
    max_iterations: int | None,
    patience: int,
    baseline: str,
    single_seed: int | None,
    seeds: str | None,
    exploration_seeds: int,
    sample_size: int | None,
    tier: str | None,
    json_out: bool,
):
    """Autonomous bench-modify-bench evolution loop."""
    from forge_world.core.evolve import EvolutionConfig, EvolutionLoop
    from forge_world.core.snapshots import SnapshotManager

    module_name = _resolve_module(ctx.obj["module_name"])
    mod = _load_module(module_name)
    analysis_cache = _build_analysis_cache(ctx.obj["cache_dir"], ctx.obj["no_cache"])
    runner = _build_runner(mod, quiet=True, analysis_cache=analysis_cache)
    sample_size = _resolve_sample_size(sample_size)

    sm = SnapshotManager(ctx.obj["snapshots_dir"])

    # Resolve config from CLI flags > TOML
    toml_config = _load_config_toml()
    evolve_toml = toml_config.get("forge-world", {}).get("evolve", {})

    resolved_agent_command = agent_command or evolve_toml.get("agent_command")
    if not resolved_agent_command:
        console.print("[red]No agent command specified. Use --agent-command or set in forge-world.toml[/red]")
        sys.exit(1)

    resolved_max_iterations = max_iterations or evolve_toml.get("max_iterations", 10)
    hard_constraints = evolve_toml.get("hard_constraints", [{"metric": "fpr", "op": "<=", "value": 0}])
    optimization_target = evolve_toml.get(
        "optimization_target", {"metric": "sensitivity", "direction": "max"}
    )

    config = EvolutionConfig(
        agent_command=resolved_agent_command,
        max_iterations=resolved_max_iterations,
        hard_constraints=hard_constraints,
        optimization_target=optimization_target,
        convergence_patience=patience,
    )

    # Build run kwargs for the evolution loop
    run_kwargs: dict[str, Any] = {"sample_size": sample_size}
    if single_seed is not None:
        run_kwargs["seed"] = single_seed
    else:
        run_kwargs["seed_strategy"] = _parse_seed_strategy(seeds, exploration_seeds)
    if tier is not None:
        run_kwargs["tier"] = tier

    # Load diagnostics if available
    diagnostics = None
    if hasattr(mod, "create_diagnostics"):
        try:
            diagnostics = mod.create_diagnostics()
        except Exception:
            pass

    config_schema = None
    try:
        config_schema = runner.pipeline.get_config_schema()
    except Exception:
        pass

    def on_iteration_start(iteration: int):
        if not json_out:
            console.print(f"\n[bold]--- Evolution Iteration {iteration} ---[/bold]")

    def on_iteration_complete(iteration: int, result):
        if not json_out:
            status = "[green]accepted[/green]" if result.accepted else "[yellow]rejected[/yellow]"
            if result.constraint_violated:
                status = "[red]constraint violated[/red]"
            elif not result.had_changes:
                status = "[dim]no changes[/dim]"
            console.print(f"  Iteration {iteration}: {status}")

    loop = EvolutionLoop(
        config=config,
        runner=runner,
        snapshot_manager=sm,
        pipeline_config_schema=config_schema,
        diagnostics=diagnostics,
        on_iteration_start=on_iteration_start,
        on_iteration_complete=on_iteration_complete,
        run_kwargs=run_kwargs,
        baseline_name=baseline,
    )

    result = loop.run()

    if json_out:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        click.echo(result.to_markdown())
