"""Command-line interface for running the Lovelace V2 pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console
from rich.table import Table

from lovelace.core import LovelaceAnalyzer, run_llm_first_pipeline_v2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lovelace",
        description="Run the Lovelace V2 LLM-first migration pipeline.",
    )

    parser.add_argument(
        "--project",
        default="spring-petclinic-main",
        help="Project name under projects when --source-dir is not provided.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Explicit source directory to analyze. Overrides --project mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to lovelace.yaml configuration file.",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.50,
        help="Maximum LLM cost per service in USD.",
    )
    parser.add_argument(
        "--gateway-url",
        default="http://localhost:8080",
        help="Base URL for generated API gateway.",
    )
    parser.add_argument(
        "--monolith-url",
        default="http://localhost:8081",
        help="Base URL for monolith proxy target.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoints.",
    )
    parser.add_argument(
        "--clear",
        "--clear-checkpoints",
        action="store_true",
        dest="clear_checkpoints",
        help="Clear checkpoints before execution.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve service generation steps.",
    )
    parser.add_argument(
        "--skip-build-verify",
        action="store_true",
        help="Skip Maven compile verification for generated services.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final result object as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--visualize",
        choices=["png", "html"],
        help="Generate dependency graph visualization after pipeline run.",
    )
    parser.add_argument(
        "--graph-json",
        type=Path,
        help="Path to save exported dependency graph JSON (default: <output>/dependency_graph.json).",
    )

    return parser


def _find_default_config() -> Optional[Path]:
    cwd = Path.cwd().resolve()

    search_roots = [cwd] + list(cwd.parents)
    relative_candidates = [Path("lovelace.yaml"), Path("src") / "lovelace.yaml"]

    for root in search_roots:
        for rel in relative_candidates:
            candidate = root / rel
            if candidate.exists():
                return candidate
    return None


def _resolve_runtime_root(project_root: Path) -> Path:
    """Pick a runtime root that contains project artifacts like test-projects/output dirs."""
    candidates = [project_root, project_root.parent]

    # Project shortcut mode depends on projects; prefer that location first.
    for candidate in candidates:
        if (candidate / "projects").exists():
            return candidate

    for candidate in candidates:
        if (candidate / "output").exists():
            return candidate
    return project_root


def _resolve_paths(args: argparse.Namespace, project_root: Path) -> tuple[Path, Path]:
    runtime_root = _resolve_runtime_root(project_root)

    if args.source_dir:
        source_dir = args.source_dir.expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        output_dir = args.output.expanduser().resolve() if args.output else (runtime_root / "output")
        return source_dir, output_dir

    monolith_root = runtime_root / "projects" / args.project
    if not monolith_root.exists():
        raise FileNotFoundError(
            f"Project '{args.project}' not found in {runtime_root / 'projects'}"
        )

    java_source = monolith_root / "src" / "main" / "java"
    source_dir = java_source if java_source.exists() else monolith_root
    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else (runtime_root / "output")
    )
    return source_dir, output_dir


def _render_service_table(console: Console, results: dict) -> None:
    table = Table(title="Service Generation Results", show_header=True)
    table.add_column("Service", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Cost", style="yellow")
    table.add_column("Iterations", style="blue")
    table.add_column("Validation", style="magenta")
    table.add_column("Docker", style="blue")

    for svc in results.get("service_results", []):
        success = bool(svc.get("success"))
        label = "OK" if success else "FAIL"
        style = "green" if success else "red"

        table.add_row(
            svc.get("name", "unknown"),
            f"[{style}]{label}[/{style}]",
            f"${svc.get('cost_usd', 0):.4f}",
            str(svc.get("iterations", 0)),
            svc.get("validation_state", "pending"),
            svc.get("docker_image", "-"),
        )

    console.print(table)


def _verify_generated_builds(console: Console, output_dir: Path) -> int:
    services_dir = output_dir / "services"
    if not services_dir.exists():
        console.print("[yellow]No services directory found; skipping build verification.[/yellow]")
        return 0

    console.print("\n[bold]Verifying builds...[/bold]")
    failed = 0

    for service_dir in sorted(services_dir.iterdir()):
        if not service_dir.is_dir() or not (service_dir / "pom.xml").exists():
            continue

        result = subprocess.run(
            ["mvn", "compile", "-q", "-B"],
            cwd=service_dir,
            capture_output=True,
            timeout=120,
            text=True,
        )
        if result.returncode == 0:
            console.print(f"[green]OK {service_dir.name} compiles[/green]")
        else:
            failed += 1
            console.print(f"[red]FAIL {service_dir.name} does not compile[/red]")
            stderr = (result.stderr or "").strip()
            if stderr:
                console.print(f"[dim]{stderr.splitlines()[-1]}[/dim]")

    return failed


def _generate_visualization(
    analyzer: LovelaceAnalyzer,
    console: Console,
    source_dir: Path,
    output_dir: Path,
    fmt: str,
    graph_json_path: Optional[Path],
) -> None:
    """Export graph JSON and generate a visualization file."""
    if analyzer.graph.graph.number_of_nodes() == 0:
        analyzer.analyze(source_dir=source_dir)

    json_path = graph_json_path.expanduser().resolve() if graph_json_path else (output_dir / "dependency_graph.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.export_graph(json_path, format="json")

    if fmt == "png":
        vis_path = json_path.with_suffix(".png")
        analyzer.graph.visualize(output_path=str(vis_path), layout="spring", figsize=(20, 16))
    else:
        vis_path = json_path.with_suffix(".html")
        analyzer.graph.visualize_interactive(output_path=str(vis_path))

    console.print(f"[green]Graph JSON exported:[/green] {json_path}")
    console.print(f"[green]Visualization generated:[/green] {vis_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console = Console()
    console.print("[bold cyan]Lovelace V2 Pipeline Runner[/bold cyan]")
    console.print("[dim]Gateway-first strangler fig migration pipeline[/dim]\n")

    config_path = args.config.expanduser().resolve() if args.config else _find_default_config()

    try:
        analyzer = LovelaceAnalyzer(config_path=config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load Lovelace configuration: {exc}[/red]")
        if not args.config:
            console.print(
                "[dim]Tip: pass --config /path/to/lovelace.yaml if auto-discovery fails.[/dim]"
            )
        return 1

    if analyzer.llm_client is None:
        console.print("[red]LLM client not available.[/red]")
        console.print(
            f"[dim]Set API key in env var: {analyzer.config.llm.api_key_env}[/dim]"
        )
        return 1

    try:
        source_dir, output_dir = _resolve_paths(args, analyzer.project_root)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    resume = not args.no_resume

    console.print(f"[dim]Source: {source_dir}[/dim]")
    console.print(f"[dim]Output: {output_dir}[/dim]")
    console.print(f"[dim]Cost limit per service: ${args.cost_limit:.2f}[/dim]")
    console.print(
        "[dim]"
        f"Resume: {resume}, Clear: {args.clear_checkpoints}, Auto-Approve: {args.auto_approve}"
        "[/dim]\n"
    )

    if args.visualize:
        console.print("[bold]Generating dependency graph before pipeline generation...[/bold]")
        try:
            _generate_visualization(
                analyzer=analyzer,
                console=console,
                source_dir=source_dir,
                output_dir=output_dir,
                fmt=args.visualize,
                graph_json_path=args.graph_json,
            )
        except ImportError as exc:
            console.print(f"[yellow]Visualization skipped: missing dependency ({exc}).[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Visualization skipped: {exc}[/yellow]")

    try:
        results = run_llm_first_pipeline_v2(
            analyzer=analyzer,
            source_dir=source_dir,
            output_dir=output_dir,
            gateway_url=args.gateway_url,
            monolith_url=args.monolith_url,
            cost_limit_per_service=args.cost_limit,
            resume=resume,
            clear_checkpoints=args.clear_checkpoints,
            auto_approve=args.auto_approve,
        )
    except Exception as exc:
        console.print(f"[red]Pipeline failed: {exc}[/red]")
        if args.verbose:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1

    console.print("\n[bold green]Pipeline Complete[/bold green]\n")
    _render_service_table(console, results)
    console.print(f"\n[cyan]Total LLM Cost: ${results.get('total_llm_cost_usd', 0):.4f}[/cyan]")
    console.print(f"[bold]Output Directory:[/bold] {output_dir}")

    build_failures = 0
    if args.skip_build_verify:
        console.print("[yellow]Skipping build verification (--skip-build-verify).[/yellow]")
    else:
        try:
            build_failures = _verify_generated_builds(console, output_dir)
        except FileNotFoundError:
            console.print("[yellow]Maven not found; skipping build verification.[/yellow]")

    if args.json:
        console.print_json(json.dumps(results, default=str))

    if build_failures:
        console.print(f"\n[red]Completed with {build_failures} build verification failure(s).[/red]")
        return 2

    console.print("\n[green]Lovelace run complete.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
