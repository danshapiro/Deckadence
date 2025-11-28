"""Deckadence CLI - Full-featured command line interface.

Provides commands for:
- serve: Launch the web UI
- export: Export deck to MP4 video
- generate: Generate slide images and transitions with AI
- config: View and manage configuration
- info: Display deck information
- init: Initialize a new project
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree

from .config import AppConfig, ConfigManager, DEFAULT_CONFIG_PATH
from .models import Deck, ExportSettings, Slide

# Default config path as string for Typer compatibility
_DEFAULT_CONFIG = str(DEFAULT_CONFIG_PATH)

# Create the main Typer app
app = typer.Typer(
    name="deckadence",
    help="Deckadence - AI-powered visual deck creation tool",
    no_args_is_help=True,
    pretty_exceptions_short=True,
)

# Sub-apps for nested commands
config_app = typer.Typer(
    help="Manage Deckadence configuration",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")

console = Console()
LOG = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _get_project_root(project_path: Optional[Path]) -> tuple[Path, Optional[Path]]:
    """Resolve project root and deck path from CLI argument."""
    if project_path:
        candidate = project_path.resolve()
        if candidate.is_file():
            return candidate.parent, candidate
        else:
            deck_file = candidate / "deck.json"
            return candidate, deck_file if deck_file.exists() else None
    else:
        cwd = Path.cwd()
        deck_file = cwd / "deck.json"
        return cwd, deck_file if deck_file.exists() else None


def _load_deck_sync(project_root: Path, deck_path: Optional[Path]) -> Deck:
    """Load a deck synchronously for CLI commands."""
    from .services import load_deck
    return asyncio.run(load_deck(project_root, deck_path))


# ============================================================================
# SERVE COMMAND
# ============================================================================


@app.command()
def serve(
    port: int = typer.Option(8080, "--port", "-p", help="Port for the NiceGUI web server."),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address (use 0.0.0.0 for external access)."),
    config_path: Optional[str] = typer.Option(
        None,
    "--config",
        "-c",
    help="Path to Deckadence JSON config file.",
    ),
    project: Optional[str] = typer.Option(
        None,
    "--project",
        "-P",
    help="Path to a project directory or deck JSON file to open on launch.",
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not automatically open a browser window."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """🚀 Launch the Deckadence web UI server.
    
    Start the interactive web interface for creating and editing visual decks.
    """
    _configure_logging(verbose)
    LOG.debug("Starting Deckadence with port=%s host=%s", port, host)

    from .ui import run_app

    cfg_path = config_path or _DEFAULT_CONFIG
    config_manager = ConfigManager(cfg_path)
    
    rprint(Panel.fit(
        f"[bold cyan]Deckadence[/] starting on [green]http://{host}:{port}[/]",
        title="🎴 Deckadence",
        border_style="cyan",
    ))
    
    run_app(
        port=port,
        host=host,
        open_browser=not no_browser,
        config_manager=config_manager,
        project_path=project,
    )


# ============================================================================
# EXPORT COMMAND
# ============================================================================


@app.command()
def export(
    project: str = typer.Argument(
        ...,
        help="Path to project directory or deck.json file.",
    ),
    output: str = typer.Option(
        "deckadence_export.mp4",
        "--output", "-o",
        help="Output video file path.",
    ),
    width: int = typer.Option(1920, "--width", "-W", help="Video width in pixels."),
    height: int = typer.Option(1080, "--height", "-H", help="Video height in pixels."),
    slide_duration: float = typer.Option(5.0, "--slide-duration", "-s", help="Duration per slide in seconds."),
    transition_duration: float = typer.Option(1.0, "--transition-duration", "-t", help="Duration per transition in seconds."),
    no_transitions: bool = typer.Option(False, "--no-transitions", help="Exclude transition videos from export."),
    fallback: str = typer.Option("cut", "--fallback", "-f", help="Behavior when no transition exists: 'cut' or 'fade'."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """📹 Export a deck to MP4 video.
    
    Renders all slides and transitions into a single video file.
    """
    _configure_logging(verbose)
    
    from .services import export_deck_to_mp4, ExportProgress

    project_path = Path(project)
    if not project_path.exists():
        rprint(f"[red]Error:[/] Path does not exist: {project}")
        raise typer.Exit(1)

    project_root, deck_path = _get_project_root(project_path)
    
    if not deck_path or not deck_path.exists():
        rprint(f"[red]Error:[/] No deck.json found in {project_root}")
        raise typer.Exit(1)

    output_path = Path(output)
    rprint(Panel.fit(
        f"Exporting [cyan]{deck_path}[/] → [green]{output_path}[/]",
        title="📹 Export",
        border_style="cyan",
    ))

    try:
        deck = _load_deck_sync(project_root, deck_path)
    except Exception as e:
        rprint(f"[red]Error loading deck:[/] {e}")
        raise typer.Exit(1)

    settings = ExportSettings(
        width=width,
        height=height,
        slide_duration=slide_duration,
        transition_duration=transition_duration,
        include_transitions=not no_transitions,
        output_path=output,
        no_transition_behavior=fallback,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting...", total=100)

        def progress_cb(p: ExportProgress) -> None:
            progress.update(task, completed=int(p.fraction * 100), description=p.message)

        try:
            result = asyncio.run(export_deck_to_mp4(deck, settings, project_root, progress_cb))
            progress.update(task, completed=100, description="Complete!")
        except Exception as e:
            rprint(f"[red]Export failed:[/] {e}")
            raise typer.Exit(1)

    rprint(f"\n[green]✓[/] Export complete: [bold]{result}[/]")


# ============================================================================
# GENERATE COMMAND
# ============================================================================


@app.command()
def generate(
    project: str = typer.Argument(
        ...,
        help="Path to project directory or deck.json file.",
    ),
    prompts_file: Optional[str] = typer.Option(
        None,
        "--prompts", "-p",
        help="JSON file with slide_prompts and transition_prompts arrays.",
    ),
    no_transitions: bool = typer.Option(False, "--no-transitions", help="Skip generating transition videos."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Generate slide images and transition videos with AI.
    
    Uses Gemini for image generation and Kling 2.5 for transitions.
    Provide prompts via a JSON file or use auto-generated prompts.
    
    Example prompts file:
    
        {
          "slide_prompts": ["Prompt for slide 1", "Prompt for slide 2"],
          "transition_prompts": ["Morph slide 1 into slide 2"]
        }
    """
    _configure_logging(verbose)
    
    from .services import generate_deck_media, ExportProgress

    project_path = Path(project)
    if not project_path.exists():
        rprint(f"[red]Error:[/] Path does not exist: {project}")
        raise typer.Exit(1)

    project_root, deck_path = _get_project_root(project_path)
    
    if not deck_path or not deck_path.exists():
        rprint(f"[red]Error:[/] No deck.json found in {project_root}")
        raise typer.Exit(1)

    cfg_path = config_path or _DEFAULT_CONFIG
    config_manager = ConfigManager(cfg_path)
    cfg = config_manager.load()

    # Check required API keys
    if not cfg.gemini_api_key:
        rprint("[red]Error:[/] Gemini API key not configured.")
        rprint("Set [cyan]GEMINI_API_KEY[/] environment variable or use [cyan]deckadence config set gemini-api-key <KEY>[/]")
        raise typer.Exit(1)

    if not no_transitions and not cfg.fal_api_key:
        rprint("[red]Error:[/] fal.ai API key not configured (required for transitions).")
        rprint("Set [cyan]FAL_KEY[/] environment variable or use [cyan]deckadence config set fal-api-key <KEY>[/]")
        rprint("Or use [cyan]--no-transitions[/] to skip transition generation.")
        raise typer.Exit(1)

    try:
        deck = _load_deck_sync(project_root, deck_path)
    except Exception as e:
        rprint(f"[red]Error loading deck:[/] {e}")
        raise typer.Exit(1)

    slide_count = deck.slide_count()
    
    # Load or generate prompts
    prompts_path = Path(prompts_file) if prompts_file else None
    if prompts_path and prompts_path.exists():
        with open(prompts_path) as f:
            prompts_data = json.load(f)
        slide_prompts = prompts_data.get("slide_prompts", [])
        transition_prompts = prompts_data.get("transition_prompts", [])
    else:
        # Auto-generate placeholder prompts
        slide_prompts = [f"Professional, visually striking slide {i+1}" for i in range(slide_count)]
        transition_prompts = [
            f"Smoothly morph and transition from slide {i+1} to slide {i+2}"
            for i in range(slide_count - 1)
        ] if not no_transitions else []
        
        if not prompts_path:
            rprint("[yellow]Note:[/] Using auto-generated prompts. Use --prompts for custom prompts.")

    # Validate prompt counts
    if len(slide_prompts) != slide_count:
        rprint(f"[red]Error:[/] Expected {slide_count} slide prompts, got {len(slide_prompts)}")
        raise typer.Exit(1)

    if not no_transitions and len(transition_prompts) != max(slide_count - 1, 0):
        rprint(f"[red]Error:[/] Expected {max(slide_count - 1, 0)} transition prompts, got {len(transition_prompts)}")
        raise typer.Exit(1)

    rprint(Panel.fit(
        f"Generating media for [cyan]{slide_count}[/] slides" + 
        (f" with [cyan]{len(transition_prompts)}[/] transitions" if not no_transitions else ""),
        title="🎨 Generate",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=100)

        def progress_cb(p: ExportProgress) -> None:
            progress.update(task, completed=int(p.fraction * 100), description=p.message)

        try:
            asyncio.run(generate_deck_media(
                deck=deck,
                slide_prompts=slide_prompts,
                transition_prompts=transition_prompts if not no_transitions else [],
                cfg=cfg,
                project_root=project_root,
                deck_path=deck_path,
                include_transitions=not no_transitions,
                progress_cb=progress_cb,
            ))
            progress.update(task, completed=100, description="Complete!")
        except Exception as e:
            rprint(f"[red]Generation failed:[/] {e}")
            raise typer.Exit(1)

    rprint(f"\n[green]✓[/] Generation complete! Updated deck saved to [bold]{deck_path}[/]")


# ============================================================================
# INFO COMMAND
# ============================================================================


@app.command()
def info(
    project: str = typer.Argument(
        ".",
        help="Path to project directory or deck.json file.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information."),
) -> None:
    """ℹ️ Display information about a deck.
    
    Shows slide count, file paths, and validation status.
    """
    _configure_logging(verbose)
    
    project_path = Path(project) if project != "." else None
    project_root, deck_path = _get_project_root(project_path)
    
    if not deck_path or not deck_path.exists():
        rprint(f"[yellow]No deck.json found in {project_root}[/]")
        rprint("\nUse [cyan]deckadence init[/] to create a new project.")
        raise typer.Exit(0)

    try:
        deck = _load_deck_sync(project_root, deck_path)
    except Exception as e:
        rprint(f"[red]Error loading deck:[/] {e}")
        raise typer.Exit(1)

    # Build info tree
    tree = Tree(f"🎴 [bold cyan]Deckadence Deck[/]")
    tree.add(f"📁 Project: [green]{project_root}[/]")
    tree.add(f"📄 Deck file: [green]{deck_path}[/]")
    tree.add(f"🖼️ Slides: [cyan]{deck.slide_count()}[/]")
    
    transitions_count = sum(1 for s in deck.slides if s.transition)
    tree.add(f"🎬 Transitions: [cyan]{transitions_count}[/]")

    rprint(tree)

    if verbose and deck.slides:
        rprint("\n[bold]Slides:[/]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Image")
        table.add_column("Transition")
        table.add_column("Status")

        for idx, slide in enumerate(deck.slides):
            img_path = project_root / slide.image if not Path(slide.image).is_absolute() else Path(slide.image)
            img_status = "[green]✓[/]" if img_path.exists() else "[red]✗ Missing[/]"
            
            if slide.transition:
                trans_path = project_root / slide.transition if not Path(slide.transition).is_absolute() else Path(slide.transition)
                trans_status = "[green]✓[/]" if trans_path.exists() else "[red]✗ Missing[/]"
                trans_text = slide.transition
            else:
                trans_status = "[dim]—[/]"
                trans_text = "[dim]None[/]"

            status = img_status
            if slide.transition:
                status = f"{img_status} / {trans_status}"

            table.add_row(str(idx + 1), slide.image, trans_text, status)

        rprint(table)


# ============================================================================
# INIT COMMAND
# ============================================================================


@app.command()
def init(
    directory: str = typer.Argument(
        ".",
        help="Directory to initialize the project in.",
    ),
    slides: int = typer.Option(5, "--slides", "-n", help="Number of placeholder slides to create."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing deck.json."),
) -> None:
    """📦 Initialize a new Deckadence project.
    
    Creates a deck.json with placeholder slides ready for generation.
    """
    project_dir = Path(directory).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    
    deck_file = project_dir / "deck.json"
    
    if deck_file.exists() and not force:
        rprint(f"[yellow]deck.json already exists in {project_dir}[/]")
        rprint("Use [cyan]--force[/] to overwrite.")
        raise typer.Exit(1)

    # Create directory structure
    (project_dir / "slides").mkdir(exist_ok=True)
    (project_dir / "transitions").mkdir(exist_ok=True)
    (project_dir / "_uploads").mkdir(exist_ok=True)

    # Create placeholder deck
    deck_slides = [
        Slide(image=f"slides/slide{i+1}.png")
        for i in range(slides)
    ]
    deck = Deck(slides=deck_slides)
    
    # Write deck.json
    with open(deck_file, "w") as f:
        json.dump(deck.model_dump(), f, indent=2)

    # Create placeholder prompts file
    prompts_file = project_dir / "prompts.json"
    prompts_data = {
        "slide_prompts": [f"Describe slide {i+1} visual content here" for i in range(slides)],
        "transition_prompts": [
            f"Describe transition from slide {i+1} to slide {i+2}"
            for i in range(slides - 1)
        ],
    }
    with open(prompts_file, "w") as f:
        json.dump(prompts_data, f, indent=2)

    tree = Tree(f"📦 [bold cyan]Project initialized[/]")
    tree.add(f"📁 {project_dir}")
    
    files_branch = tree.add("Files created:")
    files_branch.add(f"[green]deck.json[/] - Deck definition ({slides} slides)")
    files_branch.add(f"[green]prompts.json[/] - Generation prompts template")
    
    dirs_branch = tree.add("Directories:")
    dirs_branch.add("[dim]slides/[/] - Generated slide images")
    dirs_branch.add("[dim]transitions/[/] - Generated transition videos")
    dirs_branch.add("[dim]_uploads/[/] - User-uploaded assets")

    rprint(tree)
    
    rprint("\n[bold]Next steps:[/]")
    rprint("  1. Edit [cyan]prompts.json[/] with your slide descriptions")
    rprint("  2. Run [cyan]deckadence generate .[/] to create media")
    rprint("  3. Run [cyan]deckadence serve --project .[/] to preview")
    rprint("  4. Run [cyan]deckadence export .[/] to create video")


# ============================================================================
# CONFIG SUBCOMMANDS
# ============================================================================


@config_app.command("show")
def config_show(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file."),
) -> None:
    """📋 Show current configuration.
    
    Displays all configuration values (API keys are masked).
    """
    cfg_path = config_path or _DEFAULT_CONFIG
    config_manager = ConfigManager(cfg_path)
    cfg = config_manager.load()

    table = Table(title="Deckadence Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_column("Source")

    def mask_key(key: Optional[str]) -> str:
        if not key:
            return "[dim]Not set[/]"
        return key[:4] + "..." + key[-4:] if len(key) > 12 else "****"

    import os
    
    # Check if values come from environment
    gemini_source = "[green]env[/]" if os.environ.get("GEMINI_API_KEY") else "[dim]config[/]"
    fal_source = "[green]env[/]" if os.environ.get("FAL_KEY") else "[dim]config[/]"

    table.add_row("Gemini API Key", mask_key(cfg.gemini_api_key), gemini_source)
    table.add_row("fal.ai API Key", mask_key(cfg.fal_api_key), fal_source)
    table.add_row("LiteLLM Model", cfg.lite_llm_model, "[dim]config[/]")
    table.add_row("Kling Model", cfg.kling_model or "pro", "[dim]config[/]")
    table.add_row("Default Resolution", cfg.default_resolution, "[dim]config[/]")
    table.add_row("Slide Duration", f"{cfg.default_slide_duration}s", "[dim]config[/]")
    table.add_row("Transition Duration", f"{cfg.default_transition_duration}s", "[dim]config[/]")
    table.add_row("No-Transition Behavior", cfg.default_no_transition_behavior, "[dim]config[/]")

    rprint(table)
    rprint(f"\n[dim]Config file: {cfg_path}[/]")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key to set."),
    value: str = typer.Argument(..., help="Value to set."),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file."),
) -> None:
    """Set a configuration value.
    
    Available keys:
    
    - gemini-api-key: Google Gemini API key
    - fal-api-key: fal.ai API key
    - kling-model: 'standard' or 'pro'
    - resolution: Default export resolution (e.g. '1920x1080')
    - slide-duration: Default slide duration in seconds
    - transition-duration: Default transition duration in seconds
    - no-transition-behavior: 'cut' or 'fade'
    """
    cfg_path = config_path or _DEFAULT_CONFIG
    config_manager = ConfigManager(cfg_path)
    cfg = config_manager.load()

    # Map CLI key names to config attributes
    key_map = {
        "gemini-api-key": "gemini_api_key",
        "fal-api-key": "fal_api_key",
        "kling-model": "kling_model",
        "resolution": "default_resolution",
        "slide-duration": "default_slide_duration",
        "transition-duration": "default_transition_duration",
        "no-transition-behavior": "default_no_transition_behavior",
    }

    if key not in key_map:
        rprint(f"[red]Unknown key:[/] {key}")
        rprint(f"Valid keys: {', '.join(key_map.keys())}")
        raise typer.Exit(1)

    attr_name = key_map[key]
    
    # Type conversion for numeric values
    if attr_name in ("default_slide_duration", "default_transition_duration"):
        try:
            value = float(value)
        except ValueError:
            rprint(f"[red]Invalid value:[/] {value} (expected a number)")
            raise typer.Exit(1)

    # Validation
    if attr_name == "kling_model" and value not in ("standard", "pro"):
        rprint(f"[red]Invalid value:[/] {value} (must be 'standard' or 'pro')")
        raise typer.Exit(1)

    if attr_name == "default_no_transition_behavior" and value not in ("cut", "fade"):
        rprint(f"[red]Invalid value:[/] {value} (must be 'cut' or 'fade')")
        raise typer.Exit(1)

    setattr(cfg, attr_name, value)
    config_manager.save(cfg)

    rprint(f"[green]✓[/] Set [cyan]{key}[/] = [yellow]{value if 'key' not in key.lower() else '****'}[/]")


@config_app.command("path")
def config_path_cmd(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file."),
) -> None:
    """📍 Show the configuration file path."""
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    rprint(f"Config file: [cyan]{cfg_path.resolve()}[/]")
    if cfg_path.exists():
        rprint("[green]✓[/] File exists")
    else:
        rprint("[yellow]File does not exist (will be created on first save)[/]")


# ============================================================================
# VERSION CALLBACK
# ============================================================================


def version_callback(value: bool) -> None:
    if value:
        rprint("[bold cyan]Deckadence[/] version [green]0.1.0[/]")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(None, "--version", "-V", callback=version_callback, is_eager=True, help="Show version."),
) -> None:
    """🎴 Deckadence - AI-powered visual deck creation.
    
    Create stunning visual presentations with AI-generated slides and transitions.
    """
    pass


# ============================================================================
# LEGACY MAIN FUNCTION (for backwards compatibility with __main__.py)
# ============================================================================


def main() -> None:
    """Entry point for the CLI."""
    app()
