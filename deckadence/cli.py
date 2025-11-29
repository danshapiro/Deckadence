"""Deckadence CLI - Full-featured command line interface.

Provides commands for:
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
from .costs import get_tracker, reset_tracker
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
    # Suppress noisy HTTP and LiteLLM debug logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


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
    """Export a deck to MP4 video.
    
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
        f"Exporting [cyan]{deck_path}[/] -> [green]{output_path}[/]",
        title="Export",
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


async def _generate_prompts_from_topic(
    topic: str,
    slide_count: int,
    include_transitions: bool,
    cfg: AppConfig,
) -> tuple[List[str], List[str]]:
    """Use LLM to generate slide and transition prompts from a topic."""
    from litellm import acompletion
    
    system_prompt = """You are an expert presentation designer. Given a topic, create detailed visual prompts for slide images and transition animations.

For each slide, describe:
- The visual composition and layout
- Specific imagery, colors, and style
- Mood and atmosphere
- Any text overlays (keep minimal)

Make each slide visually distinct but maintain a cohesive style throughout.
For transitions, describe the motion and transformation between slides.

Respond in valid JSON format only, no markdown:
{
  "slide_prompts": ["detailed prompt for slide 1", "detailed prompt for slide 2", ...],
  "transition_prompts": ["transition from slide 1 to 2", ...]
}"""

    user_prompt = f"""Create a {slide_count}-slide presentation on: "{topic}"

Generate {slide_count} detailed slide image prompts and {slide_count - 1 if include_transitions else 0} transition prompts.
Each slide prompt should be a complete, detailed description for AI image generation.
Make it visually striking and cinematic."""

    response = await acompletion(
        model=cfg.lite_llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        api_key=cfg.gemini_api_key,
    )
    
    content = response["choices"][0]["message"]["content"]
    # Parse JSON from response (handle potential markdown wrapping)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    data = json.loads(content.strip())
    return data.get("slide_prompts", []), data.get("transition_prompts", [])


@app.command()
def generate(
    topic: Optional[str] = typer.Option(
        None,
        "--topic", "-t",
        help="Topic/prompt for the deck. LLM will generate slide and transition prompts.",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project", "-P",
        help="Path to existing project directory (optional with --topic).",
    ),
    slides: int = typer.Option(5, "--slides", "-n", help="Number of slides (used with --topic)."),
    prompts_file: Optional[str] = typer.Option(
        None,
        "--prompts", "-p",
        help="JSON file with slide_prompts and transition_prompts arrays.",
    ),
    output_dir: str = typer.Option("output", "--output", "-o", help="Output directory for generated project."),
    no_transitions: bool = typer.Option(False, "--no-transitions", help="Skip generating transition videos."),
    image_model: Optional[str] = typer.Option(
        None,
        "--image-model", "-I",
        help="Image model: 'nano_banana' (faster, $0.04) or 'nano_banana_pro' (quality, $0.14).",
    ),
    kling_model: Optional[str] = typer.Option(
        None,
        "--kling-model", "-K",
        help="Kling model: 'standard' (720p, cheaper) or 'pro' (1080p, quality).",
    ),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging."),
) -> None:
    """Generate slide images and transition videos with AI.
    
    Two modes:
    
    1. Topic mode (--topic): Provide a topic and let the LLM design the presentation.
       Example: deckadence generate --topic "AI disrupting retail" --slides 3
    
    2. Prompts mode (--prompts): Provide explicit prompts via JSON file.
       Example: deckadence generate --project ./mydeck --prompts prompts.json
    
    Model options:
    
    - Image models: nano_banana (faster, ~$0.04/image) or nano_banana_pro (quality, ~$0.14/image)
    - Kling models: standard (720p, ~$0.065/sec) or pro (1080p, ~$0.095/sec)
    """
    _configure_logging(verbose)
    
    from .services import generate_deck_media, ExportProgress

    cfg_path = config_path or _DEFAULT_CONFIG
    config_manager = ConfigManager(cfg_path)
    cfg = config_manager.load()
    
    # Override model settings from CLI options
    if image_model:
        if image_model not in ("nano_banana", "nano_banana_pro"):
            rprint(f"[red]Error:[/] Invalid image model '{image_model}'. Use 'nano_banana' or 'nano_banana_pro'.")
            raise typer.Exit(1)
        cfg.image_model = image_model
    
    if kling_model:
        if kling_model not in ("standard", "pro"):
            rprint(f"[red]Error:[/] Invalid Kling model '{kling_model}'. Use 'standard' or 'pro'.")
            raise typer.Exit(1)
        cfg.kling_model = kling_model

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

    # Determine mode and setup
    if topic:
        # Topic mode: create project and generate prompts via LLM
        project_root = Path(output_dir).resolve()
        project_root.mkdir(parents=True, exist_ok=True)
        (project_root / "slides").mkdir(exist_ok=True)
        (project_root / "transitions").mkdir(exist_ok=True)
        
        deck_path = project_root / "deck.json"
        deck_slides = [Slide(image=f"slides/slide{i+1}.png") for i in range(slides)]
        deck = Deck(slides=deck_slides)
        
        with open(deck_path, "w") as f:
            json.dump(deck.model_dump(), f, indent=2)
        
        rprint(Panel.fit(
            f"Generating [cyan]{slides}[/]-slide deck on: [yellow]{topic}[/]",
            title="Generate from Topic",
            border_style="cyan",
        ))
        
        # Generate prompts via LLM
        rprint("[dim]Designing presentation with AI...[/]")
        try:
            slide_prompts, transition_prompts = asyncio.run(
                _generate_prompts_from_topic(topic, slides, not no_transitions, cfg)
            )
        except Exception as e:
            rprint(f"[red]Failed to generate prompts:[/] {e}")
            raise typer.Exit(1)
        
        # Save generated prompts for reference
        prompts_data = {
            "topic": topic,
            "slide_prompts": slide_prompts,
            "transition_prompts": transition_prompts,
        }
        with open(project_root / "prompts.json", "w") as f:
            json.dump(prompts_data, f, indent=2)
        
        rprint(f"[green]Created {len(slide_prompts)} slide prompts[/]")
        
    else:
        # Existing project mode
        if not project:
            rprint("[red]Error:[/] Either --topic or --project is required.")
            rprint("Use [cyan]--topic[/] to generate from a topic, or [cyan]--project[/] for an existing project.")
            raise typer.Exit(1)
            
        project_path = Path(project)
        if not project_path.exists():
            rprint(f"[red]Error:[/] Path does not exist: {project}")
            raise typer.Exit(1)

        project_root, deck_path = _get_project_root(project_path)
        
        if not deck_path or not deck_path.exists():
            rprint(f"[red]Error:[/] No deck.json found in {project_root}")
            raise typer.Exit(1)

        # Load deck structure without validating assets
        try:
            with open(deck_path) as f:
                deck_data = json.load(f)
            deck = Deck(**deck_data)
        except Exception as e:
            rprint(f"[red]Error loading deck:[/] {e}")
            raise typer.Exit(1)

        slides = deck.slide_count()
        
        # Load prompts from file
        prompts_path = Path(prompts_file) if prompts_file else project_root / "prompts.json"
        if prompts_path.exists():
            with open(prompts_path) as f:
                prompts_data = json.load(f)
            slide_prompts = prompts_data.get("slide_prompts", [])
            transition_prompts = prompts_data.get("transition_prompts", [])
        else:
            rprint(f"[red]Error:[/] No prompts file found at {prompts_path}")
            rprint("Use [cyan]--topic[/] to generate prompts automatically, or provide [cyan]--prompts[/] file.")
            raise typer.Exit(1)
        
        rprint(Panel.fit(
            f"Generating media for [cyan]{slides}[/] slides" + 
            (f" with [cyan]{len(transition_prompts)}[/] transitions" if not no_transitions else ""),
            title="Generate",
            border_style="cyan",
        ))

    # Validate prompt counts
    if len(slide_prompts) != slides:
        rprint(f"[red]Error:[/] Expected {slides} slide prompts, got {len(slide_prompts)}")
        raise typer.Exit(1)

    if not no_transitions and len(transition_prompts) != max(slides - 1, 0):
        rprint(f"[red]Error:[/] Expected {max(slides - 1, 0)} transition prompts, got {len(transition_prompts)}")
        raise typer.Exit(1)

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

        # Reset cost tracker for this generation session
        reset_tracker()
        
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

    # Display cost summary
    tracker = get_tracker()
    if tracker.entries:
        rprint("\n[bold]API Costs:[/]")
        cost_table = Table(show_header=True, header_style="bold")
        cost_table.add_column("Service", style="cyan")
        cost_table.add_column("Cost", justify="right")
        cost_table.add_row("Gemini (images)", f"${tracker.gemini_cost:.4f}")
        cost_table.add_row("Kling (videos)", f"${tracker.kling_cost:.4f}")
        cost_table.add_row("[bold]Total[/]", f"[bold]${tracker.total_cost:.4f}[/]")
        rprint(cost_table)
    
    rprint(f"\n[green]✓[/] Generation complete! Project saved to [bold]{project_root}[/]")


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
    """Display information about a deck.
    
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
    tree = Tree(f"[bold cyan]Deckadence Deck[/]")
    tree.add(f"Project: [green]{project_root}[/]")
    tree.add(f"Deck file: [green]{deck_path}[/]")
    tree.add(f"Slides: [cyan]{deck.slide_count()}[/]")
    
    transitions_count = sum(1 for s in deck.slides if s.transition)
    tree.add(f"Transitions: [cyan]{transitions_count}[/]")

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
        "output",
        help="Directory to initialize the project in (default: output/).",
    ),
    slides: int = typer.Option(5, "--slides", "-n", help="Number of placeholder slides to create."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing deck.json."),
) -> None:
    """Initialize a new Deckadence project.
    
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

    tree = Tree(f"[bold cyan]Project initialized[/]")
    tree.add(f"{project_dir}")
    
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
    rprint("  3. Run [cyan]deckadence export .[/] to create video")


# ============================================================================
# CONFIG SUBCOMMANDS
# ============================================================================


@config_app.command("show")
def config_show(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file."),
) -> None:
    """Show current configuration.
    
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
    table.add_row("Image Model", cfg.image_model or "nano_banana_pro", "[dim]config[/]")
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
    - image-model: 'nano_banana' (faster, $0.04) or 'nano_banana_pro' (quality, $0.14)
    - kling-model: 'standard' (720p) or 'pro' (1080p)
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
        "image-model": "image_model",
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
    if attr_name == "image_model" and value not in ("nano_banana", "nano_banana_pro"):
        rprint(f"[red]Invalid value:[/] {value} (must be 'nano_banana' or 'nano_banana_pro')")
        raise typer.Exit(1)

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
    """Show the configuration file path."""
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
    """Deckadence - AI-powered visual deck creation.
    
    Create stunning visual presentations with AI-generated slides and transitions.
    """
    pass


# ============================================================================
# LEGACY MAIN FUNCTION (for backwards compatibility with __main__.py)
# ============================================================================


def main() -> None:
    """Entry point for the CLI."""
    app()
