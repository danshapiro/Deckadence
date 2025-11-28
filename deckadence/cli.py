from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

from .config import ConfigManager, DEFAULT_CONFIG_PATH
from .ui import run_app


LOG = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.command()
@click.option(
    "--port",
    type=int,
    default=8080,
    show_default=True,
    help="Port for the NiceGUI web server.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host address (use 0.0.0.0 for external access).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_CONFIG_PATH,
    show_default=True,
    help="Path to Deckadence JSON config file.",
)
@click.option(
    "--project",
    "project_path",
    type=click.Path(path_type=Path),
    required=False,
    help="Path to a project directory or deck JSON file to open on launch.",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Do not automatically open a browser window on startup.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def main(
    port: int,
    host: str,
    config_path: Path,
    project_path: Optional[Path],
    no_browser: bool,
    verbose: bool,
) -> None:
    """Launch the Deckadence UI server."""
    _configure_logging(verbose)
    LOG.debug("Starting Deckadence with port=%s host=%s", port, host)

    config_manager = ConfigManager(str(config_path))
    run_app(
        port=port,
        host=host,
        open_browser=not no_browser,
        config_manager=config_manager,
        project_path=str(project_path) if project_path else None,
    )
