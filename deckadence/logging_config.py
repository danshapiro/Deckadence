"""Structured logging configuration for Deckadence.

Provides:
- File-based logging with rotation (logs/deckadence.log)
- Console output with configurable verbosity
- Structured JSON logging for machine parsing
- Separate log files for different concerns (api, export, etc.)

Usage:
    from deckadence.logging_config import setup_logging, get_logger

    # At application startup
    setup_logging(verbose=True)

    # In modules
    log = get_logger(__name__)
    log.info("Processing slide", extra={"slide_num": 1, "total": 5})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# Default log directory (user's home directory)
DEFAULT_LOG_DIR = Path.home() / ".deckadence" / "logs"

# Log format constants
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
CONSOLE_FORMAT_SIMPLE = "[%(levelname)s] %(message)s"
FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Max log file size (10 MB) and backup count
MAX_LOG_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


class StructuredFormatter(logging.Formatter):
    """JSON-based structured log formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with structured fields."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include any extra fields passed to the log call
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in (
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "message", "taskName",
                ):
                    try:
                        json.dumps(value)  # Check if serializable
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with color support for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        """Format with optional color coding."""
        message = super().format(record)
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            return f"{color}{message}{self.RESET}"
        return message


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check for explicit NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        return False

    # Check for FORCE_COLOR
    if os.environ.get("FORCE_COLOR"):
        return True

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Windows-specific check
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable ANSI escape sequences on Windows 10+
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False

    return True


def _ensure_log_dir(log_dir: Path) -> Path:
    """Ensure the log directory exists."""
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(
    verbose: bool = False,
    debug: bool = False,
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    structured: bool = True,
    console_colors: bool = True,
) -> None:
    """Configure the logging system for Deckadence.

    Args:
        verbose: Enable INFO level logging to console (default shows WARNING+)
        debug: Enable DEBUG level logging (overrides verbose)
        log_dir: Custom log directory (default: ~/.deckadence/logs)
        log_to_file: Whether to write logs to files
        structured: Use JSON structured logging for file output
        console_colors: Enable colored console output
    """
    # Determine log levels
    if debug:
        console_level = logging.DEBUG
        file_level = logging.DEBUG
    elif verbose:
        console_level = logging.INFO
        file_level = logging.DEBUG
    else:
        console_level = logging.WARNING
        file_level = logging.INFO

    # Get the root logger for deckadence
    root_logger = logging.getLogger("deckadence")
    root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_fmt = CONSOLE_FORMAT if verbose or debug else CONSOLE_FORMAT_SIMPLE
    console_formatter = ColoredConsoleFormatter(console_fmt, DATE_FORMAT, use_colors=console_colors)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handlers
    if log_to_file:
        log_path = _ensure_log_dir(log_dir or DEFAULT_LOG_DIR)

        # Main rotating log file
        main_log = log_path / "deckadence.log"
        main_handler = RotatingFileHandler(
            main_log,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        main_handler.setLevel(file_level)
        if structured:
            main_handler.setFormatter(StructuredFormatter())
        else:
            main_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        root_logger.addHandler(main_handler)

        # Separate log for API calls (useful for debugging/auditing)
        api_log = log_path / "api.log"
        api_handler = RotatingFileHandler(
            api_log,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        api_handler.setLevel(logging.DEBUG)
        api_handler.setFormatter(StructuredFormatter())
        api_handler.addFilter(lambda r: "api" in r.name.lower() or "llm" in r.name.lower() or "service" in r.name.lower())
        root_logger.addHandler(api_handler)

        # Separate log for export operations
        export_log = log_path / "export.log"
        export_handler = RotatingFileHandler(
            export_log,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        export_handler.setLevel(logging.DEBUG)
        export_handler.setFormatter(StructuredFormatter())
        export_handler.addFilter(lambda r: "export" in r.name.lower() or "ffmpeg" in r.getMessage().lower())
        root_logger.addHandler(export_handler)

    # Suppress noisy third-party loggers
    _configure_third_party_loggers()


def _configure_third_party_loggers() -> None:
    """Suppress verbose logging from third-party libraries."""
    noisy_loggers = [
        "httpcore",
        "httpx",
        "LiteLLM",
        "litellm",
        "urllib3",
        "PIL",
        "fal_client",
        "google",
        "google.auth",
        "google.genai",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: The module name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        log = get_logger(__name__)
        log.info("Processing started", extra={"slide_count": 5})
    """
    # Ensure it's under the deckadence namespace
    if not name.startswith("deckadence"):
        name = f"deckadence.{name}"
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured context to log messages.

    Usage:
        with LogContext(operation="export", project="my_deck"):
            log.info("Starting operation")  # Will include operation and project fields
    """

    _context: dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        self.new_context = kwargs
        self.old_context: dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        self.old_context = LogContext._context.copy()
        LogContext._context.update(self.new_context)
        return self

    def __exit__(self, *args: Any) -> None:
        LogContext._context = self.old_context

    @classmethod
    def get_context(cls) -> dict[str, Any]:
        """Get the current logging context."""
        return cls._context.copy()


def log_operation(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
) -> Any:
    """Decorator to log function entry/exit with timing.

    Args:
        logger: Logger instance to use
        operation: Description of the operation
        level: Log level (default: INFO)

    Example:
        @log_operation(log, "generate slide")
        async def generate_slide(prompt: str) -> Path:
            ...
    """
    import functools
    import time

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            logger.log(level, "Starting: %s", operation, extra={"operation": operation, "status": "started"})
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(
                    level,
                    "Completed: %s (%.2fs)",
                    operation,
                    elapsed,
                    extra={"operation": operation, "status": "completed", "duration_seconds": elapsed},
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.exception(
                    "Failed: %s (%.2fs) - %s",
                    operation,
                    elapsed,
                    str(e),
                    extra={"operation": operation, "status": "failed", "duration_seconds": elapsed, "error": str(e)},
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            logger.log(level, "Starting: %s", operation, extra={"operation": operation, "status": "started"})
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(
                    level,
                    "Completed: %s (%.2fs)",
                    operation,
                    elapsed,
                    extra={"operation": operation, "status": "completed", "duration_seconds": elapsed},
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.exception(
                    "Failed: %s (%.2fs) - %s",
                    operation,
                    elapsed,
                    str(e),
                    extra={"operation": operation, "status": "failed", "duration_seconds": elapsed, "error": str(e)},
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

