from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from nicegui import app, ui

from .config import AppConfig, ConfigManager
from .models import ChatMessage, ConversationPhase, Deck, ExportSettings
from .services import (
    ConversationManager,
    ExportProgress,
    LLMService,
    export_deck_to_mp4,
    generate_thumbnails,
    generate_deck_media,
    load_deck,
)

# ---------------------------------------------------------------------------
# Theme & Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --bg-elevated: #30363d;
    --accent-primary: #58a6ff;
    --accent-glow: rgba(88, 166, 255, 0.25);
    --accent-warm: #f78166;
    --text-primary: #f0f6fc;
    --text-secondary: #8b949e;
    --text-muted: #6e7681;
    --border-subtle: #30363d;
    --border-muted: #21262d;
    --success: #3fb950;
    --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.4);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.3);
}

body {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.nicegui-content {
    background: var(--bg-primary) !important;
}

/* Header styling */
.deck-header {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    padding: 0.75rem 1.5rem !important;
}

.deck-logo {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
    background: linear-gradient(135deg, var(--accent-primary) 0%, #a371f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

.deck-status {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
}

/* Main panels */
.viewer-panel {
    background: var(--bg-secondary) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-md) !important;
    overflow: hidden;
}

.chat-panel {
    background: var(--bg-secondary) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Slide viewer */
.slide-container {
    background: #000 !important;
    border-radius: 8px !important;
    overflow: hidden;
    position: relative;
}

.slide-container::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.4) 100%);
    pointer-events: none;
    z-index: 1;
}

/* Transport controls */
.transport-bar {
    background: var(--bg-tertiary) !important;
    border-radius: 50px !important;
    padding: 0.5rem 1rem !important;
    gap: 0.25rem !important;
    border: 1px solid var(--border-subtle) !important;
}

.transport-btn {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    transition: all 0.15s ease !important;
}

.transport-btn:hover {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

.transport-btn-play {
    background: var(--accent-primary) !important;
    color: var(--bg-primary) !important;
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}

.transport-btn-play:hover {
    background: #79b8ff !important;
    transform: scale(1.05);
}

.slide-counter {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.875rem !important;
    color: var(--text-muted) !important;
    padding: 0 0.75rem;
}

/* Thumbnail strip */
.thumb-strip {
    background: var(--bg-tertiary) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    border: 1px solid var(--border-muted) !important;
}

.thumb-item {
    border-radius: 6px !important;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.2s ease !important;
    border: 2px solid transparent !important;
    opacity: 0.7;
}

.thumb-item:hover {
    opacity: 1;
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.thumb-active {
    border-color: var(--accent-primary) !important;
    opacity: 1;
    box-shadow: 0 0 12px var(--accent-glow) !important;
}

/* Chat styling */
.chat-header {
    padding: 1rem 1.25rem !important;
    border-bottom: 1px solid var(--border-muted) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.chat-scroll {
    background: var(--bg-primary) !important;
    border-radius: 8px !important;
}

.chat-input-area {
    background: var(--bg-tertiary) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    border: 1px solid var(--border-muted) !important;
}

.chat-input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.chat-input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

.send-btn {
    background: var(--accent-primary) !important;
    color: var(--bg-primary) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0 1.25rem !important;
    transition: all 0.15s ease !important;
}

.send-btn:hover {
    background: #79b8ff !important;
    transform: translateY(-1px);
}

/* Mode toggle */
.mode-toggle {
    background: var(--bg-tertiary) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
    border: 1px solid var(--border-muted) !important;
}

/* Header buttons */
.header-btn {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}

.header-btn:hover {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-muted) !important;
}

.header-btn-primary {
    background: var(--accent-primary) !important;
    color: var(--bg-primary) !important;
    border: none !important;
}

.header-btn-primary:hover {
    background: #79b8ff !important;
}

/* Dialogs */
.q-dialog__inner > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-lg) !important;
}

.q-card {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}

.q-expansion-item {
    background: var(--bg-tertiary) !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
}

.q-input, .q-select, .q-textarea {
    background: var(--bg-primary) !important;
}

.q-field__control {
    background: var(--bg-primary) !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--bg-elevated);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* Chat messages */
.q-message-text {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
    font-size: 0.9375rem !important;
    line-height: 1.5 !important;
}

.q-message-sent .q-message-text {
    background: var(--accent-primary) !important;
    color: var(--bg-primary) !important;
}

/* Upload area */
.q-uploader {
    background: var(--bg-tertiary) !important;
    border: 1px dashed var(--border-subtle) !important;
    border-radius: 8px !important;
}

/* Progress bars */
.q-linear-progress {
    background: var(--bg-elevated) !important;
    border-radius: 4px !important;
}

.q-linear-progress__track {
    background: var(--bg-elevated) !important;
}

.q-linear-progress__model {
    background: var(--accent-primary) !important;
}

/* Spinner */
.q-spinner {
    color: var(--accent-primary) !important;
}

/* Empty state */
.empty-state {
    color: var(--text-muted) !important;
    font-style: italic;
}

/* Attachment preview */
.attachment-preview {
    border-radius: 8px !important;
    border: 2px solid var(--border-subtle) !important;
    overflow: hidden;
}
"""

LOG = logging.getLogger(__name__)


@dataclass
class UIState:
    config_manager: ConfigManager
    config: AppConfig
    project_root: Path
    deck_path: Optional[Path] = None

    # Deck / playback
    deck: Optional[Deck] = None
    thumbnails: List[Path] = field(default_factory=list)
    current_index: int = 0
    is_playing: bool = False
    include_transitions: bool = True
    playback_phase: str = "slide"  # 'slide' or 'transition'
    playback_elapsed: float = 0.0
    is_loading_deck: bool = False
    is_generating: bool = False
    static_mount: str = "/project"

    # Conversation
    conversation: ConversationManager = field(default=None)  # type: ignore[assignment]

    # UI elements
    slide_image: Optional[ui.image] = None  # type: ignore[type-arg]
    slide_video: Optional[ui.video] = None  # type: ignore[type-arg]
    play_button: Optional[ui.button] = None  # type: ignore[type-arg]
    thumbnail_images: List[ui.image] = field(default_factory=list)  # type: ignore[type-arg]
    deck_status_label: Optional[ui.label] = None  # type: ignore[type-arg]
    counter_label: Optional[ui.label] = None  # type: ignore[type-arg]
    thumbnail_row: Optional[ui.row] = None  # type: ignore[type-arg]
    chat_column: Optional[ui.column] = None  # type: ignore[type-arg]
    input_box: Optional[ui.input] = None  # type: ignore[type-arg]
    chat_spinner: Optional[ui.spinner] = None  # type: ignore[type-arg]
    export_dialog: Optional[ui.dialog] = None  # type: ignore[type-arg]
    export_progress_label: Optional[ui.label] = None  # type: ignore[type-arg]
    export_progress_bar: Optional[ui.linear_progress] = None  # type: ignore[type-arg]
    generate_dialog: Optional[ui.dialog] = None  # type: ignore[type-arg]
    generate_progress_label: Optional[ui.label] = None  # type: ignore[type-arg]
    generate_progress_bar: Optional[ui.linear_progress] = None  # type: ignore[type-arg]
    settings_dialog: Optional[ui.dialog] = None  # type: ignore[type-arg]
    settings_error_label: Optional[ui.label] = None  # type: ignore[type-arg]
    pending_images: List[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _slide_counter_text(state: UIState) -> str:
    total = state.deck.slide_count() if state.deck else 0  # type: ignore[union-attr]
    current = state.current_index + 1 if state.deck else 0
    return f"{current} / {total}"


def _asset_url(state: UIState, path_str: str) -> str:
    """Convert a local path to a browser-accessible URL via the static mount."""
    p = Path(path_str)
    if not p.is_absolute():
        p = state.project_root / p
    try:
        rel = p.relative_to(state.project_root)
        return f"{state.static_mount}/{rel.as_posix()}"
    except ValueError:
        # Fallback to file URI; may not work cross-origin but avoids crashing.
        return p.as_uri()


def _current_slide_path(state: UIState) -> Optional[str]:
    if not state.deck or not state.deck.slides:
        return None
    return _asset_url(state, state.deck.slides[state.current_index].image)


def _current_transition_path(state: UIState) -> Optional[str]:
    if not state.deck or not state.deck.slides:
        return None
    slide = state.deck.slides[state.current_index]
    return _asset_url(state, slide.transition) if slide.transition else None


async def _update_slide_viewer(state: UIState) -> None:
    if not state.deck or not state.deck.slides:
        return

    slide_src = _current_slide_path(state)
    trans_src = _current_transition_path(state)
    show_transition = (
        state.playback_phase == "transition" and state.include_transitions and trans_src is not None
    )

    if show_transition and state.slide_video:
        state.slide_video.set_source(trans_src)
        state.slide_video.visible = True
        state.slide_video.props("autoplay muted")
        try:
            state.slide_video.run_method("play")
        except Exception:
            pass
    if state.slide_image:
        state.slide_image.set_source(slide_src or "")
        state.slide_image.visible = not show_transition

    if state.slide_video and not show_transition:
        try:
            state.slide_video.run_method("pause")
        except Exception:
            pass
        state.slide_video.visible = False

    if state.counter_label:
        state.counter_label.text = _slide_counter_text(state)
    _highlight_thumbnails(state)


def _highlight_thumbnails(state: UIState) -> None:
    if not state.thumbnail_images:
        return
    for idx, img in enumerate(state.thumbnail_images):
        base_classes = "w-28 h-16 object-cover thumb-item"
        if idx == state.current_index:
            img.classes(f"{base_classes} thumb-active")
        else:
            img.classes(base_classes)


def _render_thumbnails(state: UIState) -> None:
    """Rebuild the thumbnail strip from the current deck."""
    if not state.thumbnail_row:
        return
    state.thumbnail_row.clear()
    state.thumbnail_images.clear()

    if not state.deck or not state.thumbnails:
        with state.thumbnail_row:
            ui.label("No slides loaded").classes("empty-state text-caption")
        return

    with state.thumbnail_row:
        for idx, thumb_path in enumerate(state.thumbnails):
            url = _asset_url(state, str(thumb_path))
            img = ui.image(url).classes("w-28 h-16 object-cover thumb-item")
            img.on("click", lambda e, i=idx: asyncio.create_task(_go_to_index(state, i)))
            state.thumbnail_images.append(img)

    _highlight_thumbnails(state)


async def _load_deck_into_state(state: UIState) -> None:
    """Load deck JSON (if present) and refresh viewer and thumbnails."""
    if state.is_loading_deck:
        return
    state.is_loading_deck = True
    try:
        deck = await load_deck(state.project_root, state.deck_path)
    except FileNotFoundError:
        state.deck = None
        state.thumbnails = []
        if state.deck_status_label:
            state.deck_status_label.text = "No deck.json found."
        state.is_loading_deck = False
        return
    except Exception as exc:  # pragma: no cover - defensive
        state.deck = None
        state.thumbnails = []
        LOG.exception("Failed to load deck: %s", exc)
        ui.notify(f"Failed to load deck: {exc}", type="negative")
        state.is_loading_deck = False
        return

    loop = asyncio.get_event_loop()
    thumbs = await loop.run_in_executor(None, lambda: generate_thumbnails(deck, state.project_root))

    state.deck = deck
    state.thumbnails = thumbs
    state.current_index = 0
    state.playback_phase = "slide"
    state.playback_elapsed = 0.0
    await _update_slide_viewer(state)
    _render_thumbnails(state)

    if state.deck_status_label:
        deck_label = state.deck_path or (state.project_root / "deck.json")
        state.deck_status_label.text = f"Loaded deck: {deck_label}"
    ui.notify("Deck loaded", type="positive")
    state.is_loading_deck = False


async def _playback_tick(state: UIState, interval: float = 0.5) -> None:
    if not state.is_playing or not state.deck or not state.deck.slides:
        return

    state.playback_elapsed += interval
    slide_duration = state.config.default_slide_duration
    transition_duration = state.config.default_transition_duration

    if state.playback_phase == "slide" and state.playback_elapsed >= slide_duration:
        # Move into transition if available and enabled.
        if state.include_transitions and _current_transition_path(state):
            state.playback_phase = "transition"
            state.playback_elapsed = 0.0
            await _update_slide_viewer(state)
        elif state.config.default_no_transition_behavior == "fade":
            state.playback_phase = "transition"
            state.playback_elapsed = 0.0
            asyncio.create_task(_fake_transition_then_next(state, transition_duration))
        else:
            # Jump directly to next slide.
            await _go_next(state)
    elif state.playback_phase == "transition" and state.playback_elapsed >= transition_duration:
        await _go_next(state)


async def _fake_transition_then_next(state: UIState, duration: float) -> None:
    """Simulate a cross-fade delay when no transition asset exists."""
    await asyncio.sleep(duration)
    await _go_next(state)


async def _stop_playback(state: UIState) -> None:
    state.is_playing = False
    state.current_index = 0
    state.playback_phase = "slide"
    state.playback_elapsed = 0.0
    _sync_play_button(state)
    if state.slide_video and state.slide_video.visible:
        try:
            state.slide_video.run_method("pause")
        except Exception:
            pass
    await _update_slide_viewer(state)


async def _go_first(state: UIState) -> None:
    if not state.deck or not state.deck.slides:
        return
    state.current_index = 0
    state.playback_phase = "slide"
    state.playback_elapsed = 0.0
    await _update_slide_viewer(state)


async def _go_last(state: UIState) -> None:
    if not state.deck or not state.deck.slides:
        return
    state.current_index = state.deck.slide_count() - 1
    state.playback_phase = "slide"
    state.playback_elapsed = 0.0
    await _update_slide_viewer(state)


async def _go_prev(state: UIState) -> None:
    if not state.deck or not state.deck.slides:
        return
    if state.current_index > 0:
        state.current_index -= 1
        state.playback_phase = "slide"
        state.playback_elapsed = 0.0
        await _update_slide_viewer(state)


async def _go_next(state: UIState) -> None:
    if not state.deck or not state.deck.slides:
        return
    if state.current_index < state.deck.slide_count() - 1:
        state.current_index += 1
        state.playback_phase = "slide"
        state.playback_elapsed = 0.0
        await _update_slide_viewer(state)
    else:
        # End of deck: stop playback.
        state.is_playing = False
        state.playback_phase = "slide"
        state.playback_elapsed = 0.0
        _sync_play_button(state)


async def _go_to_index(state: UIState, index: int) -> None:
    if not state.deck or not state.deck.slides:
        return
    if 0 <= index < state.deck.slide_count():
        state.current_index = index
        state.playback_phase = "slide"
        state.playback_elapsed = 0.0
        await _update_slide_viewer(state)


async def _toggle_play(state: UIState) -> None:
    state.is_playing = not state.is_playing
    if state.is_playing:
        state.playback_elapsed = 0.0
        state.playback_phase = "slide"
    else:
        if state.slide_video and state.slide_video.visible:
            try:
                state.slide_video.run_method("pause")
            except Exception:
                pass
    _sync_play_button(state)
    await _update_slide_viewer(state)


def _append_chat_message(state: UIState, message: ChatMessage) -> None:
    if not state.chat_column:
        return
    with state.chat_column:
        ui.chat_message(message.content, sent=message.role == "user").style(
            "white-space: pre-wrap; line-height: 1.6;"
        )
        if message.images:
            with ui.row().classes("gap-2 mt-2"):
                for img in message.images:
                    ui.image(_asset_url(state, img)).classes(
                        "w-16 h-16 object-cover attachment-preview"
                    )


def _sync_play_button(state: UIState) -> None:
    if not state.play_button:
        return
    icon = "pause" if state.is_playing else "play_arrow"
    state.play_button._props["icon"] = icon
    try:
        state.play_button.update()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


def _build_settings_dialog(state: UIState) -> ui.dialog:  # type: ignore[override]
    cfg = state.config
    dialog = ui.dialog()

    # Check which keys are set via environment variables
    gemini_from_env = bool(os.environ.get("GEMINI_API_KEY"))
    fal_from_env = bool(os.environ.get("FAL_KEY"))

    # Track input references (may be None if hidden due to env var)
    gemini_input: Optional[ui.input] = None
    fal_input: Optional[ui.input] = None
    kling_model_select: Optional[ui.select] = None

    def save_settings() -> None:
        try:
            new_cfg = cfg.model_copy()
            # Only update keys that aren't set via environment
            if not gemini_from_env and gemini_input is not None:
                new_cfg.gemini_api_key = gemini_input.value or None
            if not fal_from_env and fal_input is not None:
                new_cfg.fal_api_key = fal_input.value or None
            if kling_model_select is not None:
                new_cfg.kling_model = kling_model_select.value or cfg.kling_model
            new_cfg.default_resolution = resolution_input.value or cfg.default_resolution
            new_cfg.default_slide_duration = float(slide_duration_input.value or cfg.default_slide_duration)
            new_cfg.default_transition_duration = float(
                transition_duration_input.value or cfg.default_transition_duration
            )
            new_cfg.default_no_transition_behavior = (
                no_transition_behavior.value or cfg.default_no_transition_behavior
            )

            state.config_manager.save(new_cfg)
            state.config = new_cfg
            if state.settings_error_label:
                state.settings_error_label.text = ""
            dialog.close()
            ui.notify("Settings saved", type="positive")
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("Failed to save settings: %s", exc)
            if state.settings_error_label:
                state.settings_error_label.text = f"Error: {exc}"

    with dialog, ui.card().classes("w-[500px] p-5"):
        with ui.row().classes("items-center gap-2 mb-4"):
            ui.icon("settings", size="sm").style("color: var(--accent-primary);")
            ui.label("Settings").style("font-size: 1.25rem; font-weight: 600;")

        # Only show API Keys section if at least one key needs to be configured
        if not gemini_from_env or not fal_from_env:
            with ui.expansion("API Keys", icon="vpn_key", value=True).classes("w-full"):
                visible_inputs = []
                if not gemini_from_env:
                    gemini_input = ui.input(
                        "Gemini API Key", password=True, autocomplete="off", value=cfg.gemini_api_key or ""
                    ).classes("w-full")
                    visible_inputs.append(gemini_input)
                if not fal_from_env:
                    fal_input = ui.input(
                        "fal.ai API Key", password=True, autocomplete="off", value=cfg.fal_api_key or ""
                    ).classes("w-full")
                    visible_inputs.append(fal_input)

                if visible_inputs:
                    show_keys = ui.checkbox("Show keys", value=False).classes("mt-2")

                    def toggle_password(e) -> None:
                        visible = bool(e.value)
                        for inp in visible_inputs:
                            inp.password = not visible

                    show_keys.on_value_change(toggle_password)

        with ui.expansion("Video Generation", icon="movie", value=True).classes("w-full"):
            kling_model_select = ui.select(
                {"standard": "Standard (720p, faster)", "pro": "Pro (1080p, higher quality)"},
                value=cfg.kling_model or "pro",
                label="Kling 2.5 Model",
            ).classes("w-full")
            ui.label("Pro model recommended for 2K slide decks").style(
                "font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;"
            )

        with ui.expansion("Defaults", icon="tune", value=True).classes("w-full"):
            resolution_input = ui.input(
                "Default export resolution", value=cfg.default_resolution
            ).classes("w-full")
            with ui.row().classes("gap-3 w-full"):
                slide_duration_input = ui.input(
                    "Slide duration (s)", value=str(cfg.default_slide_duration)
                ).classes("flex-1")
                transition_duration_input = ui.input(
                    "Transition duration (s)", value=str(cfg.default_transition_duration)
                ).classes("flex-1")
            no_transition_behavior = ui.select(
                ["cut", "fade"], value=cfg.default_no_transition_behavior, label="No-transition behavior"
            ).classes("w-full")

        state.settings_error_label = ui.label("").style("color: var(--accent-warm); font-size: 0.75rem;")

        with ui.row().classes("justify-end mt-4 gap-3"):
            ui.button("Cancel", on_click=dialog.close).classes("header-btn").props("flat")
            ui.button("Save", icon="check", on_click=save_settings).classes("header-btn-primary").props("unelevated")

    return dialog


def _build_export_dialog(state: UIState) -> ui.dialog:  # type: ignore[override]
    cfg = state.config
    dialog = ui.dialog()

    def parse_resolution(value: str) -> tuple[int, int]:
        try:
            w_str, h_str = value.lower().split("x", 1)
            return int(w_str), int(h_str)
        except Exception as exc:
            raise ValueError(f"Invalid resolution format: {value}") from exc

    async def do_export() -> None:
        if not state.deck:
            ui.notify("No deck loaded to export", type="warning")
            return

        try:
            selected_res = resolution_select.value or cfg.default_resolution
            res_value = custom_resolution_input.value if selected_res == "Custom" else selected_res
            width, height = parse_resolution(res_value or cfg.default_resolution)
            slide_dur = float(slide_duration_input.value or cfg.default_slide_duration)
            trans_dur = float(transition_duration_input.value or cfg.default_transition_duration)
        except Exception as exc:
            ui.notify(str(exc), type="negative")
            return

        output_path = output_input.value or "deckadence_export.mp4"

        export_settings = ExportSettings(
            width=width,
            height=height,
            slide_duration=slide_dur,
            transition_duration=trans_dur,
            include_transitions=(mode_select.value == "Slides + transitions"),
            output_path=output_path,
            no_transition_behavior=cfg.default_no_transition_behavior,
        )

        if state.export_progress_label and state.export_progress_bar:
            state.export_progress_label.text = "Starting export..."
            state.export_progress_bar.value = 0.0

        async def progress_cb(progress: ExportProgress) -> None:
            if state.export_progress_label and state.export_progress_bar:
                state.export_progress_label.text = progress.message
                state.export_progress_bar.value = progress.fraction

        async def run_export() -> None:
            try:
                await export_deck_to_mp4(
                    state.deck, export_settings, state.project_root,  # type: ignore[arg-type]
                    lambda p: asyncio.create_task(progress_cb(p)),
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOG.exception("Export failed: %s", exc)
                if state.export_progress_label:
                    state.export_progress_label.text = f"Export failed: {exc}"
                ui.notify(f"Export failed: {exc}", type="negative")
                return

            if state.export_progress_label and state.export_progress_bar:
                state.export_progress_label.text = f"Export complete: {output_path}"
                state.export_progress_bar.value = 1.0
            ui.notify("Export complete", type="positive")

        asyncio.create_task(run_export())

    with dialog, ui.card().classes("w-[500px] p-5"):
        with ui.row().classes("items-center gap-2 mb-4"):
            ui.icon("movie", size="sm").style("color: var(--accent-primary);")
            ui.label("Export Video").style("font-size: 1.25rem; font-weight: 600;")

        with ui.column().classes("gap-3 w-full"):
            resolution_select = ui.select(
                ["1280x720", "1920x1080", "2560x1440", "Custom"],
                value=cfg.default_resolution if cfg.default_resolution in {"1280x720", "1920x1080", "2560x1440"} else "Custom",
                label="Resolution",
            ).classes("w-full")

            custom_resolution_input = ui.input(
                "Custom resolution (e.g. 2048x1152)",
                value=cfg.default_resolution if resolution_select.value == "Custom" else "",
            ).classes("w-full")
            custom_resolution_input.disabled = resolution_select.value != "Custom"

            def on_res_change(e) -> None:
                custom_resolution_input.disabled = e.value != "Custom"
                if e.value != "Custom":
                    custom_resolution_input.value = ""

            resolution_select.on("update:model-value", on_res_change)

            with ui.row().classes("gap-3 w-full"):
                slide_duration_input = ui.input(
                    "Slide duration (s)", value=str(cfg.default_slide_duration)
                ).classes("flex-1")
                transition_duration_input = ui.input(
                    "Transition duration (s)", value=str(cfg.default_transition_duration)
                ).classes("flex-1")

            mode_select = ui.select(
                ["Slides only", "Slides + transitions"],
                value="Slides + transitions" if state.include_transitions else "Slides only",
                label="Mode",
            ).classes("w-full")

            output_input = ui.input("Output filename", value="deckadence_export.mp4").classes("w-full")

        with ui.column().classes("w-full mt-4 gap-1"):
            state.export_progress_label = ui.label("").style("font-size: 0.75rem; color: var(--text-muted);")
            state.export_progress_bar = ui.linear_progress(value=0.0)

        with ui.row().classes("justify-end mt-4 gap-3"):
            ui.button("Cancel", on_click=dialog.close).classes("header-btn").props("flat")
            ui.button("Export", icon="file_download", on_click=lambda: asyncio.create_task(do_export())).classes(
                "header-btn-primary"
            ).props("unelevated")

    return dialog


def _build_generation_dialog(state: UIState) -> ui.dialog:  # type: ignore[override]
    dialog = ui.dialog()
    prompt_inputs: List[ui.textarea] = []
    transition_inputs: List[ui.textarea] = []
    include_transitions_checkbox: Optional[ui.checkbox] = None

    def populate_fields() -> None:
        prompt_inputs.clear()
        transition_inputs.clear()
        slides_column.clear()
        transitions_column.clear()
        if not state.deck:
            with slides_column:
                ui.label("Load a deck first to generate media").style("color: var(--accent-warm);")
            return
        with slides_column:
            for idx, _ in enumerate(state.deck.slides):
                ta = ui.textarea(
                    f"Slide {idx + 1}",
                    placeholder="Describe foreground, background, composition, key elements, color palette...",
                ).classes("w-full").props("autogrow rows=2")
                prompt_inputs.append(ta)
        if state.deck.slide_count() > 1:
            with transitions_column:
                for idx in range(state.deck.slide_count() - 1):
                    ta = ui.textarea(
                        f"Transition {idx + 1} → {idx + 2}",
                        placeholder="Describe how elements evolve between slides.",
                    ).classes("w-full").props("autogrow rows=2")
                    transition_inputs.append(ta)

    async def do_generate() -> None:
        if not state.deck:
            ui.notify("Load a deck first", type="warning")
            return
        slide_prompts = [ta.value or f"Highly visual slide {i+1}" for i, ta in enumerate(prompt_inputs)]
        include_transitions = include_transitions_checkbox.value if include_transitions_checkbox else True
        trans_prompts = []
        if include_transitions and state.deck.slide_count() > 1:
            trans_prompts = [
                ta.value or f"Smoothly morph slide {i+1} into slide {i+2}"
                for i, ta in enumerate(transition_inputs)
            ]

        if state.generate_progress_label and state.generate_progress_bar:
            state.generate_progress_label.text = "Starting generation..."
            state.generate_progress_bar.value = 0.0

        async def progress_cb(progress: ExportProgress) -> None:
            if state.generate_progress_label and state.generate_progress_bar:
                state.generate_progress_label.text = progress.message
                state.generate_progress_bar.value = progress.fraction

        try:
            new_deck = await generate_deck_media(
                deck=state.deck,
                slide_prompts=slide_prompts,
                transition_prompts=trans_prompts,
                cfg=state.config,
                project_root=state.project_root,
                deck_path=state.deck_path or state.project_root / "deck.json",
                include_transitions=include_transitions,
                progress_cb=lambda p: asyncio.create_task(progress_cb(p)),
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("Generation failed: %s", exc)
            ui.notify(f"Generation failed: {exc}", type="negative")
            if state.generate_progress_label:
                state.generate_progress_label.text = f"Failed: {exc}"
            return

        # Refresh deck and thumbnails
        state.deck = new_deck
        state.current_index = 0
        loop = asyncio.get_event_loop()
        thumbs = await loop.run_in_executor(None, lambda: generate_thumbnails(new_deck, state.project_root))
        state.thumbnails = thumbs
        _render_thumbnails(state)
        await _update_slide_viewer(state)
        await _load_deck_into_state(state)
        ui.notify("Generation complete", type="positive")

    with dialog, ui.card().classes("w-[640px] p-5"):
        with ui.row().classes("items-center gap-2 mb-4"):
            ui.icon("auto_fix_high", size="sm").style("color: var(--accent-primary);")
            ui.label("Generate Media").style("font-size: 1.25rem; font-weight: 600;")

        include_transitions_checkbox = ui.checkbox(
            "Include animated transitions", value=True if state.include_transitions else False
        ).classes("mb-3")

        with ui.column().classes("gap-2 w-full"):
            ui.label("Slide Prompts").style("font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase;")
            slides_column = ui.scroll_area().classes("w-full").style("max-height: 200px;")

            ui.label("Transition Prompts").style(
                "font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; margin-top: 1rem;"
            )
            transitions_column = ui.scroll_area().classes("w-full").style("max-height: 150px;")

        with ui.column().classes("w-full mt-4 gap-1"):
            state.generate_progress_label = ui.label("").style("font-size: 0.75rem; color: var(--text-muted);")
            state.generate_progress_bar = ui.linear_progress(value=0.0)

        with ui.row().classes("justify-end mt-4 gap-3"):
            ui.button("Cancel", on_click=dialog.close).classes("header-btn").props("flat")
            ui.button("Generate", icon="auto_awesome", on_click=lambda: asyncio.create_task(do_generate())).classes(
                "header-btn-primary"
            ).props("unelevated")

    dialog.on("show", lambda e: populate_fields())
    return dialog


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def create_main_ui(state: UIState) -> None:
    """Compose the NiceGUI interface."""

    # Inject custom CSS
    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")

    llm_service = LLMService(state.config)
    conversation = ConversationManager(llm_service)
    state.conversation = conversation

    # Header bar
    with ui.header().classes("deck-header justify-between items-center"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("auto_awesome", size="sm").style("color: var(--accent-primary);")
            with ui.column().classes("gap-0"):
                ui.label("Deckadence").classes("deck-logo")
                state.deck_status_label = ui.label("Loading deck...").classes("deck-status")

        with ui.row().classes("items-center gap-3"):
            export_dialog = _build_export_dialog(state)
            settings_dialog = _build_settings_dialog(state)
            generate_dialog = _build_generation_dialog(state)
            state.export_dialog = export_dialog
            state.settings_dialog = settings_dialog
            state.generate_dialog = generate_dialog

            ui.button("Settings", icon="settings", on_click=settings_dialog.open).classes("header-btn").props(
                "flat unelevated"
            )
            ui.button("Generate", icon="auto_fix_high", on_click=generate_dialog.open).classes("header-btn").props(
                "flat unelevated"
            )
            ui.button("Export", icon="movie", on_click=export_dialog.open).classes("header-btn-primary").props(
                "unelevated"
            )

    # Main layout
    with ui.row().classes("w-full gap-5 p-5").style("height: calc(100vh - 70px);"):
        # Left: slide viewer and transport controls
        with ui.column().classes("viewer-panel").style("flex: 3; min-width: 0;"):
            with ui.column().classes("w-full h-full p-4 gap-4"):
                # Slide viewer keeps 16:9 aspect ratio
                with ui.element("div").classes("slide-container w-full"):
                    img = ui.image().classes("w-full").style(
                        "aspect-ratio: 16 / 9; object-fit: contain; background: #000;"
                    )
                    vid = ui.video("").classes("w-full").style(
                        "aspect-ratio: 16 / 9; object-fit: contain; background: #000;"
                    )
                    vid.visible = False
                    vid.props("controls=false muted")
                    state.slide_image = img
                    state.slide_video = vid

                # Transport controls
                with ui.row().classes("transport-bar justify-center items-center mx-auto"):
                    ui.button(on_click=lambda: asyncio.create_task(_go_first(state))).props(
                        "icon=first_page flat round"
                    ).classes("transport-btn")
                    ui.button(on_click=lambda: asyncio.create_task(_go_prev(state))).props(
                        "icon=chevron_left flat round"
                    ).classes("transport-btn")

                    state.play_button = ui.button(
                        on_click=lambda: asyncio.create_task(_toggle_play(state))
                    ).props("icon=play_arrow round unelevated").classes("transport-btn-play")

                    ui.button(on_click=lambda: asyncio.create_task(_stop_playback(state))).props(
                        "icon=stop flat round"
                    ).classes("transport-btn")
                    ui.button(on_click=lambda: asyncio.create_task(_go_next(state))).props(
                        "icon=chevron_right flat round"
                    ).classes("transport-btn")
                    ui.button(on_click=lambda: asyncio.create_task(_go_last(state))).props(
                        "icon=last_page flat round"
                    ).classes("transport-btn")

                    counter = ui.label(_slide_counter_text(state)).classes("slide-counter")
                    state.counter_label = counter

                # Thumbnail strip
                with ui.row().classes("thumb-strip w-full overflow-x-auto no-wrap items-center gap-3") as thumb_row:
                    state.thumbnail_row = thumb_row
                    ui.label("No slides loaded").classes("empty-state text-caption")

        # Right: chat panel
        with ui.column().classes("chat-panel").style("flex: 2; min-width: 320px;"):
            # Chat header
            with ui.row().classes("chat-header items-center gap-2"):
                ui.icon("chat", size="xs").style("color: var(--text-muted);")
                ui.label("Conversation")

            # Chat messages area
            with ui.column().classes("flex-1 p-3").style("min-height: 0; overflow: hidden;"):
                with ui.scroll_area().classes("chat-scroll w-full h-full"):
                    state.chat_column = ui.column().classes("gap-3 p-2")

            # Input area
            with ui.column().classes("p-3 gap-3"):
                upload_dir = state.project_root / "_uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)

                attachments_row = ui.row().classes("items-center gap-2 flex-wrap")
                attachments_row.visible = False

                def refresh_attachments() -> None:
                    attachments_row.clear()
                    if not state.pending_images:
                        attachments_row.visible = False
                        return
                    attachments_row.visible = True
                    with attachments_row:
                        for img_path in state.pending_images:
                            ui.image(_asset_url(state, str(img_path))).classes(
                                "w-12 h-12 object-cover attachment-preview"
                            )
                        ui.button(
                            icon="close",
                            on_click=lambda: (state.pending_images.clear(), refresh_attachments()),
                        ).props("flat dense round size=sm").style("color: var(--accent-warm);")

                def handle_upload(e) -> None:
                    dest = upload_dir / e.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(e.content.read())
                    state.pending_images.append(dest)
                    refresh_attachments()

                # Mode toggle
                with ui.row().classes("mode-toggle items-center gap-3 w-full"):
                    ui.icon("tune", size="xs").style("color: var(--text-muted);")
                    mode_toggle = ui.radio(
                        ["Slides only", "Slides + transitions"],
                        value="Slides + transitions" if state.include_transitions else "Slides only",
                        on_change=lambda e: setattr(state, "include_transitions", e.value == "Slides + transitions"),
                    ).props("inline dense")

                # Upload and input row
                with ui.row().classes("chat-input-area items-center gap-2 w-full"):
                    ui.upload(
                        on_upload=handle_upload,
                        multiple=True,
                    ).props("accept=image/* flat dense").classes("w-10").style(
                        "min-width: 40px; max-width: 40px;"
                    )

                    input_box = ui.input(placeholder="Describe your deck idea...").props(
                        "borderless dense"
                    ).classes("chat-input flex-1")
                    state.input_box = input_box

                    state.chat_spinner = ui.spinner("dots", size="sm")
                    state.chat_spinner.visible = False

                    async def send_message() -> None:
                        text = input_box.value or ""
                        if not text.strip():
                            return
                        images = [str(p) for p in state.pending_images] or None
                        msg = ChatMessage(role="user", content=text.strip(), images=images)
                        _append_chat_message(state, msg)
                        input_box.value = ""
                        state.pending_images.clear()
                        refresh_attachments()
                        if state.chat_spinner:
                            state.chat_spinner.visible = True

                        try:
                            reply = await state.conversation.handle_user_message(msg)
                        except Exception as exc:  # pragma: no cover - defensive
                            LOG.exception("Chat failed: %s", exc)
                            _append_chat_message(
                                state,
                                ChatMessage(
                                    role="assistant",
                                    content=f"Sorry, something went wrong: {exc}",
                                ),
                            )
                            if state.chat_spinner:
                                state.chat_spinner.visible = False
                            return

                        _append_chat_message(state, reply)
                        if state.chat_spinner:
                            state.chat_spinner.visible = False

                    ui.button(icon="send", on_click=lambda: asyncio.create_task(send_message())).classes(
                        "send-btn"
                    ).props("unelevated round")

    # Playback timer (global)
    async def on_timer() -> None:
        await _playback_tick(state)

    ui.timer(0.5, on_timer)

    # Initial assistant message for first interaction
    intro = ChatMessage(
        role="assistant",
        content=(
            "Welcome! Let's create a stunning visual deck together.\n\n"
            "To get started, tell me:\n"
            "• What's the deck's purpose and audience?\n"
            "• What tone are you going for?\n"
            "• How many slides do you need?\n"
            "• Any visual style preferences?"
        ),
    )
    _append_chat_message(state, intro)


# ---------------------------------------------------------------------------
# Entrypoint from CLI
# ---------------------------------------------------------------------------


def run_app(
    port: int,
    host: str,
    open_browser: bool,
    config_manager: ConfigManager,
    project_path: Optional[str] = None,
) -> None:
    cfg = config_manager.load()

    if project_path:
        candidate = Path(project_path).resolve()
        if candidate.is_file():
            project_root = candidate.parent
            deck_path = candidate
        else:
            project_root = candidate
            deck_path = project_root / "deck.json" if (project_root / "deck.json").exists() else None
    else:
        project_root = Path.cwd()
        deck_path = project_root / "deck.json" if (project_root / "deck.json").exists() else None

    state = UIState(
        config_manager=config_manager,
        config=cfg,
        project_root=project_root,
        deck_path=deck_path,
    )

    # Expose project assets to the browser.
    app.add_static_files(state.static_mount, str(project_root))

    create_main_ui(state)

    if config_manager.is_missing_required_keys(cfg):
        # Open settings dialog on first launch when keys are missing.
        async def open_settings_on_start() -> None:
            await asyncio.sleep(0.1)
            if state.settings_dialog:
                state.settings_dialog.open()

        ui.timer(0.2, open_settings_on_start, once=True)

    # Auto-load deck (if provided) shortly after UI mounts.
    ui.timer(0.3, lambda: asyncio.create_task(_load_deck_into_state(state)), once=True)

    ui.run(
        title="Deckadence",
        port=port,
        host=host,
        show=open_browser,
        reload=False,
    )
