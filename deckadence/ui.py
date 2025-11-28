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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-deep: #0a0a0c;
    --bg-primary: #111114;
    --bg-card: #18181b;
    --bg-elevated: #232328;
    --bg-hover: #2a2a30;
    --accent: #e5a158;
    --accent-soft: rgba(229, 161, 88, 0.15);
    --accent-glow: rgba(229, 161, 88, 0.3);
    --accent-bright: #f4b87a;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --border: rgba(255, 255, 255, 0.06);
    --border-focus: rgba(229, 161, 88, 0.4);
    --success: #4ade80;
    --error: #f87171;
}

* { box-sizing: border-box; }

body {
    font-family: 'Outfit', system-ui, sans-serif !important;
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}

.nicegui-content {
    background: var(--bg-deep) !important;
    padding: 0 !important;
}

/* ===== HEADER ===== */
.app-header {
    background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-deep) 100%) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0.875rem 1.5rem !important;
    backdrop-filter: blur(12px);
}

.logo-text {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.375rem !important;
    color: var(--accent) !important;
    letter-spacing: -0.03em;
}

.status-text {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.6875rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.02em;
}

/* ===== BUTTONS ===== */
.btn-ghost {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.8125rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.btn-ghost:hover {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

.btn-accent {
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
    font-weight: 600 !important;
    font-size: 0.8125rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.btn-accent:hover {
    background: var(--accent-bright) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--accent-glow);
}

/* ===== MAIN LAYOUT ===== */
.main-container {
    display: flex;
    gap: 1.25rem;
    padding: 1.25rem;
    height: calc(100vh - 60px);
    min-height: 0;
}

.viewer-section {
    flex: 1.4;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.chat-section {
    flex: 1;
    min-width: 340px;
    max-width: 480px;
    display: flex;
    flex-direction: column;
    background: var(--bg-card) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}

/* ===== SLIDE VIEWER ===== */
.slide-frame {
    background: var(--bg-primary) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;
}

.slide-display {
    width: 100%;
    max-height: 100%;
    aspect-ratio: 16 / 9;
    object-fit: contain;
    background: #000;
    border-radius: 8px;
}

/* ===== TRANSPORT ===== */
.transport-controls {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.375rem;
    padding: 0.625rem 1rem;
    background: var(--bg-card) !important;
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    width: fit-content;
    margin: 0 auto;
}

.transport-btn {
    background: transparent !important;
    color: var(--text-muted) !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    border-radius: 50% !important;
    transition: all 0.15s ease !important;
}

.transport-btn:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
}

.play-btn {
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
    transition: all 0.15s ease !important;
}

.play-btn:hover {
    background: var(--accent-bright) !important;
    transform: scale(1.06);
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}

.slide-counter {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    padding: 0 0.75rem;
    min-width: 60px;
    text-align: center;
}

/* ===== THUMBNAILS ===== */
.thumb-rail {
    display: flex;
    gap: 0.625rem;
    padding: 0.75rem;
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    overflow-x: auto;
    scrollbar-width: thin;
}

.thumb-card {
    flex-shrink: 0;
    width: 100px;
    height: 56px;
    border-radius: 6px !important;
    overflow: hidden;
    cursor: pointer;
    border: 2px solid transparent !important;
    opacity: 0.6;
    transition: all 0.2s ease !important;
}

.thumb-card:hover {
    opacity: 0.9;
    border-color: var(--bg-hover) !important;
}

.thumb-card.active {
    opacity: 1;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-soft);
}

/* ===== CHAT ===== */
.chat-header {
    padding: 1rem 1.25rem !important;
    border-bottom: 1px solid var(--border) !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chat-title {
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    color: var(--text-secondary) !important;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    background: var(--bg-primary) !important;
}

.chat-input-container {
    padding: 0.875rem;
    background: var(--bg-card) !important;
    border-top: 1px solid var(--border) !important;
    display: flex;
    flex-direction: column;
    gap: 0.625rem;
}

.mode-pills {
    display: flex;
    gap: 0.5rem;
    padding: 0.25rem;
    background: var(--bg-elevated) !important;
    border-radius: 8px !important;
}

.input-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem;
    background: var(--bg-elevated) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    transition: border-color 0.2s ease;
}

.input-row:focus-within {
    border-color: var(--border-focus) !important;
}

.chat-input {
    flex: 1;
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-size: 0.875rem !important;
}

.send-btn {
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.send-btn:hover {
    background: var(--accent-bright) !important;
}

/* ===== MESSAGES ===== */
.q-message-text {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    padding: 0.75rem 1rem !important;
}

.q-message-sent .q-message-text {
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
}

.q-message-name {
    font-size: 0.6875rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}

/* ===== DIALOGS ===== */
.q-dialog__inner > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.5) !important;
}

.q-card {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* ===== QUASAR OVERRIDES FOR DARK THEME ===== */

/* Expansion items */
.q-expansion-item {
    background: var(--bg-elevated) !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
    border: 1px solid var(--border) !important;
}

.q-expansion-item__container {
    background: transparent !important;
}

.q-item {
    color: var(--text-primary) !important;
}

.q-item__label {
    color: var(--text-primary) !important;
}

.q-item__label--caption {
    color: var(--text-muted) !important;
}

/* Form fields - labels */
.q-field__label {
    color: var(--text-secondary) !important;
}

.q-field__native,
.q-field__prefix,
.q-field__suffix,
.q-field__input {
    color: var(--text-primary) !important;
}

.q-field__native::placeholder {
    color: var(--text-muted) !important;
    opacity: 1;
}

/* Form field controls */
.q-field__control {
    background: var(--bg-primary) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.q-field--outlined .q-field__control:before {
    border-color: var(--border) !important;
}

.q-field--focused .q-field__control:before {
    border-color: var(--accent) !important;
}

/* Input, textarea, select text */
input, textarea, select {
    color: var(--text-primary) !important;
}

input::placeholder, textarea::placeholder {
    color: var(--text-muted) !important;
}

/* Radio buttons */
.q-radio {
    color: var(--text-primary) !important;
}

.q-radio__label {
    color: var(--text-primary) !important;
}

.q-radio__inner {
    color: var(--text-muted) !important;
}

.q-radio__inner--truthy {
    color: var(--accent) !important;
}

/* Checkboxes */
.q-checkbox {
    color: var(--text-primary) !important;
}

.q-checkbox__label {
    color: var(--text-primary) !important;
}

.q-checkbox__inner {
    color: var(--text-muted) !important;
}

.q-checkbox__inner--truthy {
    color: var(--accent) !important;
}

/* Select dropdowns */
.q-select__dropdown-icon {
    color: var(--text-muted) !important;
}

.q-menu {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
}

.q-item--active {
    color: var(--accent) !important;
}

/* Option groups in menus */
.q-virtual-scroll__content .q-item {
    color: var(--text-primary) !important;
}

.q-virtual-scroll__content .q-item:hover {
    background: var(--bg-elevated) !important;
}

/* Tooltips */
.q-tooltip {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

/* Notifications */
.q-notification {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* Icons in various contexts */
.q-icon {
    color: inherit;
}

.q-field__append .q-icon,
.q-field__prepend .q-icon {
    color: var(--text-muted) !important;
}

/* Buttons inside fields */
.q-field .q-btn {
    color: var(--text-muted) !important;
}

/* Expansion item header text */
.q-expansion-item .q-item__section--main {
    color: var(--text-primary) !important;
}

.q-expansion-item .q-item__section--side .q-icon {
    color: var(--text-muted) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--bg-hover);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ===== PROGRESS ===== */
.q-linear-progress {
    background: var(--bg-elevated) !important;
    border-radius: 4px !important;
    height: 4px !important;
}

.q-linear-progress__model {
    background: var(--accent) !important;
}

/* ===== MISC ===== */
.q-spinner {
    color: var(--accent) !important;
}

.empty-hint {
    color: var(--text-muted) !important;
    font-size: 0.8125rem !important;
}

.attachment-chip {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
}

.q-uploader {
    background: transparent !important;
    border: none !important;
}

.q-uploader__header {
    display: none !important;
}

.upload-btn {
    background: var(--bg-hover) !important;
    color: var(--text-muted) !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    border-radius: 8px !important;
}

.upload-btn:hover {
    color: var(--text-primary) !important;
}

/* ===== GLOBAL TEXT LEGIBILITY ===== */

/* Ensure all text inherits proper colors */
.nicegui-content,
.nicegui-content * {
    color: inherit;
}

/* Fix any remaining dark text */
.q-field--dark .q-field__native,
.q-field--dark .q-field__input {
    color: var(--text-primary) !important;
}

/* Labels and helper text */
.q-field__bottom {
    color: var(--text-muted) !important;
}

.q-field__messages {
    color: var(--text-muted) !important;
}

/* Table cells if any */
.q-table {
    color: var(--text-primary) !important;
}

.q-table th, .q-table td {
    color: var(--text-primary) !important;
}

/* List items */
.q-list {
    color: var(--text-primary) !important;
}

/* Tabs if used */
.q-tab {
    color: var(--text-secondary) !important;
}

.q-tab--active {
    color: var(--accent) !important;
}

/* Chip/tag components */
.q-chip {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

/* Badge */
.q-badge {
    color: var(--bg-deep) !important;
}

/* Toggle */
.q-toggle__label {
    color: var(--text-primary) !important;
}

/* Slider labels */
.q-slider__text {
    color: var(--text-primary) !important;
}

/* Breadcrumbs */
.q-breadcrumbs {
    color: var(--text-secondary) !important;
}

/* Force text colors on common elements */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: inherit;
}

/* Ensure proper contrast on focusable elements */
:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
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
        if idx == state.current_index:
            img.classes("thumb-card active", remove="thumb-card")
        else:
            img.classes("thumb-card", remove="thumb-card active")


def _render_thumbnails(state: UIState) -> None:
    """Rebuild the thumbnail strip from the current deck."""
    if not state.thumbnail_row:
        return
    state.thumbnail_row.clear()
    state.thumbnail_images.clear()

    if not state.deck or not state.thumbnails:
        with state.thumbnail_row:
            ui.label("No slides yet").classes("empty-hint")
        return

    with state.thumbnail_row:
        for idx, thumb_path in enumerate(state.thumbnails):
            url = _asset_url(state, str(thumb_path))
            img = ui.image(url).classes("thumb-card").style("object-fit: cover;")
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
        # Use markdown rendering for assistant messages, plain text for user
        if message.role == "assistant":
            with ui.chat_message(sent=False).style("line-height: 1.6;"):
                ui.markdown(message.content).classes("chat-markdown")
        else:
            ui.chat_message(message.content, sent=True).style(
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

    with dialog, ui.card().classes("w-[480px] p-6"):
        with ui.row().classes("items-center gap-3 mb-5"):
            ui.icon("settings", size="sm").style("color: var(--accent);")
            ui.label("Settings").style("font-size: 1.125rem; font-weight: 600;")

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
                "font-size: 0.6875rem; color: var(--text-muted); margin-top: 0.5rem;"
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

        state.settings_error_label = ui.label("").style("color: var(--error); font-size: 0.75rem;")

        with ui.row().classes("justify-end mt-5 gap-2"):
            ui.button("Cancel", on_click=dialog.close).classes("btn-ghost").props("flat")
            ui.button("Save", icon="check", on_click=save_settings).classes("btn-accent").props("unelevated")

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

    with dialog, ui.card().classes("w-[480px] p-6"):
        with ui.row().classes("items-center gap-3 mb-5"):
            ui.icon("movie", size="sm").style("color: var(--accent);")
            ui.label("Export Video").style("font-size: 1.125rem; font-weight: 600;")

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

        with ui.column().classes("w-full mt-5 gap-1"):
            state.export_progress_label = ui.label("").style("font-size: 0.75rem; color: var(--text-muted);")
            state.export_progress_bar = ui.linear_progress(value=0.0)

        with ui.row().classes("justify-end mt-5 gap-2"):
            ui.button("Cancel", on_click=dialog.close).classes("btn-ghost").props("flat")
            ui.button("Export", icon="download", on_click=lambda: asyncio.create_task(do_export())).classes("btn-accent").props("unelevated")

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
                ui.label("Load a deck first to generate media").style("color: var(--error);")
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

    with dialog, ui.card().classes("w-[580px] p-6"):
        with ui.row().classes("items-center gap-3 mb-5"):
            ui.icon("auto_fix_high", size="sm").style("color: var(--accent);")
            ui.label("Generate Media").style("font-size: 1.125rem; font-weight: 600;")

        include_transitions_checkbox = ui.checkbox(
            "Include animated transitions", value=True if state.include_transitions else False
        ).classes("mb-3")

        with ui.column().classes("gap-3 w-full"):
            ui.label("Slide Prompts").style("font-size: 0.6875rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;")
            slides_column = ui.scroll_area().classes("w-full").style("max-height: 180px;")

            ui.label("Transition Prompts").style("font-size: 0.6875rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.5rem;")
            transitions_column = ui.scroll_area().classes("w-full").style("max-height: 140px;")

        with ui.column().classes("w-full mt-5 gap-1"):
            state.generate_progress_label = ui.label("").style("font-size: 0.75rem; color: var(--text-muted);")
            state.generate_progress_bar = ui.linear_progress(value=0.0)

        with ui.row().classes("justify-end mt-5 gap-2"):
            ui.button("Cancel", on_click=dialog.close).classes("btn-ghost").props("flat")
            ui.button("Generate", icon="auto_awesome", on_click=lambda: asyncio.create_task(do_generate())).classes("btn-accent").props("unelevated")

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

    # Build dialogs
    export_dialog = _build_export_dialog(state)
    settings_dialog = _build_settings_dialog(state)
    generate_dialog = _build_generation_dialog(state)
    state.export_dialog = export_dialog
    state.settings_dialog = settings_dialog
    state.generate_dialog = generate_dialog

    # Header
    with ui.header().classes("app-header justify-between items-center"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("dashboard", size="sm").style("color: var(--accent);")
            with ui.column().classes("gap-0"):
                ui.label("Deckadence").classes("logo-text")
                state.deck_status_label = ui.label("Ready").classes("status-text")

        with ui.row().classes("items-center gap-2"):
            ui.button(icon="settings", on_click=settings_dialog.open).classes("btn-ghost").props("flat round")
            ui.button("Generate", icon="auto_fix_high", on_click=generate_dialog.open).classes("btn-ghost").props("flat")
            ui.button("Export", icon="download", on_click=export_dialog.open).classes("btn-accent").props("unelevated")

    # Main content
    with ui.element("div").classes("main-container"):
        # Left: Viewer
        with ui.element("div").classes("viewer-section"):
            # Slide display
            with ui.element("div").classes("slide-frame"):
                img = ui.image().classes("slide-display")
                vid = ui.video("").classes("slide-display")
                vid.visible = False
                vid.props("controls=false muted")
                state.slide_image = img
                state.slide_video = vid

            # Transport controls
            with ui.element("div").classes("transport-controls"):
                ui.button(on_click=lambda: asyncio.create_task(_go_first(state))).props("icon=skip_previous flat round").classes("transport-btn")
                ui.button(on_click=lambda: asyncio.create_task(_go_prev(state))).props("icon=chevron_left flat round").classes("transport-btn")
                state.play_button = ui.button(on_click=lambda: asyncio.create_task(_toggle_play(state))).props("icon=play_arrow round unelevated").classes("play-btn")
                ui.button(on_click=lambda: asyncio.create_task(_stop_playback(state))).props("icon=stop flat round").classes("transport-btn")
                ui.button(on_click=lambda: asyncio.create_task(_go_next(state))).props("icon=chevron_right flat round").classes("transport-btn")
                ui.button(on_click=lambda: asyncio.create_task(_go_last(state))).props("icon=skip_next flat round").classes("transport-btn")
                state.counter_label = ui.label(_slide_counter_text(state)).classes("slide-counter")

            # Thumbnail rail
            with ui.row().classes("thumb-rail") as thumb_row:
                state.thumbnail_row = thumb_row
                ui.label("No slides yet").classes("empty-hint")

        # Right: Chat
        with ui.element("div").classes("chat-section"):
            with ui.element("div").classes("chat-header"):
                ui.icon("forum", size="xs").style("color: var(--text-muted);")
                ui.label("Design Assistant").classes("chat-title")

            # Messages
            with ui.scroll_area().classes("chat-messages"):
                state.chat_column = ui.column().classes("w-full gap-2")

            # Input area
            with ui.element("div").classes("chat-input-container"):
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
                            ui.image(_asset_url(state, str(img_path))).classes("w-10 h-10 object-cover attachment-chip")
                        ui.button(icon="close", on_click=lambda: (state.pending_images.clear(), refresh_attachments())).props("flat dense round size=xs").style("color: var(--error);")

                def handle_upload(e) -> None:
                    dest = upload_dir / e.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(e.content.read())
                    state.pending_images.append(dest)
                    refresh_attachments()

                # Mode toggle
                with ui.element("div").classes("mode-pills"):
                    mode_toggle = ui.radio(
                        ["Slides", "Slides + Transitions"],
                        value="Slides + Transitions" if state.include_transitions else "Slides",
                        on_change=lambda e: setattr(state, "include_transitions", e.value == "Slides + Transitions"),
                    ).props("inline dense")

                # Input row
                with ui.element("div").classes("input-row"):
                    ui.upload(on_upload=handle_upload, multiple=True).props("accept=image/* flat dense").classes("upload-btn").style("width: 36px; height: 36px;")
                    input_box = ui.input(placeholder="What deck would you like to create?").props("borderless dense").classes("chat-input")
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
                        except Exception as exc:
                            LOG.exception("Chat failed: %s", exc)
                            _append_chat_message(state, ChatMessage(role="assistant", content=f"Sorry, something went wrong: {exc}"))
                            if state.chat_spinner:
                                state.chat_spinner.visible = False
                            return

                        _append_chat_message(state, reply)
                        if state.chat_spinner:
                            state.chat_spinner.visible = False

                    ui.button(icon="send", on_click=lambda: asyncio.create_task(send_message())).classes("send-btn").props("unelevated round")

    # Playback timer (global)
    async def on_timer() -> None:
        await _playback_tick(state)

    ui.timer(0.5, on_timer)

    # Initial assistant message
    intro = ChatMessage(
        role="assistant",
        content="Hi! I'm here to help you create a beautiful visual deck.\n\nTell me about your presentation—what's it for, who's the audience, and what vibe are you going for?",
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

    # Expose project assets to the browser.
    app.add_static_files("/project", str(project_root))

    @ui.page('/')
    def index_page():
        state = UIState(
            config_manager=config_manager,
            config=cfg,
            project_root=project_root,
            deck_path=deck_path,
        )

        create_main_ui(state)

        if config_manager.is_missing_required_keys(cfg):
            async def open_settings_on_start() -> None:
                await asyncio.sleep(0.1)
                if state.settings_dialog:
                    state.settings_dialog.open()
            ui.timer(0.2, open_settings_on_start, once=True)

        ui.timer(0.3, lambda: asyncio.create_task(_load_deck_into_state(state)), once=True)

    ui.run(
        title="Deckadence",
        port=port,
        host=host,
        show=open_browser,
        reload=False,
    )
