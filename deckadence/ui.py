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
        classes = "w-28 h-16 object-cover rounded cursor-pointer border"
        if idx == state.current_index:
            classes += " border-primary shadow-md"
        else:
            classes += " border-grey-5"
        img.classes(classes)


def _render_thumbnails(state: UIState) -> None:
    """Rebuild the thumbnail strip from the current deck."""
    if not state.thumbnail_row:
        return
    state.thumbnail_row.clear()
    state.thumbnail_images.clear()

    if not state.deck or not state.thumbnails:
        ui.label("No slides loaded yet").classes("text-grey-6 text-caption")
        return

    for idx, thumb_path in enumerate(state.thumbnails):
        url = _asset_url(state, str(thumb_path))
        img = ui.image(url).classes("w-28 h-16 object-cover rounded cursor-pointer border")
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
        ui.chat_message(message.content, sent=message.role == "user").style("white-space: pre-wrap;")
        if message.images:
            with ui.row().classes("gap-1 mt-1"):
                for img in message.images:
                    ui.image(_asset_url(state, img)).classes(
                        "w-16 h-16 object-cover rounded border border-grey-5"
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

    with dialog, ui.card().classes("w-[480px]"):
        ui.label("Deckadence Settings").classes("text-h6 mb-2")

        # Only show API Keys section if at least one key needs to be configured
        if not gemini_from_env or not fal_from_env:
            with ui.expansion("API Keys", icon="vpn_key", value=True):
                visible_inputs = []
                if not gemini_from_env:
                    gemini_input = ui.input(
                        "Gemini API Key", password=True, autocomplete="off", value=cfg.gemini_api_key or ""
                    )
                    visible_inputs.append(gemini_input)
                if not fal_from_env:
                    fal_input = ui.input(
                        "fal.ai API Key (FAL_KEY)", password=True, autocomplete="off", value=cfg.fal_api_key or ""
                    )
                    visible_inputs.append(fal_input)

                if visible_inputs:
                    show_keys = ui.checkbox("Show keys", value=False).classes("mt-1")

                    def toggle_password(e) -> None:
                        visible = bool(e.value)
                        for inp in visible_inputs:
                            inp.password = not visible

                    show_keys.on_value_change(toggle_password)

        with ui.expansion("Video Generation", icon="movie", value=True):
            kling_model_select = ui.select(
                {"standard": "Standard (720p, faster)", "pro": "Pro (1080p, higher quality)"},
                value=cfg.kling_model or "pro",
                label="Kling 2.5 Model",
            )
            ui.label("Pro model recommended for 2K slide decks").classes("text-caption text-grey-6 mt-1")

        with ui.expansion("Defaults", icon="settings", value=True):
            resolution_input = ui.input(
                "Default export resolution (e.g. 1920x1080)", value=cfg.default_resolution
            )
            slide_duration_input = ui.input(
                "Default slide duration (seconds)", value=str(cfg.default_slide_duration)
            )
            transition_duration_input = ui.input(
                "Default transition duration (seconds)", value=str(cfg.default_transition_duration)
            )
            no_transition_behavior = ui.select(
                ["cut", "fade"], value=cfg.default_no_transition_behavior, label="No-transition behavior"
            )

        state.settings_error_label = ui.label("").classes("text-negative text-caption mt-1")

        with ui.row().classes("justify-end mt-2"):
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Save", on_click=save_settings).props("color=primary")

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

    with dialog, ui.card().classes("w-[480px]"):
        ui.label("Export Video").classes("text-h6 mb-2")

        resolution_select = ui.select(
            ["1280x720", "1920x1080", "2560x1440", "Custom"],
            value=cfg.default_resolution if cfg.default_resolution in {"1280x720", "1920x1080", "2560x1440"} else "Custom",
            label="Resolution",
        )
        custom_resolution_input = ui.input(
            "Custom resolution (e.g. 2048x1152)",
            value=cfg.default_resolution if resolution_select.value == "Custom" else "",
        )
        custom_resolution_input.disabled = resolution_select.value != "Custom"

        def on_res_change(e) -> None:
            custom_resolution_input.disabled = e.value != "Custom"
            if e.value != "Custom":
                custom_resolution_input.value = ""

        resolution_select.on("update:model-value", on_res_change)

        slide_duration_input = ui.input(
            "Slide duration (seconds)", value=str(cfg.default_slide_duration)
        )
        transition_duration_input = ui.input(
            "Transition duration (seconds)", value=str(cfg.default_transition_duration)
        )
        mode_select = ui.select(
            ["Slides only", "Slides + transitions"],
            value="Slides + transitions" if state.include_transitions else "Slides only",
            label="Mode",
        )
        output_input = ui.input("Output MP4 path", value="deckadence_export.mp4")

        state.export_progress_label = ui.label("")
        state.export_progress_bar = ui.linear_progress(value=0.0).classes("mt-1")

        with ui.row().classes("justify-end mt-2"):
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Export", on_click=lambda: asyncio.create_task(do_export())).props("color=primary")

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
            slides_column.clear()
            transitions_column.clear()
            ui.label("Load a deck first to generate media").classes("text-negative")
            return
        for idx, _ in enumerate(state.deck.slides):
            ta = ui.textarea(
                f"Slide {idx + 1} visual prompt",
                placeholder="Describe foreground, background, composition, key elements, color palette...",
                autogrow=True,
            ).classes("w-full")
            prompt_inputs.append(ta)
        if state.deck.slide_count() > 1:
            for idx in range(state.deck.slide_count() - 1):
                ta = ui.textarea(
                    f"Transition {idx + 1} -> {idx + 2} prompt",
                    placeholder="Describe how elements evolve between slides.",
                    autogrow=True,
                ).classes("w-full")
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

    with dialog, ui.card().classes("w-[640px]"):
        ui.label("Generate Slides & Transitions").classes("text-h6 mb-2")
        include_transitions_checkbox = ui.checkbox(
            "Generate transitions", value=True if state.include_transitions else False
        )
        slides_column = ui.column().classes("gap-2 max-h-64 overflow-y-auto")
        transitions_column = ui.column().classes("gap-2 max-h-48 overflow-y-auto mt-2")

        state.generate_progress_label = ui.label("").classes("text-caption mt-1")
        state.generate_progress_bar = ui.linear_progress(value=0.0).classes("mt-1")

        with ui.row().classes("justify-end mt-2 gap-2"):
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Generate", on_click=lambda: asyncio.create_task(do_generate())).props("color=primary")

    dialog.on("show", lambda e: populate_fields())
    return dialog


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def create_main_ui(state: UIState) -> None:
    """Compose the NiceGUI interface."""

    llm_service = LLMService(state.config)
    conversation = ConversationManager(llm_service)
    state.conversation = conversation

    # Header bar
    with ui.header().classes("justify-between items-center px-4"):
        with ui.column():
            ui.label("Deckadence").classes("text-h5")
            state.deck_status_label = ui.label("Loading deck...").classes("text-caption text-grey-6")
        with ui.row().classes("items-center gap-2"):
            export_dialog = _build_export_dialog(state)
            settings_dialog = _build_settings_dialog(state)
            generate_dialog = _build_generation_dialog(state)
            state.export_dialog = export_dialog
            state.settings_dialog = settings_dialog
            state.generate_dialog = generate_dialog

            ui.button("Settings", on_click=settings_dialog.open).props("flat dense")
            ui.button("Generate", on_click=generate_dialog.open).props("flat dense")
            ui.button("Export", on_click=export_dialog.open).props("color=primary")

    # Main layout
    with ui.row().classes("w-full h-full gap-4 p-4"):
        # Left: slide viewer and transport controls
        with ui.column().classes("w-3/5 items-center"):
            # Slide viewer keeps 16:9 aspect ratio with letterboxing
            slide_container = ui.card().classes("w-full bg-black")
            with slide_container:
                wrapper = ui.element("div").classes("w-full relative")
                with wrapper:
                    img = ui.image().classes("w-full").style(
                        "aspect-ratio: 16 / 9; object-fit: contain; background-color: black;"
                    )
                    vid = ui.video("").classes("w-full").style(
                        "aspect-ratio: 16 / 9; object-fit: contain; background-color: black;"
                    )
                    vid.visible = False
                    vid.props("controls=false muted")
                state.slide_image = img
                state.slide_video = vid

            # Transport controls
            with ui.row().classes("w-full justify-center items-center mt-2 gap-2"):
                ui.button(on_click=lambda: asyncio.create_task(_go_first(state))).props("icon=first_page")
                ui.button(on_click=lambda: asyncio.create_task(_go_prev(state))).props("icon=chevron_left")

                state.play_button = ui.button(
                    on_click=lambda: asyncio.create_task(_toggle_play(state))
                ).props("icon=play_arrow")

                ui.button(on_click=lambda: asyncio.create_task(_stop_playback(state))).props("icon=stop")
                ui.button(on_click=lambda: asyncio.create_task(_go_next(state))).props("icon=chevron_right")
                ui.button(on_click=lambda: asyncio.create_task(_go_last(state))).props("icon=last_page")
                counter = ui.label(_slide_counter_text(state)).classes("ml-2")
                state.counter_label = counter

            # Thumbnail strip
            with ui.row().classes(
                "w-full overflow-x-auto no-wrap mt-2 items-center gap-2 border-t pt-2"
            ) as thumb_row:
                state.thumbnail_row = thumb_row
                ui.label("No slides loaded yet").classes("text-grey-6 text-caption")

        # Right: chat panel
        with ui.column().classes("w-2/5 h-full"):
            ui.label("Conversation").classes("text-subtitle1 mb-1")

            with ui.scroll_area().classes("flex-1 border rounded-lg p-2") as chat_scroll:
                state.chat_column = ui.column()

            upload_dir = state.project_root / "_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            attachments_row = ui.row().classes("items-center gap-2 mt-1 flex-wrap")
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
                            "w-12 h-12 object-cover rounded border border-grey-5"
                        )
                    ui.button(
                        "Clear attachments",
                        on_click=lambda: (state.pending_images.clear(), refresh_attachments()),
                    ).props("flat dense color=negative")

            def handle_upload(e) -> None:
                dest = upload_dir / e.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(e.content.read())
                state.pending_images.append(dest)
                refresh_attachments()

            with ui.row().classes("items-center gap-2 mt-2"):
                ui.upload(
                    label="Attach images",
                    on_upload=handle_upload,
                    multiple=True,
                ).props("accept=image/*")
                ui.label("Optional: add style/content reference images").classes("text-caption text-grey-7")

            with ui.row().classes("items-center mt-2 gap-2"):
                mode_toggle = ui.radio(
                    ["Slides only", "Slides + transitions"],
                    value="Slides + transitions" if state.include_transitions else "Slides only",
                    on_change=lambda e: setattr(state, "include_transitions", e.value == "Slides + transitions"),
                )
                mode_toggle.props("inline")
                ui.space()
            with ui.row().classes("items-center mt-2 gap-2"):
                input_box = ui.input("Type a message...").props("clearable").classes("flex-1")
                state.input_box = input_box
                state.chat_spinner = ui.spinner("dots")
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
                                content=f"Sorry, something went wrong while talking to the model: {exc}",
                            ),
                        )
                        if state.chat_spinner:
                            state.chat_spinner.visible = False
                        return

                    _append_chat_message(state, reply)
                    if state.chat_spinner:
                        state.chat_spinner.visible = False

                ui.button("Send", on_click=lambda: asyncio.create_task(send_message())).props("color=primary")

    # Playback timer (global)
    async def on_timer() -> None:
        await _playback_tick(state)

    ui.timer(0.5, on_timer)

    # Initial assistant message for first interaction
    intro = ChatMessage(
        role="assistant",
        content=(
            "Hi, I'm Deckadence. Let's design a highly visual slide deck together.\n\n"
            "- What's the purpose of the deck and who is the audience?\n"
            "- What's the tone (e.g., visionary, analytical, playful)?\n"
            "- Roughly how many slides do you want?\n"
            "- Do you prefer static slides only, or slides plus animated transitions?\n"
            "- Any visual style preferences or reference images?"
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
