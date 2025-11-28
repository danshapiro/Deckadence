from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import aiofiles
import fal_client
import ffmpeg  # type: ignore
import httpx
from PIL import Image, ImageOps
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from .config import AppConfig
from .prompts import get_prompt, load_prompts
from .models import ChatMessage, ConversationPhase, Deck, ExportSettings, Slide

LOG = logging.getLogger(__name__)
VALID_2K_RESOLUTIONS: set[Tuple[int, int]] = {(1920, 1080), (2048, 1152)}
DOWNLOAD_DIR_NAME = "_downloads"
NORMALIZED_DIR_NAME = "_normalized"
THUMBNAIL_DIR_NAME = "_thumbnails"


# ---------------------------------------------------------------------------
# LLM service via LiteLLM
# ---------------------------------------------------------------------------


class LLMService:
    """Wrapper around LiteLLM for conversational behavior.

    This uses the async `acompletion` API. The concrete model name and API
    key are provided via configuration. Tool definitions (e.g. web_search)
    can be added in `extra_params` if desired.
    """

    def __init__(self, cfg: AppConfig, prompt_path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.prompts = load_prompts(prompt_path)

    async def chat(
        self,
        messages: List[ChatMessage],
        phase: ConversationPhase,
        allow_web_search: bool = False,
        extra_params: Optional[dict] = None,
    ) -> ChatMessage:
        from litellm import acompletion  # imported lazily

        if not self.cfg.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured. Please set GEMINI_API_KEY environment variable or configure it in Settings.")

        system_prefix = self._system_prompt_for_phase(phase)
        llm_messages = [{"role": "system", "content": system_prefix}]
        llm_messages += [self._to_llm_message(m) for m in messages]

        params = {
            "model": self.cfg.lite_llm_model,
            "messages": llm_messages,
            "api_key": self.cfg.gemini_api_key,
        }
        if allow_web_search:
            params["tools"] = self._default_tools()
            params["tool_choice"] = "auto"
        if extra_params:
            params.update(extra_params)

        LOG.debug("Calling LiteLLM with model=%s, phase=%s", self.cfg.lite_llm_model, phase.value)

        try:
            resp = await acompletion(**params)
        except Exception as exc:
            LOG.exception("LiteLLM completion failed: %s", exc)
            raise

        try:
            choice = resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("Unexpected LiteLLM response format: %r", resp)
            raise RuntimeError("Unexpected LiteLLM response format") from exc

        # Handle web_search tool calls manually if present.
        if tool_calls:
            tool_messages = []
            for call in tool_calls:
                if call.get("function", {}).get("name") != "web_search":
                    continue
                try:
                    args = json.loads(call.get("function", {}).get("arguments") or "{}")
                except Exception:
                    args = {}
                query = args.get("query") or ""
                recency_days = args.get("recency_days")
                try:
                    result = await self._perform_web_search(query, recency_days)
                except Exception as exc:  # pragma: no cover - defensive
                    LOG.warning("Web search failed for query '%s': %s", query, exc)
                    result = f"(web search failed for '{query}': {exc})"
                tool_messages.append(
                    {"role": "tool", "tool_call_id": call.get("id"), "content": result}
                )

            if tool_messages:
                followup_messages = llm_messages + [{"role": "assistant", "tool_calls": tool_calls, "content": ""}]
                followup_messages.extend(tool_messages)
                LOG.debug("Calling LiteLLM follow-up with tool results for %d searches", len(tool_messages))
                resp2 = await acompletion(
                    model=self.cfg.lite_llm_model,
                    messages=followup_messages,
                    api_key=self.cfg.gemini_api_key,
                )
                choice = resp2["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or content

        return ChatMessage(role="assistant", content=content)

    def _default_tools(self) -> List[dict]:
        """Provide the web_search tool definition expected by LiteLLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for up-to-date information to inform slide content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string."},
                            "recency_days": {
                                "type": "integer",
                                "description": "Optional recency limit in days.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def _perform_web_search(self, query: str, recency_days: Optional[int]) -> str:
        """Very lightweight web search using DuckDuckGo instant answer API."""
        if not query:
            return "(empty search query)"
        params = {"q": query, "format": "json", "no_html": 1, "no_redirect": 1}
        if recency_days:
            params["timelimit"] = f"d{recency_days}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://api.duckduckgo.com/", params=params)
            resp.raise_for_status()
            data = resp.json()

        snippets = []
        abstract = data.get("AbstractText")
        if abstract:
            snippets.append(abstract)
        related = data.get("RelatedTopics") or []
        for item in related:
            if isinstance(item, dict) and item.get("Text"):
                snippets.append(item["Text"])
            if len(snippets) >= 5:
                break

        if not snippets:
            return f"No results found for '{query}'."
        return "\n".join(f"- {s}" for s in snippets[:5])

    def _to_llm_message(self, msg: ChatMessage) -> dict:
        """Convert ChatMessage to LiteLLM message payload, including images if present."""

        if msg.images:
            content_parts = [{"type": "text", "text": msg.content}]
            for img in msg.images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": self._as_data_uri(img),
                    }
                )
            return {"role": msg.role, "content": content_parts}
        return {"role": msg.role, "content": msg.content}

    def _as_data_uri(self, image_path_or_uri: str) -> str:
        """Encode a local image as a data URI if needed."""
        if image_path_or_uri.startswith("http"):
            return image_path_or_uri
        if image_path_or_uri.startswith("data:"):
            return image_path_or_uri
        path = Path(image_path_or_uri)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path_or_uri}")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def _system_prompt_for_phase(self, phase: ConversationPhase) -> str:
        """Return a tailored system prompt for a given phase from the prompt catalog."""

        key_map = {
            ConversationPhase.ONBOARDING: "onboarding",
            ConversationPhase.OUTLINE: "outline",
            ConversationPhase.VISUAL: "visual",
            ConversationPhase.GENERATION_READY: "generation_ready",
        }

        if phase not in key_map:
            raise ValueError(f"Unhandled conversation phase: {phase}")

        key = key_map[phase]
        return get_prompt(self.prompts, "system", key)


# ---------------------------------------------------------------------------
# Asset utilities (download / normalization)
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
async def download_asset(url: str, dest: Path) -> Path:
    """Download a remote asset (image or video) with retries.

    This is used for normalizing decks that reference HTTP(S) URLs so that
    downstream processing (thumbnails, ffmpeg export) can work on local
    files.
    """
    LOG.info("Downloading asset %s -> %s", url, dest)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
    return dest


async def load_deck(project_root: Path, deck_path: Optional[Path] = None, normalize_remote: bool = True) -> Deck:
    """Load a deck JSON file, optionally downloading remote assets."""

    deck_file = _resolve_deck_file(project_root, deck_path)
    async with aiofiles.open(deck_file, "r", encoding="utf-8") as f:
        raw = await f.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Deck JSON is invalid: {exc}") from exc

    deck = Deck(**data)
    if normalize_remote:
        deck = await _normalize_deck_assets(deck, project_root)
    return deck


async def save_deck(deck: Deck, deck_path: Path) -> None:
    """Persist a deck to disk."""
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(deck_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(deck.model_dump(), indent=2))


async def _normalize_deck_assets(deck: Deck, project_root: Path) -> Deck:
    """Download remote assets and ensure slides are 2K, returning a new Deck."""
    normalized_slides: List[Slide] = []
    for idx, slide in enumerate(deck.slides):
        image_path = await _normalize_asset(slide.image, project_root, "slides", idx)
        transition_path = None
        if slide.transition:
            transition_path = await _normalize_asset(slide.transition, project_root, "transitions", idx)
        normalized_slides.append(Slide(image=image_path, transition=transition_path))
    return Deck(slides=normalized_slides)


async def _normalize_asset(path_str: str, project_root: Path, kind: str, index: int) -> str:
    """Ensure an asset is local; download if remote and return relative path where possible."""
    is_remote = "://" in path_str
    if is_remote:
        hashed = hashlib.sha1(path_str.encode("utf-8")).hexdigest()[:10]
        ext = Path(path_str).suffix or ".bin"
        dest = project_root / DOWNLOAD_DIR_NAME / kind / f"{index + 1}_{hashed}{ext}"
        await download_asset(path_str, dest)
        local_path = dest
    else:
        candidate = Path(path_str)
        local_path = candidate if candidate.is_absolute() else project_root / candidate
        if not local_path.exists():
            raise FileNotFoundError(f"Asset not found: {path_str}")

    if kind == "slides":
        local_path = _ensure_2k_slide(local_path, project_root)

    try:
        return str(local_path.relative_to(project_root))
    except ValueError:
        return str(local_path)


def _ensure_2k_slide(path: Path, project_root: Path) -> Path:
    """Validate slide resolution; if not 2K, pad/resize to 1920x1080."""
    try:
        with Image.open(path) as im:
            width, height = im.size
            if (width, height) in VALID_2K_RESOLUTIONS:
                return path
            LOG.warning("Slide %s is %sx%s, normalizing to 1920x1080", path, width, height)
            normalized_dir = project_root / NORMALIZED_DIR_NAME / "slides"
            normalized_dir.mkdir(parents=True, exist_ok=True)
            target_size = (1920, 1080)
            padded = ImageOps.pad(im.convert("RGB"), target_size, color=(0, 0, 0))
            out_path = normalized_dir / path.name
            padded.save(out_path, format="PNG")
            return out_path
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Could not inspect slide %s: %s", path, exc)
        return path


def generate_thumbnails(deck: Deck, project_root: Path, max_width: int = 280) -> List[Path]:
    """Generate downscaled thumbnails for the current deck."""
    thumbs_dir = project_root / THUMBNAIL_DIR_NAME
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    thumbs: List[Path] = []
    target_height = int(max_width * 9 / 16)

    for idx, slide in enumerate(deck.slides):
        src = _resolve_path(slide.image, project_root)
        if not src.exists():
            raise FileNotFoundError(f"Slide image not found for thumbnail: {src}")
        thumb_path = thumbs_dir / f"slide_{idx + 1}.png"
        with Image.open(src) as im:
            thumb = ImageOps.pad(im.convert("RGB"), (max_width, target_height), color=(16, 16, 16))
            thumb.save(thumb_path, format="PNG")
        thumbs.append(thumb_path)

    return thumbs


def _resolve_deck_file(project_root: Path, deck_path: Optional[Path]) -> Path:
    """Determine the deck JSON path from CLI/project arguments."""
    if deck_path:
        candidate = Path(deck_path)
        if candidate.is_dir():
            candidate = candidate / "deck.json"
    else:
        candidate = project_root / "deck.json"

    if not candidate.exists():
        raise FileNotFoundError(f"No deck JSON found at {candidate}")
    return candidate


# ---------------------------------------------------------------------------
# Media generation (Nano Banana Pro / Kling 2.5)
# ---------------------------------------------------------------------------


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
async def generate_slide_image(
    prompt: str,
    cfg: AppConfig,
    dest: Path,
    reference_images: Optional[Sequence[Path]] = None,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Generate a 2K slide image via Nano Banana Pro."""
    if not cfg.nano_banana_base_url or not cfg.gemini_api_key:
        raise RuntimeError("Nano Banana Pro is not configured (base URL and Gemini API key required).")

    payload: dict = {"prompt": prompt, "width": width, "height": height}
    if reference_images:
        payload["reference_images"] = [str(p) for p in reference_images]

    headers = {"Authorization": f"Bearer {cfg.gemini_api_key}"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{cfg.nano_banana_base_url}/v1/generate", json=payload, headers=headers)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    return dest


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
async def generate_slide_image_edit(
    prompt: str,
    reference_image: Path,
    cfg: AppConfig,
    dest: Path,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """Generate a 2K slide image using Nano Banana Pro Edit with a reference image."""
    if not cfg.nano_banana_base_url or not cfg.gemini_api_key:
        raise RuntimeError("Nano Banana Pro is not configured (base URL and Gemini API key required).")

    payload: dict = {
        "prompt": prompt,
        "reference_image": str(reference_image),
        "width": width,
        "height": height,
    }

    headers = {"Authorization": f"Bearer {cfg.gemini_api_key}"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{cfg.nano_banana_base_url}/v1/edit", json=payload, headers=headers)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    return dest


# ---------------------------------------------------------------------------
# fal.ai helpers for Kling 2.5 video generation
# ---------------------------------------------------------------------------


async def upload_image_to_fal(image_path: Path, cfg: AppConfig) -> str:
    """Upload a local image to fal.ai storage and return the URL.
    
    Args:
        image_path: Path to the local image file.
        cfg: Application config containing the fal.ai API key.
        
    Returns:
        The fal.ai URL for the uploaded image.
    """
    if not cfg.fal_api_key:
        raise RuntimeError("fal.ai API key is not configured.")
    
    # Set the API key for fal_client
    os.environ["FAL_KEY"] = cfg.fal_api_key
    
    loop = asyncio.get_event_loop()
    # fal_client.upload_file is synchronous, run in executor
    url = await loop.run_in_executor(
        None,
        lambda: fal_client.upload_file(str(image_path))
    )
    LOG.debug("Uploaded %s to fal.ai: %s", image_path, url)
    return url


def _get_kling_endpoint(cfg: AppConfig) -> str:
    """Get the appropriate Kling endpoint based on the configured model."""
    model = cfg.kling_model or "pro"
    if model == "standard":
        return "fal-ai/kling-video/v2.5-turbo/std/image-to-video"
    return "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"


def _get_kling_duration(duration: float) -> str:
    """Convert duration to Kling API format (5 or 10 seconds only)."""
    # Kling only supports 5 or 10 second videos
    return "10" if duration > 7 else "5"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
async def generate_transition_clip(
    prompt: str,
    first_frame: Path,
    last_frame: Path,
    cfg: AppConfig,
    dest: Path,
    duration: float = 5.0,
) -> Path:
    """Generate a transition video via Kling 2.5 on fal.ai.
    
    Uses the Kling 2.5 API through fal.ai to generate a video that
    smoothly transitions from first_frame to last_frame.
    
    Args:
        prompt: Text description guiding the transition animation.
        first_frame: Path to the starting image (first keyframe).
        last_frame: Path to the ending image (last keyframe).
        cfg: Application config containing fal.ai API key and model preference.
        dest: Destination path for the output video file.
        duration: Approximate video duration in seconds (will be 5 or 10).
        
    Returns:
        Path to the generated video file.
    """
    if not cfg.fal_api_key:
        raise RuntimeError("fal.ai API key is not configured. Please set FAL_KEY environment variable or configure it in Settings.")

    # Set the API key for fal_client
    os.environ["FAL_KEY"] = cfg.fal_api_key
    
    # Upload both images to fal.ai storage
    LOG.info("Uploading images to fal.ai for transition generation...")
    start_url = await upload_image_to_fal(first_frame, cfg)
    end_url = await upload_image_to_fal(last_frame, cfg)
    
    endpoint = _get_kling_endpoint(cfg)
    kling_duration = _get_kling_duration(duration)
    
    LOG.info("Generating transition via %s (duration=%ss)...", endpoint, kling_duration)
    
    # Prepare the request arguments
    arguments = {
        "prompt": prompt,
        "image_url": start_url,
        "tail_image": end_url,
        "duration": kling_duration,
        "aspect_ratio": "16:9",
    }
    
    # Call fal.ai API using subscribe (handles polling for completion)
    loop = asyncio.get_event_loop()
    
    def run_fal_subscribe():
        return fal_client.subscribe(
            endpoint,
            arguments=arguments,
            with_logs=True,
        )
    
    result = await loop.run_in_executor(None, run_fal_subscribe)
    
    # Extract video URL from result
    video_url = result.get("video", {}).get("url")
    if not video_url:
        raise RuntimeError(f"No video URL in fal.ai response: {result}")
    
    LOG.info("Downloading generated transition video from %s", video_url)
    
    # Download the video file
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(video_url)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    
    LOG.info("Transition video saved to %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Batch generation for slides + transitions
# ---------------------------------------------------------------------------


async def generate_deck_media(
    deck: Deck,
    slide_prompts: Sequence[str],
    transition_prompts: Sequence[str],
    cfg: AppConfig,
    project_root: Path,
    deck_path: Path,
    include_transitions: bool = True,
    progress_cb: Optional[Callable[[ExportProgress], None]] = None,
) -> Deck:
    """Generate slide images (and optional transitions) and save the updated deck."""

    slide_count = deck.slide_count()
    if len(slide_prompts) != slide_count:
        raise ValueError("slide_prompts length must match number of slides in deck")
    if include_transitions and len(transition_prompts) != max(slide_count - 1, 0):
        raise ValueError("transition_prompts length must match slide_count - 1")

    slides_dir = project_root / "slides"
    transitions_dir = project_root / "transitions"
    slides_dir.mkdir(parents=True, exist_ok=True)
    transitions_dir.mkdir(parents=True, exist_ok=True)

    updated_slides: List[Slide] = []
    slide_paths: List[Path] = []

    for idx, slide in enumerate(deck.slides):
        prompt = slide_prompts[idx]
        dest = slides_dir / f"slide{idx + 1}.png"

        if idx == 0:
            await generate_slide_image(prompt=prompt, cfg=cfg, dest=dest, width=1920, height=1080)
        else:
            ref_img = slide_paths[-1]
            await generate_slide_image_edit(
                prompt=prompt,
                reference_image=ref_img,
                cfg=cfg,
                dest=dest,
                width=1920,
                height=1080,
            )

        slide_paths.append(dest)
        updated_slides.append(Slide(image=str(dest.relative_to(project_root)), transition=None))

        if progress_cb:
            frac = 0.05 + 0.6 * (idx + 1) / max(slide_count, 1)
            progress_cb(ExportProgress(message=f"Generated slide {idx + 1}/{slide_count}", fraction=frac))

    if include_transitions and slide_count > 1:
        for idx in range(slide_count - 1):
            trans_prompt = transition_prompts[idx] if idx < len(transition_prompts) else ""
            trans_dest = transitions_dir / f"slide{idx + 1}_to_slide{idx + 2}.mp4"
            await generate_transition_clip(
                prompt=trans_prompt,
                first_frame=slide_paths[idx],
                last_frame=slide_paths[idx + 1],
                cfg=cfg,
                dest=trans_dest,
                duration=cfg.default_transition_duration,
            )
            updated_slides[idx].transition = str(trans_dest.relative_to(project_root))

            if progress_cb:
                frac = 0.65 + 0.3 * (idx + 1) / max(slide_count - 1, 1)
                progress_cb(
                    ExportProgress(
                        message=f"Generated transition {idx + 1}/{slide_count - 1}", fraction=frac
                    )
                )

    new_deck = Deck(slides=updated_slides)
    await save_deck(new_deck, deck_path)
    if progress_cb:
        progress_cb(ExportProgress(message="Generation complete", fraction=1.0))
    return new_deck


# ---------------------------------------------------------------------------
# Video export pipeline using ffmpeg-python
# ---------------------------------------------------------------------------


@dataclass
class ExportProgress:
    message: str
    fraction: float  # 0.0 - 1.0


async def export_deck_to_mp4(
    deck: Deck,
    settings: ExportSettings,
    project_root: Path,
    progress_cb: Optional[Callable[[ExportProgress], None]] = None,
) -> Path:
    """Export a deck to a single MP4 file.

    Each slide image is turned into a video segment of `slide_duration`
    seconds, optionally followed by the transition clip. All segments are
    concatenated into a single H.264 MP4 with the configured resolution and
    a constant frame rate.
    """

    loop = asyncio.get_event_loop()
    out_path = Path(settings.output_path)

    def _run_export() -> Path:
        return _export_deck_to_mp4_sync(deck, settings, project_root, progress_cb)

    return await loop.run_in_executor(None, _run_export)


def _export_deck_to_mp4_sync(
    deck: Deck,
    settings: ExportSettings,
    project_root: Path,
    progress_cb: Optional[Callable[[ExportProgress], None]] = None,
) -> Path:
    width, height = settings.width, settings.height
    slide_duration = settings.slide_duration
    transition_duration = settings.transition_duration
    include_transitions = settings.include_transitions
    frame_rate = 30

    if progress_cb:
        progress_cb(ExportProgress(message="Preparing ffmpeg graph...", fraction=0.05))

    streams: List = []
    slide_count = deck.slide_count()

    for idx, slide in enumerate(deck.slides):
        # Slide image -> video segment
        slide_path = _resolve_path(slide.image, project_root)
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide image not found: {slide_path}")

        slide_stream = _make_slide_stream(slide_path, width, height, frame_rate, slide_duration)
        streams.append(slide_stream)

        # Transition or fallback after this slide (except after last slide)
        if idx < slide_count - 1:
            next_slide_path = _resolve_path(deck.slides[idx + 1].image, project_root)

            if include_transitions and slide.transition:
                trans_path = _resolve_path(slide.transition, project_root)
                if not trans_path.exists():
                    raise FileNotFoundError(f"Transition clip not found: {trans_path}")
                trans_stream = _make_transition_stream(
                    trans_path, width, height, frame_rate, transition_duration
                )
                streams.append(trans_stream)
            elif settings.no_transition_behavior == "fade":
                fade_stream = _make_crossfade_stream(
                    slide_path, next_slide_path, width, height, frame_rate, transition_duration
                )
                streams.append(fade_stream)

        if progress_cb:
            frac = 0.05 + 0.7 * (idx + 1) / max(slide_count, 1)
            progress_cb(ExportProgress(message=f"Prepared slide {idx + 1}/{slide_count}", fraction=frac))

    if not streams:
        raise ValueError("Deck has no slides to export")

    if progress_cb:
        progress_cb(ExportProgress(message="Concatenating segments...", fraction=0.8))

    concat_stream = ffmpeg.concat(*streams, v=1, a=0)
    out_path = Path(settings.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_kwargs = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "r": frame_rate,
        "movflags": "+faststart",
    }

    try:
        (
            concat_stream.output(str(out_path), **output_kwargs)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:  # type: ignore[assignment]
        stderr = getattr(exc, "stderr", b"")
        decoded = stderr.decode("utf-8", errors="ignore") if isinstance(stderr, (bytes, bytearray)) else str(stderr)
        LOG.exception("ffmpeg export failed: %s", decoded)
        raise RuntimeError(f"ffmpeg export failed: {decoded}") from exc

    if progress_cb:
        progress_cb(ExportProgress(message="Export complete", fraction=1.0))

    return out_path


def _make_slide_stream(
    slide_path: Path, width: int, height: int, frame_rate: int, duration: float
):
    """Create a video stream from a still image."""
    img_input = ffmpeg.input(str(slide_path), loop=1, framerate=frame_rate)
    img_stream = (
        img_input.filter("scale", width, height)
        .filter("fps", fps=frame_rate)
        .filter("format", "yuv420p")
    )
    return img_stream.filter("trim", duration=duration).setpts("PTS-STARTPTS")


def _make_transition_stream(
    trans_path: Path, width: int, height: int, frame_rate: int, duration: float
):
    """Normalize a transition clip to the target resolution and duration."""
    trans_input = ffmpeg.input(str(trans_path))
    trans_stream = (
        trans_input.filter("scale", width, height)
        .filter("fps", fps=frame_rate)
        .filter("format", "yuv420p")
    )
    return trans_stream.filter("trim", duration=duration).setpts("PTS-STARTPTS")


def _make_crossfade_stream(
    slide_a: Path, slide_b: Path, width: int, height: int, frame_rate: int, duration: float
):
    """Create a generated crossfade transition between two still slides."""
    a = ffmpeg.input(str(slide_a), loop=1, framerate=frame_rate)
    b = ffmpeg.input(str(slide_b), loop=1, framerate=frame_rate)
    a = a.filter("scale", width, height).filter("fps", fps=frame_rate).filter("format", "yuv420p")
    b = b.filter("scale", width, height).filter("fps", fps=frame_rate).filter("format", "yuv420p")
    a = a.filter("trim", duration=duration).setpts("PTS-STARTPTS")
    b = b.filter("trim", duration=duration).setpts("PTS-STARTPTS")
    return ffmpeg.filter([a, b], "xfade", transition="fade", duration=duration, offset=0)


def _resolve_path(path_str: str, project_root: Path) -> Path:
    """Resolve a possibly-relative path against the project root.

    HTTP(S) URLs are passed through unchanged as strings, but for export
    purposes we expect all assets to be local. The calling code should
    normalize remote assets via `download_asset` before invoking export.
    """
    if "://" in path_str:
        # For ffmpeg we need a local path; remote URLs should have been
        # downloaded beforehand.
        raise ValueError(f"Remote URLs are not supported for export: {path_str}")
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


# ---------------------------------------------------------------------------
# Conversation orchestration
# ---------------------------------------------------------------------------


@dataclass
class ConversationState:
    phase: ConversationPhase = ConversationPhase.ONBOARDING
    messages: List[ChatMessage] = field(default_factory=list)

    def add_user_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def add_assistant_message(self, message: ChatMessage) -> None:
        self.messages.append(message)


class ConversationManager:
    """High-level orchestration around the LLM service.

    This class tracks the design phase and delegates actual text generation
    to the LLMService.
    """

    def __init__(self, llm: LLMService):
        self.llm = llm
        self.state = ConversationState()

    async def handle_user_message(self, message: ChatMessage) -> ChatMessage:
        self.state.add_user_message(message)
        use_search = self._should_use_web_search(message.content)
        reply = await self.llm.chat(self.state.messages, self.state.phase, allow_web_search=use_search)
        self.state.add_assistant_message(reply)
        self._maybe_advance_phase(message.content)
        return reply

    def _maybe_advance_phase(self, content: str) -> None:
        lowered = content.lower()
        if self.state.phase is ConversationPhase.ONBOARDING and "outline" in lowered:
            self.state.phase = ConversationPhase.OUTLINE
        elif self.state.phase is ConversationPhase.OUTLINE and (
            "looks good" in lowered or "approve" in lowered or "proceed" in lowered
        ):
            self.state.phase = ConversationPhase.VISUAL
        elif self.state.phase is ConversationPhase.VISUAL and (
            "ready" in lowered or "generate" in lowered or "export" in lowered
        ):
            self.state.phase = ConversationPhase.GENERATION_READY

    def _should_use_web_search(self, content: str) -> bool:
        """Lightweight detection to enable research mode."""
        lowered = content.lower()
        triggers = ("research", "latest", "current data", "up to date", "news", "recent")
        return any(t in lowered for t in triggers)
