**Name:** Deckadence

**Purpose:**
Deckadence designs and produces highly visual slide decks and optional stitched videos. Each deck consists of static 2K slide images and optional animated transitions between them. The system includes a built-in slide player for interactive preview and a video export pipeline that produces MP4 files at user-selected resolutions.

---

## 2. Output Specification

### 2.1 Slides JSON Schema

```json
{
  "slides": [
    {
      "image": "slides/slide1.png",
      "transition": "transitions/slide1_to_slide2.mp4"
    },
    {
      "image": "slides/slide2.png",
      "transition": "transitions/slide2_to_slide3.mp4"
    },
    {
      "image": "slides/slide3.png"
    }
  ]
}
```

* `image` (required): Path or URL to a 2K slide image (PNG or JPG). 2K means 1920×1080 or 2048×1152.
* `transition` (optional): Path or URL to a video transitioning to the next slide. The last slide typically omits this.
* Paths may be local file paths or HTTP(S) URLs. The system defaults to local paths in local environments.

---

## 3. Conversation & UX

### 3.1 First-Turn Behavior

On first interaction, Deckadence asks about:
* The deck's purpose, audience, tone, and approximate slide count.
* Whether the user wants static slides only or slides plus animated transitions (also exposed as a UI control).
* Visual style preferences (e.g., "minimalist," "corporate," "hand-drawn," "dark sci-fi," or a reference image).
* Whether the user has reference images for style or content.

### 3.2 Deck Design Iteration

Design proceeds in two phases:

* **Outline phase:** Deckadence proposes a numbered list of slide titles and purposes (e.g., "Slide 1: Big vision," "Slide 2: Market landscape"). The user edits or approves.
* **Visual design phase:** For each approved slide, Deckadence provides a detailed visual description suitable for an image generator—describing foreground, background, composition, key elements, and color palette as if for a blind person.

The agent iterates until the user indicates readiness for image generation.

---

## 4. Slide Design Requirements

Slides are highly visual with minimal on-slide text (short titles or labels only). Each slide must stand alone as a compelling image. Descriptions must specify concrete compositions, not generic prompts.

Canonical example:

> "A dark blue gradient background, a central glowing rocket launching upward, subtle grid lines, tiny icons of charts and dollar signs orbiting around it, with a clean sans-serif title 'Go-To-Market Launch Plan' at the top center."

Slides in a deck share style, vibe, texture, and font while differing in layout and scene content to make transitions visually interesting.

---

## 5. Tooling & System Architecture

### 5.1 Text LLM via LiteLLM (Local)

All text LLM interactions use the local LiteLLM library (not a remote server). System and user messages define behavior, with tools like web search enabled via LiteLLM's tool interface.

Reference: `https://docs.litellm.ai/docs/completion/web_search`

For research-driven decks, the LLM formulates search queries, summarizes results into slide outlines, and converts information into visual ideas rather than verbose on-screen text.

Canonical research prompt:

> "Use the web_search tool to gather up-to-date information about [topic]. Summarize the most important points for a 10-slide deck and propose visual metaphors and scenes for each slide, keeping text on slides minimal."

### 5.2 Image and Media Understanding

The agent can submit one or two images plus a natural language prompt to the text LLM through LiteLLM for:
* Describing a reference image's style, color palette, and layout.
* Comparing two slides for style consistency.
* Identifying elements to preserve in variations.

Canonical prompts:

> "Analyze this image and describe its illustration style, dominant colors, and font characteristics in detail."

> "Compare these two slide images and explain whether they share the same style, vibe, texture, and font; if not, suggest specific changes to make them match."

> "Using this slide image as a reference, describe the elements that define its style so we can generate further slides in the exact same style, vibe, texture, and font."

### 5.3 Visual Generation – Nano Banana Pro

* First/reference slides are generated from text prompts using Nano Banana Pro Text-to-Image.
* Subsequent slides use Nano Banana Pro Edit with a reference image and style-preserving prompt.
* All images are 2K resolution with minimal on-image text.

Canonical edit prompts:

> "In the exact same style, vibe, texture and font as this image, generate a new 2K slide that shows [detailed description of the new scene]."

> "In the EXACT style, vibe and texture as this image, generate a new 2K slide showing [desired visual], but remove all the other stuff from the image; only [desired visual] should be visible in this style."

For multiple style references (up to four images):

> "Using these reference images to define the style, vibe, texture, and font, generate a new 2K slide that shows [scene description] with a layout suitable for a presentation slide."

### 5.4 Transitions – Kling 2.5

For each adjacent slide pair, Deckadence passes slide n as the first frame and slide n+1 as the last frame to Kling 2.5. The prompt describes how elements evolve between slides.

Canonical transition prompts:

> "Starting from the first frame slide, smoothly transform the central rocket into the upward bars of the bar chart in the last frame slide, while the background grid lines morph into a subtle graph behind the chart."

> "From the first frame slide, slowly fade the city skyline into night as the lights morph into glowing data points that rearrange into the funnel diagram visible in the last frame slide."

> "From the first frame slide, have the scattered icons swirl toward the center and merge into the single bold icon featured in the last frame slide, keeping the same color palette and illustration style throughout."

### 5.5 Video Assembly and Export

The export panel collects:
* Desired resolution (e.g., 1280×720, 1920×1080, or custom).
* Per-slide display duration in seconds.
* Transition duration in seconds (if applicable).

Deckadence confirms before creating the video, then uses ffmpeg to:
* Convert each slide image into a video segment at the chosen resolution and duration.
* Insert transition clips between corresponding slide segments.
* Concatenate all segments into a single MP4 with consistent resolution and frame rate.

Canonical ffmpeg concept:

> "Generate an MP4 at [width]x[height] resolution that plays each slide image for [slide_duration] seconds, with the corresponding transition clip between slides where available, using a constant frame rate and reasonable bitrate."

Video models and ffmpeg are accessed directly from the host environment, not through LiteLLM.

---

## 6. User Interface (NiceGUI)

The UI is built with NiceGUI, providing a reactive web interface at `localhost:8080`.

### 6.1 Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Deckadence                            [Settings] [Export]      │
├───────────────────────────────────┬─────────────────────────────┤
│                                   │                             │
│                                   │      Chat / Conversation    │
│         Slide Viewer              │                             │
│      (image or video)             │   - LLM messages            │
│                                   │   - User input              │
│                                   │   - Mode toggle             │
│                                   │     (slides only /          │
│                                   │      slides + transitions)  │
├───────────────────────────────────┤                             │
│  [⏮] [⏪] [⏸/▶] [⏩] [⏭]  1/12   │                             │
├───────────────────────────────────┤                             │
│  ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢ ▢        │                             │
│  (thumbnail strip, scrollable)    │                             │
└───────────────────────────────────┴─────────────────────────────┘
```

### 6.2 Components

**Header bar:**
* App title
* Settings button → opens API key entry dialog
* Export button → opens export panel (resolution, durations, confirm)

**Slide viewer (left panel, ~60% width):**
* Displays the current slide image via `ui.image()`
* During transitions, swaps to `ui.video()` for the transition clip, then displays next slide
* Maintains 16:9 aspect ratio with letterboxing if needed

**Transport controls:**
* First / Previous / Play-Pause / Next / Last buttons
* Slide counter showing current position (e.g., "3 / 12")
* Clicking a thumbnail jumps to that slide

**Thumbnail strip:**
* Horizontal scrollable row of slide thumbnails
* Current slide highlighted
* Thumbnails generated on deck load (downscaled from 2K images)

**Chat panel (right panel, ~40% width):**
* Scrollable message history showing LLM and user messages
* Text input at bottom for user messages
* Mode toggle (radio buttons): "Slides only" / "Slides + transitions"
* Displays outline and visual descriptions during design phase
* Shows generation progress with `ui.spinner()` and status text

### 6.3 Playback Behavior

* **Play:** Auto-advance through slides. Display slide for configured duration → play transition video (if present) → next slide.
* **Pause:** Freeze on current frame. If mid-transition, pause the video.
* **Stop:** Return to slide 1, paused.
* **Next/Previous:** Immediate jump, skipping any in-progress transition.
* **No transition defined:** Cross-fade or hard cut (configurable default).

### 6.4 Export Panel (Dialog)

Opened via Export button. Contains:
* Resolution dropdown (720p, 1080p, 1440p, custom)
* Slide duration input (seconds per slide)
* Transition duration input (seconds, if applicable)
* Confirm / Cancel buttons
* Progress bar and status text during export

### 6.5 Settings Dialog

Opened via Settings button. Contains:
* Input fields for each API key (LiteLLM, Nano Banana Pro, Kling 2.5)
* Keys masked by default, with show/hide toggle
* Default export settings (resolution, slide duration, transition duration)
* Default playback settings (no-transition behavior: cut vs fade)
* Save / Cancel buttons
* Validation feedback (key format, test call if feasible)

All settings are persisted to a local JSON config file (e.g., `~/.deckadence/config.json`) and loaded on startup.

---

## 7. Configuration and Credentials

All configuration is managed through the NiceGUI Settings dialog and persisted to `~/.deckadence/config.json`.

**Stored settings:**
* API keys (LiteLLM, Nano Banana Pro, Kling 2.5)
* Default export resolution and timing
* Playback preferences

**Startup behavior:**
* Load config file if present; environment variables override file values for API keys.
* If required API keys are missing, the Settings dialog opens automatically on first launch.
* Keys are validated on save; invalid keys show an error and are not persisted.

---

## 8. Error Handling

* **Image generation failure:** Report the failure with the visual description; offer to retry, tweak the description, or skip.
* **Transition failure:** Report which transition failed; offer to retry or skip (skipped transitions are simply omitted from the JSON).
* **Video export failure:** Surface the ffmpeg error in readable form; offer to retry with adjusted parameters (e.g., lower resolution).

The JSON deck always remains in a consistent, working state.

---

## 9. Command Line Options

The app is launched via `python -m deckadence` or `deckadence` (if installed). Options control startup behavior; all other configuration happens in the UI.

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `8080` | Port for the NiceGUI web server |
| `--host` | `127.0.0.1` | Host address (`0.0.0.0` to allow external access) |
| `--config` | `~/.deckadence/config.json` | Path to config file |
| `--project` | (none) | Path to a project directory or deck JSON to open on launch |
| `--no-browser` | (flag) | Don't auto-open browser on startup |

Examples:

```bash
deckadence                              # default: localhost:8080, opens browser
deckadence --port 3000                  # run on port 3000
deckadence --project ./my-deck/         # open existing project
deckadence --config ./custom-config.json --no-browser
```

---

## 10. Python libraries to use

* `litellm` — Unified LLM API for text generation, image understanding, and web search tool calls.
* `pydantic` + `pydantic-settings` — Data models for Deck/Slide schemas with validation; config management with environment variable override support.
* `httpx` — Async HTTP client for Nano Banana Pro and Kling 2.5 API calls. Supports concurrent requests.
* `Pillow` — Image loading, format conversion, and resizing for resolution normalization.
* `ffmpeg-python` — Pythonic wrapper for ffmpeg. Cleaner than subprocess for building complex filter graphs.
* `nicegui` — Reactive web UI. Handles the entire interface including settings, player, chat, and export.
* `aiofiles` — Async file I/O for non-blocking reads/writes during batch operations.
* `tenacity` — Retry logic with exponential backoff for flaky API calls (image/video generation).
* `click` — Command line option parsing for startup configuration.
