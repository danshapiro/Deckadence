# Deckadence

Deckadence creates stunning visual slide decks with AI-generated images and animated transitions. Generate professional presentations from a simple topic prompt, then export to MP4 video.

> **Note:**  
> Besides this paragraph, this README is AI generated and probably wrong. The code is barely tested, the UI isn't tested at all, and usage may get expensive quickly—use with caution. But it's also pretty awesome.

## Features

- AI-powered slide image generation via Google Gemini
- Animated transitions between slides via Kling 2.5 (fal.ai)
- Export decks to MP4 video with ffmpeg
- JSON-based deck schema for easy editing
- Full CLI for scripting and automation

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.10+
- ffmpeg (must be in PATH for video export)
- API keys:
  - `GEMINI_API_KEY` - Google Gemini API key (required)
  - `FAL_KEY` - fal.ai API key (required for transitions)

## CLI Reference

Deckadence provides a full-featured command line interface.

### Quick Start

```bash
# Generate a 5-slide deck from a topic
deckadence generate --topic "The Future of AI" --slides 5

# Export to video
deckadence export output -o presentation.mp4

# View deck info
deckadence info output
```

### Commands

#### `deckadence init`

Initialize a new project directory with placeholder files.

```bash
deckadence init <directory> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--slides`, `-n` | 5 | Number of placeholder slides to create |
| `--force`, `-f` | false | Overwrite existing deck.json |

Creates:
- `deck.json` - Deck definition
- `prompts.json` - Template for slide/transition prompts
- `slides/` - Directory for generated images
- `transitions/` - Directory for generated videos
- `_uploads/` - Directory for user assets

#### `deckadence generate`

Generate slide images and transition videos with AI.

**Topic mode** - Let AI design the presentation:

```bash
deckadence generate --topic "Climate Change Solutions" --slides 8
```

**Prompts mode** - Use explicit prompts from a JSON file:

```bash
deckadence generate --project ./mydeck --prompts prompts.json
```

| Option | Description |
|--------|-------------|
| `--topic`, `-t` | Topic for AI to design a presentation around |
| `--project`, `-P` | Path to existing project directory |
| `--slides`, `-n` | Number of slides (used with --topic, default: 5) |
| `--prompts`, `-p` | JSON file with slide_prompts and transition_prompts arrays |
| `--output`, `-o` | Output directory for generated project (default: output) |
| `--no-transitions` | Skip generating transition videos |
| `--config`, `-c` | Path to config file |
| `--verbose`, `-v` | Enable verbose logging |

**Prompts JSON format:**

```json
{
  "slide_prompts": [
    "A dramatic wide shot of a city skyline at sunset...",
    "Close-up of hands typing on a futuristic keyboard..."
  ],
  "transition_prompts": [
    "The sunset fades as city lights flicker on..."
  ]
}
```

#### `deckadence export`

Export a deck to MP4 video.

```bash
deckadence export <project> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | deckadence_export.mp4 | Output video file path |
| `--width`, `-W` | 1920 | Video width in pixels |
| `--height`, `-H` | 1080 | Video height in pixels |
| `--slide-duration`, `-s` | 5.0 | Duration per slide in seconds |
| `--transition-duration`, `-t` | 1.0 | Duration per transition in seconds |
| `--no-transitions` | false | Exclude transition videos from export |
| `--fallback`, `-f` | cut | Behavior when no transition exists: `cut` or `fade` |

**Examples:**

```bash
# Export at 720p with 3-second slides
deckadence export ./mydeck -o output.mp4 -W 1280 -H 720 -s 3

# Export without transitions
deckadence export ./mydeck --no-transitions

# Export with crossfade fallback
deckadence export ./mydeck --fallback fade
```

#### `deckadence info`

Display information about a deck.

```bash
deckadence info [project] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Show detailed slide-by-slide information |

**Example output:**

```
Deckadence Deck
├── Project: /path/to/project
├── Deck file: /path/to/project/deck.json
├── Slides: 5
└── Transitions: 4
```

#### `deckadence config`

Manage configuration settings.

**Show current configuration:**

```bash
deckadence config show
```

**Set a configuration value:**

```bash
deckadence config set <key> <value>
```

Available keys:

| Key | Description |
|-----|-------------|
| `gemini-api-key` | Google Gemini API key |
| `fal-api-key` | fal.ai API key |
| `kling-model` | `standard` (720p) or `pro` (1080p) |
| `resolution` | Default export resolution (e.g., `1920x1080`) |
| `slide-duration` | Default slide duration in seconds |
| `transition-duration` | Default transition duration in seconds |
| `no-transition-behavior` | `cut` or `fade` |

**Show config file path:**

```bash
deckadence config path
```

Configuration is stored in `~/.deckadence/config.json`. Environment variables (`GEMINI_API_KEY`, `FAL_KEY`) override config file values.

#### `deckadence serve`

Launch the web UI server (under development).

```bash
deckadence serve [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--port`, `-p` | 8080 | Port for the web server |
| `--host`, `-h` | 127.0.0.1 | Host address |
| `--project`, `-P` | none | Project to open on launch |
| `--no-browser` | false | Don't auto-open browser |

## Deck Schema

Decks are defined in `deck.json`:

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

- `image` (required): Path to slide image (PNG or JPG)
- `transition` (optional): Path to transition video to next slide

## Web UI

The web UI (`deckadence serve`) is currently under development. It provides:

- Interactive slide viewer with playback controls
- AI chat assistant for deck design
- Settings management
- Export dialog

## License

MIT License - see [LICENSE](LICENSE) for details.
