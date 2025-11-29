# Deckadence

Deckadence creates stunning visual slide decks with AI-generated images and animated transitions. Generate professional presentations from a simple topic prompt, then export to MP4 video.

> **Note:**  
> Besides this paragraph, this README is AI generated and probably wrong. The code is barely tested and usage may get expensive quickly—use with caution. But it's also pretty awesome.

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

Deckadence provides a full-featured command line interface for creating AI-powered visual presentations.

```bash
deckadence [COMMAND] [OPTIONS]
deckadence --help           # Show all commands
deckadence --version        # Show version
```

---

### Quick Start Examples

```bash
# 1. Generate a complete deck from a topic (simplest workflow)
deckadence generate --topic "The Future of Space Travel" --slides 5

# 2. Export to video
deckadence export output -o presentation.mp4
```

---

### Commands Overview

| Command | Description |
|---------|-------------|
| `generate` | Generate slides and transitions using AI |
| `export` | Export deck to MP4 video |
| `info` | Display deck information |
| `config` | Manage settings and API keys |

---

### `deckadence generate`

Generate slide images and transition videos with AI. Two modes available:

#### Topic Mode (Recommended for new decks)

Let the AI design your entire presentation from a topic:

```bash
deckadence generate --topic "Your Topic Here" [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--topic` | `-t` | — | Topic for AI to design a presentation around |
| `--slides` | `-n` | 5 | Number of slides to generate |
| `--output` | `-o` | output | Output directory for generated project |
| `--no-transitions` | — | false | Skip generating transition videos (faster, cheaper) |
| `--image-model` | `-I` | nano_banana_pro | `nano_banana` (faster, ~$0.04) or `nano_banana_pro` (quality, ~$0.14) |
| `--kling-model` | `-K` | pro | `standard` (720p, ~$0.065/sec) or `pro` (1080p, ~$0.095/sec) |
| `--config` | `-c` | — | Path to config file |
| `--verbose` | `-v` | false | Enable verbose logging |

#### Prompts Mode (For fine-grained control)

Use explicit prompts from a JSON file:

```bash
deckadence generate --project ./mydeck --prompts prompts.json
```

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-P` | Path to existing project directory |
| `--prompts` | `-p` | JSON file with slide_prompts and transition_prompts arrays |

#### Examples

```bash
# Generate a 3-slide deck on AI
deckadence generate -t "Artificial Intelligence in Healthcare" -n 3

# Full control with all options
deckadence generate \
  --topic "Quantum Computing Explained" \
  --slides 6 \
  --output quantum-deck \
  --image-model nano_banana_pro \
  --kling-model pro \
  --verbose
```

#### Prompts JSON Format

When using `--prompts`, provide a JSON file:

```json
{
  "slide_prompts": [
    "A dramatic wide shot of a city skyline at sunset, golden hour lighting, cinematic composition",
    "Close-up of hands typing on a futuristic holographic keyboard, blue glow, tech aesthetic",
    "Aerial view of a winding river through autumn forest, vibrant red and orange leaves"
  ],
  "transition_prompts": [
    "The sunset fades as city lights flicker on, smooth zoom transition",
    "The holographic display dissolves into particles that reform as leaves falling"
  ]
}
```

---

### `deckadence export`

Export a deck to MP4 video using ffmpeg.

```bash
deckadence export <project> [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | deckadence_export.mp4 | Output video file path |
| `--width` | `-W` | 1920 | Video width in pixels |
| `--height` | `-H` | 1080 | Video height in pixels |
| `--slide-duration` | `-s` | 5.0 | Duration each slide is shown (seconds) |
| `--transition-duration` | `-t` | 1.0 | Playback speed for transitions (seconds) |
| `--no-transitions` | — | false | Exclude transition videos from export |
| `--config` | `-c` | — | Path to config file |
| `--verbose` | `-v` | false | Enable verbose logging |

**Note:** The `--transition-duration` option controls playback speed, not cropping. If a generated transition is 5 seconds and you set `-t 2.5`, the video plays at 2x speed. If you set `-t 10`, it plays at 0.5x speed. When a slide has no transition video, there is a hard cut to the next slide.

#### Examples

```bash
# Basic export
deckadence export ./my-deck -o presentation.mp4

# Full control: 4K, custom timing, faster transitions
deckadence export ./my-deck \
  --output 4k-export.mp4 \
  --width 3840 \
  --height 2160 \
  --slide-duration 4 \
  --transition-duration 2.5
```

---

### `deckadence info`

Display information about a deck project.

```bash
deckadence info [project] [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Show detailed slide-by-slide information with file status |

#### Examples

```bash
# Show info for current directory
deckadence info

# Show info for a specific project
deckadence info ./my-deck

# Show detailed slide-by-slide information
deckadence info ./my-deck --verbose
```

#### Example Output

```
Deckadence Deck
  Project: /path/to/my-deck
  Deck file: /path/to/my-deck/deck.json
  Slides: 5
  Transitions: 4
```

With `--verbose`:

```
Slides:
  #   Image                  Transition                         Status
  1   slides/slide1.png      transitions/slide1_to_slide2.mp4   OK / OK
  2   slides/slide2.png      transitions/slide2_to_slide3.mp4   OK / OK
  3   slides/slide3.png      None                               OK
```

---

### `deckadence config`

Manage configuration settings and API keys.

#### Show Current Configuration

```bash
deckadence config show
```

Displays all settings with masked API keys and shows whether values come from environment variables or the config file.

#### Set a Configuration Value

```bash
deckadence config set <key> <value>
```

| Key | Description | Valid Values |
|-----|-------------|--------------|
| `gemini-api-key` | Google Gemini API key | Your API key |
| `fal-api-key` | fal.ai API key (for transitions) | Your API key |
| `image-model` | Default image generation model | `nano_banana`, `nano_banana_pro` |
| `kling-model` | Default video transition model | `standard` (720p), `pro` (1080p) |
| `resolution` | Default export resolution | e.g., `1920x1080`, `1280x720` |
| `slide-duration` | Default slide duration | Seconds (e.g., `5.0`) |
| `transition-duration` | Default transition playback speed | Seconds (e.g., `1.0`) |

#### Show Config File Path

```bash
deckadence config path
```

#### Examples

```bash
# Set API keys
deckadence config set gemini-api-key YOUR_GEMINI_KEY
deckadence config set fal-api-key YOUR_FAL_KEY

# Use the faster/cheaper image model by default
deckadence config set image-model nano_banana

# Use standard Kling model (720p, more affordable)
deckadence config set kling-model standard

# Set default timing
deckadence config set slide-duration 6.0
deckadence config set transition-duration 1.0

# Set default resolution
deckadence config set resolution 1920x1080

# View current configuration
deckadence config show
```

#### Configuration Priority

1. Command-line options (highest priority)
2. Environment variables (`GEMINI_API_KEY`, `FAL_KEY`)
3. Config file (`~/.deckadence/config.json`)
4. Default values

---

### Complete Workflow Examples

#### Basic: Quick Presentation from Topic

```bash
# Generate a 5-slide deck and export to video
deckadence generate --topic "Introduction to Machine Learning"
deckadence export output -o ml-intro.mp4
```

#### Comprehensive: Custom Prompts with Premium Models

```bash
# 1. Create a prompts.json with your custom descriptions
cat > prompts.json << 'EOF'
{
  "slide_prompts": [
    "Dramatic wide shot of a futuristic city at sunset, golden hour, cinematic",
    "Close-up of hands on holographic keyboard, blue glow, tech aesthetic",
    "Aerial view of data streams flowing through a digital landscape"
  ],
  "transition_prompts": [
    "City lights blur and transform into digital particles",
    "Holographic display dissolves into flowing data streams"
  ]
}
EOF

# 2. Generate with premium models and custom prompts
deckadence generate \
  --topic "placeholder" \
  --slides 3 \
  --image-model nano_banana_pro \
  --kling-model pro \
  --output brand-video \
  --verbose

# (Or use existing prompts: deckadence generate --project brand-video --prompts prompts.json)

# 3. Export 4K video with custom timing
deckadence export brand-video \
  -o brand-video-4k.mp4 \
  -W 3840 -H 2160 \
  --slide-duration 5 \
  --transition-duration 2.5
```

---

### Model Options and Pricing

#### Image Models

| Model | Resolution | Speed | Cost per Image |
|-------|-----------|-------|----------------|
| `nano_banana` | 1024x1024 | Faster | ~$0.04 |
| `nano_banana_pro` | 1024x1024 | Slower | ~$0.14 |

#### Kling Video Models

| Model | Resolution | Quality | Cost per Second |
|-------|-----------|---------|-----------------|
| `standard` | 720p | Good | ~$0.065 |
| `pro` | 1080p | Best | ~$0.095 |

*Prices are approximate and may vary.*

---

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key (required) |
| `FAL_KEY` | fal.ai API key (required for transitions) |

Set these in your shell profile or before running commands:

```bash
export GEMINI_API_KEY="your-key-here"
export FAL_KEY="your-fal-key-here"
```

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

## License

MIT License - see [LICENSE](LICENSE) for details.
