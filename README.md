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

# 3. Launch web UI to preview and edit
deckadence serve --project output
```

---

### Commands Overview

| Command | Description |
|---------|-------------|
| `generate` | Generate slides and transitions using AI |
| `export` | Export deck to MP4 video |
| `serve` | Launch the web UI |
| `init` | Create a new empty project |
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
# Generate a 3-slide deck on AI (minimal)
deckadence generate -t "Artificial Intelligence in Healthcare" -n 3

# Generate without transitions (faster, less expensive)
deckadence generate -t "Climate Change" -n 5 --no-transitions

# Use the faster/cheaper image model
deckadence generate -t "Modern Architecture" --image-model nano_banana

# Use standard Kling model (720p, more affordable)
deckadence generate -t "Ocean Wildlife" --kling-model standard

# Output to a specific directory
deckadence generate -t "Startup Pitch" -n 8 -o my-pitch-deck

# Generate from existing prompts file
deckadence generate --project ./my-project --prompts custom-prompts.json

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
| `--transition-duration` | `-t` | 1.0 | Duration of transitions (seconds) |
| `--no-transitions` | — | false | Exclude transition videos from export |
| `--fallback` | `-f` | cut | Behavior when transition is missing: `cut` or `fade` |
| `--config` | `-c` | — | Path to config file |
| `--verbose` | `-v` | false | Enable verbose logging |

#### Examples

```bash
# Basic export
deckadence export ./my-deck -o presentation.mp4

# Export at 720p resolution
deckadence export ./my-deck -o output.mp4 -W 1280 -H 720

# Longer slides, shorter transitions
deckadence export ./my-deck -s 8 -t 0.5 -o slow-paced.mp4

# Export without transitions (slides only)
deckadence export ./my-deck --no-transitions -o slides-only.mp4

# Use crossfade when transition videos are missing
deckadence export ./my-deck --fallback fade

# Export 4K video with custom timing
deckadence export ./my-deck \
  --output 4k-export.mp4 \
  --width 3840 \
  --height 2160 \
  --slide-duration 4 \
  --transition-duration 1.5

# Export from current directory
deckadence export . -o output.mp4
```

---

### `deckadence serve`

Launch the interactive web UI for creating and editing decks.

```bash
deckadence serve [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--port` | `-p` | 8080 | Port for the web server |
| `--host` | `-h` | 127.0.0.1 | Host address (use `0.0.0.0` for network access) |
| `--project` | `-P` | — | Project directory to open on launch |
| `--no-browser` | — | false | Don't automatically open browser |
| `--config` | `-c` | — | Path to config file |
| `--verbose` | `-v` | false | Enable verbose logging |

#### Examples

```bash
# Start with default settings (opens browser automatically)
deckadence serve

# Open with an existing project
deckadence serve --project ./my-deck

# Run on a different port
deckadence serve --port 3000

# Allow access from other devices on the network
deckadence serve --host 0.0.0.0 --port 8080

# Start without opening browser (for remote servers)
deckadence serve --no-browser

# Full example
deckadence serve \
  --project ./my-presentation \
  --port 9000 \
  --host 0.0.0.0 \
  --no-browser \
  --verbose
```

---

### `deckadence init`

Initialize a new empty project with placeholder files.

```bash
deckadence init [directory] [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--slides` | `-n` | 5 | Number of placeholder slides to create |
| `--force` | `-f` | false | Overwrite existing deck.json if present |

#### What Gets Created

```
my-project/
  deck.json         # Deck definition with placeholder slides
  prompts.json      # Template for generation prompts (edit this!)
  slides/           # Directory for generated slide images
  transitions/      # Directory for generated transition videos
  _uploads/         # Directory for user-uploaded assets
```

#### Examples

```bash
# Initialize in default 'output' directory with 5 slides
deckadence init

# Initialize in a specific directory with 8 slides
deckadence init my-presentation --slides 8

# Force overwrite existing project
deckadence init ./existing-project --force

# Create a minimal 3-slide project
deckadence init quick-deck -n 3
```

#### Workflow After Init

```bash
# 1. Initialize project
deckadence init my-project -n 5

# 2. Edit prompts.json with your slide descriptions
# 3. Generate the media
deckadence generate --project my-project

# 4. Preview in web UI
deckadence serve --project my-project

# 5. Export to video
deckadence export my-project -o final.mp4
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
| `transition-duration` | Default transition duration | Seconds (e.g., `1.0`) |
| `no-transition-behavior` | Fallback when no transition exists | `cut`, `fade` |

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
deckadence config set transition-duration 1.5

# Set default resolution
deckadence config set resolution 1920x1080

# Use fade instead of cut for missing transitions
deckadence config set no-transition-behavior fade

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

#### Example 1: Quick Presentation from Topic

```bash
# Generate a 5-slide deck
deckadence generate --topic "Introduction to Machine Learning"

# Export immediately
deckadence export output -o ml-intro.mp4
```

#### Example 2: Budget-Conscious Workflow

```bash
# Use cheaper models and skip transitions
deckadence generate \
  --topic "Quarterly Sales Report" \
  --slides 4 \
  --image-model nano_banana \
  --no-transitions \
  --output sales-deck

# Export as slides-only video with 8-second slides
deckadence export sales-deck -o quarterly.mp4 --slide-duration 8
```

#### Example 3: High-Quality Production

```bash
# Use premium models for best quality
deckadence generate \
  --topic "Product Launch Keynote" \
  --slides 10 \
  --image-model nano_banana_pro \
  --kling-model pro \
  --output keynote

# Preview and refine in web UI
deckadence serve --project keynote

# Export 4K video
deckadence export keynote \
  -o keynote-4k.mp4 \
  -W 3840 -H 2160 \
  --slide-duration 5 \
  --transition-duration 1.5
```

#### Example 4: Custom Prompts Workflow

```bash
# Initialize project structure
deckadence init brand-video --slides 6

# Edit prompts.json with your custom descriptions
# (use your favorite editor)

# Generate from your prompts
deckadence generate --project brand-video

# Preview
deckadence serve --project brand-video

# Export
deckadence export brand-video -o brand-video.mp4
```

#### Example 5: Batch Processing

```bash
# Generate multiple decks
for topic in "AI Ethics" "Cloud Computing" "Cybersecurity Basics"; do
  dir=$(echo "$topic" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')
  deckadence generate --topic "$topic" --slides 5 --output "$dir"
  deckadence export "$dir" -o "${dir}.mp4"
done
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

## Web UI

The web UI (`deckadence serve`) is currently under development. It provides:

- Interactive slide viewer with playback controls
- AI chat assistant for deck design
- Settings management
- Export dialog

## License

MIT License - see [LICENSE](LICENSE) for details.
