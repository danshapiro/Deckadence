# Deckadence

Deckadence designs and plays highly visual slide decks backed by LLM-assisted
conversation. It includes:

- A NiceGUI web UI for conversation, slide preview, and playback
- A JSON-based deck schema describing slide images and optional transitions
- An ffmpeg-based MP4 export pipeline
- Configurable API keys for LiteLLM and media backends

## Installation

```bash
pip install -e .
```

## Usage

```bash
deckadence                               # Runs on http://localhost:8080
deckadence --port 3000                   # Custom port
deckadence --config ./config.json        # Custom config path
deckadence --project ./my-deck/          # Project root or deck JSON
deckadence --config ./cfg.json --no-browser
```

On first launch, if the LiteLLM API key is missing, the Settings dialog
opens automatically so you can paste your credentials.
