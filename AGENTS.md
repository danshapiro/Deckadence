# Agent Guidelines for Deckadence

## Development Environment

- **Use uv** for Python package management and virtual environments

## Testing Requirements

- **Test all changes end-to-end with the command line** before committing
- Run `python -m deckadence --help` to verify CLI works
- Test specific commands: `export`, `generate`, `info`, `init`, `config`

## Style Guidelines

- **Don't use emoji** in code, logs, or user-facing messages in the codebase
- Keep CLI output clean and professional
- Use Rich library for formatted terminal output (tables, progress bars, panels)

## CLI Commands Reference

```bash
# Initialize a project
python -m deckadence init <directory> --slides <n>

# View project info
python -m deckadence info <project>

# Generate media
python -m deckadence generate <project> --prompts prompts.json

# Export to video
python -m deckadence export <project> -o output.mp4

# Configure
python -m deckadence config show
python -m deckadence config set <key> <value>
```

