from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from ruamel.yaml import YAML

DEFAULT_PROMPTS_PATH = Path(__file__).with_name("prompts.yaml")


@lru_cache(maxsize=2)
def load_prompts(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load prompts from a YAML file using ruamel.yaml.

    Falls back to the bundled prompts.yaml if no path is provided.
    """
    yaml = YAML(typ="safe")
    file_path = Path(path) if path else DEFAULT_PROMPTS_PATH
    with file_path.open("r", encoding="utf-8") as f:
        data = yaml.load(f) or {}
    return data


def get_prompt(prompts: Dict[str, Any], category: str, key: str) -> str:
    """Retrieve a prompt string from the loaded prompts dictionary or fail loudly."""
    if not prompts:
        raise KeyError("Prompt catalog is empty; cannot fetch any prompts.")
    try:
        value = prompts[category][key]
    except KeyError as exc:
        raise KeyError(f"Missing prompt entry '{category}.{key}' in prompt catalog.") from exc

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Prompt entry '{category}.{key}' is empty in prompt catalog.")
    return text
