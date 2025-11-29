from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.deckadence/config.json")


class FileConfig(BaseModel):
    """Configuration stored on disk.

    Environment variables may override individual values at runtime, but are
    never written back to disk.
    """

    lite_llm_model: str = Field(
        default="gemini/gemini-3-pro-preview",
        description="Default LiteLLM model identifier.",
    )
    gemini_api_key: Optional[str] = Field(
        default=None, description="API key for Google Gemini (used for LLM and image generation)."
    )
    fal_api_key: Optional[str] = Field(
        default=None, description="API key for fal.ai (used for Kling 2.5 video generation)."
    )
    kling_model: str = Field(
        default="pro",
        description="Kling 2.5 model to use: 'standard' (720p) or 'pro' (1080p).",
    )
    image_model: str = Field(
        default="nano_banana_pro",
        description="Image generation model: 'nano_banana' (faster) or 'nano_banana_pro' (higher quality).",
    )

    default_resolution: str = Field(
        default="1920x1080",
        description="Default export resolution in WIDTHxHEIGHT format.",
    )
    default_slide_duration: float = Field(
        default=5.0,
        description="Default slide display duration in seconds.",
    )
    default_transition_duration: float = Field(
        default=1.0,
        description="Default transition duration in seconds.",
    )
    default_no_transition_behavior: str = Field(
        default="cut",
        description="Behavior when no transition video is available: 'cut' or 'fade'.",
    )

    @field_validator("default_no_transition_behavior")
    @classmethod
    def validate_transition_behavior(cls, value: str) -> str:
        if value not in {"cut", "fade"}:
            raise ValueError("default_no_transition_behavior must be 'cut' or 'fade'")
        return value

    @field_validator("kling_model")
    @classmethod
    def validate_kling_model(cls, value: str) -> str:
        if value not in {"standard", "pro"}:
            raise ValueError("kling_model must be 'standard' or 'pro'")
        return value

    @field_validator("image_model")
    @classmethod
    def validate_image_model(cls, value: str) -> str:
        if value not in {"nano_banana", "nano_banana_pro"}:
            raise ValueError("image_model must be 'nano_banana' or 'nano_banana_pro'")
        return value


class EnvConfig(BaseSettings):
    """Environment-driven overrides.

    Uses standard provider environment variable names (GEMINI_API_KEY, FAL_KEY).
    Any non-null values here override those in FileConfig.
    """

    model_config = SettingsConfigDict(extra="ignore")

    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    fal_api_key: Optional[str] = Field(default=None, alias="FAL_KEY")


class AppConfig(FileConfig):
    """Runtime configuration after merging file and env settings."""


class ConfigManager:
    """Load and persist Deckadence configuration.

    The precedence is:

    1. FileConfig loaded from JSON file (if present)
    2. EnvConfig (environment variables) overriding any non-null values
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or DEFAULT_CONFIG_PATH)

    def load(self) -> AppConfig:
        """Load configuration from disk (if present) and merge env overrides."""
        file_data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    file_data = json.load(f)
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("Failed to load config file %s: %s", self.path, exc)

        file_cfg = FileConfig(**file_data)
        env_cfg = EnvConfig()

        merged = file_cfg.model_dump()
        for key, value in env_cfg.model_dump(exclude_unset=True).items():
            if value is not None:
                merged[key] = value

        return AppConfig(**merged)

    def save(self, cfg: AppConfig) -> None:
        """Persist configuration to disk.

        Environment variable overrides are *not* written back to disk.
        """
        self._validate_keys(cfg)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = cfg.model_dump()

        # Remove any values provided by environment variables so that file
        # only contains user-managed configuration.
        env_cfg = EnvConfig()
        env_values = env_cfg.model_dump(exclude_unset=True)
        for key, value in env_values.items():
            if value is not None and key in data:
                data.pop(key, None)

        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _validate_keys(self, cfg: AppConfig) -> None:
        """Lightweight sanity checks for API keys so we fail fast in the UI."""

        def _invalid(value: Optional[str]) -> bool:
            if value is None:
                return False
            trimmed = value.strip()
            if not trimmed:
                return True
            return len(trimmed) < 8  # heuristic: too short to be genuine

        for label, value in [
            ("Gemini", cfg.gemini_api_key),
            ("fal.ai", cfg.fal_api_key),
        ]:
            if label == "Gemini":
                if value is None or _invalid(value):
                    raise ValueError("Gemini API key is required and appears invalid or empty.")
            elif _invalid(value):
                raise ValueError(f"{label} API key appears invalid or empty.")

    def is_missing_required_keys(self, cfg: AppConfig) -> bool:
        """Check if any critical API keys are missing.

        Gemini API key is required for the conversational interface and
        Nano Banana image generation. Kling is optional.
        """
        return not bool(cfg.gemini_api_key)

    def as_redacted_dict(self, cfg: AppConfig) -> Dict[str, Any]:
        """Return a version of the config with secrets redacted for UI display."""
        data = cfg.model_dump()
        for key in list(data.keys()):
            if key.endswith("api_key") and data[key]:
                data[key] = "********"
        return data

    @staticmethod
    def is_key_from_env(key_name: str) -> bool:
        """Check if a specific API key is set via environment variable.
        
        Args:
            key_name: One of 'gemini' or 'fal'
        
        Returns:
            True if the corresponding env var is set
        """
        env_var_map = {
            "gemini": "GEMINI_API_KEY",
            "fal": "FAL_KEY",
        }
        env_var = env_var_map.get(key_name)
        if not env_var:
            return False
        return bool(os.environ.get(env_var))
