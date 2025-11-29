from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, FieldValidationInfo, field_validator


class Slide(BaseModel):
    """A single slide in a deck.

    The `image` and optional `transition` fields are paths or URLs. Paths
    may be absolute or relative to the project directory.
    """

    image: str = Field(..., description="Path or URL to a 2K slide image (PNG or JPG).")
    transition: Optional[str] = Field(
        default=None,
        description="Path or URL to a video transitioning to the *next* slide.",
    )
    fast_transition: bool = Field(
        default=False,
        description="If True, play transition at 2x speed (2.5s instead of 5s).",
    )

    @field_validator("image", "transition")
    @classmethod
    def validate_path_or_url(
        cls, value: Optional[str], info: FieldValidationInfo
    ) -> Optional[str]:
        if value is None:
            return value
        if not isinstance(value, str):
            raise TypeError("Path or URL must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Path or URL cannot be empty")
        lower = value.lower()
        if info.field_name == "image":
            allowed_ext = (".png", ".jpg", ".jpeg")
            if not any(lower.endswith(ext) for ext in allowed_ext):
                raise ValueError("Slides must be PNG or JPG images")
        else:
            allowed_ext = (".mp4", ".mov", ".webm", ".mkv")
            if lower and not any(lower.endswith(ext) for ext in allowed_ext):
                raise ValueError("Transitions must be a video file (mp4, mov, webm, mkv)")
        # We accept both local paths and HTTP(S) URLs; deeper validation is
        # handled at usage time.
        return value


class Deck(BaseModel):
    """A Deckadence deck JSON representation."""

    slides: List[Slide] = Field(default_factory=list)

    def slide_count(self) -> int:
        return len(self.slides)


class ConversationPhase(str, Enum):
    """High-level phases of the design conversation."""

    ONBOARDING = "onboarding"
    OUTLINE = "outline"
    VISUAL = "visual"
    GENERATION_READY = "generation_ready"


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'.")
    content: str = Field(..., description="Message text.")
    images: Optional[List[str]] = Field(
        default=None,
        description="Optional list of image paths or data URIs attached to the message.",
    )


class ExportSettings(BaseModel):
    width: int = 1920
    height: int = 1080
    slide_duration: float = 5.0
    transition_duration: float = 1.0
    include_transitions: bool = True
    output_path: str = "deckadence_export.mp4"
    no_transition_behavior: str = Field(
        default="cut", description="Fallback when no transition exists: 'cut' or 'fade'."
    )

    @field_validator("width", "height")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Dimensions must be positive integers")
        return value

    @field_validator("slide_duration", "transition_duration")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Durations must be positive numbers")
        return value

    @field_validator("no_transition_behavior")
    @classmethod
    def validate_behavior(cls, value: str) -> str:
        if value not in {"cut", "fade"}:
            raise ValueError("no_transition_behavior must be 'cut' or 'fade'")
        return value
