"""Cost tracking for API usage in Deckadence.

Tracks costs for:
- Gemini image generation (slides)
- fal.ai Kling video generation (transitions)

Pricing is loaded from pricing.yaml for easy maintenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load pricing from YAML
# ---------------------------------------------------------------------------

def _load_pricing() -> Dict:
    """Load pricing data from pricing.yaml."""
    pricing_file = Path(__file__).parent / "pricing.yaml"
    if pricing_file.exists():
        with open(pricing_file, "r") as f:
            return yaml.safe_load(f)
    # Fallback defaults if file is missing
    return {
        "gemini": {"image_generation": 0.02},
        "kling": {
            "standard_5s": 0.054,
            "standard_10s": 0.108,
            "pro_5s": 0.108,
            "pro_10s": 0.216,
        }
    }


_PRICING: Optional[Dict] = None


def get_pricing() -> Dict:
    """Get cached pricing data."""
    global _PRICING
    if _PRICING is None:
        _PRICING = _load_pricing()
    return _PRICING


def get_gemini_image_cost(model: str = "nano_banana_pro") -> float:
    """Get cost for Gemini image generation.
    
    Args:
        model: "nano_banana" or "nano_banana_pro"
    
    Returns:
        Cost in USD per image.
    """
    gemini = get_pricing().get("gemini", {})
    return gemini.get(model, gemini.get("nano_banana_pro", 0.14))


def get_kling_cost(model: str, duration: int) -> float:
    """Get cost for Kling video generation.
    
    Args:
        model: "standard" or "pro"
        duration: 5 or 10 (seconds)
    
    Returns:
        Cost in USD.
    """
    kling = get_pricing().get("kling", {})
    key = f"{model}_{duration}s"
    return kling.get(key, kling.get("pro_5s", 0.108))


@dataclass
class CostEntry:
    """A single cost entry for an API call."""
    service: str  # "gemini" or "kling"
    operation: str  # e.g., "slide_image", "transition_clip"
    cost_usd: float
    details: str = ""  # e.g., "slide 1", "transition 1->2"


@dataclass
class CostTracker:
    """Tracks cumulative API costs for a session."""
    entries: List[CostEntry] = field(default_factory=list)
    
    def add(self, entry: CostEntry) -> None:
        """Add a cost entry and log it."""
        self.entries.append(entry)
        LOG.info(
            "Cost: $%.4f - %s %s%s",
            entry.cost_usd,
            entry.service,
            entry.operation,
            f" ({entry.details})" if entry.details else ""
        )
    
    def add_gemini_image(self, model: str = "nano_banana_pro", details: str = "") -> float:
        """Record a Gemini image generation cost.
        
        Args:
            model: "nano_banana" or "nano_banana_pro"
            details: Optional description (e.g., "slide 1")
        
        Returns:
            The cost in USD.
        """
        cost = get_gemini_image_cost(model)
        self.add(CostEntry(
            service="gemini",
            operation=f"image_generation ({model})",
            cost_usd=cost,
            details=details
        ))
        return cost
    
    def add_kling_video(self, model: str, duration: int, details: str = "") -> float:
        """Record a Kling video generation cost.
        
        Args:
            model: "standard" or "pro"
            duration: 5 or 10 (seconds)
            details: Optional description (e.g., "transition 1->2")
        
        Returns:
            The cost in USD.
        """
        cost = get_kling_cost(model.lower(), duration)
        self.add(CostEntry(
            service="kling",
            operation="video_generation",
            cost_usd=cost,
            details=details
        ))
        return cost
    
    @property
    def total_cost(self) -> float:
        """Get the total cost of all entries."""
        return sum(e.cost_usd for e in self.entries)
    
    @property
    def gemini_cost(self) -> float:
        """Get total Gemini costs."""
        return sum(e.cost_usd for e in self.entries if e.service == "gemini")
    
    @property
    def kling_cost(self) -> float:
        """Get total Kling costs."""
        return sum(e.cost_usd for e in self.entries if e.service == "kling")
    
    def summary(self) -> str:
        """Get a formatted cost summary."""
        lines = []
        if self.entries:
            lines.append(f"Gemini (images): ${self.gemini_cost:.4f}")
            lines.append(f"Kling (videos):  ${self.kling_cost:.4f}")
            lines.append(f"Total cost:      ${self.total_cost:.4f}")
        else:
            lines.append("No API costs incurred.")
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Clear all cost entries."""
        self.entries.clear()


# Global cost tracker instance for the session
_tracker: Optional[CostTracker] = None


def get_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def reset_tracker() -> None:
    """Reset the global cost tracker."""
    global _tracker
    _tracker = CostTracker()

