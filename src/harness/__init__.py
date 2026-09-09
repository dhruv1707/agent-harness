"""An AI creative strategist harness: top ads -> hooks -> iteration briefs."""

from .prompt import AssembledPrompt, Layer, RunContext, build_effective_system_prompt

__version__ = "0.1.0"

__all__ = [
    "AssembledPrompt",
    "Layer",
    "RunContext",
    "build_effective_system_prompt",
]
