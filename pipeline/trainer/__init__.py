"""Training backends. V1 ships only the pinned sd-scripts adapter."""

from .base import TrainerBackend
from .sd_scripts import SdScriptsTrainer

__all__ = ["TrainerBackend", "SdScriptsTrainer"]
