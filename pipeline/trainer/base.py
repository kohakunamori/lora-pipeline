from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import TrainingRequest, TrainingResult


class TrainerBackend(ABC):
    """Thin adapter contract; implementations must not contain a custom training loop."""

    @abstractmethod
    def train(self, request: TrainingRequest, *, dry_run: bool = False, verbose: int = 0) -> TrainingResult:
        raise NotImplementedError
