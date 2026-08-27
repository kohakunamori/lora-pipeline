from __future__ import annotations

from .interactive_batch_tags import InteractiveWizard as BatchTagWizard
from .interactive_composition import InteractiveWizard as CompositionWizard


class InteractiveWizard(CompositionWizard, BatchTagWizard):
    """Final CLI wizard combining composition metadata and batch Tag editing."""

    pass
