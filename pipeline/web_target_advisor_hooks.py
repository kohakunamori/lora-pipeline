from __future__ import annotations

from . import web_outfit
from .web_target_advisor import TargetAdvisorHandler


# Final Web entrypoints import this hook before importing the protected-deletion
# layer. That layer then subclasses the target-aware handler while preserving all
# earlier metadata/semantic monkeypatches installed on web_outfit.
web_outfit.OutfitHandler = TargetAdvisorHandler
