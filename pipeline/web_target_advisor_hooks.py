from __future__ import annotations

import sys

from . import web_outfit
from .web_target_advisor import TargetAdvisorHandler


# Final Web entrypoints normally import this hook before the protected-deletion
# layer, so that layer naturally subclasses TargetAdvisorHandler. Also repair an
# already-imported protected-deletion module (for test/plugin import-order safety)
# without changing its deletion methods or make_server implementation.
web_outfit.OutfitHandler = TargetAdvisorHandler

_loaded = sys.modules.get(f"{__package__}.web_protected_deletion")
if _loaded is not None:
    current = getattr(_loaded, "ProtectedDeletionHandler", None)
    if current is not None and not issubclass(current, TargetAdvisorHandler):
        class ProtectedTargetAdvisorHandler(current, TargetAdvisorHandler):
            pass

        _loaded.ProtectedDeletionHandler = ProtectedTargetAdvisorHandler
