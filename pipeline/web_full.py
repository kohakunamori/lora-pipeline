from __future__ import annotations

from . import web_metadata_hooks as _web_metadata_hooks  # noqa: F401
from . import semantic_project_hooks as _semantic_project_hooks  # noqa: F401
from . import web_target_advisor_hooks as _web_target_advisor_hooks  # noqa: F401
from .web_protected_deletion import ProtectedDeletionHandler as FullHandler
from .web_protected_deletion import main, make_server, serve

__all__ = ["FullHandler", "make_server", "serve", "main"]


if __name__ == "__main__":
    main()
