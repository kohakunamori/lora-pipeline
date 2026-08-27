from __future__ import annotations

from . import web_metadata_hooks as _web_metadata_hooks  # noqa: F401
from .web_outfit import OutfitHandler as FullHandler
from .web_outfit import main, make_server, serve

__all__ = ["FullHandler", "make_server", "serve", "main"]


if __name__ == "__main__":
    main()
