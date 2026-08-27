from __future__ import annotations

from .web_metadata import MetadataHandler as FullHandler
from .web_metadata import main, make_server, serve

__all__ = ["FullHandler", "make_server", "serve", "main"]


if __name__ == "__main__":
    main()
