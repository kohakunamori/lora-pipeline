from __future__ import annotations

from .web_batch_tags import FinalHandler as FullHandler
from .web_batch_tags import main, make_server, serve

__all__ = ["FullHandler", "make_server", "serve", "main"]


if __name__ == "__main__":
    main()
