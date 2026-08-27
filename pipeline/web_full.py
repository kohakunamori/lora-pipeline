from __future__ import annotations

from .web_safety import FullHandler, main, make_server, serve

__all__ = ["FullHandler", "make_server", "serve", "main"]


if __name__ == "__main__":
    main()
