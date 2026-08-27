from __future__ import annotations

import sys

from .wizard import Wizard


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import main as cli_main

        cli_main()
        return
    Wizard().home()


if __name__ == "__main__":
    main()
