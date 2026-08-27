from __future__ import annotations

import sys

from .i18n import initialize_interactive
from .interactive_video_auth import InteractiveWizard


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import main as cli_main

        cli_main()
        return
    initialize_interactive()
    InteractiveWizard().home()


if __name__ == "__main__":
    main()
