from __future__ import annotations

import sys

from . import interactive_metadata_hooks as _interactive_metadata_hooks  # noqa: F401
from .i18n import initialize_interactive
from .interactive_training_parameters import InteractiveWizard
from .tty_compat import configure_interactive_input


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import main as cli_main

        cli_main()
        return
    configure_interactive_input()
    initialize_interactive()
    InteractiveWizard().home()


if __name__ == "__main__":
    main()
