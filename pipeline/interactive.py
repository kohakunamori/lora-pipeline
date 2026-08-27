from __future__ import annotations

import sys

from .interactive_app import InteractiveWizard
from .i18n import initialize_interactive
from .i18n_menu import install_language_menu


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import main as cli_main

        cli_main()
        return
    initialize_interactive()
    install_language_menu()
    InteractiveWizard().home()


if __name__ == "__main__":
    main()
