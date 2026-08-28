from __future__ import annotations

import sys

from . import interactive_metadata_hooks as _interactive_metadata_hooks  # noqa: F401
from .i18n import initialize_interactive
from .interactive_local_video_import import install_local_video_import_modes
from .interactive_menu_descriptions import install_menu_descriptions
from .interactive_menu_navigation import install_menu_navigation
from .interactive_semantic_concepts import InteractiveWizard
from .tty_compat import configure_interactive_input


def main() -> None:
    if len(sys.argv) > 1:
        from .cli import main as cli_main

        cli_main()
        return
    configure_interactive_input()
    initialize_interactive()
    install_menu_descriptions()
    install_menu_navigation()
    install_local_video_import_modes(InteractiveWizard)
    InteractiveWizard().home()


if __name__ == "__main__":
    main()
