from __future__ import annotations

import sys

from .i18n import initialize_interactive
from .interactive_local_video_import import install_local_video_import_modes
from .interactive_materialization import InteractiveWizard
from .interactive_menu_descriptions import install_menu_descriptions
from .interactive_menu_navigation import install_menu_navigation
from .interactive_menu_plaintext import install_plain_menu_labels
from .interactive_result_reports import install_result_report_menu
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
    install_plain_menu_labels()
    install_local_video_import_modes(InteractiveWizard)
    install_result_report_menu(InteractiveWizard)
    InteractiveWizard().home()


if __name__ == "__main__":
    main()
