"""Restartable pipeline step implementations."""

# Keep target-aware policies outside the core step implementations so the base
# safety logic remains easy to audit. Importing the package installs wrappers for
# both direct step-module use and service-driven workflows.
from . import caption as _caption
from . import preflight as _preflight
from ..style_caption_policy import install_style_caption_policy_hook
from ..target_preflight import install_target_preflight_hook

install_style_caption_policy_hook(_caption)
install_target_preflight_hook(_preflight)
