"""Restartable pipeline step implementations."""

# Keep target-aware policy outside the core preflight implementation so the base
# safety checks remain easy to audit. Importing the package installs the wrapper
# for both direct ``pipeline.steps.preflight`` use and service-driven workflows.
from . import preflight as _preflight
from ..target_preflight import install_target_preflight_hook

install_target_preflight_hook(_preflight)
