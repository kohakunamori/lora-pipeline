"""Canonical restartable Project steps: preflight and train.

Materialization lives in :mod:`pipeline.materialization`; Results operations live
in :mod:`pipeline.evaluation`. Importing this package has no side effects.

``promote`` is a temporary compatibility alias for older interactive modules. It
points at the run-scoped Results implementation and is not a Project step.
"""

from ..evaluation import promotion as promote
