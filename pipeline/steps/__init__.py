"""Canonical restartable Project steps: preflight and train.

Materialization lives in :mod:`pipeline.materialization`; Results operations live
in :mod:`pipeline.evaluation`. Importing this package has no side effects.

The ``promote`` name below is a temporary import bridge for older TUI mixins while
this refactor removes their historical ``pipeline.steps`` dependency. It is not a
Project step and has no state-machine registration.
"""

from ..evaluation import promotion as promote
