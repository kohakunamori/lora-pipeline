from __future__ import annotations

import re

from .caption_cleaner import CATEGORY_PATTERNS, normalize_tag


IDENTITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:black|blonde|brown|red|blue|green|purple|pink|white|grey|gray|silver|"
        r"aqua|orange|multicolored|two-tone) hair$"
    ),
    re.compile(
        r"^(?:black|brown|red|blue|green|purple|pink|yellow|gold|golden|white|grey|"
        r"gray|silver|aqua|orange) eyes$"
    ),
    re.compile(
        r"^(?:very long hair|long hair|medium hair|short hair|twintails|twin tails|"
        r"ponytail|side ponytail|braid|double braid|single braid|bob cut|hime cut|"
        r"straight hair|wavy hair|curly hair|ahoge|blunt bangs|bangs|hair over one eye|"
        r"heterochromia|hair ribbon|hair ornament|glasses)$"
    ),
)


def is_identity_tag(tag: str) -> bool:
    normalized = normalize_tag(tag)
    return any(pattern.fullmatch(normalized) for pattern in IDENTITY_PATTERNS)


def is_outfit_tag(tag: str) -> bool:
    normalized = normalize_tag(tag)
    return any(pattern.search(normalized) for pattern in CATEGORY_PATTERNS.get("outfit", ()))
