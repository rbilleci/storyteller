"""The identifier grammar and word tokenizer shared across the narrator package.

Four modules (``interactions.py``, ``social.py``, ``decisions.py``,
``player_directory.py``) each defined the same ``^[a-z0-9][a-z0-9_-]{0,63}$``
pattern under a locally-scoped name, and two of them (``interactions.py``,
``social.py``) also each defined the same case-folded ``[a-z]+`` word tokenizer.
Zero package imports, so every other narrator module can depend on this one
without risking a cycle.
"""

from __future__ import annotations

import re

#: A campaign-safe identifier: lower-case letters, digits, hyphen, and underscore,
#: 1 to 64 characters, starting with a letter or digit.
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_WORDS = re.compile(r"[a-z]+")


def words(text: str) -> frozenset[str]:
    """Every case-folded run of letters in ``text``, as a set of whole words."""
    return frozenset(_WORDS.findall(text.casefold()))
