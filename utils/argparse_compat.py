from __future__ import annotations

import argparse
from typing import Any


def add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool | None = None,
    **kwargs: Any,
) -> None:
    """Add a --flag/--no-flag option on Python 3.8 and newer."""
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument(name, action=action, default=default, **kwargs)
        return

    option = name.lstrip("-")
    dest = option.replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, **kwargs)
    parser.add_argument(f"--no-{option}", dest=dest, action="store_false")
