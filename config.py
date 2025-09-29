"""Configuration helpers for the Alpaca edge service."""

from __future__ import annotations

import os
from typing import Final

DEFAULT_BASE_URL: Final[str] = "https://paper-api.alpaca.markets"


def _normalise_base_url(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return DEFAULT_BASE_URL
    return value


def _derive_base_url() -> str:
    apca_env = os.environ.get("APCA_API_BASE_URL")
    legacy_env = os.environ.get("ALPACA_API_BASE_URL")

    if apca_env:
        resolved = _normalise_base_url(apca_env)
    elif legacy_env:
        resolved = _normalise_base_url(legacy_env)
        os.environ["APCA_API_BASE_URL"] = resolved
    else:
        resolved = DEFAULT_BASE_URL
        os.environ["APCA_API_BASE_URL"] = resolved

    return resolved


APCA_API_BASE_URL: Final[str] = _derive_base_url()

__all__ = ["APCA_API_BASE_URL", "DEFAULT_BASE_URL"]
