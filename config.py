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


def _synchronise_base_urls(base_url: str) -> str:
    os.environ["ALPACA_API_BASE_URL"] = base_url
    os.environ["APCA_API_BASE_URL"] = base_url
    return base_url


def _derive_base_url() -> str:
    return _normalise_base_url(
        os.environ.get("ALPACA_API_BASE_URL")
        or os.environ.get("APCA_API_BASE_URL")
    )


APCA_API_BASE_URL: Final[str] = _synchronise_base_urls(_derive_base_url())


__all__ = ["APCA_API_BASE_URL", "DEFAULT_BASE_URL"]
