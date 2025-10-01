"""Configuration helpers for the Alpaca edge service."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import Final

DEFAULT_BASE_URL: Final[str] = "https://paper-api.alpaca.markets"
DEFAULT_DATA_BASE_URL: Final[str] = "https://data.alpaca.markets"


load_dotenv(override=True)


def _normalise_base_url(value: str | None, *, default: str = DEFAULT_BASE_URL) -> str:
    value = (value or "").strip()
    if not value:
        return default
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

def _derive_data_base_url() -> str:
    data_env = os.environ.get("APCA_DATA_BASE_URL")
    legacy_data_env = os.environ.get("ALPACA_DATA_BASE_URL")

    if data_env:
        resolved = _normalise_base_url(data_env, default=DEFAULT_DATA_BASE_URL)
    elif legacy_data_env:
        resolved = _normalise_base_url(legacy_data_env, default=DEFAULT_DATA_BASE_URL)
        os.environ["APCA_DATA_BASE_URL"] = resolved
    else:
        resolved = DEFAULT_DATA_BASE_URL
        os.environ.setdefault("APCA_DATA_BASE_URL", resolved)

    return resolved


APCA_API_BASE_URL: Final[str] = _derive_base_url()
APCA_DATA_BASE_URL: Final[str] = _derive_data_base_url()

__all__ = ["APCA_API_BASE_URL", "APCA_DATA_BASE_URL", "DEFAULT_BASE_URL", "DEFAULT_DATA_BASE_URL"]
