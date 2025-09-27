import os
from urllib.parse import urlparse


def _normalize_base_url() -> str:
    """
    Canonical source of truth for Alpaca base URL.
    - Prefer APCA_API_BASE_URL
    - Accept ALPACA_API_BASE_URL as an alias
    - Default to PAPER
    Also write-through both env vars so all callers see the same value.
    """

    base = (
        os.getenv("APCA_API_BASE_URL")
        or os.getenv("ALPACA_API_BASE_URL")
        or "https://paper-api.alpaca.markets"
    )
    os.environ.setdefault("APCA_API_BASE_URL", base)
    os.environ.setdefault("ALPACA_API_BASE_URL", base)
    return base


# Export canonical base and helpers
APCA_API_BASE_URL = _normalize_base_url()


def is_paper(base: str | None = None) -> bool:
    b = base or APCA_API_BASE_URL
    netloc = urlparse(b).netloc.lower()
    return "paper" in netloc

