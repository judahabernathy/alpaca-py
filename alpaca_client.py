from __future__ import annotations

import os

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from config import is_paper


class AlpacaClient:
    """Thin wrapper around ``alpaca-py``'s :class:`TradingClient`."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        """Instantiate the client using standard Alpaca environment variables."""

        # Be tolerant of either APCA_/ALPACA_ naming conventions.
        api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
        # Default to paper; detect live only if base URL says so
        paper = is_paper()
        return cls(api_key=api_key, secret_key=secret_key, paper=paper)

    def get_account(self) -> dict:
        """Fetch account details using the official SDK."""

        try:
            account = self.client.get_account()
        except APIError as exc:
            raise APIError(
                f"Failed to fetch account: {exc.status_code} {exc.code} {exc.message}"
            ) from exc
        return account.model_dump()
