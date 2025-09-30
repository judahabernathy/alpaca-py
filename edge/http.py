"""HTTP client abstractions for upstream Alpaca calls."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

import aiohttp
from fastapi import HTTPException

from .logging import get_correlation_id, log_error, log_request


class EdgeHttpClient:
    """Wrapper around aiohttp that injects headers and logs requests."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        resolved_base = (base_url or os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets").strip()
        if not resolved_base:
            resolved_base = "https://paper-api.alpaca.markets"
        self._base_url = resolved_base.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None

    async def startup(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=None)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def shutdown(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _auth_headers(self) -> Dict[str, str]:
        key_id = os.getenv("APCA_API_KEY_ID", "").strip()
        secret_key = os.getenv("APCA_API_SECRET_KEY", "").strip()
        if not key_id or not secret_key:
            raise HTTPException(status_code=503, detail="Alpaca credentials are not configured")
        return {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }


    @property
    def base_url(self) -> str:
        return self._base_url


    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> tuple[int, Dict[str, str], str]:
        if self._session is None:
            raise RuntimeError("HTTP client has not been started")

        url = f"{self._base_url}{path}"
        request_headers: Dict[str, str] = self._auth_headers()
        if headers:
            request_headers.update(headers)
        correlation_id = get_correlation_id()
        if correlation_id:
            request_headers.setdefault("X-Correlation-ID", correlation_id)

        start = time.perf_counter()
        try:
            response = await self._session.request(
                method.upper(), url, headers=request_headers, **kwargs
            )
            body = await response.text()
            elapsed = time.perf_counter() - start
            log_request(method.upper(), url, response.status, elapsed)
            response_headers = {k: v for k, v in response.headers.items()}
            await response.release()
            return response.status, response_headers, body
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            log_error("http_error", method=method.upper(), url=url, error=str(exc))
            raise
