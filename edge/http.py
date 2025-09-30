"""HTTP client abstractions for upstream Alpaca calls."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import aiohttp

from .logging import get_correlation_id, log_error, log_request


class EdgeHttpClient:
    """Wrapper around aiohttp that injects headers and logs requests."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None

    async def startup(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=None)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def shutdown(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

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
        request_headers: Dict[str, str] = headers.copy() if headers else {}
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
