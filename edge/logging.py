"""Structured logging utilities for the edge service."""

from __future__ import annotations

import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any, Dict

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_logger: logging.Logger | None = None


def configure_logging() -> logging.Logger:
    """Configure application logging for structured output."""
    global _logger
    if _logger is not None:
        return _logger

    level_name = os.getenv("EDGE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    _logger = logging.getLogger("edge")
    _logger.setLevel(level)
    return _logger


def set_correlation_id(value: str | None):
    """Bind a correlation ID for the current context and return the token."""
    return correlation_id_var.set(value)


def reset_correlation_id(token) -> None:
    correlation_id_var.reset(token)


def get_correlation_id() -> str | None:
    return correlation_id_var.get()


def _serialise(event: str, **fields: Any) -> str:
    record: Dict[str, Any] = {"event": event, **fields}
    correlation_id = get_correlation_id()
    if correlation_id:
        record.setdefault("correlation_id", correlation_id)
    return json.dumps(record, default=str)


def _get_logger() -> logging.Logger:
    if _logger is None:
        return configure_logging()
    return _logger


def log_event(event: str, **fields: Any) -> None:
    _get_logger().info(_serialise(event, **fields))


def log_error(event: str, **fields: Any) -> None:
    _get_logger().error(_serialise(event, **fields))


def log_request(method: str, url: str, status: int, latency_s: float, **fields: Any) -> None:
    log_event(
        "http_request",
        method=method,
        url=url,
        status=status,
        latency_ms=round(latency_s * 1000, 2),
        **fields,
    )


def log_duration(event: str, start_time: float, **fields: Any) -> None:
    duration = time.perf_counter() - start_time
    log_event(event, latency_ms=round(duration * 1000, 2), **fields)
