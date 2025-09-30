"""Supplementary routes for readiness and dynamic OpenAPI metadata."""

from __future__ import annotations

from copy import deepcopy
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _app_module():
    from . import app as edge_app  # Local import avoids circular dependency at module load time.

    return edge_app


@router.get("/readyz", include_in_schema=False)
def readyz(request: Request) -> dict:
    edge_app = _app_module()
    edge_app._require_gateway_key_from_request(request)
    credentials_ok = edge_app._alpaca_credentials_present()
    base_url = edge_app._resolved_api_base_url()
    return {
        "ok": credentials_ok and bool(base_url),
        "env": {
            "alpaca_credentials": credentials_ok,
            "api_base_url": base_url,
        },
    }


@router.get("/.well-known/openapi.json", include_in_schema=False)
def well_known_openapi(request: Request) -> JSONResponse:
    edge_app = _app_module()
    base_schema = deepcopy(edge_app._build_openapi_schema(edge_app.app.routes))
    base_schema["servers"] = [{"url": str(request.base_url).rstrip("/")}]
    return JSONResponse(base_schema)


__all__ = ["router"]
