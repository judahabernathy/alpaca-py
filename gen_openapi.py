#!/usr/bin/env python3
"""Utility for generating the public OpenAPI specification."""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the OpenAPI specification")
    parser.add_argument("--in", dest="input_path", default="openapi.yaml", help="Path to the base specification")
    parser.add_argument("--out", dest="output_path", default="-", help="Path to write the rendered spec (default stdout)")
    parser.add_argument("--full", action="store_true", help="Include advanced endpoints useful for power users")
    return parser.parse_args()


def _load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _dump_spec(spec: dict, path: str) -> None:
    output = yaml.safe_dump(spec, sort_keys=False)
    if path == "-" or not path:
        sys.stdout.write(output)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(output)


def _apply_public_base(spec: dict) -> None:
    base_url = os.getenv("PUBLIC_BASE_URL")
    if not base_url:
        return
    servers = spec.setdefault("servers", [])
    if servers:
        servers[0]["url"] = base_url
    else:
        servers.append({"url": base_url})


def _enable_advanced_routes(spec: dict) -> None:
    paths = spec.setdefault("paths", {})
    paths.setdefault(
        "/v2/orders:preview",
        {
            "post": {
                "summary": "Preview trade (advanced)",
                "security": [{"ApiKeyAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"type": "object"}}
                    },
                },
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {}}},
                    }
                },
            }
        },
    )


def main() -> None:
    args = parse_args()
    spec = _load_spec(args.input_path)
    _apply_public_base(spec)
    if args.full:
        _enable_advanced_routes(spec)
    _dump_spec(spec, args.output_path)


if __name__ == "__main__":
    main()
