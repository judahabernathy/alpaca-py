#!/usr/bin/env python
"""Export the FastAPI OpenAPI specification to .well-known files."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from edge.app import app


def main() -> None:
    spec = app.openapi()
    root = Path(__file__).resolve().parent.parent
    target = root / ".well-known"
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "openapi.json"
    yaml_path = target / "openapi.yaml"

    json_path.write_text(json.dumps(spec, indent=2))
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))

    print(f"Wrote {json_path}")
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
