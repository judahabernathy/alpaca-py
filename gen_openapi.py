import json
import os
from typing import Iterable, Tuple

import yaml

from app import app

# Only expose the subset of operations we allow in GPT Actions (cap at 30).
EXCLUDED_OPERATIONS: Iterable[Tuple[str, str]] = {
    ("/v2/orders/sync", "post"),
    ("/v2/account/portfolio/history", "get"),
    ("/v2/corporate_actions/announcements/{announcement_id}", "get"),
    ("/v2/positions/{symbol_or_contract_id}/exercise", "post"),
    ("/v2/watchlists/{watchlist_id}", "post"),
    ("/v2/watchlists/{watchlist_id}", "delete"),
    ("/v2/watchlists/{watchlist_id}/{symbol}", "delete"),
}


def _prune_operations(schema: dict) -> None:
    paths = schema.get("paths", {})
    for path, method in EXCLUDED_OPERATIONS:
        path_item = paths.get(path)
        if not isinstance(path_item, dict):
            continue
        path_item.pop(method, None)
        remaining = [name for name in path_item if not name.startswith("x-")]
        if not remaining:
            paths.pop(path, None)


schema = app.openapi()
_prune_operations(schema)

base = (
    os.environ.get("PUBLIC_BASE_URL")
    or os.environ.get("RAILWAY_STATIC_URL")
    or os.environ.get("SERVER_URL")
    or "https://alpaca-edge-production.up.railway.app"
)
schema["servers"] = [{"url": base}]

paths = schema.get("paths", {})
if "/v1/order/bracket" in paths and "post" in paths["/v1/order/bracket"]:
    paths["/v1/order/bracket"]["post"]["operationId"] = "placeBracketOrder"
if "/v1/order/stop" in paths and "post" in paths["/v1/order/stop"]:
    paths["/v1/order/stop"]["post"]["operationId"] = "placeStopOrder"
if "/v1/order/trailing" in paths and "post" in paths["/v1/order/trailing"]:
    paths["/v1/order/trailing"]["post"]["operationId"] = "placeTrailingStopOrder"

with open("openapi.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
with open("openapi.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(schema, f, sort_keys=False, allow_unicode=True)

print("Wrote openapi.json & openapi.yaml with servers[0].url =", base)
