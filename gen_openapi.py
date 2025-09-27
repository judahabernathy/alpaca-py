#!/usr/bin/env python3
import argparse
import os
import sys
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="openapi.yaml")
    parser.add_argument("--out", dest="out", default="-")
    parser.add_argument(
        "--full", action="store_true", help="optionally add advanced helpers"
    )
    args = parser.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    base = os.getenv("PUBLIC_BASE_URL")
    if base:
        if not doc.get("servers"):
            doc["servers"] = [{"url": base}]
        else:
            doc["servers"][0]["url"] = base

    if args.full:
        paths = doc.setdefault("paths", {})
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

    out = yaml.safe_dump(doc, sort_keys=False)
    if args.out == "-" or not args.out:
        sys.stdout.write(out)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)


if __name__ == "__main__":
    main()
