# Alpha Knowledge Pack v1 — 2025‑09‑28
Scope: supporting docs for the 6 actions. Source of truth = Actions OpenAPI + Router OAS + these files. Archive all old docs.

## Folders
- adapters/: classifier + FinRL + journals mappings
- policies/: routing, risk, session, idempotency, errors
- workflows/: fast_lane, watchlist, memory
- specs/: endpoint summaries (not full OpenAPI)
- tests/: golden prompts + pass criteria
- scripts/: PowerShell 7 smokes
- snippets/: reusable response text

## Rules
- Do not copy rules from Instructions; reference them.
- Charts basis: CP‑CAP‑1, CP‑FAST‑1, CP‑ACCT‑1, CP‑RISK‑1, CP‑IDEMP‑1, CP‑ENV‑1, CP‑ERR‑1, CP‑CLF‑1, CP‑CLF‑ADAPT‑1, CP‑WL‑1, CP‑JRN‑1, CP‑MEM‑1, CP‑RL‑CORE‑1, CP‑RL‑TRAIN‑1, CP‑RL‑JOBS‑1, CP‑TI‑ALIGN‑1.

## Contents
- adapters/classifier_adapter.md
- adapters/finrl_adapter.md
- adapters/journal_mapping.md
- policies/routing_map.md
- policies/risk_policy.md
- policies/idempotency.md
- policies/error_policy.md
- workflows/fast_lane.md
- workflows/watchlist.md
- workflows/memory.md
- specs/alpaca_summary.md
- specs/finrl_summary.md
- specs/journals_summary.md
- specs/classifier_summary.md
- specs/finnhub_summary.md
- tests/golden_prompts.md
- scripts/ps7_smoke.ps1
