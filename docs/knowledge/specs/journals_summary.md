# Journals summary

## Auth
- Header `X-API-Key`

## Endpoints
- /health, /journal (GET/POST), /journal/bulk (POST), /journal/{id} (GET/PATCH/DELETE), /stats (GET), /params (POST), /params/latest (GET)

## Shapes
- JournalIn requires symbol; PATCH uses JournalIn; include symbol.
- Params: {version, source?, blob{…}}
