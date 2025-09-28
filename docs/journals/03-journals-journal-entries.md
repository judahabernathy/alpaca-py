# 03-journals-journal-entries.md
version: 2025-09-08
status: canonical
scope: journals/journal-entries

## Models

### JournalIn (request)
Fields:
`symbol` (string, max 32, required); `strategy` (string, max 64); `side` ("long" or "short"); `entry_time`, `exit_time` (datetime); `entry_price`, `exit_price`, `stop_price`, `tp_price` (floats); `r_multiple` (float); `outcome` (string); `notes` (string); `tags` (JSON object).
Unknown fields are ignored.

### JournalOut (response)
Extends `JournalIn` with `id` (int), `created_at` and `updated_at` timestamps.

## Endpoints

- **POST /journal** – Accepts a `JournalIn`, returns the created entry with ID and timestamps.
- **POST /journal/bulk** – Accepts `{ "items": [JournalIn, …] }`, inserts all entries and returns a list of `JournalOut`.
- **GET /journal** – Query parameters: `symbol`, `strategy`, `start`, `end`, `limit` (default 200). Returns a list of entries ordered by creation time.
- **GET /journal/{id}** – Retrieves a single entry by ID; returns 404 if not found.
- **PATCH /journal/{id}** – Updates provided fields; missing fields are left unchanged.
- **DELETE /journal/{id}** – Deletes the entry; returns 204 on success.

### GET /stats
Aggregates journal data. Query parameters: `symbol`, `strategy`, `start`, `end`. Returns objects with `symbol`, `strategy`, `trades` (count), `win_rate` (mean of positive R) and `avg_r` (average R).

