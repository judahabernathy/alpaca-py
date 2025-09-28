# Journal mapping

## Router → Journals (POST /journal)
- ts→entry_time
- symbol→symbol
- avg_price→entry_price
- sl→stop_price
- tp→tp_price
- side: buy→long, sell→short
- notes→notes

## Tags (tags.*)
- mode, route, action, qty, client_order_id, order_id, order_class, type, time_in_force, session, quote_asof, equity, buying_power, request_id.

## Updates
- On fills/exits PATCH /journal/{id} with exit_time, exit_price, outcome.

## Bulk
- Use {items:[JournalIn,...]} for /journal/bulk.
