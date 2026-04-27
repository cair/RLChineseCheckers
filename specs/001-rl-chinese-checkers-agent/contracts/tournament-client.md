# Tournament Client Contract

The tournament-compatible player must match the provided server protocol without modifying original files.

## Join

Request:

```json
{"op": "join", "player_name": "<name>"}
```

Expected response fields:
- `ok`
- `game_id`
- `player_id`
- `colour`
- `status`

## Start

Request:

```json
{"op": "start", "game_id": "<game-id>", "player_id": "<player-id>"}
```

## State

Request:

```json
{"op": "get_state", "game_id": "<game-id>"}
```

Relevant state fields:
- `status`
- `players`
- `pins`
- `move_count`
- `current_turn_colour`
- `turn_order`
- `last_move`
- `turn_timeout_notice`

## Legal Moves

Request:

```json
{"op": "get_legal_moves", "game_id": "<game-id>", "player_id": "<player-id>"}
```

Expected response:

```json
{"ok": true, "legal_moves": {"0": [10, 11], "1": [12]}}
```

The client must choose only a move present in this response.

## Move

Request:

```json
{
  "op": "move",
  "game_id": "<game-id>",
  "player_id": "<player-id>",
  "pin_id": 0,
  "to_index": 10
}
```

Required behavior:
- Submit within the turn timeout.
- If the learned policy cannot score the move, use heuristic fallback.
- If the server rejects the move, log the rejection and continue polling state.
