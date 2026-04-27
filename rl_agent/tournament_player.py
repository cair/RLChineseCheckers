"""Tournament-compatible client for the provided socket game server."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import board_adapter
from .config import COLOUR_ORDER
from .paths import SELECTED_CANDIDATE_PATH
from .policies import HeuristicPolicy
from .agent import MoveRanker
from .state import GameState, LegalMove


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50555


def rpc(payload: Mapping[str, Any], host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 10.0) -> dict:
    """Send one JSON request to the provided server."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.sendall(json.dumps(payload).encode("utf-8"))
        data = sock.recv(1_000_000)
    finally:
        sock.close()
    if not data:
        return {"ok": False, "error": "no-response"}
    return json.loads(data.decode("utf-8"))


def state_from_server(server_state: Mapping[str, Any], own_colour: str) -> GameState:
    """Convert server JSON state into the project-owned state snapshot."""

    pins = {
        colour: tuple(int(idx) for idx in indices)
        for colour, indices in server_state.get("pins", {}).items()
    }
    turn_order = tuple(server_state.get("turn_order") or [c for c in COLOUR_ORDER if c in pins])
    current_colour = server_state.get("current_turn_colour") or own_colour
    current_index = turn_order.index(current_colour) if current_colour in turn_order else 0
    return GameState(
        pins_by_colour=pins,
        player_colours=tuple(pins.keys()),
        turn_order=turn_order,
        current_turn_index=current_index,
        move_count=int(server_state.get("move_count", 0)),
        status=server_state.get("status", "PLAYING"),
    )


def legal_moves_from_server(
    server_state: Mapping[str, Any],
    own_colour: str,
    legal_moves: Mapping[str, Sequence[int]],
) -> tuple[LegalMove, ...]:
    """Build `LegalMove` objects from server-provided legal moves."""

    pins = server_state.get("pins", {}).get(own_colour, [])
    moves: list[LegalMove] = []
    for pin_id_text, destinations in legal_moves.items():
        pin_id = int(pin_id_text)
        if pin_id >= len(pins):
            continue
        from_index = int(pins[pin_id])
        before = board_adapter.min_distance_to_goal(from_index, own_colour)
        for dest in destinations:
            to_index = int(dest)
            dist = board_adapter.axial_distance(from_index, to_index)
            after = board_adapter.min_distance_to_goal(to_index, own_colour)
            moves.append(
                LegalMove(
                    pin_id=pin_id,
                    from_index=from_index,
                    to_index=to_index,
                    is_step=dist == 1,
                    is_jump=dist > 1,
                    distance_delta=before - after,
                    enters_goal=to_index in board_adapter.metadata().goal_indices[own_colour],
                )
            )
    return tuple(sorted(moves, key=lambda m: (m.pin_id, m.to_index)))


def choose_move(
    server_state: Mapping[str, Any],
    own_colour: str,
    legal_moves: Mapping[str, Sequence[int]],
    checkpoint: str | None = None,
) -> LegalMove:
    """Choose a legal move using selected candidate if available, else heuristic."""

    # Candidate loading is extended later. For now, heuristic is the tournament-safe fallback.
    state = state_from_server(server_state, own_colour)
    legal = legal_moves_from_server(server_state, own_colour, legal_moves)
    if checkpoint and Path(checkpoint).suffix == ".json" and Path(checkpoint).exists():
        try:
            return MoveRanker.load(checkpoint).select_move(state, legal)
        except Exception:
            pass
    return HeuristicPolicy().select_move(state, legal)


def load_selected_candidate_path(path: Path = SELECTED_CANDIDATE_PATH) -> Path | None:
    """Return selected candidate checkpoint path if metadata exists."""

    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = data.get("checkpoint_path")
        return Path(checkpoint) if checkpoint else None
    except Exception:
        return None


def run_client(name: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, checkpoint: str | None = None) -> int:
    if checkpoint is None:
        selected = load_selected_candidate_path()
        checkpoint = str(selected) if selected else None
    join = rpc({"op": "join", "player_name": name}, host, port)
    if not join.get("ok"):
        print("JOIN ERROR:", join.get("error"))
        return 1
    game_id = join["game_id"]
    player_id = join["player_id"]
    colour = join["colour"]
    print(f"Joined game {game_id} as {colour}")

    while True:
        state_resp = rpc({"op": "get_state", "game_id": game_id}, host, port)
        state = state_resp.get("state", {})
        if state.get("status") in ("READY_TO_START", "PLAYING"):
            break
        time.sleep(0.2)

    rpc({"op": "start", "game_id": game_id, "player_id": player_id}, host, port)

    while True:
        state_resp = rpc({"op": "get_state", "game_id": game_id}, host, port)
        if not state_resp.get("ok"):
            print("STATE ERROR:", state_resp.get("error"))
            return 1
        state = state_resp["state"]
        if state.get("status") == "FINISHED":
            print("Game finished")
            return 0
        if state.get("current_turn_colour") == colour and state.get("status") == "PLAYING":
            legal_resp = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id}, host, port)
            if not legal_resp.get("ok"):
                print("LEGAL MOVE ERROR:", legal_resp.get("error"))
                time.sleep(0.2)
                continue
            move = choose_move(state, colour, legal_resp.get("legal_moves", {}), checkpoint)
            rpc(
                {
                    "op": "move",
                    "game_id": game_id,
                    "player_id": player_id,
                    "pin_id": move.pin_id,
                    "to_index": move.to_index,
                },
                host,
                port,
            )
        time.sleep(0.2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tournament-compatible RLCC agent player.")
    parser.add_argument("--name", default="RLCC-Agent")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)
    return run_client(args.name, args.host, args.port, args.checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
