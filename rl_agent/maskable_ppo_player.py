"""Server client that can play a trained MaskablePPO checkpoint.

This file is intentionally separate from the provided course server/client
files. It uses the same JSON socket protocol as `multi system .../player.py`,
but chooses moves with either a MaskablePPO checkpoint or a simple baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_space import action_to_legal_move, legal_action_mask
from .features import state_features
from .policies import HeuristicPolicy, RandomPolicy
from .state import GameState, LegalMove
from .tournament_player import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    legal_moves_from_server,
    load_selected_candidate_path,
    rpc,
    state_from_server,
)


class MaskablePPOPolicy:
    """Small adapter from server state/legal moves to MaskablePPO actions."""

    def __init__(self, checkpoint: str | Path, deterministic: bool = True):
        try:
            import numpy as np
            from sb3_contrib import MaskablePPO
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "MaskablePPO play requires numpy, stable-baselines3, and sb3-contrib "
                "in the active Python environment."
            ) from exc

        self.np = np
        self.checkpoint = resolve_checkpoint(checkpoint)
        self.deterministic = deterministic
        self.model = MaskablePPO.load(str(self.checkpoint))
        self.fallback = HeuristicPolicy()

    def select_move(self, state: GameState, legal_moves: Sequence[LegalMove]) -> LegalMove:
        if not legal_moves:
            raise ValueError("No legal moves available")

        obs = self.np.asarray(state_features(state), dtype=self.np.float32)
        mask = self.np.asarray(legal_action_mask(legal_moves), dtype=bool)
        action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        move = action_to_legal_move(int(action), legal_moves)
        if move is None:
            print("PPO selected an illegal masked action; falling back to heuristic.")
            return self.fallback.select_move(state, legal_moves)
        return move


class BaselinePolicy:
    """Uniform interface for heuristic/random visual opponents."""

    def __init__(self, policy: str, seed: int):
        self.policy_name = policy
        if policy == "random":
            self.policy = RandomPolicy(seed=seed)
        elif policy == "heuristic":
            self.policy = HeuristicPolicy()
        else:
            raise ValueError(f"Unknown baseline policy: {policy}")

    def select_move(self, state: GameState, legal_moves: Sequence[LegalMove]) -> LegalMove:
        return self.policy.select_move(state, legal_moves)


def resolve_checkpoint(checkpoint: str | Path | None) -> Path:
    """Accept a .zip checkpoint, sidecar metadata JSON, or selected candidate."""

    if checkpoint is None:
        selected = load_selected_candidate_path()
        if selected is None:
            raise FileNotFoundError(
                "No checkpoint was provided and checkpoints/selected_candidate.json was not found."
            )
        checkpoint = selected

    path = Path(checkpoint)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        path = Path(data.get("checkpoint_path", path))
    if path.suffix != ".zip":
        raise ValueError(f"MaskablePPO checkpoints must be .zip files, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def make_policy(policy: str, checkpoint: str | None, deterministic: bool, seed: int):
    if policy == "maskable_ppo":
        return MaskablePPOPolicy(checkpoint, deterministic=deterministic)
    return BaselinePolicy(policy, seed)


def print_scoreboard(state: Mapping[str, Any]) -> None:
    print("\n=== GAME FINISHED ===")
    for player in state.get("players", []):
        score = player.get("score") or {}
        if not score:
            print(f"{player.get('name')} ({player.get('colour')}): no score")
            continue
        print(
            f"{player.get('name')} ({player.get('colour')}): "
            f"final={score.get('final_score', 0.0):.1f} "
            f"pins={score.get('pins_in_goal', 0)} "
            f"dist={score.get('total_distance', 0)} "
            f"moves={score.get('moves', 0)} "
            f"time={score.get('time_taken_sec', 0.0):.2f}s"
        )


def render_state_summary(state: Mapping[str, Any]) -> None:
    print("Board pins:")
    for colour, pins in state.get("pins", {}).items():
        print(f"  {colour}: {pins}")


def run_client(
    name: str,
    policy_name: str,
    checkpoint: str | None,
    host: str,
    port: int,
    deterministic: bool,
    seed: int,
    poll_seconds: float,
    render: bool,
) -> int:
    policy = make_policy(policy_name, checkpoint, deterministic, seed)
    checkpoint_text = f" checkpoint={getattr(policy, 'checkpoint', None)}" if policy_name == "maskable_ppo" else ""
    join = rpc({"op": "join", "player_name": name}, host, port)
    if not join.get("ok"):
        print("JOIN ERROR:", join.get("error"))
        return 1

    game_id = join["game_id"]
    player_id = join["player_id"]
    colour = join["colour"]
    print(f"{name} joined game {game_id} as {colour} using {policy_name}{checkpoint_text}")

    while True:
        state_resp = rpc({"op": "get_state", "game_id": game_id}, host, port)
        state = state_resp.get("state", {})
        if state.get("status") in ("READY_TO_START", "PLAYING"):
            break
        time.sleep(poll_seconds)

    rpc({"op": "start", "game_id": game_id, "player_id": player_id}, host, port)
    last_move_seen = -1

    while True:
        state_resp = rpc({"op": "get_state", "game_id": game_id}, host, port)
        if not state_resp.get("ok"):
            print("STATE ERROR:", state_resp.get("error"))
            return 1

        server_state = state_resp["state"]
        if server_state.get("status") == "FINISHED":
            print_scoreboard(server_state)
            return 0

        move_count = int(server_state.get("move_count", 0))
        if move_count != last_move_seen:
            last_move_seen = move_count
            last_move = server_state.get("last_move")
            if last_move:
                print(
                    f"move {move_count}: {last_move.get('by')} ({last_move.get('colour')}) "
                    f"{last_move.get('from')} -> {last_move.get('to')}"
                )
                if render:
                    render_state_summary(server_state)

        if server_state.get("current_turn_colour") != colour or server_state.get("status") != "PLAYING":
            time.sleep(poll_seconds)
            continue

        legal_resp = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id}, host, port)
        if not legal_resp.get("ok"):
            print("LEGAL MOVE ERROR:", legal_resp.get("error"))
            time.sleep(poll_seconds)
            continue

        legal = legal_moves_from_server(server_state, colour, legal_resp.get("legal_moves", {}))
        if not legal:
            print("No legal moves available.")
            time.sleep(poll_seconds)
            continue

        state = state_from_server(server_state, colour)
        move = policy.select_move(state, legal)
        response = rpc(
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
        if not response.get("ok"):
            print("MOVE REJECTED:", response.get("error"))
            time.sleep(poll_seconds)
            continue

        print(f"{name} played pin {move.pin_id}: {move.from_index} -> {move.to_index}")
        time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MaskablePPO or baseline RLCC server player.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--policy", choices=["maskable_ppo", "heuristic", "random"], default="maskable_ppo")
    parser.add_argument("--checkpoint", default=None, help="MaskablePPO .zip checkpoint or metadata .json.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--stochastic", action="store_true", help="Sample from the PPO policy instead of deterministic play.")
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--render", action="store_true", help="Print pin positions after observed moves.")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    name = args.name or {
        "maskable_ppo": "RLCC-MaskablePPO",
        "heuristic": "RLCC-Heuristic",
        "random": "RLCC-Random",
    }[args.policy]
    return run_client(
        name=name,
        policy_name=args.policy,
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        deterministic=not args.stochastic,
        seed=args.seed,
        poll_seconds=args.poll_seconds,
        render=args.render,
    )


if __name__ == "__main__":
    raise SystemExit(main())
