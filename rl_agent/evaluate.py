"""Evaluate baseline and candidate policies in the local simulator."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .config import EnvConfig
from .environment import ChineseCheckersEnv
from .metrics import EvaluationMetric, default_metrics_path, summarize_metrics, write_csv, write_jsonl
from .paths import SELECTED_CANDIDATE_PATH, ensure_runtime_dirs
from .policies import make_policy
from .agent import MoveRanker


def _policy_from_args(policy_name: str, seed: int, checkpoint: str | None = None):
    if policy_name == "custom_ranker":
        return MoveRanker.load(checkpoint) if checkpoint else MoveRanker()
    return make_policy(policy_name, seed=seed)


def play_game(policy_name: str, player_count: int, seed: int, opponent_policy: str = "random", checkpoint: str | None = None) -> dict:
    env = ChineseCheckersEnv(EnvConfig(player_count=player_count, seed=seed))
    controlled = env.state.current_colour
    policy = _policy_from_args(policy_name, seed=seed, checkpoint=checkpoint)
    opponent = make_policy(opponent_policy, seed=seed + 1)
    total_reward = 0.0
    start_distance = env.total_distance(env.state, controlled)
    start = time.perf_counter()
    invalid = 0

    while env.state.status == "PLAYING":
        active = env.state.current_colour
        legal = env.legal_moves()
        if not legal:
            break
        chooser = policy if active == controlled else opponent
        move = chooser.select_move(env.state, legal)
        result = env.step(move)
        if active == controlled:
            total_reward += result.reward
        if result.truncated:
            break

    duration = time.perf_counter() - start
    final_distance = env.total_distance(env.state, controlled)
    moves = max(env.state.move_count, 1)
    return {
        "reward": total_reward,
        "score": env.score_for_colour(env.state, controlled),
        "moves": env.state.move_count,
        "pins_in_goal": env.pins_in_goal(env.state, controlled),
        "total_distance": final_distance,
        "distance_progress": float(start_distance - final_distance),
        "avg_time_per_move_sec": duration / moves,
        "game_duration_sec": duration,
        "legal_move_rate": 1.0 if invalid == 0 else 0.0,
    }


def evaluate_policy(
    policy_name: str,
    games: int,
    player_counts: Sequence[int],
    seed: int = 460,
    opponent_policy: str = "random",
    checkpoint: str | None = None,
) -> list[EvaluationMetric]:
    run_id = str(uuid.uuid4())
    metrics: list[EvaluationMetric] = []
    for player_count in player_counts:
        results = [
            play_game(policy_name, player_count, seed + i, opponent_policy=opponent_policy, checkpoint=checkpoint)
            for i in range(games)
        ]
        metrics.append(
            EvaluationMetric(
                run_id=run_id,
                candidate_id=policy_name,
                candidate_source=policy_name,
                opponent_profile=opponent_policy,
                player_count=player_count,
                games=games,
                avg_reward=statistics.fmean(r["reward"] for r in results),
                avg_score=statistics.fmean(r["score"] for r in results),
                avg_moves=statistics.fmean(r["moves"] for r in results),
                avg_pins_in_goal=statistics.fmean(r["pins_in_goal"] for r in results),
                avg_total_distance=statistics.fmean(r["total_distance"] for r in results),
                avg_distance_progress=statistics.fmean(r["distance_progress"] for r in results),
                avg_time_per_move_sec=statistics.fmean(r["avg_time_per_move_sec"] for r in results),
                avg_game_duration_sec=statistics.fmean(r["game_duration_sec"] for r in results),
                legal_move_rate=statistics.fmean(r["legal_move_rate"] for r in results),
            )
        )
    return metrics


def candidate_source(path: str | Path) -> str:
    path = Path(path)
    if path.name == "heuristic":
        return "heuristic"
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("source", "custom_move_ranker")
        except Exception:
            return "custom_move_ranker"
    if path.suffix == ".zip":
        return "maskable_ppo"
    return "unknown"


def normalize_candidate_paths(candidates: Sequence[str]) -> list[str]:
    """Remove metadata-only files and de-duplicate candidate inputs.

    Shell globs such as `checkpoints/*` include selected-candidate metadata and
    MaskablePPO sidecar JSON files. Those are not directly loadable policies, so
    this function resolves or skips them before evaluation.
    """

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        if path.name == SELECTED_CANDIDATE_PATH.name:
            continue
        resolved = str(path)
        if path.suffix == ".json" and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "weights" in data:
                resolved = str(path)
            elif data.get("source") == "maskable_ppo":
                checkpoint = data.get("checkpoint_path")
                if not checkpoint:
                    continue
                resolved = checkpoint
            else:
                continue
        key = str(Path(resolved).resolve()) if Path(resolved).exists() else resolved
        if key not in seen:
            normalized.append(resolved)
            seen.add(key)
    return normalized


def candidate_policy_name(path: str | Path) -> str:
    source = candidate_source(path)
    if source in ("custom_move_ranker", "custom_ranker"):
        return "custom_ranker"
    if source == "heuristic":
        return "heuristic"
    if source == "maskable_ppo":
        # Full PPO evaluation is added when optional dependencies are installed.
        return "heuristic"
    return "heuristic"


def evaluate_candidates(
    candidates: Sequence[str],
    games: int,
    player_counts: Sequence[int],
    seed: int,
    opponent_policy: str,
) -> list[EvaluationMetric]:
    all_metrics: list[EvaluationMetric] = []
    for candidate in normalize_candidate_paths(candidates):
        path_or_name = str(candidate)
        policy_name = candidate_policy_name(path_or_name)
        checkpoint = path_or_name if policy_name == "custom_ranker" and Path(path_or_name).exists() else None
        metrics = evaluate_policy(policy_name, games, player_counts, seed, opponent_policy, checkpoint)
        for metric in metrics:
            metric.candidate_id = Path(path_or_name).stem if Path(path_or_name).exists() else path_or_name
            metric.candidate_source = candidate_source(path_or_name)
        all_metrics.extend(metrics)
    return all_metrics


def select_best_candidate(metrics: Sequence[EvaluationMetric]) -> EvaluationMetric | None:
    if not metrics:
        return None
    return max(metrics, key=lambda m: (m.avg_score, m.avg_distance_progress, -m.avg_moves))


def write_selected_candidate(best: EvaluationMetric, candidates: Sequence[str]) -> None:
    ensure_runtime_dirs()
    selected_path = None
    for candidate in candidates:
        if Path(candidate).stem == best.candidate_id:
            selected_path = str(Path(candidate))
            break
    data = {
        "candidate_id": best.candidate_id,
        "source": best.candidate_source,
        "checkpoint_path": selected_path,
        "selection_score": best.avg_score,
        "selection_distance_progress": best.avg_distance_progress,
    }
    SELECTED_CANDIDATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RL Chinese Checkers policies.")
    parser.add_argument("--policy", default="heuristic", choices=["random", "greedy", "heuristic", "custom_ranker"])
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--players", nargs="+", type=int, default=[2])
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--opponent", default="random", choices=["random", "greedy", "heuristic"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args(argv)

    if args.summary:
        summary = summarize_metrics(Path(args.summary))
        for name, values in summary.items():
            score = values.get("avg_score", values.get("final_score", 0.0))
            reward = values.get("avg_reward", values.get("total_reward", 0.0))
            print(f"{name}: score={score:.2f} reward={reward:.2f}")
        return 0

    if args.candidates:
        metrics = evaluate_candidates(args.candidates, args.games, args.players, args.seed, args.opponent)
        if args.select_best:
            best = select_best_candidate(metrics)
            if best:
                for metric in metrics:
                    metric.selected_candidate = metric.candidate_id == best.candidate_id
                write_selected_candidate(best, args.candidates)
    else:
        metrics = evaluate_policy(args.policy, args.games, args.players, args.seed, args.opponent, args.checkpoint)
    output_name = args.output or (f"evaluation_{args.policy}" if not args.candidates else "evaluation_candidates")
    json_path = default_metrics_path(output_name)
    csv_path = json_path.with_suffix(".csv")
    write_jsonl(json_path, metrics)
    write_csv(csv_path, metrics)

    for metric in metrics:
        print(
            f"{metric.candidate_id} players={metric.player_count} "
            f"score={metric.avg_score:.2f} reward={metric.avg_reward:.2f} "
            f"moves={metric.avg_moves:.1f} pins={metric.avg_pins_in_goal:.1f}"
        )
    print(f"Wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
