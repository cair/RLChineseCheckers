"""Student-readable training entry point for dependency-free policies."""

from __future__ import annotations

import argparse
import time
import uuid
from typing import Sequence

from .agent import MoveRanker, new_candidate_path
from .config import COLOUR_ORDER, DEFAULT_WANDB_PROJECT, EnvConfig, RANDOM_CONTROLLED_COLOUR
from .environment import ChineseCheckersEnv
from .policies import HeuristicPolicy, RandomPolicy
from .wandb_utils import WandbTracker, flatten_for_wandb


def train_custom_ranker(
    episodes: int,
    player_count: int,
    seed: int,
    resume: str | None = None,
    tracker: WandbTracker | None = None,
    profile: str = "practice",
    max_moves: int | None = None,
    controlled_colour: str = "red",
) -> str:
    ranker = MoveRanker.load(resume) if resume else MoveRanker()
    teacher = HeuristicPolicy()
    opponent = RandomPolicy(seed=seed + 10)
    session_id = str(uuid.uuid4())

    for episode in range(episodes):
        env = ChineseCheckersEnv(
            EnvConfig(
                player_count=player_count,
                controlled_colour=controlled_colour,
                seed=seed + episode,
                max_moves=max_moves,
            )
        )
        controlled = env.state.current_colour
        start_distance = env.total_distance(env.state, controlled)
        total_reward = 0.0
        reward_totals: dict[str, float] = {}
        start = time.perf_counter()
        while env.state.status == "PLAYING":
            legal = env.legal_moves()
            if not legal:
                break
            if env.state.current_colour == controlled:
                chosen = ranker.select_move(env.state, legal)
                target = teacher.select_move(env.state, legal)
                ranker.update_toward(env.state, chosen, target)
                result = env.step(chosen)
                total_reward += result.reward
                for key, value in result.info["reward_breakdown"].items():
                    reward_totals[key] = reward_totals.get(key, 0.0) + float(value)
            else:
                env.step(opponent.select_move(env.state, legal))
        duration = time.perf_counter() - start
        final_distance = env.total_distance(env.state, controlled)
        moves = max(env.state.move_count, 1)
        log_data = {
            "training/session_id": session_id,
            "training/profile": profile,
            "training/method": "custom_ranker",
            "episode": episode + 1,
            "episode/total_reward": total_reward,
            "episode/final_score": env.score_for_colour(env.state, controlled),
            "episode/moves": env.state.move_count,
            "episode/pins_in_goal": env.pins_in_goal(env.state, controlled),
            "episode/total_distance": final_distance,
            "episode/distance_progress": start_distance - final_distance,
            "episode/game_duration_sec": duration,
            "episode/avg_time_per_move_sec": duration / moves,
        }
        log_data.update(flatten_for_wandb("reward", reward_totals))
        if tracker:
            tracker.log(log_data, step=episode + 1)

    path = ranker.save(new_candidate_path(), parent_candidate_id=resume, episodes=episodes)
    return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train dependency-free RLCC move ranker.")
    parser.add_argument("--algo", default="custom_ranker", choices=["custom_ranker", "heuristic", "maskable_ppo"])
    parser.add_argument("--profile", default="practice")
    parser.add_argument("--curriculum-stage", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--players", nargs="+", type=int, default=[2])
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--log-every-episodes", type=int, default=200)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--controlled-colour", choices=[*COLOUR_ORDER, RANDOM_CONTROLLED_COLOUR], default=RANDOM_CONTROLLED_COLOUR)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)
    wandb_config = {
        "algo": args.algo,
        "profile": args.profile,
        "curriculum_stage": args.curriculum_stage,
        "episodes": args.episodes,
        "players": args.players,
        "seed": args.seed,
        "resume": args.resume,
        "save_every": args.save_every,
        "log_every_episodes": args.log_every_episodes,
        "max_moves": args.max_moves,
        "controlled_colour": args.controlled_colour,
    }

    if args.algo == "maskable_ppo":
        from .train_maskable_ppo import main as ppo_main

        forwarded = [
            "--episodes",
            str(args.episodes),
            "--players",
            *[str(p) for p in args.players],
            "--profile",
            args.profile,
            "--seed",
            str(args.seed),
            "--wandb-project",
            args.wandb_project,
            "--log-every-episodes",
            str(args.log_every_episodes),
        ]
        if args.max_moves is not None:
            forwarded.extend(["--max-moves", str(args.max_moves)])
        forwarded.extend(["--controlled-colour", args.controlled_colour])
        if args.curriculum_stage:
            forwarded.extend(["--curriculum-stage", args.curriculum_stage])
        if args.save_every:
            forwarded.extend(["--save-every", str(args.save_every)])
        if args.resume:
            forwarded.extend(["--resume", args.resume])
        if args.wandb_run_name:
            forwarded.extend(["--wandb-run-name", args.wandb_run_name])
        if args.no_wandb:
            forwarded.append("--no-wandb")
        return ppo_main(forwarded)
    if args.algo == "heuristic":
        print("Heuristic policy does not require training.")
        return 0

    tracker = WandbTracker(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=wandb_config,
        tags=["custom_ranker", args.profile],
    )
    checkpoint = train_custom_ranker(
        args.episodes,
        args.players[0],
        args.seed,
        args.resume,
        tracker=tracker,
        profile=args.profile,
        max_moves=args.max_moves,
        controlled_colour=args.controlled_colour,
    )
    tracker.finish()
    print(f"Saved custom ranker candidate: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
