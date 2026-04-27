"""Optional MaskablePPO training entry point.

Stable-Baselines is used only as the optimizer. The environment, action masks,
reward components, and evaluation remain project-owned and reportable.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from pathlib import Path
from typing import Sequence

from .config import COLOUR_ORDER, DEFAULT_WANDB_PROJECT, EnvConfig, RANDOM_CONTROLLED_COLOUR
from .errors import OptionalDependencyMissing
from .paths import CHECKPOINT_DIR, ensure_runtime_dirs
from .wandb_utils import WandbTracker


def require_maskable_ppo():
    """Import optional RL dependencies or raise a clear project error."""

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from .gym_env import RLChineseCheckersGymEnv
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise OptionalDependencyMissing(
            "MaskablePPO training requires gymnasium, stable-baselines3, and sb3-contrib. "
            "The heuristic/evaluation/tournament paths still work without them."
        ) from exc
    return MaskablePPO, ActionMasker, RLChineseCheckersGymEnv


def mask_fn(env):
    return env.action_masks()


def checkpoint_stem(model_type: str, curriculum_stage: str, episode: int) -> str:
    safe_stage = re.sub(r"[^A-Za-z0-9_]+", "_", curriculum_stage.strip().lower())
    return f"{model_type}_{safe_stage}_ep{episode}"


def infer_episode_from_checkpoint(path: str | None) -> int:
    if not path:
        return 0
    match = re.search(r"_ep(\d+)", Path(path).stem)
    return int(match.group(1)) if match else 0


def player_count_for_stage(curriculum_stage: str, requested_player_count: int) -> int:
    if curriculum_stage == "empty_board":
        return 1
    return requested_player_count


def opponent_for_stage(curriculum_stage: str) -> str:
    if curriculum_stage == "empty_board":
        return "none"
    if curriculum_stage in ("self_play", "heuristic_self_play"):
        return "heuristic"
    if curriculum_stage == "random_opponents":
        return "random"
    return "heuristic"


def write_metadata(path: Path, metadata: dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_maskable_ppo(
    episodes: int,
    player_count: int,
    seed: int,
    profile: str,
    curriculum_stage: str,
    timesteps: int | None = None,
    resume: str | None = None,
    save_every: int = 0,
    log_every_episodes: int = 200,
    max_moves: int | None = None,
    controlled_colour: str = RANDOM_CONTROLLED_COLOUR,
    tracker: WandbTracker | None = None,
) -> Path:
    MaskablePPO, ActionMasker, RLChineseCheckersGymEnv = require_maskable_ppo()
    ensure_runtime_dirs()
    actual_player_count = player_count_for_stage(curriculum_stage, player_count)
    opponent_policy = opponent_for_stage(curriculum_stage)
    env = RLChineseCheckersGymEnv(
        EnvConfig(
            player_count=actual_player_count,
            controlled_colour=controlled_colour,
            seed=seed,
            max_moves=max_moves,
        ),
        opponent_policy=opponent_policy,
    )
    masked_env = ActionMasker(env, mask_fn)
    from stable_baselines3.common.callbacks import BaseCallback

    start_episode = infer_episode_from_checkpoint(resume)
    target_episode = start_episode + episodes
    effective_timesteps = timesteps if timesteps is not None else max(1, episodes) * env.local_env.config.effective_max_moves

    class EpisodeMetricsCheckpointCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode = start_episode
            self.window = deque(maxlen=log_every_episodes)
            self.last_checkpoint_path: Path | None = None

        def _save_checkpoint(self, episode: int) -> Path:
            path = CHECKPOINT_DIR / f"{checkpoint_stem('maskable_ppo', curriculum_stage, episode)}.zip"
            self.model.save(str(path))
            metadata = {
                "candidate_id": path.stem,
                "source": "maskable_ppo",
                "checkpoint_path": str(path),
                "parent_candidate_id": resume,
                "training_profile": profile,
                "curriculum_stage": curriculum_stage,
                "training_method": "maskable_ppo",
                "episodes_completed": episode,
                "timesteps": self.num_timesteps,
                "player_count": actual_player_count,
                "max_moves": env.local_env.config.effective_max_moves,
                "opponent_policy": opponent_policy,
                "controlled_colour": controlled_colour,
                "actual_controlled_colour": env.local_env.controlled_colour,
                "created_at": time.time(),
                "summary_metrics": self._rolling_means(),
            }
            write_metadata(path, metadata)
            self.last_checkpoint_path = path
            if tracker:
                tracker.log(
                    {
                        "checkpoint/episode": episode,
                        "checkpoint/timesteps": self.num_timesteps,
                        "checkpoint/path": str(path),
                    },
                    step=episode,
                )
            return path

        def _rolling_means(self) -> dict[str, float]:
            if not self.window:
                return {}
            keys = set().union(*(row.keys() for row in self.window))
            means = {}
            for key in keys:
                values = [row[key] for row in self.window if isinstance(row.get(key), (int, float))]
                if values:
                    means[key] = sum(values) / len(values)
            return means

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for done, info in zip(dones, infos):
                if not done:
                    continue
                self.episode += 1
                summary = dict(info.get("episode_summary", {}))
                summary["episode"] = self.episode
                summary["timesteps"] = self.num_timesteps
                self.window.append(summary)
                if tracker:
                    tracker.log({f"episode/{k}": v for k, v in summary.items()}, step=self.episode)
                if log_every_episodes and self.episode % log_every_episodes == 0:
                    rolling = self._rolling_means()
                    if tracker:
                        tracker.log({f"rolling_{log_every_episodes}/{k}": v for k, v in rolling.items()}, step=self.episode)
                    print(
                        f"[{curriculum_stage}] ep={self.episode} "
                        f"score={rolling.get('score', 0.0):.2f} "
                        f"reward={rolling.get('reward', 0.0):.2f} "
                        f"pins={rolling.get('pins_in_goal', 0.0):.2f} "
                        f"dist={rolling.get('total_distance', 0.0):.2f} "
                        f"progress={rolling.get('distance_progress', 0.0):.2f} "
                        f"moves={rolling.get('agent_moves', 0.0):.2f}"
                    )
                if save_every and self.episode % save_every == 0:
                    self._save_checkpoint(self.episode)
                if self.episode >= target_episode:
                    return False
            return True

    if resume:
        model = MaskablePPO.load(resume, env=masked_env)
        reset_num_timesteps = False
    else:
        model = MaskablePPO("MlpPolicy", masked_env, verbose=0, seed=seed)
        reset_num_timesteps = True
    callback = EpisodeMetricsCheckpointCallback()
    model.learn(total_timesteps=effective_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)

    final_episode = callback.episode
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_stem('maskable_ppo', curriculum_stage, final_episode)}.zip"
    model.save(str(checkpoint_path))
    metadata = {
        "candidate_id": checkpoint_path.stem,
        "source": "maskable_ppo",
        "checkpoint_path": str(checkpoint_path),
        "parent_candidate_id": resume,
        "training_profile": profile,
        "curriculum_stage": curriculum_stage,
        "training_method": "maskable_ppo",
        "episodes_completed": final_episode,
        "timesteps": effective_timesteps,
        "player_count": actual_player_count,
        "max_moves": env.local_env.config.effective_max_moves,
        "opponent_policy": opponent_policy,
        "controlled_colour": controlled_colour,
        "created_at": time.time(),
        "summary_metrics": callback._rolling_means(),
    }
    write_metadata(checkpoint_path, metadata)
    if tracker:
        tracker.log(
            {
                "checkpoint/source": "maskable_ppo",
                "checkpoint/path": str(checkpoint_path),
                "checkpoint/timesteps": effective_timesteps,
                "checkpoint/player_count": actual_player_count,
                "checkpoint/max_moves": env.local_env.config.effective_max_moves,
                "checkpoint/episode": final_episode,
                "checkpoint/curriculum_stage": curriculum_stage,
                "checkpoint/controlled_colour": controlled_colour,
            },
            step=final_episode,
        )
    return checkpoint_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train optional MaskablePPO RL Chinese Checkers agent.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--players", nargs="+", type=int, default=[2])
    parser.add_argument("--profile", default="practice")
    parser.add_argument("--curriculum-stage", default=None, choices=["empty_board", "self_play", "heuristic_self_play", "random_opponents"])
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--log-every-episodes", type=int, default=200)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--controlled-colour", choices=[*COLOUR_ORDER, RANDOM_CONTROLLED_COLOUR], default=RANDOM_CONTROLLED_COLOUR)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)

    curriculum_stage = args.curriculum_stage or ("empty_board" if args.profile == "empty_board" else "self_play")
    timesteps = args.timesteps
    tracker = WandbTracker(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config={
            "algo": "maskable_ppo",
            "timesteps": timesteps,
            "episodes": args.episodes,
            "players": args.players,
            "profile": args.profile,
            "curriculum_stage": curriculum_stage,
            "seed": args.seed,
            "resume": args.resume,
            "save_every": args.save_every,
            "log_every_episodes": args.log_every_episodes,
            "max_moves": args.max_moves,
            "controlled_colour": args.controlled_colour,
        },
        tags=["maskable_ppo", args.profile, curriculum_stage],
    )
    try:
        checkpoint = train_maskable_ppo(
            episodes=args.episodes,
            player_count=args.players[0],
            seed=args.seed,
            profile=args.profile,
            curriculum_stage=curriculum_stage,
            timesteps=timesteps,
            resume=args.resume,
            save_every=args.save_every,
            log_every_episodes=args.log_every_episodes,
            max_moves=args.max_moves,
            controlled_colour=args.controlled_colour,
            tracker=tracker,
        )
    except OptionalDependencyMissing as exc:
        print(str(exc))
        tracker.finish()
        return 2
    tracker.finish()
    print(f"Saved MaskablePPO candidate: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
