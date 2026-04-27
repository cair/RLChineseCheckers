"""Optional Gymnasium wrapper around the project-owned local simulator."""

from __future__ import annotations

from typing import Any

from .action_space import ACTION_SPACE_SIZE, action_to_legal_move, legal_action_mask
from .config import EnvConfig
from .environment import ChineseCheckersEnv
from .errors import IllegalActionError, OptionalDependencyMissing
from .features import state_features
from .policies import GreedyPolicy, HeuristicPolicy, RandomPolicy


try:  # Optional by design.
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - depends on local environment
    gym = None
    spaces = None


def require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise OptionalDependencyMissing(
            "Gymnasium is not installed. Direct simulator, heuristic evaluation, "
            "and tournament play still work; install gymnasium for RL wrappers."
        )


class RLChineseCheckersGymEnv(gym.Env if gym is not None else object):
    """Gymnasium-compatible environment with MaskablePPO action masks."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, opponent_policy: str = "heuristic"):
        require_gymnasium()
        import numpy as np

        self.local_env = ChineseCheckersEnv(config)
        self.controlled_colour = self.local_env.controlled_colour
        self.opponent_policy_name = opponent_policy
        self.opponent_policy = self._make_opponent_policy(opponent_policy)
        self.episode_reward = 0.0
        self.episode_agent_moves = 0
        self.episode_reward_components: dict[str, float] = {}
        self.episode_start_distance = self.local_env.total_distance(self.local_env.state, self.controlled_colour)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        obs_len = len(state_features(self.local_env.state))
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(obs_len,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.local_env.reset(seed=seed)
        else:
            self.local_env.reset()
        self.controlled_colour = self.local_env.controlled_colour
        self.episode_reward = 0.0
        self.episode_agent_moves = 0
        self.episode_reward_components = {}
        self.episode_start_distance = self.local_env.total_distance(self.local_env.state, self.controlled_colour)
        self._play_opponents_until_controlled_or_done()
        return self._obs(), self._info()

    def step(self, action: int):
        legal = self.local_env.legal_moves()
        move = action_to_legal_move(int(action), legal)
        if move is None:
            raise IllegalActionError(f"Action id {action} is not legal for the current state")
        result = self.local_env.step(move)
        self.episode_reward += result.reward
        self.episode_agent_moves += 1
        for key, value in result.info.get("reward_breakdown", {}).items():
            if isinstance(value, (int, float)):
                self.episode_reward_components[key] = self.episode_reward_components.get(key, 0.0) + float(value)
        terminated = result.terminated
        truncated = result.truncated
        info = dict(result.info)
        if not terminated and not truncated:
            terminated, truncated = self._play_opponents_until_controlled_or_done()
        if terminated or truncated:
            final_distance = self.local_env.total_distance(self.local_env.state, self.controlled_colour)
            info["episode_summary"] = {
                "reward": self.episode_reward,
                "agent_moves": self.episode_agent_moves,
                "total_moves": self.local_env.state.move_count,
                "score": self.local_env.score_for_colour(self.local_env.state, self.controlled_colour),
                "pins_in_goal": self.local_env.pins_in_goal(self.local_env.state, self.controlled_colour),
                "total_distance": final_distance,
                "distance_progress": self.episode_start_distance - final_distance,
                "status": self.local_env.state.status,
                "controlled_colour": self.controlled_colour,
            }
            for key, value in self.episode_reward_components.items():
                info["episode_summary"][f"reward_{key}"] = value
        return self._obs(), result.reward, terminated, truncated, info

    def action_masks(self):
        return legal_action_mask(self.local_env.legal_moves())

    def _obs(self):
        import numpy as np

        return np.asarray(state_features(self.local_env.state), dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        legal = self.local_env.legal_moves()
        return {
            "legal_move_count": len(legal),
            "move_count": self.local_env.state.move_count,
            "current_colour": self.local_env.state.current_colour,
            "controlled_colour": self.controlled_colour,
            "opponent_policy": self.opponent_policy_name,
        }

    def _make_opponent_policy(self, name: str):
        if name == "random":
            return RandomPolicy(seed=self.local_env.config.seed + 1000)
        if name == "greedy":
            return GreedyPolicy()
        return HeuristicPolicy()

    def _play_opponents_until_controlled_or_done(self) -> tuple[bool, bool]:
        while (
            self.local_env.state.status == "PLAYING"
            and self.local_env.state.current_colour != self.controlled_colour
        ):
            legal = self.local_env.legal_moves()
            if not legal:
                return True, False
            result = self.local_env.step(self.opponent_policy.select_move(self.local_env.state, legal))
            if result.terminated or result.truncated:
                return result.terminated, result.truncated
        return self.local_env.state.status != "PLAYING", self.local_env.state.status == "TRUNCATED"
