# RL Chinese Checkers Agent

This package contains the project-owned implementation for training and running
an RL Chinese Checkers agent without modifying the provided course files.

## Main Modules

- `board_adapter.py`: imports the provided board and pin rules, then exposes a
  small immutable interface for the rest of the project.
- `environment.py`: direct local simulator used for training and evaluation.
- `action_space.py`: fixed action id mapping from discrete actions to
  `(pin_id, to_index)` moves.
- `gym_env.py`: optional Gymnasium wrapper and action masks for MaskablePPO.
- `rewards.py`: named reward components for report-friendly reward design.
- `policies.py`: random, greedy, and strong heuristic policies.
- `train_maskable_ppo.py`: optional Stable-Baselines3/sb3-contrib training path.
- `train.py`: dependency-free custom move-ranker training path.
- `evaluate.py`: local evaluation, candidate comparison, and summaries.
- `tournament_player.py`: client wrapper compatible with the provided server.

## State Representation

The local state stores participating colours, pin indices by colour, turn order,
current turn index, move count, status, and the last move. Feature extraction is
kept in `features.py` so the report can describe exactly what the agent sees.
The PPO observation is colour-relative: the active/controlled player is always
encoded in the first player slot, then opponents follow turn order. This is
important because the tournament server may assign the agent any colour.

## Action Masks

The fixed action space maps every `(pin_id, to_index)` pair to one integer.
`action_space.py` builds a boolean mask where `True` means the move is legal in
the current state. `gym_env.py` exposes this through `action_masks()` for
MaskablePPO. Tests compare these masks against the direct simulator legal moves.

## Reward Components

Rewards are split into named components:

- `distance_progress`
- `goal_entry`
- `move_penalty`
- `regression_penalty`
- `jump_progress`
- `future_jump_setup`
- `finish_bonus`
- `invalid_or_timeout_penalty`

The active reward keeps the original shaping terms for distance progress,
regression penalties, useful jumps, and future jump setup. The anti-hacking
change is that `goal_entry` is only paid once per pin per episode, so a single
pin cannot repeatedly leave and re-enter goal to collect entry reward. Terminal
bonus is based on how many pins are actually in goal when the episode ends. The
local simulator truncates episodes at the tournament training cap of 100 moves
for each player/agent by default; use `--max-moves` to override this for
experiments.

## Training Paths

The intended high-performance path is optional MaskablePPO when Gymnasium,
Stable-Baselines3, and sb3-contrib are installed. If they are unavailable, the
heuristic and custom move-ranker paths still work.

### MaskablePPO Curriculum Commands

The first curriculum stage can train on an empty board, meaning only the
controlled colour's pins are present:

```bash
python -m rl_agent.train_maskable_ppo \
  --curriculum-stage empty_board \
  --episodes 1000 \
  --save-every 1000 \
  --log-every-episodes 20 \
  --controlled-colour random
```

The next stage can resume from that checkpoint and train with automated
heuristic opponents:

```bash
python -m rl_agent.train_maskable_ppo \
  --curriculum-stage self_play \
  --episodes 100000 \
  --players 2 \
  --resume checkpoints/maskable_ppo_empty_board_ep1000.zip \
  --save-every 5000 \
  --log-every-episodes 200 \
  --controlled-colour random
```

`--controlled-colour random` is the default for MaskablePPO training. Keep it
enabled for tournament checkpoints so the model learns all colour assignments.

To force a different per-agent local truncation length:

```bash
python -m rl_agent.train_maskable_ppo \
  --curriculum-stage empty_board \
  --episodes 1000 \
  --max-moves 160
```

Checkpoints are named as:

```text
checkpoints/maskable_ppo_<curriculum_stage>_ep<episode>.zip
```

The `self_play` stage currently means the learning agent plays against the
project heuristic opponent between its turns. Full model-vs-model snapshot
self-play can be added later if needed.

## Weights & Biases Tracking

Training commands try to use Weights & Biases when `wandb` is installed. The
default W&B project is `rlcc`. The logged configuration includes algorithm,
profile, episode/timestep budget, player counts, seed, resume path, and save
interval.

Custom ranker training logs per-episode metrics:

- total reward
- final score
- move count
- pins in goal
- total distance
- distance progress
- game duration
- average time per move
- named reward components

MaskablePPO training logs the run configuration and PPO timestep progress when
the optional RL dependencies are installed. Terminal progress prints rolling
means for score, reward, pins in goal, total distance, distance progress, and
controlled-agent moves over the configured episode window.

Disable W&B with:

```bash
python -m rl_agent.train --algo custom_ranker --episodes 100 --players 2 --no-wandb
```

Use a custom project/run name with:

```bash
python -m rl_agent.train --algo custom_ranker --episodes 100 --players 2 \
  --wandb-project rlcc --wandb-run-name practice-custom-ranker
```

## Tournament Play

`tournament_player.py` uses the provided server protocol. It requests legal
moves from the server and submits only a legal `(pin_id, to_index)` pair. If no
selected candidate is available, it falls back to the strong heuristic policy.
