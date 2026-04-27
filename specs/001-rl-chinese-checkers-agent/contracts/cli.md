# CLI Contract

The implementation exposes student-readable command-line entry points.

## Train

```bash
python -m rl_agent.train [options]
```

Required behavior:
- Runs without the socket server.
- Supports short and extended profiles.
- Supports checkpoint resume.
- Saves candidate metadata and report-ready metrics.
- Logs named reward components.
- Supports optional MaskablePPO training when Gymnasium, Stable-Baselines3, and sb3-contrib are installed.
- Provides a clear error and points to heuristic/custom fallback commands when optional dependencies are missing.

Important options:
- `--episodes <int>`
- `--players <ints...>`
- `--profile practice|extended|curriculum`
- `--algo heuristic|maskable_ppo|custom_ranker`
- `--resume <candidate-path>`
- `--save-every <int>`
- `--seed <int>`

## Evaluate

```bash
python -m rl_agent.evaluate [options]
```

Required behavior:
- Evaluates heuristic fallback and/or saved candidates.
- Evaluates MaskablePPO candidates when available.
- Supports random and goal-directed opponents.
- Supports 2-6 total players.
- Can select the best tournament candidate.
- Writes aggregate metrics.

Important options:
- `--policy heuristic|random|greedy|maskable_ppo|custom_ranker`
- `--candidates <paths...>`
- `--games <int>`
- `--players <ints...>`
- `--select-best`
- `--summary <metrics-dir>`

## Tournament Player

```bash
python -m rl_agent.tournament_player [options]
```

Required behavior:
- Joins the existing server flow.
- Requests legal moves from the server.
- Selects from legal moves only.
- Submits `pin_id` and `to_index`.
- Falls back to heuristic when no selected candidate exists.
- Falls back to heuristic when a selected optional-dependency candidate cannot be loaded.

Important options:
- `--name <player-name>`
- `--checkpoint <candidate-path>`
- `--host <host>`
- `--port <port>`
