# Quickstart: RL Chinese Checkers Agent

This quickstart describes the intended workflow after implementation.

## 1. Verify Local Environment

```bash
python -m rl_agent.evaluate --policy heuristic --games 5 --players 2
```

Expected result: evaluation completes without starting the socket server and reports legal move rate, score, pins in goal, distance progress, move count, and timing.

## 2. Run a Short Practice Training Session

```bash
python -m rl_agent.train --algo maskable_ppo --profile practice --episodes 100 --players 2 3 4 --save-every 25
```

Expected result: if optional Gymnasium/Stable-Baselines dependencies are installed, MaskablePPO checkpoints and metrics are written under `checkpoints/` and `metrics/`. Reward components and action-mask statistics are logged by name.

If optional dependencies are unavailable, use the dependency-free fallback:

```bash
python -m rl_agent.train --algo custom_ranker --profile practice --episodes 100 --players 2 3 4 --save-every 25
```

## 3. Continue Training Later

```bash
python -m rl_agent.train --algo maskable_ppo --resume checkpoints/<candidate>.zip --profile extended --episodes 1000
```

Expected result: a new candidate is produced while preserving the earlier candidate and metrics.

## 4. Compare Candidates

```bash
python -m rl_agent.evaluate --candidates checkpoints/* --games 20 --players 2 3 4 5 6 --select-best
```

Expected result: candidates are ranked against random and goal-directed opponents, and a selected tournament candidate is recorded.

## 5. Run Tournament-Compatible Player

Start the provided game manager separately, then run:

```bash
python -m rl_agent.tournament_player --name RLCC-Agent
```

Expected result: the player joins the existing server, waits for turns, selects moves from server-provided legal moves, and submits `(pin_id, to_index)` quickly. If no selected checkpoint exists, the strong heuristic fallback is used.

## 6. Generate Report Inputs

Use metrics files from training and evaluation:

```bash
python -m rl_agent.evaluate --summary metrics/
```

Expected result: report-ready tables summarize average reward, score, moves per game, time per move, pins in goal, distance progress, and reward component totals.
