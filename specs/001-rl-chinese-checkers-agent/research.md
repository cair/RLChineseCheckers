# Research: RL Chinese Checkers Agent

## Decision: Heuristic-first implementation

**Rationale**: A strong deterministic heuristic provides immediate tournament-safe behavior, a fallback when no checkpoint exists, a baseline for evaluation, and an explainable scaffold for training. This reduces the risk of relying on unstable reinforcement learning from scratch under limited time.

**Alternatives considered**:
- Pure self-play from scratch: higher research appeal but high risk and likely unstable within the time limit.
- Random baseline only: easy but poor tournament performance.
- Black-box RL library: faster to try but harder to explain and may not fit the dynamic legal-action structure cleanly.

## Decision: Gymnasium adapter with fixed action mapping and legal masks

**Rationale**: A Gymnasium-compatible wrapper lets the project use standard RL tooling while keeping the core simulator project-owned. MaskablePPO needs a fixed discrete action space and a per-state legal action mask. Mapping action ids to `(pin_id, to_index)` pairs keeps the mask testable and lets the tournament wrapper translate model choices back to the provided server protocol.

**Alternatives considered**:
- Train only through a custom Python API: simpler but gives up safer standard PPO tooling.
- Use an unmasked fixed action space: would waste learning capacity on illegal moves and risks invalid training behavior.
- Put rules directly inside a Gym environment: works, but hides game logic inside an RL wrapper and makes heuristic/evaluation reuse harder.

## Decision: Optional Stable-Baselines3 `sb3-contrib` MaskablePPO path

**Rationale**: MaskablePPO is a safer performance-oriented choice for large dynamic legal-action games than a fully hand-rolled policy-gradient implementation. The course allows Stable-Baselines usage, and this plan keeps the reportable parts project-owned: environment, action mapping, legal masks, reward design, curriculum, baselines, evaluation, and candidate selection.

**Alternatives considered**:
- From-scratch PPO: more report credit but too risky and time-consuming.
- DQN-style fixed action learning: possible, but large invalid action space and sparse rewards make it less reliable.
- Heuristic-only submission: reliable but less compelling as an RL project and may underperform trained agents.

## Decision: Custom move-ranking remains fallback/stretch

**Rationale**: Chinese Checkers legal moves vary by board state and pin. A custom ranking model over legal moves remains useful if optional dependencies are unavailable, and it is easy to explain in the report. It is secondary to MaskablePPO for performance, but can compare well against heuristics and provide a student-readable learning baseline.

**Alternatives considered**:
- Fixed global action space without masking: requires considering many invalid actions and is less reliable.
- Predict destination only: cannot choose which pin to move cleanly when many pins can reach the same or similar destinations.

## Decision: Named reward components

**Rationale**: The final report needs reward design evidence. Separate named components make reward shaping debuggable and explainable: distance progress, goal entry, regression penalty, immediate jump value, future jump-chain opportunity, and finishing progress.

**Alternatives considered**:
- Single scalar reward only: simpler but opaque and harder to tune.
- Only final score reward: sparse and likely too slow for limited training.

## Decision: Candidate checkpoint comparison

**Rationale**: The latest checkpoint is not always the strongest. Evaluation across random and goal-directed opponents, multiple player counts, and report metrics gives an empirical basis for selecting the tournament candidate.

**Alternatives considered**:
- Always use latest checkpoint: simpler but risky.
- Manual selection by inspecting logs: less repeatable and weaker for the report.

## Decision: Local simulator instead of socket-server training

**Rationale**: Training through the socket server would be slow, hard to parallelize, and harder to control. A local simulator based on the same board and move rules enables faster training/evaluation while the tournament wrapper preserves server compatibility.

**Alternatives considered**:
- Train directly through server/client: maximum compatibility but too slow for many episodes.
- Reimplement all rules from scratch without using provided code: risks subtle rule mismatch.

## Decision: Student-readable package structure with optional dependency isolation

**Rationale**: The implementation must support the final report. One clear module per concept makes the code easier to understand: environment, action space, Gym wrapper, features, rewards, policies, train, MaskablePPO train, evaluate, tournament player, metrics. Optional RL library imports stay in dedicated modules so the rest of the project remains runnable.

**Alternatives considered**:
- Highly generic RL framework design: reusable but harder to explain.
- Single large script: quick initially but difficult to test and report.
