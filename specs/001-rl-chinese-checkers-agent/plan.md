# Implementation Plan: RL Chinese Checkers Agent

**Branch**: `001-rl-chinese-checkers-agent` | **Date**: 2026-04-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/001-rl-chinese-checkers-agent/spec.md`

## Summary

Build a tournament-ready Chinese Checkers agent around the existing repository without modifying the provided files. The implementation will add a student-readable local simulator, deterministic baseline policies, evaluation tooling, a Gymnasium-compatible training adapter, an optional Stable-Baselines3 `sb3-contrib` MaskablePPO training path, a tournament-compatible player wrapper, and candidate checkpoint comparison. The plan is heuristic-first for reliability and early practice tournaments, then uses MaskablePPO as the main performance-oriented training path when optional dependencies are available. A simple custom move-ranking agent remains a fallback/stretch path for explainability and dependency resilience.

The design intentionally favors clear modules and explicit data structures over heavy abstraction. If reinforcement learning underperforms, the selected tournament candidate can fall back to the strongest heuristic while training evidence and evaluation comparisons still support the report.

## Technical Context

**Language/Version**: Python 3.x matching the local course environment; current repo runs under Python 3.13 in this workspace.  
**Primary Dependencies**: Existing standard-library game code; NumPy for numeric features; optional Gymnasium for RL environment compatibility; optional Stable-Baselines3 plus `sb3-contrib` for MaskablePPO; optional PyTorch through the RL stack or for a custom fallback move-ranking model. The core simulator, action mapping, reward logic, heuristics, and evaluation must work without Stable-Baselines dependencies.  
**Storage**: Files under new project-owned directories: checkpoints, metrics CSV/JSON, selected-candidate metadata, and evaluation summaries.  
**Testing**: Python unit tests with `pytest` if available, otherwise standard-library `unittest`; smoke-test commands must work without the socket server.  
**Target Platform**: Local student machine running the provided Python Chinese Checkers server/client scripts.  
**Project Type**: Python CLI/tooling plus a tournament-compatible client wrapper.  
**Performance Goals**: 100% legal submitted moves; tournament move choice under 2 seconds for at least 95% of turns; local evaluation across 2-6 total players; short practice training of at least 100 games.  
**Constraints**: Do not modify original provided files; preserve server/client protocol compatibility; implementation must be understandable enough to explain in the final report; RL training must be interruptible and resumable; MaskablePPO dependencies are optional and isolated.  
**Scale/Scope**: Chinese Checkers board with 121 cells, 10 pins per player, 2-6 players, one controlled learning agent, 1-5 opponents, multiple candidate checkpoints.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution file is still the generated placeholder and contains no active project-specific gates. The plan therefore uses the feature specification and repository constraints as the active gates:

- PASS: Original provided project files remain unchanged; new files will be added in separate modules/directories.
- PASS: Tournament compatibility is preserved by matching the existing server player's legal-move request and `(pin_id, to_index)` move submission shape.
- PASS: Understandability is a design goal; modules have clear student-readable responsibilities.
- PASS: Baseline-first plan manages risk if learned policy or optional RL dependencies underperform or are unavailable.
- PASS: Report metrics and named reward components are included from the start.
- PASS: Gymnasium and MaskablePPO integration is isolated from the core rules/reward/evaluation logic to preserve explainability.

## Milestones

### Milestone 1 - Rules-Compatible Local Environment (MVP Required)

Create a local simulator that mirrors the provided board and move rules while staying independent of the socket server. It should expose clear state, legal actions, step/reset behavior, scoring helpers, win/draw detection, and reward components. This is the foundation for training, evaluation, and tests.

### Milestone 2 - Gymnasium Adapter and Action Masks (MVP Required for RL)

Wrap the local simulator in a Gymnasium-compatible environment for RL libraries. Define a fixed discrete action mapping over possible `(pin_id, to_index)` pairs, expose legal action masks for the current state, and add tests that prove masked actions match legal moves from the simulator. This adapter is only a wrapper; game logic and rewards stay project-owned.

### Milestone 3 - Baseline Policies and Evaluation Metrics (MVP Required)

Implement random, greedy goal-directed, and stronger deterministic heuristic policies. The strong heuristic should consider distance progress, pins entering the goal, avoiding regressions, immediate jump length, and future jump-chain opportunities. Add evaluation that records score, reward, moves, timing, pins in goal, distance progress, and reward components.

### Milestone 4 - Tournament-Compatible Player Wrapper (MVP Required)

Add a new player script/wrapper that joins the existing game manager, requests legal moves, chooses quickly using the selected checkpoint or heuristic fallback, and submits `(pin_id, to_index)` without modifying the provided `player.py`.

### Milestone 5 - MaskablePPO Training and Checkpointing (Competitive Core)

Add an optional `sb3-contrib` MaskablePPO training command using the Gymnasium adapter and legal action masks. Save checkpoints with metadata, support resume training, and keep older candidates for comparison. Stable-Baselines is used only as the optimizer/training algorithm; environment dynamics, rewards, action mapping, evaluation, and tournament logic remain owned by this project.

### Milestone 6 - Custom Move-Ranking Agent (Fallback/Stretch)

Implement a simple, explainable move-ranking policy if time allows or if optional MaskablePPO dependencies are unavailable. Prefer an easy-to-explain approach: per-legal-move feature vectors scored by a compact model, trained from heuristic imitation and/or reinforcement updates. This path is secondary to MaskablePPO for performance, but useful for report comparison and dependency resilience.

### Milestone 7 - Curriculum, Self-Play, Candidate Selection (Performance Improvements)

Add staged training: movement practice, baseline-opponent play, mixed player counts, and self-play snapshots. Evaluation should compare candidates across opponent types and player counts, then mark/export the selected tournament candidate. Reward shaping must not blindly favor the longest immediate jump; it should balance short-term progress with future positional value.

### Milestone 8 - Report-Friendly Outputs and Quickstart (Required for Delivery)

Produce clear CSV/JSON summaries and concise docs explaining state features, reward components, baseline policies, training loop, checkpointing, and evaluation. Outputs should support report tables/figures directly.

## Project Structure

### Documentation (this feature)

```text
specs/001-rl-chinese-checkers-agent/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   ├── gym-env.md
│   ├── tournament-client.md
│   └── metrics.md
├── checklists/
│   └── requirements.md
└── tasks.md              # Created later by /speckit-tasks
```

### Source Code (repository root)

```text
rl_agent/
├── __init__.py
├── board_adapter.py       # Imports/reuses provided board + pin rules safely
├── environment.py         # Local reset/step/legal-action simulation
├── action_space.py        # Fixed action id <-> (pin_id, to_index) mapping
├── gym_env.py             # Optional Gymnasium-compatible wrapper + masks
├── features.py            # State and move feature extraction
├── rewards.py             # Named reward components
├── policies.py            # Random, greedy, strong heuristic, checkpoint policy
├── agent.py               # Optional custom trainable move-ranking agent
├── train.py               # Student-readable training entry point
├── train_maskable_ppo.py  # Optional SB3/sb3-contrib training path
├── evaluate.py            # Baseline/candidate comparison and selection
├── tournament_player.py   # Compatible socket client wrapper
├── metrics.py             # CSV/JSON metric recording
└── config.py              # Small dataclasses/defaults

tests/
├── test_environment.py
├── test_action_space.py
├── test_gym_env.py
├── test_policies.py
├── test_rewards.py
├── test_metrics.py
└── test_tournament_adapter.py

checkpoints/               # Generated locally, gitignored if needed
metrics/                   # Generated locally, gitignored if needed
```

**Structure Decision**: Use a single new package, `rl_agent/`, at repository root. This keeps all new work separate from the provided code while making responsibilities obvious for the report. Generated runtime artifacts go under `checkpoints/` and `metrics/`.

## Implementation Strategy

1. Build the local environment first and test it against known properties from the provided code: board size, starting zones, legal moves, opposite goals, win/draw conditions, and score components.
2. Define a fixed action mapping for RL wrappers, then prove that legal masks exactly match simulator legal actions. Invalid actions should be impossible for MaskablePPO and safely rejected in direct APIs.
3. Implement heuristic policies before learned policies. The strong heuristic becomes a fallback, a baseline, and a teacher/scaffold for training.
4. Implement evaluation before training. Every training improvement must be measured against random and heuristic baselines.
5. Add MaskablePPO training as the main performance path, isolated behind optional imports so the project still works without Gymnasium/SB3.
6. Keep a simple custom move-ranking path as fallback/stretch and as an understandable comparison point.
7. Use candidate checkpoints as first-class objects: save metadata, evaluate many candidates, and select one for tournament play.
8. Keep logs report-ready: each episode row includes total reward, score, move count, pins in goal, distance progress, timing, named reward components, and training method.

## Risk Management

- **MaskablePPO dependencies unavailable**: all simulator, heuristic, evaluation, tournament wrapper, and custom fallback pieces still run; document dependency status in metrics.
- **RL underperforms heuristic**: submit the strongest heuristic or best checkpoint selected by evaluation; report the comparison honestly and use the learned policy results as experimentation evidence.
- **Action mask bug permits illegal moves**: block training progress until action-mask validation tests pass; tournament wrapper still only chooses from server-provided legal moves.
- **Training time is too short**: use the short practice run to produce/compare a candidate, then continue later from the saved candidate.
- **Move selection too slow**: fall back to heuristic ranking for tournament play; keep learned model compact and evaluate inference timing.
- **Reward shaping creates greedy jumping**: monitor distance progress, regression penalties, pins in goal, and future jump-chain opportunity metrics separately.
- **Compatibility mismatch with server**: tournament wrapper only submits moves from the server-provided legal move set.
- **Code becomes hard to explain**: keep a direct one-file-per-concept structure and document non-obvious choices near the relevant code.

## Complexity Tracking

No constitution violations. The optional MaskablePPO path adds dependencies, but they are isolated and justified by tournament performance. Core rules, rewards, masks, evaluation, and fallback behavior remain project-owned and explainable.
