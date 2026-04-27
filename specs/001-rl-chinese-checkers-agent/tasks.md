# Tasks: RL Chinese Checkers Agent

**Input**: Design documents from `specs/001-rl-chinese-checkers-agent/`  
**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Included because the plan explicitly requires action-map/action-mask validation and because legal move correctness is tournament-critical.

**Organization**: Tasks are grouped by implementation milestones from the plan and labeled with user stories where applicable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P] tasks in the same phase after dependencies are satisfied.
- **[Story]**: Maps to user stories from `spec.md`.
- **MVP Required**: Needed for a usable practice-tournament agent path.
- **Optional/Performance**: Improves performance or robustness, but core heuristic/evaluation/tournament path should still work without it.

## Phase 1: Setup (Shared Infrastructure) - MVP Required

**Purpose**: Create the new project-owned package and test scaffolding without modifying provided original files.

- [x] T001 Create `rl_agent/__init__.py` with package docstring describing the project-owned RL agent modules
- [x] T002 Create `rl_agent/config.py` with small dataclasses/defaults for colours, player counts, paths, training profiles, and random seeds
- [x] T003 Create `tests/__init__.py` so local tests can import `rl_agent`
- [x] T004 Create `.gitignore` entries for generated `checkpoints/`, `metrics/`, `games/`, `__pycache__/`, and model artifact files
- [x] T005 [P] Create `rl_agent/errors.py` with clear custom exceptions for missing optional dependencies, illegal actions, invalid state, and checkpoint loading failures
- [x] T006 [P] Create `rl_agent/paths.py` with repository-root path helpers for provided code, checkpoints, metrics, and selected-candidate metadata

**Checkpoint**: `python -m unittest discover -s tests` should run without import errors once tests exist.

---

## Phase 2: Foundational Rules Adapter (Blocking Prerequisites) - MVP Required

**Purpose**: Build the project-owned interface to the provided board/pin rules. This blocks all user stories.

- [x] T007 Implement safe imports for provided board and pin classes in `rl_agent/board_adapter.py`
- [x] T008 Implement board metadata helpers in `rl_agent/board_adapter.py` for board size, colours, opposite colours, starting indices, goal indices, axial coordinates, and hex distance
- [x] T009 Implement immutable/lightweight state dataclasses for game state and legal moves in `rl_agent/state.py`
- [x] T010 Implement legal move extraction in `rl_agent/board_adapter.py` that returns project-owned `LegalMove` objects without exposing mutable provided pins directly
- [x] T011 [P] Add board metadata tests in `tests/test_board_adapter.py` for 121 cells, 6 colours, 10 start cells per colour, and opposite-colour mapping
- [x] T012 [P] Add legal move adapter tests in `tests/test_board_adapter.py` confirming returned legal moves match provided pin `getPossibleMoves()` for initial states
- [x] T013 Run `python -m unittest tests.test_board_adapter` and fix adapter issues in `rl_agent/board_adapter.py`

**Checkpoint**: Board adapter reflects the existing rules and no original files are changed.

---

## Phase 3: Milestone 1 - Rules-Compatible Local Environment and State/Action Representation (US1, MVP Required)

**Goal**: A local simulator can run games without the socket server and expose legal actions, transitions, rewards, and terminal status.

**Independent Test**: Start local games with 2-6 players, step through legal moves, and verify no illegal moves occur.

- [x] T014 [US1] Implement `ChineseCheckersEnv` reset/setup logic in `rl_agent/environment.py` for 2-6 total players and deterministic seeding
- [x] T015 [US1] Implement turn order, current-player lookup, and opponent profile hooks in `rl_agent/environment.py`
- [x] T016 [US1] Implement legal action listing in `rl_agent/environment.py` using `rl_agent.board_adapter`
- [x] T017 [US1] Implement state transition and move application in `rl_agent/environment.py` for `(pin_id, to_index)` moves
- [x] T018 [US1] Implement win, draw, max-move, and timeout/truncation status helpers in `rl_agent/environment.py`
- [x] T019 [US1] Implement scoring helpers in `rl_agent/environment.py` mirroring tournament score components
- [x] T020 [P] [US1] Implement state feature extraction in `rl_agent/features.py` for board occupancy, active colour, goal zones, player count, and progress signals
- [x] T021 [P] [US1] Implement named reward components in `rl_agent/rewards.py` for distance progress, goal entry, goal occupancy, regression penalty, jump progress, future jump setup, finish bonus, and invalid/timeout penalty
- [x] T022 [US1] Integrate `rl_agent.rewards` into `rl_agent.environment` step results with `reward_breakdown` in info
- [x] T023 [P] [US1] Add environment reset/step tests in `tests/test_environment.py` for 2-player and 6-player games
- [x] T024 [P] [US1] Add reward breakdown tests in `tests/test_rewards.py` confirming named components sum to total reward
- [x] T025 [US1] Run `python -m unittest tests.test_environment tests.test_rewards` and fix issues in `rl_agent/environment.py` and `rl_agent/rewards.py`

**Checkpoint**: Milestone 1 passes with direct Python API only. No Gymnasium, SB3, or socket server required.

---

## Phase 4: Milestone 2 - Gymnasium Adapter and Legal Action Masks (US1, MVP Required for RL)

**Goal**: Provide a Gymnasium-compatible wrapper and action masks suitable for MaskablePPO while keeping game logic in the local simulator.

**Independent Test**: Action masks exactly match direct environment legal actions after reset and after steps.

- [x] T026 [US1] Implement fixed action id mapping in `rl_agent/action_space.py` for all `(pin_id, to_index)` combinations
- [x] T027 [US1] Implement action id to `LegalMove` conversion and `LegalMove` to action id conversion in `rl_agent/action_space.py`
- [x] T028 [US1] Implement legal action mask creation in `rl_agent/action_space.py` from current direct simulator legal moves
- [x] T029 [P] [US1] Add action mapping tests in `tests/test_action_space.py` for one-to-one mapping, invalid ids, and mask length
- [x] T030 [US1] Implement optional dependency guards in `rl_agent/gym_env.py` so direct simulator still works when Gymnasium is unavailable
- [x] T031 [US1] Implement Gymnasium-compatible `RLChineseCheckersGymEnv` in `rl_agent/gym_env.py` with reset, step, action space, observation space, and info fields
- [x] T032 [US1] Implement MaskablePPO-compatible `action_masks()` method in `rl_agent/gym_env.py`
- [x] T033 [P] [US1] Add Gym wrapper tests in `tests/test_gym_env.py` that skip cleanly when Gymnasium is unavailable
- [x] T034 [P] [US1] Add action-mask validation tests in `tests/test_gym_env.py` comparing `action_masks()` to direct simulator legal moves across reset and several steps
- [x] T035 [US1] Run `python -m unittest tests.test_action_space tests.test_gym_env` and fix mask/action mapping issues in `rl_agent/action_space.py` and `rl_agent/gym_env.py`

**Checkpoint**: Mask validation tests pass before any MaskablePPO training task begins.

---

## Phase 5: Milestone 3 - Baseline Policies and Evaluation Metrics (US1, US2, MVP Required)

**Goal**: Provide random, greedy, and strong heuristic policies plus evaluation metrics before optional learned training.

**Independent Test**: Evaluate heuristic and random policies locally without the socket server and produce metrics files.

- [x] T036 [P] [US1] Implement random legal policy in `rl_agent/policies.py`
- [x] T037 [P] [US1] Implement greedy goal-directed policy in `rl_agent/policies.py`
- [x] T038 [US1] Implement strong heuristic policy in `rl_agent/policies.py` using distance progress, goal entry, regression avoidance, jump progress, and future jump setup
- [x] T039 [P] [US2] Implement metrics row dataclasses and CSV/JSON writers in `rl_agent/metrics.py`
- [x] T040 [US2] Implement local game evaluation loop in `rl_agent/evaluate.py` for one candidate/policy against random and greedy opponents
- [x] T041 [US2] Implement CLI arguments in `rl_agent/evaluate.py` for `--policy`, `--games`, `--players`, `--seed`, and output directory
- [x] T042 [P] [US2] Add policy legality tests in `tests/test_policies.py` confirming random, greedy, and heuristic policies only choose legal moves
- [x] T043 [P] [US2] Add metrics writer tests in `tests/test_metrics.py` confirming required fields from `contracts/metrics.md`
- [x] T044 [US2] Run `python -m rl_agent.evaluate --policy heuristic --games 5 --players 2` and fix evaluation issues in `rl_agent/evaluate.py`
- [x] T045 [US2] Run `python -m unittest tests.test_policies tests.test_metrics` and fix policy/metric issues

**Checkpoint**: A complete heuristic/evaluation path exists and can produce report-friendly metrics.

---

## Phase 6: Milestone 4 - Tournament-Compatible Player Wrapper (US3, MVP Required)

**Goal**: Add a new tournament client that uses selected candidate or heuristic fallback and never modifies provided `player.py`.

**Independent Test**: Against the existing server, the new player joins, waits for turns, requests legal moves, and submits valid `(pin_id, to_index)` moves quickly.

- [x] T046 [P] [US3] Implement JSON RPC helper compatible with provided server protocol in `rl_agent/tournament_player.py`
- [x] T047 [US3] Implement state-to-policy input conversion in `rl_agent/tournament_player.py` using server `pins`, `colour`, and legal moves
- [x] T048 [US3] Implement heuristic fallback move selection in `rl_agent/tournament_player.py` using server-provided legal moves only
- [x] T049 [US3] Implement selected-candidate loading stub in `rl_agent/tournament_player.py` that falls back cleanly when no candidate exists
- [x] T050 [US3] Implement non-interactive main loop in `rl_agent/tournament_player.py` with `--name`, `--host`, `--port`, and `--checkpoint`
- [x] T051 [P] [US3] Add tournament adapter tests in `tests/test_tournament_adapter.py` for legal move filtering and fallback behavior without opening sockets
- [x] T052 [US3] Run `python -m unittest tests.test_tournament_adapter` and fix wrapper issues in `rl_agent/tournament_player.py`

**Checkpoint**: MVP practice-tournament path is available: strong heuristic + evaluation + tournament wrapper.

---

## Phase 7: Milestone 5 - Optional MaskablePPO Training and Checkpointing (US1, US2, Optional/Performance)

**Goal**: Add the main performance-oriented training path while keeping optional dependencies isolated.

**Independent Test**: If Gymnasium/SB3/sb3-contrib are installed, a short MaskablePPO run saves a checkpoint and metrics; if not installed, the command fails with a clear dependency message and other paths still work.

- [x] T053 [US1] Implement optional import/dependency checks for Gymnasium, Stable-Baselines3, and sb3-contrib in `rl_agent/train_maskable_ppo.py`
- [x] T054 [US1] Implement MaskablePPO environment factory in `rl_agent/train_maskable_ppo.py` using `rl_agent.gym_env.RLChineseCheckersGymEnv`
- [x] T055 [US1] Implement MaskablePPO training command in `rl_agent/train_maskable_ppo.py` with `--episodes` or `--timesteps`, `--players`, `--profile`, `--seed`, `--save-every`, and `--resume`
- [x] T056 [US1] Implement checkpoint metadata saving for MaskablePPO candidates in `rl_agent/train_maskable_ppo.py`
- [x] T057 [US2] Extend `rl_agent/evaluate.py` to load and evaluate MaskablePPO candidates when optional dependencies are available
- [x] T058 [P] [US1] Add dependency-missing tests in `tests/test_optional_dependencies.py` for clear MaskablePPO unavailable errors
- [x] T059 [P] [US2] Add MaskablePPO candidate metadata tests in `tests/test_metrics.py`
- [x] T060 [US1] Run `python -m rl_agent.train_maskable_ppo --help` and verify it works or fails only with the planned dependency message

**Checkpoint**: Optional MaskablePPO path is wired, isolated, and does not break MVP paths.

---

## Phase 8: Milestone 6 - Custom Move-Ranking Fallback/Stretch Agent (US1, US2, Optional/Performance)

**Goal**: Provide an understandable custom trained candidate path if time allows or optional RL dependencies are unavailable.

**Independent Test**: A custom ranker can train briefly from heuristic labels or simple updates, save a checkpoint, and be evaluated against baselines.

- [x] T061 [P] [US1] Implement per-move feature vectors in `rl_agent/features.py` for distance delta, goal entry, jump indicators, future jump setup, and congestion
- [x] T062 [US1] Implement simple custom move-ranking model and checkpoint format in `rl_agent/agent.py`
- [x] T063 [US1] Implement custom ranker training mode in `rl_agent/train.py` using heuristic imitation and/or simple reward-weight updates
- [x] T064 [US1] Implement resume support for custom ranker checkpoints in `rl_agent/train.py`
- [x] T065 [US2] Extend `rl_agent/evaluate.py` to load and evaluate custom ranker candidates
- [x] T066 [P] [US1] Add custom ranker save/load tests in `tests/test_agent.py`
- [x] T067 [US1] Run `python -m rl_agent.train --algo custom_ranker --profile practice --episodes 20 --players 2` and fix training issues

**Checkpoint**: Custom learning path can produce and evaluate a candidate independent of MaskablePPO.

---

## Phase 9: Milestone 7 - Curriculum, Self-Play, and Candidate Selection (US1, US2, Optional/Performance)

**Goal**: Improve performance through staged training and robust candidate selection.

**Independent Test**: Multiple candidates can be evaluated and one selected tournament candidate is recorded.

- [x] T068 [US1] Implement curriculum profile definitions in `rl_agent/config.py` for movement practice, baseline-opponent play, mixed player counts, and self-play snapshots
- [x] T069 [US1] Extend `rl_agent/train.py` and `rl_agent/train_maskable_ppo.py` to apply curriculum profiles consistently
- [x] T070 [US2] Implement candidate discovery and metadata loading in `rl_agent/evaluate.py`
- [x] T071 [US2] Implement multi-candidate comparison across random and greedy opponents in `rl_agent/evaluate.py`
- [x] T072 [US2] Implement selected tournament candidate export in `rl_agent/evaluate.py` to `checkpoints/selected_candidate.json`
- [x] T073 [US2] Extend `rl_agent/tournament_player.py` to load `checkpoints/selected_candidate.json` by default
- [x] T074 [P] [US2] Add candidate selection tests in `tests/test_candidate_selection.py`
- [x] T075 [US2] Run `python -m rl_agent.evaluate --candidates checkpoints/* --games 5 --players 2 3 --select-best` and fix candidate selection issues

**Checkpoint**: The project can pick the best available candidate instead of blindly using the newest checkpoint.

---

## Phase 10: Milestone 8 - Report-Friendly Outputs, Documentation, and Quickstart (US4, US5, Required for Delivery)

**Goal**: Make the implementation easy to explain and generate report-ready evidence.

**Independent Test**: A student can run quickstart commands and use outputs for report tables/figures.

- [x] T076 [US4] Add summary generation command in `rl_agent/evaluate.py` for `--summary metrics/`
- [x] T077 [US4] Ensure `rl_agent/metrics.py` writes reward component totals and action-mask validation fields from `contracts/metrics.md`
- [x] T078 [US4] Add report-ready aggregate CSV/JSON output in `rl_agent/metrics.py`
- [x] T079 [US5] Add concise module docstrings to `rl_agent/environment.py`, `rl_agent/action_space.py`, `rl_agent/gym_env.py`, `rl_agent/rewards.py`, `rl_agent/policies.py`, and `rl_agent/train_maskable_ppo.py`
- [x] T080 [US5] Create `rl_agent/README.md` explaining state representation, action masks, reward components, heuristic policy, MaskablePPO path, evaluation, and tournament wrapper
- [x] T081 [US4] Update `specs/001-rl-chinese-checkers-agent/quickstart.md` if implementation commands differ from the planned commands
- [x] T082 [P] [US5] Add explainability smoke test in `tests/test_docs.py` checking key docs files exist and mention reward components/action masks
- [x] T083 [US4] Run quickstart validation commands from `specs/001-rl-chinese-checkers-agent/quickstart.md` where dependencies are available

**Checkpoint**: Implementation is deliverable and report-supporting.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational Rules Adapter**: Depends on Phase 1.
- **Phase 3 Local Environment**: Depends on Phase 2.
- **Phase 4 Gymnasium Adapter and Masks**: Depends on Phase 3.
- **Phase 5 Baselines/Evaluation**: Depends on Phase 3; can start before Phase 4 is complete except metrics fields involving masks.
- **Phase 6 Tournament Wrapper**: Depends on Phase 5 for heuristic fallback.
- **Phase 7 MaskablePPO**: Depends on Phase 4 action-mask tests and Phase 5 metrics.
- **Phase 8 Custom Ranker**: Depends on Phase 5; independent of Phase 7.
- **Phase 9 Curriculum/Selection**: Depends on Phase 5 and benefits from Phase 7 or Phase 8 candidates.
- **Phase 10 Report/Docs**: Can start after Phase 5 but final validation depends on chosen delivery scope.

### User Story Dependencies

- **US1 Train a Competitive Tournament Candidate Locally**: Requires Phases 1-4 for direct/Gym training, Phase 7 or 8 for trained candidates.
- **US2 Select the Best Tournament Candidate**: Requires Phase 5 evaluation and Phase 9 candidate selection.
- **US3 Play Through Tournament Interface**: Requires Phase 6, optionally Phase 9 selected-candidate loading.
- **US4 Collect Report Evidence**: Requires Phase 5 metrics and Phase 10 summaries.
- **US5 Understand and Explain Implementation**: Cross-cutting; Phase 10 finalizes docs and explanation quality.

### MVP Scope

Minimal competitive practice-tournament version:

1. Complete T001-T052.
2. Validate `python -m rl_agent.evaluate --policy heuristic --games 5 --players 2`.
3. Validate `python -m unittest discover -s tests`.
4. Use `python -m rl_agent.tournament_player --name RLCC-Agent` against the provided server.

This MVP intentionally ships a strong heuristic before learned training so there is always a legal, fast tournament-capable agent.

### Performance Scope

Performance improvement path after MVP:

1. Complete T053-T060 for MaskablePPO if dependencies are available.
2. Complete T061-T067 for custom fallback/stretch training if useful.
3. Complete T068-T075 for curriculum and candidate selection.
4. Complete T076-T083 for report-ready delivery.

## Parallel Opportunities

- T005 and T006 can run in parallel after T001-T004.
- T011 and T012 can run in parallel after T007-T010.
- T020 and T021 can run in parallel after T014-T019 interfaces are clear.
- T023 and T024 can run in parallel after T022.
- T029, T033, and T034 can run in parallel after T026-T032.
- T036, T037, and T039 can run in parallel after Phase 3.
- T042 and T043 can run in parallel after T036-T041.
- T058 and T059 can run in parallel after T053-T057.
- T061 and T066 can start while T062-T065 are being implemented if interfaces are agreed.
- T076-T082 can mostly run in parallel after metrics formats settle.

## Parallel Examples

### Milestone 1 Environment

```text
Task: "T020 [P] [US1] Implement state feature extraction in rl_agent/features.py"
Task: "T021 [P] [US1] Implement named reward components in rl_agent/rewards.py"
```

### Milestone 2 Action Masks

```text
Task: "T029 [P] [US1] Add action mapping tests in tests/test_action_space.py"
Task: "T033 [P] [US1] Add Gym wrapper tests in tests/test_gym_env.py"
Task: "T034 [P] [US1] Add action-mask validation tests in tests/test_gym_env.py"
```

### Milestone 3 Baselines

```text
Task: "T036 [P] [US1] Implement random legal policy in rl_agent/policies.py"
Task: "T037 [P] [US1] Implement greedy goal-directed policy in rl_agent/policies.py"
Task: "T039 [P] [US2] Implement metrics row dataclasses and CSV/JSON writers in rl_agent/metrics.py"
```

## Implementation Strategy

### MVP First

1. Build setup and board adapter.
2. Build local environment and reward components.
3. Build action mapping and validate masks.
4. Build heuristic policies and evaluation.
5. Build tournament wrapper with heuristic fallback.
6. Stop and validate MVP before adding optional training dependencies.

### Incremental Delivery

1. Environment works locally.
2. Evaluation works locally with heuristic and random baselines.
3. Tournament wrapper works against provided server.
4. Optional MaskablePPO training works if dependencies are available.
5. Candidate selection picks best trained/heuristic candidate.
6. Report outputs and docs are finalized.

### Notes

- Do not modify provided original files.
- Keep optional imports inside optional modules.
- Legal action masks must pass tests before MaskablePPO training.
- Tournament wrapper must always choose from server-provided legal moves.
- If learned policies underperform, selected candidate may be the strong heuristic.
